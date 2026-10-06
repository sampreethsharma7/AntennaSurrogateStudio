"""Atomic replacement must survive a transient Windows lock, and only that.

A staged write is swapped in with a single `os.replace`. On Windows a scanner or
indexer can hold a brief handle and make that call raise `PermissionError`
(WinError 5) although nothing is wrong, which is how a training run came to fail
with "Access is denied" while promoting its run directory.

These tests pin the three properties that matter: a transient lock is absorbed,
a persistent one is still reported with its own type and message after a bounded
number of attempts, and an error that is never transient is not waited out.
"""

from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio.atomic_replace import (
    ATOMIC_REPLACE_ATTEMPTS,
    ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS,
    replace_with_retry,
)
from studio.dataset_registry import register_dataset
from studio.dataset_validation import validate_dataset
from studio.model_training import ModelTrainingRequest, submit_model_training_request
from studio.parser_engine import TrainingRequest
from studio.project_store import ProjectStore


def failing_replace(failures: int, *, only_directories: bool = False):
    """A real replace that raises a transient lock the first `failures` times.

    `only_directories` narrows the injection to a staged directory swap. A
    training run performs six replacements and only the last promotes the run
    directory, so without the filter the lock would land on an earlier manifest
    write and the test would not exercise the site it names.
    """

    real = os.replace
    attempts = {"count": 0}

    def replace(source, destination):
        if only_directories and not Path(source).is_dir():
            return real(source, destination)
        attempts["count"] += 1
        if attempts["count"] <= failures:
            raise PermissionError(5, "simulated transient file lock")
        return real(source, destination)

    return replace


class ReplaceWithRetryTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _staged_file(self, name: str = "staged.txt", content: str = "new") -> Path:
        staged = self.root / name
        staged.write_text(content, encoding="utf-8")
        return staged

    def _staged_directory(self, name: str = "staging") -> Path:
        staged = self.root / name
        staged.mkdir()
        (staged / "payload.txt").write_text("new", encoding="utf-8")
        return staged

    def test_a_transient_lock_is_absorbed(self):
        staged = self._staged_file()
        destination = self.root / "destination.txt"
        destination.write_text("old", encoding="utf-8")

        with (
            patch("studio.atomic_replace.os.replace", side_effect=failing_replace(1)) as replace,
            patch("studio.atomic_replace.time.sleep") as sleep,
        ):
            replace_with_retry(staged, destination)

        self.assertEqual(destination.read_text(encoding="utf-8"), "new")
        self.assertEqual(replace.call_count, 2)
        sleep.assert_called_once()
        self.assertFalse(staged.exists())

    def test_a_staged_directory_is_promoted_through_a_transient_lock(self):
        """The case that actually failed: promoting a staged run directory."""

        staged = self._staged_directory()
        destination = self.root / "run-0001"

        with (
            patch("studio.atomic_replace.os.replace", side_effect=failing_replace(2)) as replace,
            patch("studio.atomic_replace.time.sleep") as sleep,
        ):
            replace_with_retry(staged, destination)

        self.assertTrue(destination.is_dir())
        self.assertEqual(
            (destination / "payload.txt").read_text(encoding="utf-8"), "new"
        )
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertFalse(staged.exists())

    def test_a_lock_that_clears_on_the_last_attempt_still_succeeds(self):
        staged = self._staged_file()
        destination = self.root / "late.txt"

        with (
            patch(
                "studio.atomic_replace.os.replace",
                side_effect=failing_replace(ATOMIC_REPLACE_ATTEMPTS - 1),
            ) as replace,
            patch("studio.atomic_replace.time.sleep"),
        ):
            replace_with_retry(staged, destination)

        self.assertEqual(destination.read_text(encoding="utf-8"), "new")
        self.assertEqual(replace.call_count, ATOMIC_REPLACE_ATTEMPTS)

    def test_a_persistent_lock_is_reported_not_retried_away(self):
        staged = self._staged_file()
        destination = self.root / "locked.txt"
        destination.write_text("old", encoding="utf-8")

        with (
            patch(
                "studio.atomic_replace.os.replace",
                side_effect=PermissionError(5, "simulated persistent file lock"),
            ) as replace,
            patch("studio.atomic_replace.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(PermissionError, "persistent file lock"):
                replace_with_retry(staged, destination)

        # Bounded, and the destination is left exactly as it was.
        self.assertEqual(replace.call_count, ATOMIC_REPLACE_ATTEMPTS)
        self.assertEqual(sleep.call_count, ATOMIC_REPLACE_ATTEMPTS - 1)
        self.assertEqual(destination.read_text(encoding="utf-8"), "old")

    def test_a_missing_source_fails_immediately(self):
        """Never transient, so it must not wait out the whole backoff."""

        with (
            patch("studio.atomic_replace.os.replace", side_effect=FileNotFoundError(2, "no such file")) as replace,
            patch("studio.atomic_replace.time.sleep") as sleep,
        ):
            with self.assertRaises(FileNotFoundError):
                replace_with_retry(self.root / "absent", self.root / "destination")

        self.assertEqual(replace.call_count, 1)
        sleep.assert_not_called()

    def test_the_backoff_is_bounded_and_grows(self):
        with (
            patch(
                "studio.atomic_replace.os.replace",
                side_effect=PermissionError(5, "locked"),
            ),
            patch("studio.atomic_replace.time.sleep") as sleep,
        ):
            with self.assertRaises(PermissionError):
                replace_with_retry(self.root / "a", self.root / "b")

        delays = [call.args[0] for call in sleep.call_args_list]
        self.assertEqual(
            delays,
            [
                ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS * (2**index)
                for index in range(ATOMIC_REPLACE_ATTEMPTS - 1)
            ],
        )
        self.assertLess(sum(delays), 1.0)

    def test_no_retry_is_spent_when_nothing_is_locked(self):
        staged = self._staged_file()
        destination = self.root / "clean.txt"

        with patch("studio.atomic_replace.time.sleep") as sleep:
            replace_with_retry(staged, destination)

        self.assertEqual(destination.read_text(encoding="utf-8"), "new")
        sleep.assert_not_called()


class TrainingRunPromotionTests(unittest.TestCase):
    """The originally failing path, end to end through the real backend."""

    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.store = ProjectStore(Path(self.temp_dir.name) / "library")
        self.project = self.store.create_project("Atomic Replace")
        self._register_dataset()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _register_dataset(self) -> None:
        prepared = self.project.path / "data" / "prepared"
        input_path = prepared / "inputs.csv"
        output_path = prepared / "outputs.csv"
        with input_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "P2", "P3"])
            for index in range(1, 21):
                writer.writerow([f"Design_{index:03d}", float(index), float((index * 2) % 7)])
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Sample ID", "gain"])
            for index in range(1, 21):
                writer.writerow(
                    [f"Design_{index:03d}", 2.0 * index + 3.0 * ((index * 2) % 7) + 1.0]
                )
        register_dataset(
            self.project.path,
            validate_dataset(
                TrainingRequest(
                    input_csv_path=input_path,
                    output_csv_path=output_path,
                    feature_columns=["P2", "P3"],
                    target_columns=["gain"],
                    sample_id_column="Sample ID",
                )
            ),
        )

    def _train(self):
        return submit_model_training_request(
            ModelTrainingRequest(
                model_name="linear_regression",
                training_mode="auto",
                search_level="medium",
            ),
            project_path=self.project.path,
        )

    def test_training_survives_a_transient_lock_on_the_run_directory(self):
        with (
            patch(
                "studio.atomic_replace.os.replace",
                side_effect=failing_replace(1, only_directories=True),
            ),
            patch("studio.atomic_replace.time.sleep"),
        ):
            run = self._train()

        self.assertTrue(run.success, run.error_message)
        self.assertTrue(run.run_directory.is_dir())
        self.assertEqual(run.run_directory.name, run.run_id)
        self.assertTrue(run.model_artifact_path.is_file())

    def test_training_still_reports_a_persistent_lock(self):
        with (
            patch(
                "studio.atomic_replace.os.replace",
                side_effect=failing_replace(
                    ATOMIC_REPLACE_ATTEMPTS, only_directories=True
                ),
            ),
            patch("studio.atomic_replace.time.sleep"),
        ):
            run = self._train()

        self.assertFalse(run.success)
        self.assertIn("transient file lock", run.error_message)


if __name__ == "__main__":
    unittest.main()
