import tempfile
import unittest
from pathlib import Path

from studio.assistant import LIGHTWEIGHT_MODEL, SnowBuddyService
from studio.project_store import ProjectStore
from studio.settings import (
    load_appearance_mode,
    load_studio_settings,
    save_appearance_mode,
    studio_settings_path,
    update_studio_settings,
)
from studio.theme import COLORS, DARK_COLORS, LIGHT_COLORS


class StudioSettingsTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / ".test_runs"
        test_root.mkdir(exist_ok=True)
        self.temp_dir = tempfile.TemporaryDirectory(dir=test_root)
        self.library_root = Path(self.temp_dir.name) / "library"
        self.library_root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_appearance_defaults_to_light_for_missing_or_invalid_settings(self):
        self.assertEqual(load_appearance_mode(self.library_root), "light")

        update_studio_settings(
            self.library_root,
            {"ui": {"appearance_mode": "system"}},
        )
        self.assertEqual(load_appearance_mode(self.library_root), "light")

        studio_settings_path(self.library_root).write_text(
            "not valid json",
            encoding="utf-8",
        )
        self.assertEqual(load_appearance_mode(self.library_root), "light")

    def test_dark_appearance_persists_and_preserves_assistant_settings(self):
        update_studio_settings(
            self.library_root,
            {"assistant": {"model": LIGHTWEIGHT_MODEL}},
        )

        save_appearance_mode(self.library_root, "dark")
        payload = load_studio_settings(self.library_root)

        self.assertEqual(load_appearance_mode(self.library_root), "dark")
        self.assertEqual(payload["assistant"]["model"], LIGHTWEIGHT_MODEL)

    def test_model_change_preserves_appearance_setting(self):
        store = ProjectStore(self.library_root)
        save_appearance_mode(self.library_root, "dark")

        SnowBuddyService(store, model=LIGHTWEIGHT_MODEL).set_model(
            LIGHTWEIGHT_MODEL
        )

        self.assertEqual(load_appearance_mode(self.library_root), "dark")
        self.assertEqual(
            SnowBuddyService(store).model,
            LIGHTWEIGHT_MODEL,
        )

    def test_core_palette_tokens_have_light_and_dark_values(self):
        for name in (
            "app_bg",
            "surface",
            "sidebar",
            "ink",
            "border",
            "primary",
            "control",
            "chat_user",
            "chat_assistant",
        ):
            self.assertEqual(
                COLORS[name],
                (LIGHT_COLORS[name], DARK_COLORS[name]),
            )
            self.assertNotEqual(LIGHT_COLORS[name], DARK_COLORS[name])


if __name__ == "__main__":
    unittest.main()
