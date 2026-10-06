"""One home for atomic file and directory replacement.

Every durable write in the Studio stages its output beside the destination and
then swaps it in with a single `os.replace`, which is atomic on both POSIX and
Windows. On Windows that swap can fail transiently: an indexer, a virus
scanner, or a preview handler may hold a brief handle on the file or folder, and
the call raises `PermissionError` (WinError 5) or a sharing violation (WinError
32) even though nothing is wrong.

A short bounded retry absorbs that without changing the semantics: the swap is
still one atomic operation, and a lock that is not transient still raises the
original error rather than being retried away.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


# Five attempts spend at most 0.375 s sleeping (0.025 + 0.05 + 0.1 + 0.2), which
# covers the scanner-sized window without making a genuine failure feel hung.
ATOMIC_REPLACE_ATTEMPTS = 5
ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS = 0.025


def replace_with_retry(source: Path | str, destination: Path | str) -> None:
    """Atomically move `source` onto `destination`, retrying transient locks.

    Works for a staged file and for a staged directory. On Windows a directory
    swap additionally requires `destination` not to exist, which is the caller's
    business and is left to raise.

    `FileNotFoundError` is never retried: a missing source or a missing
    destination parent is a logic or cleanup error, not contention, so it fails
    immediately instead of waiting out the full backoff.

    After the final attempt the original exception propagates unchanged, so a
    persistent permission problem stays visible with its own type and message.
    """

    for attempt in range(ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except FileNotFoundError:
            raise
        except OSError:
            if attempt == ATOMIC_REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(ATOMIC_REPLACE_INITIAL_BACKOFF_SECONDS * (2**attempt))
