# SnowBuddy Runtime Contract

This folder is loaded by SnowBuddy at runtime:

- `SNOWBUDDY_CHARACTER.md` defines identity, voice, grounding, and boundaries.
- `BLIND_GUI_READ.md` is the current nonvisual map of the Studio interface.

The character file is behavioral instruction. The GUI file is reference data.
The application also supplies a live UI-state snapshot for the active page and,
after training, compact deterministic findings calculated from the latest saved
run artifacts. SnowBuddy explains those grounded facts instead of relying on a
separate What This Means results panel.

When the GUI changes, update `BLIND_GUI_READ.md` and refresh its `UI source
SHA-256`, `Results UI source SHA-256`, and `Theme source SHA-256` values. The
contract test intentionally fails if a GUI source changes without that review.
