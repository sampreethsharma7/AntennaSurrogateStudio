# Antenna Surrogate Studio Development Contract

## SnowBuddy GUI awareness

Any change that modifies `studio/ui.py`, `studio/theme.py`, visible labels,
navigation, layout, dialogs, user-facing states, or interaction behavior must
review and update `snowbuddy/BLIND_GUI_READ.md` in the same change.

After the GUI review:

1. Correct all affected descriptions in `BLIND_GUI_READ.md`.
2. Recompute SHA-256 for `studio/ui.py` and `studio/theme.py`.
3. Replace the two hash values at the top of `BLIND_GUI_READ.md`.
4. Run `python -m unittest discover -s tests -v`.

Do not weaken or remove the GUI-contract test to bypass this requirement.

Changes to SnowBuddy’s identity, tone, scope, or response rules must also update
`snowbuddy/SNOWBUDDY_CHARACTER.md`.
