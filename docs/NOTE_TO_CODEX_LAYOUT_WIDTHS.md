# Note to Codex — parameter table widths (commit `892fc5b`)

This documents a change made directly on top of `85f73d8`, why the previous
approach kept failing, and what is deliberately left for you.

## What was wrong

`85f73d8` fixed reachability but reintroduced the horizontal clipping that
`3e9c517` had fixed. The scrollbar needed ~24px in a pane already at its
minimum, and that width was taken out of the columns:

| | `3e9c517` | `85f73d8` |
|---|---|---|
| `INPUT_COLUMN_MIN_WIDTHS` | 140, 150, 90, 90 | 70, 140, 66, 66 |
| label width cap | 190 | 90 |
| px-per-character estimate | 11 | 6 |

`CONFIGURATION_MIN_WIDTH` was left at 522 — the one number that should have
moved.

Measured against the real fonts:

```
body_small = Segoe UI 17
  SlotRadius   108 px   (10.8 px/char)
  PatchW        78 px   (13.0 px/char)
  Max           44 px   (14.7 px/char)
mono = Cascadia Mono 16
  "LOW / VALUE" 132 px
```

Nothing in that table is under 10.5 px/char, so 6 px/char was low by about
half. `SlotRadius` was given `width=68, wraplength=66` for text needing 108px
and broke into `SlotRadi` / `us`. The 66px entries could not fit their own
`Max` placeholder. `LOW / VALUE` needed 132px in a 66px column, which is why a
hand-placed `\n` had to be inserted into the heading.

## Root cause — two of them

**1. Character counts are not text widths.** The estimate was wrong twice in
opposite directions (11/char too wide in `3e9c517`, 6/char too narrow in
`85f73d8`). Both times a constant was tuned until the screenshots looked
acceptable, which is why the same defect kept moving rather than closing.

**2. CustomTkinter scales `width=`; Tk's grid `minsize` is raw.** This is the
subtler one and it is easy to reintroduce. `CTkBaseClass` multiplies a widget's
`width` and `wraplength` by the current widget scaling, while
`grid_columnconfigure(minsize=...)` goes straight to Tk unscaled. Setting both
from the same number therefore drifts by the scaling factor: at 1.5 a label
asked for 110 and rendered 165, overflowing its own 120px column and widening
that row alone. That is what staggered the columns. Tuple fonts are *not*
scaled (only `CTkFont` is), so text never grew to match.

## What the code does now

**`studio/theme.py`**

- `text_width(font_key, text)` — measured width in unscaled pixels, via
  `tkinter.font.Font.measure`, with a cached font per key. Falls back to a
  per-character estimate only when no Tk root exists, and drops a cached font
  whose root has been destroyed.
- `widest_text_width(font_key, texts, *, padding, minimum, maximum)` — the
  width that fits every string, clamped.
- `column_safe_width(widget, target)` — divides by
  `ScalingTracker.get_widget_scaling(widget)` so an explicit `width=` renders
  at `target` rather than `target × scaling`.

**`studio/inverse_design_ui.py`**

- `configuration_width_for(label_width)` derives the pane width from the column
  budget. `CONFIGURATION_MIN_WIDTH` is now `configuration_width_for(120)` = 524
  rather than a literal.
- `_apply_configuration_min_width()` raises the pane's `minsize` at runtime
  when measured names need a wider name column, then re-clamps the sash. A long
  name widens the pane; it never squeezes a column.
- `INPUT_COLUMN_PADDING` is shared by the heading grid and every row grid, and
  the heading row is inset by `INPUT_HEADING_SCROLL_INSET` so both grids have
  identical usable width. Verified: heading and cell centres now match exactly
  (192/192, 327/327, 455/455, 561/561).
- Heading is `LOW` and `HIGH`, no embedded newline. A Fixed row's single field
  carries its own `Value` placeholder, which is what the removed `/ VALUE` was
  trying to say.
- Card prose wraps at `CONFIGURATION_TEXT_WRAP_WIDTH` instead of clipping.

**`studio/ui.py`** — sidebar icons restored to pictorial marks.

## About the sidebar monograms

Replacing all eight glyphs overshot, and the premise was partly mine: I wrote
that four were tofu. It was **two** — `⌁` U+2301 (Text / CAD Design) and `⌾`
U+233E (Inverse Design). The other six rendered correctly in the same
screenshots.

`ST/DS/DP/MT/TR/ML/IN/ID` costs more than it fixed. Expanded, the button reads
`ML    Model Library` with a redundant prefix. Collapsed, `ML` reads as
*machine learning* in a machine-learning tool, and `IN`/`ID` and `DS`/`DP` are
mutually confusable.

Worth knowing before you touch these again — I checked all of this by rendering
and diffing pixels, not by reasoning about fonts:

- **None of the eight glyphs are native to Segoe UI.** All eight rely on
  fallback to Segoe UI Symbol. That the rail worked at all was luck.
- **`Font.measure()` is not a coverage test.** It returns a plausible 27px for
  U+233E, which drew as a box.
- **The cmap is not a coverage test either.** All eight are present in
  `seguisym.ttf`, including the two that failed.
- **I could not reproduce the tofu on this machine.** All eight render here in
  a `CTkButton`, at the real nav font, DPI-aware, through the canvas path.
  The capture environment evidently lacks the fallback the dev machine has.

So the two that failed now use characters native to Segoe UI itself, needing no
fallback: `∆` U+2206 for Text / CAD Design and `↔` U+2194 for Inverse Design.
The native symbol set is thin (`← ↑ → ↓ ↔ ∂ ∆ ∏ ∑ √ ∞ ∩ ∫ ≈ ≠ ≡ ≤ ≥ ■ □ ▪ ▫ ▬
▲ ► ▼ ◄ ◊ ○ ◌ ● ◦ ❶–❿`), so if more marks ever need replacing, the durable fix
is to bundle an icon font rather than keep hunting codepoints.

The test that asserted `icon.isascii() and icon.isalpha()` locked in the
monograms. It now asserts what actually matters: one character per
destination, all distinct, label carried by tooltip and accessible name.

## Verification

- Full suite: **957 tests pass**.
- New regression tests in `tests/test_inverse_design_page.py`:
  - `test_input_table_gives_every_control_room_for_its_own_text` — no label,
    heading or placeholder is narrower than its own measured text, no heading
    contains a newline, no row overflows its host.
  - `test_long_names_widen_the_pane_rather_than_the_columns` — the pane
    minimum grows with the measured label width and stops at the cap.
- Screenshots re-captured at 100% and 150% in
  `docs/screenshots/e2e_run_03`.
- `snowbuddy/BLIND_GUI_READ.md` updated with the sidebar and inverse-inputs
  descriptions, and all four affected source hashes recomputed.

One fix to the capture harness (`.test_runs/capture_group_b.py`, gitignored):
it collapsed the sidebar and SnowBuddy *before* `set_project`, and loading a
project re-docked SnowBuddy. The 100% captures were of a squeezed page. It now
collapses after the project loads.

## Left for you

**1. Inference has the same unreachable-row defect that `85f73d8` fixed for
Inverse Design.** `studio/inference_ui.py` still paginates: `input_host` is a
plain `CTkFrame` and the pager only appears above `INPUTS_PER_PAGE = 8`. With
exactly 8 inputs in a short window the last row is cut off with no pager, no
scrollbar and no cue — visible in `inference_after_100.png`, where `ArrayDy` is
missing.

I implemented the scrollable fix and **reverted it**. It is not a
drop-in mirror of the inverse page: Inference lays its cards out in a plain
grid with `content.grid_columnconfigure(0, weight=0, minsize=270)`, so widening
the input card takes width directly from the plot, and
`test_laptop_layout_keeps_snowbuddy_and_plot_settings_clear_of_actions` asserts
the plot canvas stays ≥ `MIN_USABLE_PREDICTION_PLOT_WIDTH` (420). Adding the
scrollbar dropped it to 351. Budget that width explicitly — the way
`configuration_width_for` now does for Inverse Design — rather than letting the
card's natural width absorb it.

Two things that cost me time and will cost you the same:

- The old test mutates the class-shared `active_book.feature_columns` and never
  restores it. It only worked because its name sorted *after* the laptop test.
  Restore the columns in `addCleanup`, and re-run `_refresh()` there.
- A narrow window leaves the plot workbench's internal sash clamped, and that
  position persists into later tests in the same class.

**2. The plot toolbar hint clips at narrow widths.** "Move over a curve to
inspect X and Y." renders as "…e to inspect X and Y." in
`inference_after_150.png`. Pre-existing; it needs the same wrap-or-shorten
treatment the empty-plot message got.

**3. Two scrollbar constants in `inverse_design_ui.py` are close but not
equal** — `INPUT_SCROLLBAR_ALLOWANCE = 24` (the bar's own width, for the pane
budget) and `INPUT_HEADING_SCROLL_INSET = 17` (what the host withholds from its
rows, for heading alignment). Both were measured against a built page. If
CustomTkinter's scrollable frame changes, re-measure rather than assume they
should be the same number.

## The general rule

Measure, do not estimate — and when a widget's width has to agree with a grid
`minsize`, put the width through `column_safe_width` so CustomTkinter's scaling
does not silently break the agreement. Three passes moved a hardcoded pixel
constant; that is the cycle this change is meant to end.
