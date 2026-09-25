# Parametric Antenna Builder — UI and UI-architecture review

**Date:** 2026-09-24
**Reviewer:** Claude (Opus 5), reviewing as a first-time user, then as a returning one
**Basis:** `artifacts/antenna_builder_after_vtk_full.png`, the maximized 1361×756 capture,
a full read of `studio/antenna_builder_ui.py` (1470 lines), and `snowbuddy/BLIND_GUI_READ.md`.
Companion to `BUILDER_USER_TRIAL_REPORT.md`, which covers agent behaviour.
**Code changed:** none.

Second pass. The first pass reviewed the screenshots and the layout constants; it missed the
conversation surface and the dialog behaviour, which turn out to contain the worst problems in
the feature. Those are §5 and §6. Corrections to the first pass are marked.

Goal: right controls, right place, right size, nothing on screen that isn't earning its
position.

---

## 0. What is already right — do not touch

- **The two-pane split.** Describe on the left, see the result on the right.
- **The Design Start page.** Two cards, clear titles, clear consequences, one accented.
  Cleanest screen in the feature.
- **The 3D preview.** Big, clean, honest control hints, and a caption strip
  (`FR4 · 2.45 GHz · 1 element · 1 validated port`).
- **The amber limitation line** under the viewer ("No corporate feed network is generated").
- **`Enter` adds a line · `Ctrl+Enter` applies**, stated inline.
- **The threading model is correct.** Planning and native CST export run on daemon threads and
  marshal results back with `self.after(0, …)`. The stale-response guard
  (`if self.session is not planned_session`) works. This is the right pattern and it is
  properly implemented — see §7 for what is missing around it, not wrong in it.

---

## 1. First run: the guidance exists, but is hidden where nobody will see it

**Correction to the first pass.** I wrote that there were no starter examples. There are —
`antenna_builder_ui.py:1246`:

> *Builder: Try "Create an inset-fed rectangular patch at 2.45 GHz on FR4," "Create a circular
> patch," or "Create a simple dipole at 915 MHz."*

Three genuinely good examples. But:

- they render **inside the Conversation tab**, and `tabs.add("Parameters")` is called first, so
  **Parameters is the default tab and Conversation is hidden**;
- they are **plain text in a disabled `CTkTextbox`**, so they cannot be clicked;
- the *visible* empty state, in the preview pane, says only *"No antenna design yet. Describe
  the antenna you want to create."*

So the work is done and then placed where a first-time user will not find it. The fix is
placement, not authoring: surface those examples in the default view, make them clickable to
fill the composer, and add the scope line that is genuinely absent:

```
Supported: 3 antenna families · arrays to 16×16 · circular and rectangular
slots · corner cutouts
Not supported: horns, Vivaldi, spirals, feed networks, solver runs
```

That last line matters more than any polish. This system refuses `conductor radius` on a
circular patch because the word belongs to the dipole. Three refusals in a row read as "broken"
unless scope was set first.

---

## 2. Planner configuration outranks the actual task

The left card reads, top to bottom:

```
DESIGN REQUEST                          0 chars · 1 line
Provider:  [Local Ollama    ▾]   [✓] Free only  [Refresh]
Model:     [qwen3:8b · Tested            ▾]
Private/offline: models are discovered from local Ollama; design context stays
┌──────────────────────────────────────────────────────┐
│                  (the actual text box)               │
└──────────────────────────────────────────────────────┘
```

Five pieces of infrastructure above the thing the user came to do. Specific faults:

1. **"Free only" is visible while Local Ollama is selected**, where it does nothing — it is an
   OpenRouter catalogue filter.
2. **The privacy notice is truncated mid-sentence** — *"…design context stays"*. The words "on
   this computer" are clipped. This is the one line telling a user whether their design leaves
   the machine.
3. **Provider and Model are two full-width rows** for a setting that changes about once.

The older capture had a single `Planner: [Local Qwen ▾]` line. This area grew as backends were
added — fine as research surface, not as shipping surface.

**Fix:** collapse to one status line under the composer, clickable to expand —
`Local · qwen3:8b · stays on this computer  ⚙`. Cloud swaps the third segment to
`Gemini · sent to Google`. Render "Free only" only when OpenRouter is active.

---

## 3. The parameter table — the actual workspace — is a porthole

At 1361×756 the table shows **two rows**, the second clipped mid-row. An inset patch with a
modifier and a composed slot has **eighteen** parameters.

The cause is structural: the left pane stacks composer + planner rows + tabs + material row +
table header + table + `LHS:` summary in one column's vertical budget. The table is last in
line. §9.1 shows the arithmetic.

**Fix:** move the parameter table into the right pane, under the preview it describes. Full
spec in §9.

---

## 4. What the agent just did is clipped into the footer

The footer status reads:

```
er_circle_cutouts_v1 applied; Corner radius / patch width to 0.25.
```

The beginning is cut off. This is the result of the user's action, rendered as a truncated
single line squeezed between a back button and three export buttons — and it is replaced on
the next action, so nothing in view records what the design became.

---

## 5. The conversation surface is broken in four independent ways

This is the section the first pass missed, and it is the most serious. The builder is a
conversational tool whose conversation surface is inverted, non-scrolling, unstyled, and
destructive of user input.

### 5.1 The composer sits above the transcript

`_build_editor` places the instruction frame at `row=1` and the tabview containing the
conversation at `row=2`. So the input is **above** the history.

Every chat interface in existence — and every one your users have habits from — puts history
above and the input anchored at the bottom. The reason is not fashion: reading flows downward,
so the newest message should be adjacent to where you type your reply. Here the newest message
is as far from the composer as the layout allows.

### 5.2 There is no autoscroll — at all

`grep -nE "yview|\.see\(|moveto"` over the entire 1470-line file returns **nothing**.

`_render_conversation` rebuilds the transcript with `delete("1.0","end")` then re-inserts every
message. After a rebuild Tk leaves the viewport at line 1. So **after every single turn, the
user is looking at the top of the conversation and must scroll down manually to read the
response they just waited for.**

The longer the session, the further they scroll. This alone makes long sessions — the thing you
are building toward — unpleasant.

### 5.3 The transcript is flat undifferentiated text

```python
self.conversation_box.insert("end", f"{speaker}: {message['content']}\n\n")
```

A single disabled `CTkTextbox`. No role styling, no indentation, no separators. A success, a
clarification, and a refusal are visually identical — `Builder: …` in every case. The user
cannot scan the history to find where things went wrong, and cannot tell at a glance whether
the last turn changed the design or declined to.

The change summaries the executor produces (`created SlotRadius`, `subtracted from patch`,
`composition replayed`) are exactly what should be rendered here as a distinct, quieter row
type. Instead they go to the truncating footer label of §4.

### 5.4 The user's text is destroyed on clarify and refuse

`_llm_plan_failed` calls `self.instruction_var.set("")` before showing the dialog.

So: you type three careful sentences, the agent asks *"What target width?"* — and **your three
sentences are gone**. You must retype them to add the one missing number. `apply_instruction`
also never disables `instruction_entry`, so anything typed while the model is working is wiped
by the same line.

Clarification is not failure. It is the single most common non-executing outcome — in the
agent trial it fired on *"Make the patch a bit wider"* and *"Add a slot"*, both reasonable
things to say. Treating it as a failure path that clears state is the wrong model.

---

## 6. Modal dialogs interrupt the conversation

`_llm_plan_failed` ends with:

```python
messagebox.showwarning(title, str(exc), parent=self)
```

Clarify, refuse, and genuine errors all take this path. So when the agent asks a question, the
user gets a **modal Win32 warning dialog** titled *"Clarification needed"*, with a warning icon
and an OK button, which must be dismissed before they can type the answer — into a box that has
already been emptied.

A conversational tool stopping the conversation to ask its question in an alert box is the
worst interaction in the feature. Two further points:

- **A clarification is not a warning.** `PlannerClarificationRequired` gets the same
  `showwarning` treatment — icon and all — as a hard failure.
- **Dismissing it drops you at the top of the conversation** (§5.2), so the question you just
  dismissed is now off-screen.

There are **seven** `messagebox` call sites. Triage:

| Site | Current | Should be |
|---|---|---|
| Clarification | `showwarning` modal | **transcript entry**, `?` glyph, composer focused, text preserved |
| Refusal | `showwarning` modal | **transcript entry**, `✕` glyph, text preserved |
| Plan error | `showwarning` modal | **transcript entry**, `✕` glyph |
| Empty instruction | `showwarning` modal | **nothing** — or a one-line inline hint. A modal to say the visible empty box is empty is gratuitous |
| No project open | `showwarning` modal | inline; also disable Apply when no project |
| Export succeeded | `showinfo` modal | defensible — it reports file paths and a caution. Could be a non-modal toast with a "show in folder" action |
| Export failed | `showerror` modal | keep modal |

Only the last two earn a dialog. The conversational outcomes belong in the conversation.

---

## 7. UI-layer architecture

The agent architecture is strong (see the trial report). The UI layer has four issues worth
naming, in descending order.

### 7.1 Conversational outcomes are modelled as exceptions

`PlannerClarificationRequired` and `PlannerRefusal` are raised, caught by a handler literally
named `_llm_plan_failed`, and rendered with a warning icon. But `clarify` and `refuse` are
**normal terminal outcomes of the agent protocol** — `AgentTerminalResult.outcome` is one of
`finished | clarify | refuse`, all valid, none exceptional.

The exception encoding is why the UI treats a question as a failure: it clears input, switches
tabs, and alerts. Everything in §5.4 and §6 follows from this one modelling choice.

`execute_builder_turn` already returns a structured `BuilderTurnResult` with
`terminal_result.outcome`. The UI could branch on that value and render three outcomes as three
transcript styles, reserving exceptions for genuine faults — transport failures, capability
errors, malformed state. This is the highest-leverage change in the UI layer, because §5.4, §6,
and half of §5.3 collapse into it.

### 7.2 Full widget teardown on every state change

`_rebuild_parameter_table` does `for child in …winfo_children(): child.destroy()` and recreates
every row. Eighteen parameters × 4 widgets ≈ **72 widgets destroyed and rebuilt per turn**, plus
on every material change and every debounced field edit.

Consequences beyond cost: focus and cursor position are lost, so editing several parameters in
sequence is jumpy; the `selected`/`sweep_vars` state has to be manually salvaged and restored
across the rebuild (the function opens by doing exactly that), which is a recurring source of
subtle bugs. Diffing against existing rows — or rebuilding only when the *recipe* changes rather
than when any value changes — would remove both.

### 7.3 No cancel, and no busy state beyond a button label

The only progress signal is `apply_button.configure(text="Planning...")`. There is no
cancel path anywhere in the file. In the agent trial, Gemini turns took 5–22 s; a local model on
a large design will be far slower, and §9 of the trial report shows the planner payload is
already ~17–22 k tokens, so local turns will get slower as designs grow.

A user who realises mid-turn that they phrased something wrong has no way to stop it, and no
indication of whether anything is happening beyond three static dots in a label. Meanwhile the
composer stays editable and whatever they type will be erased.

### 7.4 Planner construction happens inside the worker

`_create_planner()` is called on the worker thread. A missing `GEMINI_API_KEY` therefore
surfaces as a modal warning *after* a round-trip delay, rather than being known when the
provider is selected. Validating credentials at provider-selection time — and reflecting it in
the §2 status line — would turn a late failure into an upfront one.

---

## 8. Smaller things, in order of cost to a user

| | Problem | Change |
|---|---|---|
| a | **No revision or undo affordance.** Design carries a revision and full conversation; UI shows neither. Only recourse for an unwanted change is describing the inverse. | Show `rev 5` in a design identity strip. State to support stepping back already exists. |
| b | **Two CST buttons, unclear difference.** | One primary `Create CST project`, `Export .bas script` demoted to a `▾` menu item. |
| c | **Three footer buttons compete with the primary action.** | One accented primary; group the CST outputs. |
| d | **`LHS: PatchL, PatchW, Inset`** is small grey text far from the button it governs. | Move to the footer beside the LHS button as `3 selected for sweep`. |
| e | **Header chrome.** `3 VALIDATED RECIPES · CST ADAPTER` means nothing to a newcomer. | Keep `EXPERIMENTAL`; drop the other two; recipe count moves into the scope text of §1. |
| f | **Unlabeled left icon rail**, eight icons, no text. | Tooltips at minimum. |
| g | **Developer-facing subtitle.** | Cut to *"Geometry is deterministic. Electromagnetic performance still requires simulation."* |

---

## 9. Concrete layout specification

Reuses existing tokens in `studio/theme.py` and sizes already in `antenna_builder_ui.py`.
**No changes to `theme.py`** — no new fonts, colours, or radii.

### 9.1 Why the table is two rows today

The split pane occupies roughly **y = 155 → 690, so 535 px**, at 1361×756. The left column
spends it:

| Element | Height |
|---|---:|
| Card title "Describe the design" | 39 |
| `DESIGN REQUEST` header row | 24 |
| Provider row (`height=28` + pad) | 34 |
| Free-only / Refresh row | 28 |
| Model row (`height=28` + pad) | 34 |
| Privacy notice | 20 |
| Composer textbox (`INSTRUCTION_COMPOSER_MIN_HEIGHT`) | 92 |
| Hint + Apply row (`height=34` + pad) | 46 |
| Tabview header | 30 |
| Substrate material row | 40 |
| `PARAMETER / VALUE / VARY` header | 22 |
| `LHS:` summary | 20 |
| Card padding | ~25 |
| **Subtotal** | **454** |
| **Left for the table** | **~81** |

Row pitch is `CTkEntry(height=30)` + `pady=3` = **36 px**. 81 ÷ 36 = **2 rows**, second clipped.
Matches the screenshot. No trimming inside that column reaches ten rows; the table must leave it.

### 9.2 Pane geometry

Keep `PanedWindow` and `sashwidth=8`. The panes swap roles, so:

| | Current | Proposed |
|---|---|---|
| Left `minsize` / `width` | 480 / 535 | **420 / 480** |
| Right `minsize` | 580 | **660** |

Minimum window ≈ 420 + 8 + 660 + ~110 chrome = **~1200 px**, versus ~1170 today. Still inside
1366 with ~165 px slack. Above 1366, prefer growing the **right** pane; the transcript does not
improve past ~520 px of width.

### 9.3 Left pane — the conversation column

Transcript above, composer anchored at the bottom. This is §5.1 fixed.

```
┌─ LEFT PANE (420 min / 480 start) ────────────────────┐
│  Inset-fed rectangular patch  ·  rev 5        [⤺]    │  34 px  fixed
│──────────────────────────────────────────────────────│   1 px
│                                                      │
│   you   Add a 3 mm circular slot at the centre       │
│   ✓     created SlotRadius                           │  FLEX
│         subtracted from patch                        │  min 160 px
│                                                      │  scrolls
│   you   Make it a bit wider                          │  AUTOSCROLLED
│   ?     What target width, in mm?                    │  to bottom
│                                                      │
│   you   Make it a 1×2 array                          │
│   ✓     array columns → 2 · composition replayed     │
│                                                      │
│──────────────────────────────────────────────────────│   1 px
│  ┌────────────────────────────────────────────────┐  │
│  │ Describe a change…                             │  │  92 → 154 px
│  └────────────────────────────────────────────────┘  │  (constants unchanged)
│   Local · qwen3:8b · stays on this computer  ⚙       │  28 px
│                                        [  Apply  ]   │  36 px
└──────────────────────────────────────────────────────┘
```

Budget: 34 + 160 + 92 + 28 + 36 + ~20 = **370 px** against 535. Transcript absorbs the surplus —
~325 px at rest, 8–10 exchanges.

**Transcript specification**

- `CTkScrollableFrame`, **scrolled to bottom after every render** — the missing behaviour of
  §5.2. One call after rebuild.
- Role gutter 44 px. `you` in `FONTS["mono"]` / `COLORS["muted"]`.
- Outcome glyph per §5.3: `✓` `COLORS["success"]` executed · `?` `COLORS["warning"]` clarify ·
  `✕` `COLORS["danger"]` refuse · `·` `COLORS["muted"]` completed-without-change.
- Message body `FONTS["body_small"]` (16), **wrapped, never truncated** — replaces the clipped
  footer label of §4.
- Change summaries as a quieter sub-row: `FONTS["caption"]` (15), `COLORS["muted"]`, indented to
  the gutter.
- Row pitch 22, block gap 10.

**Composer**

- Keep `INSTRUCTION_COMPOSER_MIN_HEIGHT = 92` / `MAX = 154` — that growth already works.
- Placeholder `Describe a change…`, or `Describe the antenna you want…` at rev 0.
- **Preserve text on clarify and refuse** (§5.4). Clear only on `executed`.
- **Disable during planning**, re-enable on completion.
- **Apply becomes Cancel while planning** (§7.3) — same slot, no new control.
- Grow Apply from `106×34` to **`112×36`**; it currently reads smaller than the footer's
  secondary buttons (`162×40`), inverting the hierarchy.

**Status line** — 28 px, `FONTS["caption"]`, `COLORS["muted"]` local / `COLORS["warning"]` cloud.
Gear 28×28 expands the existing Provider/Model/Refresh widgets at their **current** sizes
(`150×28` / `320×28` / `70×28`); nothing about those widgets is wrong except permanent visibility.

**Empty state (rev 0)** occupies the transcript, which is empty anyway — zero layout cost. The
examples of §1, promoted here and made clickable:

```
   Start from one of these, or describe your own:
   ┌────────────────────────────────────────────────────┐   34 px each
   │  Inset-fed rectangular patch at 2.45 GHz on FR4    │   4 px gap
   │  Probe-fed circular patch at 5.8 GHz on RT5880     │   surface_alt fill
   │  Centre-fed dipole at 915 MHz                      │   FONTS["body_small"]
   │  Patch at 2.45 GHz as a 1×4 array, 0.55λ spacing   │   click → fills composer
   └────────────────────────────────────────────────────┘
   Supported: 3 antenna families · arrays to 16×16 · circular and
   rectangular slots · corner cutouts
   Not supported: horns, Vivaldi, spirals, feed networks, solver runs
```

4×34 + 3×4 + ~70 = **218 px**, inside the ~325 px available. Scope text `FONTS["caption"]`,
`COLORS["muted"]`.

### 9.4 Right pane — the design column

```
┌─ RIGHT PANE (660 min) ───────────────────────────────────────┐
│  Live deterministic geometry                  [ Reset view ] │  40 px  fixed
│  ┌────────────────────────────────────────────────────────┐  │
│  │                    VTK preview                         │  │  FLEX
│  │              Left-drag orbit · Wheel zoom              │  │  min 240, weight 3
│  └────────────────────────────────────────────────────────┘  │
│  FR4 · 2.45 GHz · 2 elements · 2 validated ports             │  22 px
│  No array feed network is generated.                         │  20 px
│──────────────────────────────────────────────────────────────│   1 px
│  PARAMETER              VALUE          VARY                  │  22 px  sticky
│  Substrate material     [ FR4     ▾ ]    —                   │
│  Frequency              [ 2.45  ] GHz    ☐                   │  FLEX
│  Substrate thickness    [ 1.6   ] mm     ☐                   │  min 200, weight 2
│  Patch length           [ 28.81 ] mm     ☑                   │  scrolls
│  Patch width            [ 38.0  ] mm     ☑                   │
│  …                                                           │
└──────────────────────────────────────────────────────────────┘
```

Two elements move, both improvements in their own right:

- **Substrate material becomes the first table row** rather than a separate 40 px row. It *is*
  a parameter; it sits outside only because it renders as a dropdown. Vary cell shows `—`.
- **The `LHS:` summary moves to the footer**, beside the button it describes.

**Row pitch 36 → 30 px:** `CTkEntry(height=30)` → `26`, `pady=3` → `2`. At 16 pt a 26 px entry is
still comfortable; 6 px per row is a 20 % capacity gain where it is scarce.

**Budget.** Fixed: toolbar 40 + caption 22 + limitation 20 + divider 1 + table header 22 +
padding 20 = **125 px**. Rest is flexible.

| Window | Pane height | Preview | Table | Rows |
|---|---:|---:|---:|---:|
| 1366×768 | 565 | 240 (pinned to min) | 200 | **6** |
| 1600×900 | 697 | 343 | 229 | **7** |
| 1920×1080 | 877 | 451 | 301 | **10** |

Six at the smallest supported size, ten where an engineer actually works, against two today.

Two honest caveats: at 768 px tall you cannot have both a usable 3D view and all eighteen rows —
the table still scrolls, and the goal is a table you can *work* in. And the 768 figure reaches
six only because the preview is pinned at its 240 px minimum there; keep the preview larger on
small screens and it drops to four.

### 9.5 Footer

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [← Design Start]      [Create CST project ▾]   3 selected  [Send to LHS  →]  │
└──────────────────────────────────────────────────────────────────────────────┘
   150×38                     174×40            caption,muted   200×40 accent
```

| Change | From | To |
|---|---|---|
| Status label | truncating, `column=1` | **removed** — lives in the transcript |
| `Export CST script` | separate 162×40 | **merged** into `▾` on Create CST project |
| `Send selected to LHS →` | ~190×40 accent | **200×40** accent, rightmost |
| `← Design Start` | 150×38 | unchanged |

24 px gap between the CST group and the LHS primary; 8 px between related controls.

### 9.6 Header trim

~110 px today. Reclaiming ~30 gives every row count above one more row.

- Title `Parametric Antenna Builder` at `FONTS["title"]` — keep.
- Keep `EXPERIMENTAL` badge; drop `3 VALIDATED RECIPES` and `CST ADAPTER`.
- Subtitle to one line: *"Geometry is deterministic. Electromagnetic performance still requires
  simulation."*

### 9.7 Summary of numeric changes

| Item | From | To |
|---|---|---|
| left pane `minsize` / `width` | 480 / 535 | 420 / 480 |
| right pane `minsize` | 580 | 660 |
| parameter `CTkEntry(height=)` | 30 | 26 |
| parameter row `pady` | 3 | 2 |
| substrate material row | separate 40 px row | first table row |
| `LHS:` summary | bottom of editor pane | footer, beside LHS button |
| Apply button | 106×34 | 112×36 |
| `Send selected to LHS` | ~190×40 | 200×40 |
| `Export CST script` | 162×40 button | menu item under Create CST project |
| footer status label | present | removed |
| composer ↔ transcript order | composer above | **transcript above, composer anchored bottom** |
| transcript autoscroll | absent | **scroll to bottom after render** |
| `INSTRUCTION_COMPOSER_MIN/MAX_HEIGHT` | 92 / 154 | **unchanged** |
| Provider/Model/Refresh widget sizes | 150/320/70 × 28 | **unchanged**, collapsed behind ⚙ |
| `theme.py` | — | **no change** |

---

## 10. Deliberately not recommending

Per the streamlined-not-fancy constraint, I am **not** suggesting: animations or transitions;
a new colour system or iconography; a wizard or multi-step onboarding; inline charts or a
dashboard; a command palette or theming; drag-to-reorder or collapsible parameter groups.

Every item above either removes something, moves something to where it is already needed,
replaces truncated text with complete text, or adds a scroll call that should already exist.
The only genuinely new UI is the clickable starter block — and even that is promoting text
that already exists.

---

## 11. If only four things get done

1. **§7.1 — stop modelling clarify and refuse as exceptions.** Branch on
   `terminal_result.outcome` and render three outcomes as three transcript styles. §5.4, §6,
   and half of §5.3 collapse into this one change. It is also the change that most directly
   serves the conversational-ownership goal in the trial report.
2. **§5.1 + §5.2 — flip the composer below the transcript, and autoscroll.** A missing
   `scroll-to-bottom` is a one-line defect currently costing the user a manual scroll on every
   single turn.
3. **§3 / §9.4 — move the parameter table to the right pane.** Two visible rows out of eighteen
   is the difference between a demo and a tool.
4. **§1 — promote the starter examples and add the scope line.** The examples already exist;
   they are behind an unselected tab.

§4 and the tab split resolve themselves once 1–3 are done.

---

## 12. Note on the stale capture

`.test_runs_stale_20260921/builder_1366x768.png` is a screenshot of the Windows lock screen,
not the application. Whatever produced it captured the wrong surface. If 1366×768 reachability
is asserted from that artifact, the assertion is not backed by evidence — the claim may still
be true, but that file does not demonstrate it.
