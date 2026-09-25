# Consolidated open defects

**Compiled:** 2026-09-25, after commits `75ecc91`–`51ac915`
**Sources:** `BUILDER_USER_TRIAL_REPORT.md` (instrumented, Gemini, 34 turns),
`BUILDER_UX_REVIEW.md` (UI + UI-architecture), `PILOT_RUN_01.md` (blind pilot,
antenna engineer, no code access).

Every item below was verified against the current tree. Items already fixed are in §5
so they are not reworked.

---

## 1. Trust defects — the tool states something untrue

These are the highest priority. The geometry is correct; the honesty is not.

### T1. The board-width constraint is acknowledged, then violated silently
**Source:** trial report D2, confirmed by pilot Task 5.
User: *"my board cannot exceed 60 mm wide, keep that from now on."*
Tool: *"Future modifications will take this 60 mm limit into account."*
Next instruction widened the board 114.5 mm → **127.0 mm with no warning**.

The constraint IS stored correctly (`board_width_limit = 60.0`, `constraint_operator:
max`, normalized) and the agent can reason about it perfectly when asked. It is never
consulted while acting.

This is now worse than the original silent-ignore behaviour, because the tool
explicitly promises enforcement it does not perform.

**Direction:** evaluate active `constraint`-kind memory against the candidate design
inside the transactional gate, before publication. A violation must at minimum force a
disclosure in the turn message; a `clarify` would be better. Also display active
constraints somewhere persistent, and mark a violating design as violating. If
enforcement is not implemented, the planner must stop promising it.

### T2. Refused requests are recorded as project requirements
**Source:** pilot Task 9. New — not found by instrumented testing.
The user asked for a corporate feed network and RHCP via corner truncation. Both were
**refused**. After reload, stored project memory listed *"Intended Feed: Single 50 Ω
corporate feed network"* and *"Polarization Goal: Circular polarization (RHCP)"* as
active requirements.

`AgentStep` permits `memory_proposals` on terminal `refuse` steps, so a refusal writes
durable intent. The stored record now contradicts the delivered design, and nothing in
the UI surfaces it.

**Direction:** a refused capability must not become an active requirement. Either block
`memory_proposals` on `refuse`, or record them under a distinct status such as
`requested_unsupported` that is never presented as a goal. Memory must also be visible
in the UI — it is currently invisible until the user thinks to ask.

---

## 2. Correctness defects

### C1. Substrate thickness changes do not re-derive feed width
**Source:** pilot Task 2. New — instrumented testing never checked impedance.
`h` 1.6 mm → 0.8 mm re-derived `PatchL` and `Inset` but left `FeedW` at 3.0 mm. On
0.8 mm FR4 that is roughly **30 Ω, not 50 Ω** (≈1.5 mm would be correct). No flag.

The tool re-derived the dependent dimensions it knows about and silently left the one
that sets the match. The pilot called this the defect that would end their trust:
*"the first time an unflagged mismatch like the feed width made it into a sim queue
unnoticed, I'd stop trusting it."*

**Direction:** either derive `FeedW` from the 50 Ω microstrip condition for the current
`SubH`/`εr`, or raise an explicit engineering finding when the ratio drifts from the
design impedance. Silence is the unacceptable option.

### C2. Sweeping frequency in LHS physically moves the array
**Source:** pilot Task 7 (read from the macro); confirmed against the LHS handoff.
`ElementSpacing = 299.792458/FreqGHz*SpacingLambda`, and `FreqGHz` is offered as a
sweep variable (measured bounds 2.205–2.695 GHz). The solver frequency range also
derives from `FreqGHz`.

So an LHS table that varies frequency varies **geometry and frequency together**. A
surrogate trained on it learns a confounded relationship. This is the Studio's core
workflow.

**Direction:** distinguish design variables from operating-point variables in the LHS
handoff. Either pin `ElementSpacing` to a physical value when `FreqGHz` is swept, or
exclude `FreqGHz` from the sweep set, or surface the coupling explicitly at selection
time. Related: trial report item 14 (no per-row LHS feasibility check) is still open.

### C3. `ArrayRows` / `ArrayCols` appear sweepable in CST but are topology-fixed
**Source:** pilot Task 7. Known caveat (overview §13.4), but sharper than documented:
they appear in CST's Parameter List looking like ordinary sweep variables, while
element positions are hard-coded (`(1-(2-1)/2)*ElementSpacing`). Sweeping `ArrayCols`
resizes the board and leaves two elements. Only a comment in the macro warns.

**Direction:** emit them in a visually distinct way, or omit them from `StoreParameter`
and bake the counts, or add a CST-visible parameter description stating they are
structural. Parameter descriptions are currently empty throughout.

---

## 3. Blocking usability defects

### U1. The conversation pane does not scroll
**Source:** pilot Task 3 onward — *"for the rest of the session I could not read the
assistant's replies."* Combined with U2, this left the tool with no working feedback
channel.

Two root causes, both verified:
- **No mousewheel bindings anywhere.** `grep -nE "MouseWheel|Button-4|Button-5"` over
  `antenna_builder_ui.py` returns nothing. In CustomTkinter, children inside a
  `CTkScrollableFrame` swallow wheel events unless bound explicitly, so hovering over
  message text kills the wheel.
- **Stale scrollregion.** `_scroll_conversation_to_bottom` calls
  `canvas.update_idletasks()` then `yview_moveto(1.0)`, but label `wraplength` reflow
  happens after that. The canvas scrolls to a bottom that no longer exists, and the
  scrollbar has nothing to drag.

**Direction:** refresh scrollregion on `<Configure>` of the inner frame, scroll after
reflow completes, and bind wheel events on child widgets.

### U2. The footer status bar was not removed
**Source:** `antenna_builder_ui.py:800` still binds `textvariable=self.status_var`.
UX review §4 called for its removal once change summaries moved into the transcript.
It remains, still truncating at the left, and became the pilot's only feedback channel
once U1 bit. They had to zoom in to read refusals.

One later message overflowed it so badly it *pushed the footer buttons into the middle
of the text*.

**Direction:** remove it. All turn feedback belongs in the transcript.

### U3. The new-project dialog loses its buttons at non-100% display scaling
**Source:** pilot Task 0 — *"the dialog has no Create/OK button… cut off at the bottom
and can't be resized or scrolled."* Escaped only by guessing Enter.

The buttons exist (`Cancel`, `Create project →`, plus a `<Return>` binding). The cause
is `ui.py:5387`: `self.geometry("540x430")` with `self.resizable(False, False)`.
Content sums to roughly **428 px against a 430 px box**. `set_window_scaling(1.0/dpi)`
and `set_widget_scaling(ui_scaling/dpi)` are different factors, so at non-100% Windows
scaling the content and the box grow at different rates and the actions row falls off.
Not resizable, so unrecoverable.

The pilot machine reports 1707×1067 logical = 2560×1600 at 150%. `ui.py:5503`
(`580x540`) has the same pattern.

**Direction:** size dialogs to their content rather than fixed pixels, or allow resize,
or place the actions row so it cannot be clipped. **Verify at 100 %, 125 % and 150 %.**

### U4. Generated `.cst` is locked by a hidden background CST instance
**Source:** pilot Task 7. Opening the generated project failed with *"already open in
another instance of CST Studio Suite"*, still locked after 4+ minutes. Only released
when the Studio was closed. The fallback (import the `.bas`) is unavailable in CST
Learning Edition, where macro import is greyed out.

So on a Learning Edition machine there is currently **no way to view the output while
the Studio is running**.

**Direction:** release the COM instance after `SaveAs`, or run creation out-of-process
and terminate it, or tell the user plainly that the Studio holds the project and offer
a "release" action.

---

## 4. Discoverability and clarity

| ID | Defect | Source |
|---|---|---|
| D01 | Home screen never mentions text description or geometry generation. Sidebar gives no hint. User told "it turns text into antennas" needed ~2 min and two wrong turns to find it. | Pilot Task 0 |
| D02 | `+ Create project` on the hero panel appeared to do nothing (2 clicks); `File › New project…` worked. Both call `create_project_dialog`. **Needs human confirmation** — likely z-order/focus, possibly test-bridge related. | Pilot Task 0 |
| D03 | `Design Start` is not disabled before a project exists; the precondition is discovered by hitting a modal. | Pilot Task 0 |
| D04 | Starter example chips are truncated mid-word (`"Inset-fed rectangular patch at 2.45 GH…"`). The feature landed; the chips are too narrow. | Pilot Task 1 |
| D05 | εr and tanδ appear nowhere in the UI. Learned only by reading the exported macro. | Pilot Task 1 |
| D06 | Nothing indicates which provider has a configured key. Default is Local Ollama. `Free only` is unexplained and shown when irrelevant (UX review §2 — not yet done). | Pilot Task 1 |
| D07 | Slot position/offset is settable by text but never appears as a parameter — cannot be read, edited, swept, or seen in CST. (Trial report D6.) | Pilot Task 3 |
| D08 | A single table edit is logged as *"Updated Frequency… Substrate thickness… Copper thickness… Patch length…"* — every parameter, including untouched ones. | Pilot Task 2 |
| D09 | Internal jargon in user-facing status: *"stored composition composition_1 operation call_4eace2c0bda22601 updated"*. Internal IDs also leak into refusals (`corner_circle_cutouts_v1`). | Pilot Tasks 3, 8 |
| D10 | *"Corner cutouts"* in the scope line reads as CP corner truncation to any patch engineer. It means circular notches. | Pilot Task 8 |
| D11 | No undo and no revision history in the UI. `Edit` menu contains only *"Editing tools will be added here"*. Revisions exist (`revision 8` surfaced on reload) but are never exposed. Text `undo that last change` works. (UX review §8a.) | Pilot Task 6 |
| D12 | `Element spacing` and array rows/cols rows are shown for single-element designs. | Pilot Task 1 |
| D13 | Before any design exists the table shows a `Substrate material` row reading `"No des…"` (truncated). | Pilot Task 1 |
| D14 | Export default name `Create_inset_patch_v2_in_CST` says "patch" for an array; `v2` is never explained. | Pilot Task 7 |
| D15 | Project card still reads *"New project"* after 8 revisions; the name wraps one word per line. | Pilot Task 9 |
| D16 | When a material request is refused, an accompanying thickness in the same sentence is silently dropped. | Pilot Task 8 |
| D17 | No farfield or field monitors in the export, so an array model yields no pattern without hand-editing. | Pilot Task 7 |
| D18 | Feed-network refusal offers no alternative (e.g. independent ports + post-processing). Material refusal does this well and is the model to copy. | Pilot Task 8 |
| D19 | Two CST buttons still separate and equally weighted (UX review §8b — not yet done). | Code |

---

## 5. Already fixed — do not rework

| Item | Evidence |
|---|---|
| Non-uniform composed arrays (trial D1) | `21838a5`; pilot verified slot copied to both elements, in 3D and in CST |
| Contract-fingerprint drift, red suite | Re-frozen deliberately; Stage 5E retired as research |
| `.env` BOM | `3a5f492` |
| `evaluate_scalar` RecursionError | `ef3a32f` |
| Work uncommitted | `75ecc91`–`51ac915`, nine chunks |
| Parameter table moved to right pane (UX §3) | Pilot: *"the fastest and most trustworthy way to iterate in this tool"* |
| Transcript above composer (UX §5.1) | `conversation_frame` row 1, composer row 2 |
| Clarify/refuse no longer exceptions (UX §7.1) | Outcome branching at `antenna_builder_ui.py:1335-1346` |
| Input preserved on clarify/refuse (UX §5.4) | Pilot: *"the request text stayed in the input box"* |
| Cancel during planning (UX §7.3) | `BuilderTurnCancelled`, `_active_plan_cancel` |
| Starter examples + scope line (UX §1) | Pilot: *"genuinely useful; it set my expectations correctly"* |
| Outcome glyph styling (UX §5.3) | Outcome mapping at `:557-559` |

---

## 6. Deferred — known, not urgent

- Local Ollama `num_ctx: 16384` against a measured 17–22 k token payload (trial D3).
  Cloud masks it; local users hit truncation.
- `design/planner_ab.jsonl` grows ~100 KB/turn with no rotation (trial D4).
- Composed features keep stale names after being moved (`center_slot_tool` at x = 6)
  (trial D5).
- 30–40 s per text turn with Gemini Flash. Inherent to the design; the instant table
  edits compensate.
- Ports are discrete edge ports; PEC metals; no de-embedding. Documented and accepted
  for a starting model.

---

## 7. Suggested order

1. **T1, T2** — the tool currently says untrue things. Cheapest large trust win.
2. **U1, U2** — restore a working feedback channel. U1 has a diagnosed root cause.
3. **C1** — the defect the pilot said would end their trust.
4. **C2** — protects the surrogate pipeline, which is the point of the product.
5. **U3, U4** — first-run blocker and output-inspection blocker.
6. **§4 items** — cheap individually; D01/D04/D09 have the best ratio.

C3 and D17 are export-quality items worth a pass once the above land.
