# Pilot Run 01 — Antenna Surrogate Studio, text-to-parametric builder

**Participant:** antenna engineer, first contact with the tool (no docs, no source, no README read)
**Date:** 2026-09-25, ~13:05–13:35 PT (≈30 min of actual tool time; ~8 min lost to launch/window issues)
**Build seen:** "Studio Preview · v0.33.2", Parametric Antenna Builder labelled "EXPERIMENTAL · 3 VALIDATED RECIPES · CST ADAPTER"
**Planner used:** Gemini → gemini-3.8-flash ("Tested")
**Environment note:** the session was driven through a screen-control bridge. Some friction below (window visibility, flicker) belongs to that bridge rather than the Studio, and is labelled as such.

---

## Task 0 — Orient

**First impression (home screen, before touching anything):**
"Your surrogate workspace — Build trusted antenna models. Save them as books." Sidebar: Start, Design Start, Data Prep, Model Training, Training Results, Model Library, Inference, plus Inverse Design set apart lower down. A "SnowBuddy" button sits top right.

- *What I think it does:* a pipeline for training surrogate models from EM simulation data. Nothing on this screen mentions a text description or generating geometry.
- *Where I think I start:* "Design Start" looked like the best guess.
- *Confidence:* low–moderate. "Save them as books" means nothing to me as an antenna engineer. I also don't know what SnowBuddy is.

**Path to the text feature (~2 min from window visible):**
1. Clicked **Design Start** → got the modal "Create or open a project before continuing the workflow." That was expected and fine.
2. Clicked **+ Create project** (the big teal button). **Nothing happened.** Clicked again: still nothing. *Wrong turn #1.*
3. Found **File › New project…**, which opened a "Create a portable project" dialog with name and description fields.
4. **The dialog has no Create/OK button.** It is cut off at the bottom and can't be resized or scrolled. *Wrong turn #2.* I also typed a description, but it never landed in the field.
5. Guessed that pressing **Enter** in the name field would confirm. It did, and the project was created. **This was a pure guess.**
6. That landed me on "How do you want to start this design?", with two cards: "I already have a design" and **"Design with antenna agent — EXPERIMENTAL"**. The second card says: "Describe a supported antenna, then edit its parameters, inspect the live 3D geometry, and create CST output." That matched what I'd been told, and **Open Experimental Builder** took me in.

*Launch-level issue (may be environment-related):* my first double-click on the launcher produced no visible window. On relaunch, the window appeared in ~20 s. The Studio also opened un-maximised both times.

## Task 1 — First contact: 2.4 GHz patch on FR4

**Builder layout on arrival:**
- Left panel ("Describe the design") holds four example chips. All are truncated mid-word, e.g. "Inset-fed rectangular patch at 2.45 GH…" and "Patch at 2.45 GHz as a 1×4 array, 0.55λ…".
- Under the chips is a scope statement: "Supported: 3 antenna families · arrays to 16×16 · circular and rectangular slots · corner cutouts. Not supported: horns, Vivaldi, spirals, feed networks, solver runs."
- The request box has a Provider dropdown, set to Local Ollama / qwen3:8b by default.
- Right side: a 3D view and a parameter table.

The scope statement is genuinely useful; it set my expectations correctly for most of the session.

**Provider choice:** the options were Local Ollama, Gemini, Groq and OpenRouter. I'd been told a cloud key was configured, but **nothing tells you which provider has the key**. I guessed Gemini. It discovered 38 models and selected "Gemini 3.8 Flash · Tested", so the guess was right. The "Free only" checkbox is unexplained.

**Typed:** `2.4 GHz microstrip patch on FR4` (Ctrl+Enter)
**Expected:** a rectangular patch with some feed and sensible dimensions.
**Got (≈10 s):** "Inset-fed rectangular patch", envelope 52.0 × 43.4 × 1.7 mm, 1 port.
- Parameters: h = 1.6, Cu 0.035, **L = 29.42, W = 38.01**, feed 3.0, inset 8.83, inset gap 1, board margin 7 (all mm).
- I hand-checked the numbers. W matches the standard formula for εr ≈ 4.4, and L is within 0.02 mm of my Hammerstad-based estimate. A 3 mm feed is correct for 50 Ω on 1.6 mm FR4, and an 8.8 mm inset is in the plausible range.
- Three parameters (L, W, inset) came pre-ticked in the "VARY" column. The footer read "3 selected for sweep".

**Time to confidence:** ~1 minute after the first result. Good first contact.

**Confusions:**
- **εr and tanδ are never shown in the UI.** The material is just "FR4" in a dropdown. I only learned εr = 4.4 and tanδ = 0.02 by reading the exported macro.
- "Element spacing 0.55 λ₀" and array rows/cols are shown for a single element.
- Before any design existed, the table already showed a "Substrate material" row with a dropdown reading "No des…" (truncated).

## Task 2 — Iterate

**Typed:** `go to 0.8 mm FR4 instead`
**Expected:** h = 0.8, with L, the inset *and the feed width* re-derived.
**Got (≈35 s):**
- h = 0.8, L → 29.66, inset → 8.90, W unchanged (fine, since W doesn't depend on h).
- **Feed width stayed at 3.0 mm.** On 0.8 mm FR4 that line is roughly 30 Ω, not 50 Ω; 50 Ω should be about 1.5 mm.
- The tool re-derived some dependent dimensions and silently left the one that sets the match. I had to catch it myself.

**Direct table edit:** I changed Feed width to 1.5 in the table and pressed Enter. The change applied instantly and the 3D view updated. **This is the fastest and most trustworthy way to iterate in this tool.**

**Oddity:** a single table edit is logged in the chat as "Updated Frequency to 2.4; Substrate thickness to 0.8; Copper thickness to 0.035; Patch length…" — every parameter, including the ones I didn't touch. That makes it hard to tell what actually changed.

**Latency:** each text turn took 30–40 s with Gemini Flash. That's usable, but it discourages rapid "try this, try that" iteration. The table is instant.

## Task 3 — Load it with a slot

**Typed:** `cut a 1 mm wide, 20 mm long slot across the middle of the patch, parallel to the radiating edges, to pull the resonance down`
**Expected:** a centred transverse slot (perpendicular to the current).
**Got (≈30 s):** exactly that. I checked the orientation in 3D: the slot runs parallel to the feed-side (radiating) edge, is centred, and is about half of W. **Correct.** New table rows: Slot length 20, Slot width 1.

**Placement:**
- **The table has no slot position or orientation parameters.** Only length and width are exposed.
- **Typed:** `move the slot 5 mm toward the far radiating edge, away from the feed`
- The first attempt failed on the input side: my text never reached the box, and the tool correctly said "Instruction not applied — Enter an antenna instruction first."
- On retry it worked. The slot visibly moved toward the far edge.
- **But the offset never appeared as a parameter.** It is a hidden value I can't read, edit or sweep.
- The status line said "Updated stored composition composition_1 operation call_4eace2c0bda22601 updated; stored composition composition_1 replayed." That is internal jargon with no meaning to me.

**Chat panel stuck:** from this point on, the chat pane would not scroll to show the newest replies, neither with the mouse wheel nor by dragging the scrollbar. **For the rest of the session I could not read the assistant's replies.** My only feedback was the one-line status bar at the bottom, which was usually truncated.

## Task 4 — Scale it (two-element array)

**Typed:** `make it a two-element array, elements side by side across the width (H-plane), half-wavelength centre-to-centre spacing`
**Got (≈35 s):** "1 × 2 linear inset-fed rectangular patch array", envelope 114.5 × 43.7 × 0.9 mm, 2 ports. Array columns = 2, Element spacing = 0.5 λ₀. The slot and its hidden offset were copied onto both elements.

**Inspection, as I would before a sim run:**
- The array axis is along the width, i.e. the H-plane. That's correct.
- Board width = 62.46 + 38.01 + 2 × 7 = 114.47 mm. That's consistent, and leaves a 24.5 mm edge gap.
- **Feed:** there are two independent edge ports and no combiner. The UI says so plainly ("Inset-fed elements use independent ports. No corporate feed network is generated."), which I appreciate. It's fine as a starting point for coupling studies, but it is not an array I could sign off for pattern or match without post-processing the port weights.
- **Would I sign it off?** As a *geometry starting point*, yes. As an array model, no: there's no feed network, the slot offset is hidden, and there are no farfield monitors (see Task 7).

## Task 5 — Constrain it (≤ 60 mm board width)

**Typed:** `constraint: my board cannot exceed 60 mm wide. keep that from now on.`
**Expected:** it should either tell me the current 114.5 mm design violates the limit and propose options (a tighter array isn't physically possible here; two 38 mm patches plus margins can't fit in 60 mm), or refuse clearly.
**Got:** only a truncated status line: "…rd width of approximately 114.5 mm due to the 1x2 array configuration (two elements at 0.5 lambda spacing with 7 mm margins). **Future modifications will take this 60 mm limit into account.**" Also:
- the design stayed at 114.5 mm;
- nothing on screen marks the design as violating anything;
- the constraint isn't displayed anywhere;
- the request text stayed in the input box.

**Then kept working. Typed:** `open the element spacing up to 0.6 wavelength to cut coupling`
**Got:** spacing 0.6 λ₀ and board 127.0 × 43.7 mm, applied **with no warning.**

**The interface told me something I then watched it contradict.** It said future changes would respect the 60 mm limit, and the very next change made the board wider.

## Task 6 — Change your mind

- **Edit menu:** its only content is "Editing tools will be added here". There is no Undo, and no Ctrl+Z that I could find.
- **Typed:** `undo that last change` → spacing went back to 0.5 λ₀ and the board to 114.5 mm (≈35 s). **This worked.**
- **Typed:** `actually drop the slot, go back to plain patches` → the slot was removed from both elements and the slot rows disappeared from the table. Everything else was intact. Status: "Updated stored composition composition_1 deleted." **This worked cleanly.**
- **Difficulty:** easy once I gave up on looking for a menu. Everything costs 35 s per step, though, and there's no visible history or revision list to jump back to. (A "revision 8" counter surfaced later on reload, so revisions exist but aren't exposed.)

## Task 7 — Get it out

**Export CST script:**
- A readable save dialog opened ("Save parameterized CST construction macro", type "CST VBA macro"), defaulting to the project's `design` folder.
- The default name was `Create_inset_patch_v2_in_CST`. It says "patch" for an array, and "v2" is never explained.
- Confirmation: "CST script ready — Run the macro in a new CST Microwave Studio project. Review ports, mesh, and materials before solving." It also wrote a design-record JSON.

**What the macro contains (read as an engineer):**
- Good:
  - Everything is a StoreParameter: FreqGHz, SubH, CopperT, PatchL, PatchW, FeedW, Inset, Gap, Margin, ArrayRows, ArrayCols and SpacingLambda.
  - Derived values are formulas: ElementSpacing = c/FreqGHz·SpacingLambda, and BoardW/BoardL in terms of the patch, margin and spacing.
  - The patch, notch and feed bricks are built from those parameters.
  - Boundaries are open (add space); solver is time domain; range is 0.65–1.35·f.
  - FR4 is εr 4.4, tanδ 0.02. That's plausible, though I'd have liked to see it in the UI.
  - **The core dimensions would support a parameter sweep.**
- Concerns:
  - **ArrayRows/ArrayCols are exposed as CST parameters, but the element count and positions are hard-coded** (e.g. `(1-(2-1)/2)*ElementSpacing`). Sweeping ArrayCols in CST would resize the board and leave two elements. A comment in the file admits this. It's a trap for anyone who only looks at the Parameter List.
  - **ElementSpacing and the solver frequency range both hang off FreqGHz.** Sweeping frequency as a "design" variable would move the elements.
  - There are **no farfield or field monitors**, so I'd get no pattern from an array model without adding them myself.
  - The ports are discrete edge ports with radius 0 between ground and the trace underside. That's acceptable for a first pass, but I'd swap in waveguide ports before trusting S11.
  - Metals are PEC and the ground is the same size as the substrate. That's fine for a starting model.
  - Parameter descriptions are empty in CST.

**Create CST project:**
- A readable save dialog opened ("Create native CST antenna project", default `Generated_inset_patch_v2`).
- After saving, **a window I could not read appeared over the Studio**. I stopped here as instructed.
- Once I was given CST access it resolved into a readable dialog: "CST project ready … The solver was not started." The .cst and its folder were created.
- **Blocking finding:** opening the .cst myself in CST failed with *"This project … is already open in another instance of CST Studio Suite."* The Studio keeps a hidden background CST instance holding the lock, and it was still locked after 4+ minutes.
- The fallback, running the .bas in a new CST project, was **also blocked**: in this CST 2026 Learning Edition, "Import VBA Macro…" and the macro editor are greyed out.
- I only got to see the native model after closing the Studio (Task 9). The hidden CST instance then surfaced as a visible window with the project open.
- **In CST the model matched the Studio exactly:** two inset patches, ports 1 and 2 on the board edge, and the full parameter chain (ElementSpacing 62.457, BoardW 114.467, BoardL 43.66).

**Would I trust this as a starting model?**
- *Geometry and parameterisation:* yes. This is better than most hand-built starting models I've seen from juniors.
- *Simulation setup:* no, not without adding monitors and revisiting the ports, and not before fixing the ArrayRows/Cols trap.

## Task 8 — Push the edges

| Asked for | Response | Left me knowing what to do? |
|---|---|---|
| "combine the two elements with a microstrip T-junction / quarter-wave transformer corporate feed so I have a single 50 ohm port" | Status bar only (truncated): "…combining the elements into a single physical 50 ohm port) is not supported by the installed antenna recipes and tools." | **Partly.** A clear no, consistent with the "Not supported: feed networks" line. No alternative was suggested (e.g. "use independent ports and combine in post-processing"). |
| "switch the laminate to Rogers RO4350B, 0.762 mm" | "…RO4350B is not supported by the installed antenna recipes or material library. Available substrate materials are FR4, Rogers RO4003C, and Rogers RT5880." | **Yes.** This is the best refusal in the tool: it tells me what I *can* pick, and RO4003C is the obvious substitute. Note that the thickness part of my request was silently dropped along with the material. |
| "make each element RHCP using truncated corners" | "…recipes are limited to linearly polarized inset-fed rectangular patches and probe-fed circular patches, and the only installed corner modifier is circular corner cutouts (corner_circle_cutouts_v1)." | **Yes, but it exposed a misleading label.** The landing panel advertises "corner cutouts", and any patch engineer reads that as CP corner truncation. It actually means circular notches. An internal ID also leaked into the message. |

Every refusal appeared **only in the one-line status bar**, truncated at the left, and the request text was left in the input box. With the chat pane stuck, I had to zoom into the status bar to read the "no" at all.

## Task 9 — Come back

- **Closing:** the window closed on the first click of X, with no "save?" prompt. That was fine, since everything turned out to be saved. As noted above, closing also brought the hidden CST instance to the foreground.
- **Reopening:** it relaunched un-maximised, with "No project open", and Pilot01 was first under **Recent projects**. The card still said "New project", and the name wrapped one word per line.
- Clicking the card took me **straight back into the builder** with "Restored 1 × 2 linear inset-fed rectangular patch array revision 8". Geometry, table and chat history were all intact, and the provider and model were retained. **The work survived.**
- **Does it still know what I was doing?** I typed `what constraints am I working under right now?`
  - It remembered **"Board Width Constraint: Maximum 60.0 mm (note: the current 1x2 array board width is 114.47 mm, exceeding this limit)."** So the constraint *is* stored and it *knows* it's violated. It simply never enforced or flagged it.
  - **It also listed "Intended Feed: Single 50 Ω corporate feed network" and "Polarization Goal: Circular polarization (RHCP)" as project requirements.** Both were things it had *refused*. The recorded intent now contradicts the actual design, and nothing in the UI shows this memory.
  - The answer rendered as a multi-line block in the status area. It overflowed the window, pushed the footer buttons into the middle of the text, and clipped lines on the left.

---

## Where I got stuck, and what I tried

1. **Creating a project.** The "+ Create project" button did nothing (2 clicks). File › New project worked, but the dialog has no confirm button (I tried scrolling and resizing). I guessed Enter, and it worked.
2. **Reading assistant replies.** The chat pane wouldn't scroll (I tried the wheel and dragging the scrollbar). I gave up and relied on the truncated status bar for the rest of the session.
3. **Opening the generated CST project.** The file was locked by the hidden CST instance and macro import was greyed out in CST LE. I waited over 4 minutes, then got in only after closing the Studio.
4. **Finding the text feature.** It sits behind "Design Start" → a project gate → a card labelled "EXPERIMENTAL". Nothing on the home screen hints at it.

## Things I had to guess, and whether I was right

- Enter confirms the new-project dialog: **right**, but only by luck.
- "Design Start" is where the text feature lives: **right**.
- Gemini is the provider with the key: **right** (but unverifiable before trying).
- Ctrl+Enter submits: **right**; the footer says so ("Enter adds a new line · Ctrl+Enter applies").
- "Corner cutouts" means CP truncation: **wrong**; they're circular notches.
- Stating a constraint means it will be enforced: **wrong**.
- ArrayRows/ArrayCols in the CST parameter list control the element count: **would have been wrong**. I only caught it by reading the macro.

## Things that looked wrong to me as an antenna engineer

- **Changing substrate thickness did not re-derive the 50 Ω feed width.** A 3 mm line on 0.8 mm FR4 is a ~30 Ω mismatch that the tool never flagged.
- **The slot offset exists, but isn't a parameter.** It can't be read, swept, or seen in CST as a named variable.
- **The ArrayRows/ArrayCols CST parameters don't control the element count.** Sweeping them yields a wrong board.
- **Element spacing and the solver frequency range are both tied to FreqGHz.** A frequency sweep would also move the geometry.
- **There are no farfield monitors on an array model.**
- **εr and tanδ are hidden from the UI.**
- The "Element spacing" row is shown for single-element designs.
- The patch sizing itself was textbook-correct, and so were the array arithmetic and slot orientation. **I would not dispute any geometry it actually drew.**

## Things the interface told me that I didn't believe, or that contradicted the screen

- "Future modifications will take this 60 mm limit into account." The next modification widened the board to 127 mm without a word.
- After a reload, the stored "requirements" list a corporate feed and RHCP, both of which it had refused. Meanwhile the design is linear, with independent ports.
- "Supported: … corner cutouts" on the landing panel versus "the only corner modifier is circular corner cutouts" in the refusal.
- The chat log for a single table edit claims every parameter was "updated".
- The project card still says "New project" after eight revisions.
- A "Create project" button that does nothing, next to a menu item that works.

## The three things that most got in my way

1. **Feedback lived in a truncated one-line status bar, and the chat pane stopped scrolling.** Refusals, constraint acknowledgements and planner explanations were all hard or impossible to read.
2. **Constraints are acknowledged but not enforced or displayed.** The tool told me the limit would be respected and then broke it on the next change, silently.
3. **Getting from "Create CST project" to actually looking at the model in CST.** A hidden CST instance held the file lock, and I could only open the project after closing the Studio.

(Honourable mention: the project-creation dialog with no confirm button. If I hadn't guessed Enter, I'd have stopped there.)

## Would I use this on a real project?

For generating a correctly-sized, fully parameterised *starting geometry* for a patch or small patch array, and dropping it into a sweep, I would. The dimensions were right on the first try, table edits are instant, and the exported macro is cleaner than what I'd hand-write in 20 minutes. But I would use it with my eyes open and my own calculator next to me. It doesn't keep dependent quantities honest (the feed width), it forgets to enforce what it promises to remember (the board limit), and some of what it produces is invisible (the slot offset, the stored "intent"). Its main feedback channel is a status line I had to zoom into to read. The first time an unflagged mismatch like the feed width made it into a sim queue unnoticed, I'd stop trusting it. Today it's a fast sketchpad for one of three recipes, not something I'd hand to a junior unsupervised.
