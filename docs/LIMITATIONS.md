# Current limitations

Antenna Surrogate Studio has two halves. The **surrogate pipeline** — data
preparation, model training and comparison, inference, and inverse design — is
the mature half. The **text-parametric antenna builder** is explicitly
experimental and is where most of the limits below apply.

Read this before using generated geometry for anything that matters. Everything
here is a product boundary, not a hidden defect.

## Scope of the geometry builder

- Three base antenna families are certified: an inset-fed rectangular
  microstrip patch, a probe-fed circular patch, and a center-fed dipole. One
  modifier is certified: four subtractive circular corner cutouts on the
  rectangular patch.
- The substrate library holds three materials with constant, frequency-
  independent properties: FR4 (εr 4.4, tanδ 0.02), Rogers RO4003C (3.55,
  0.0027) and Rogers RT5880 (2.2, 0.0009).
- Importing or extracting parameters from an existing `.cst` project is out of
  scope.

## Exported CST models

A generated model is a **starting point for you to verify**, not a solved
design. Validate mesh, ports, materials and results before engineering use.

- **Array row and column counts are structural.** `ArrayRows` and `ArrayCols`
  appear in the CST parameter list, but element positions are baked into the
  construction history. Changing either inside an existing CST project resizes
  the board and leaves the original element count. Regenerate the model from the
  Studio instead.
- **No field or farfield monitors are exported.** An array model therefore
  produces no radiation pattern until you add monitors in CST yourself.
- **Ports are simplified discrete excitations.** Conductors export as PEC even
  though conductivity is stored. Connectors, baluns, waveguide ports, coaxial
  clearances and de-embedding are all absent.
- **Array elements get independent ports and no feed network.** Feed-network
  synthesis and any corporate-feed design remain your work in CST.

## Element spacing and frequency sweeps

Each array recipe carries a `SpacingMode` of either `lambda` or `fixed_mm`.

- In `lambda` mode — the default, and the historical behaviour — element spacing
  is held electrically, so changing `FreqGHz` re-derives the spacing and resizes
  the board. Correct for a scaled design, but geometry depends on frequency.
- In `fixed_mm` mode spacing is held physically in millimetres, and changing
  `FreqGHz` moves the operating point only.

**Inside the Studio this is enforced.** In `lambda` mode frequency is not
offered as a sweep variable at all — its Vary box is disabled and you vary
`SpacingLambda` instead. In `fixed_mm` mode frequency becomes sweepable and
`SpacingLambda` becomes the derived, read-only row. You cannot accidentally
build an LHS table that varies frequency and geometry together.

**Outside the Studio it is not, and cannot be.** `FreqGHz` is always written
into the exported CST parameter list, so a frequency sweep set up *inside CST*
while the design is in `lambda` mode will still move every element and resize
the board. The generated macro carries a comment stating which behaviour
applies. Switch to `fixed_mm` before building a frequency sweep in CST.

## Natural-language editing

- Cloud planners (Gemini, Groq, OpenRouter) need network access and your own API
  key, and **send design-planning context to the selected provider**. The local
  Ollama path keeps everything on your machine.
- The local planner requests a 16,384-token context. Measured planner payloads
  reach 17–22k tokens on larger designs, so local models may see a truncated
  request where cloud models do not. `qwen3:8b` is the validated local default.
- Expect roughly 30–40 s per text turn against a cloud planner. Direct parameter
  table edits are immediate and are the faster way to iterate.
- The planner may only call registered tools. Materials, ports, solver setup and
  arbitrary transforms are deliberately unavailable to it, so requests touching
  those are refused rather than approximated.
- No undo or redo is exposed in the UI. Design revisions are recorded and the
  text request `undo that last change` works, but the **Edit** menu is a
  placeholder.
- The interactive 3D preview is not a general Boolean engine. Planner-composed
  slot holes appear in the canonical design and the CST export, but are not
  carved into the preview mesh.

## Housekeeping

- The per-project planner audit log at `design/planner_ab.jsonl` grows by
  roughly 100 KB per text turn and is never rotated. Delete it if a long-lived
  project's folder gets large; nothing depends on its history.

## Further reading

Section 21 of
[Text-Parametric Builder: Deep Technical and Design Overview](TEXT_PARAMETRIC_BUILDER_DEEP_OVERVIEW.txt)
carries the longer engineering-debt list, including items that are internal
rather than user-facing.
