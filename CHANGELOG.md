# Changelog

Notable changes to Antenna Surrogate Studio. Newest first.

Versions before `0.34.0-beta` predate this file; their history is in the commit
log.

## [0.34.1-beta] — 2026-10-06

A focused usability release. The experimental Text-to-CAD builder now asks for
its cloud API key in the application instead of expecting the user to create a
file. Nothing about the planner, the recipes, the geometry or the CST output
changes.

### Added

- **Text-to-CAD API key setup.** Opening the experimental builder for the first
  time shows a short **Set up Text-to-CAD** dialog recommending Gemini, with a
  masked key field, a **Get an API key** action that opens the provider's own
  key page in the system browser, and **Verify & Continue**. Verification is a
  real authenticated request to the provider's model catalog, so an unusable key
  is caught at setup rather than deep inside a later design request. It carries
  no geometry, design state or instruction.
- **Operating system credential storage.** A verified key is saved through
  `keyring`, which resolves to the Windows Credential Manager on Windows, under
  one Antenna Surrogate Studio service entry per provider. Keys are held per
  provider rather than per project, and are never written to a project file, a
  log, the planner audit, or `.env`. A saved key is not displayed again.
- **An API keys button** beside the builder's Model menu reopens the same
  dialog to replace or remove a stored key, and names which source each
  provider's key currently comes from.
- **A separate help window**, *About Text-to-CAD and your API key*, covering
  where the key is stored, what the provider receives, cost, and the offline
  option. It opens from the setup dialog, which itself stays small.

### Changed

- Credentials now resolve from the process environment, then an ignored `.env`,
  then the credential store. The first two are unchanged, so existing
  environment variables and `.env` files keep working and never trigger setup.
  An exported variable deliberately outranks a key saved in the dialog, and the
  dialog names the source in effect so a shadowed key is visible.
- README and the User Manual describe pasting a key into the dialog as the
  normal path, with environment variables and `.env` moved to an advanced
  section.
- Groq, OpenRouter and Local Ollama remain fully supported, under **Other
  providers and offline use**. Local Ollama is no longer the recommended first
  run, and its lower capability is stated where it is offered.

### Fixed

- Status, error and help text wrapped wider than the window showing it, which
  cut the last characters off several messages on laptop layouts.
- A provider verified during setup now becomes the builder's active provider
  immediately, instead of leaving the workspace on the previous one.
- The Provider selector stays visible alongside the new API keys control.
- A recorded cloud planner whose saved credential is missing now asks for a key
  again instead of failing during a later design request.

### Notes

- Entering the builder settles how the planner will be reached before the
  workspace opens, so a first run can no longer proceed with no usable
  credential and fail later. A recorded offline choice is never interrupted. A
  recorded cloud provider is asked about again only when its credential no
  longer resolves, so a missing key is raised at the dialog rather than during a
  later design request.
- `ANTENNA_STUDIO_NO_CREDENTIAL_STORE=1` keeps the Studio away from the
  credential store, leaving the environment and `.env` as the only sources.
- No OpenAI provider was added, and the builder still performs no solving or
  optimisation.

## [0.34.0-beta] — 2026-10-05

The release that introduces the **experimental text-driven antenna builder**.
The surrogate-modelling pipeline is unchanged in behaviour and remains the
stable half of the tool.

### Added

- **Parametric Antenna Builder (experimental).** Plain-language construction of
  antenna geometry, seeded from three validated recipes — an inset-fed
  rectangular microstrip patch, a probe-fed circular patch and a centre-fed
  dipole — with compositional editing on top. See
  [TEXT_TO_CAD.md](TEXT_TO_CAD.md).
- **Compositional geometry editing.** Nine planner-exposed primitives create
  named parameters, build rectangular and circular cutting geometry, translate,
  rotate and duplicate it, and apply validated union and subtraction
  operations. Compositions persist with the project and replay against semantic
  element targets after a later recipe or parameter rebuild.
- **Arrays and excitation.** 1×N linear and M×N planar replication where
  geometrically valid, with independent ports per element.
- **Read-only engineering analysis.** Four tools — design summary, array
  spacing, rectangular-patch analytical baseline and dipole electrical-length
  baseline — documented in
  [docs/ENGINEERING_ANALYSIS_REFERENCE.md](docs/ENGINEERING_ANALYSIS_REFERENCE.md).
- **CST output.** Parameterised `.bas` construction macro on any platform, and
  direct native `.cst` project creation on Windows through an isolated
  automation server.
- **Interactive 3D preview** of the evaluated canonical geometry, with true Z
  dimensions, drawn port endpoints, and an explicit warning when a Boolean
  cannot be rendered faithfully.
- **Element spacing basis.** Array recipes carry a `SpacingMode` of `lambda` or
  `fixed_mm`, with a matching physical `ElementSpacing` parameter in mm.
- **Multi-provider planners.** Local Ollama, Gemini, Groq and OpenRouter, with
  per-project provider and model selection, Tested/Untested labelling, and a
  per-project planner audit log at `design/planner_ab.jsonl`.
- **Documentation:** [docs/LIMITATIONS.md](docs/LIMITATIONS.md),
  [docs/ENGINEERING_ANALYSIS_REFERENCE.md](docs/ENGINEERING_ANALYSIS_REFERENCE.md),
  [docs/FUTURE_DIRECTION.md](docs/FUTURE_DIRECTION.md),
  [TEXT_TO_CAD.md](TEXT_TO_CAD.md), [CONTRIBUTING.md](CONTRIBUTING.md) and this
  changelog.

### Fixed

- **A completed training run could be silently discarded.** If the project
  record could not be rewritten — for example while another process briefly held
  the file — the rest of the completion path was abandoned and the exception was
  swallowed by Tk, leaving the page with no results and no message. The run is
  now recorded first, and a *Project record not updated* warning reports the
  failure.
- **Transient Windows file locks could fail an atomic replacement.** All twelve
  file and directory replacements now retry a transient `PermissionError` or
  sharing violation with bounded exponential backoff (5 attempts, ≤ 0.375 s
  total). A persistent error is still reported unchanged, and a missing source
  is never retried. Previously this could fail a finished training run while
  promoting its run directory.
- **Sweeping frequency silently moved array geometry.** Element spacing was
  always locked to the wavelength, so a Latin Hypercube table that varied
  frequency also varied the physical geometry, confounding any surrogate trained
  on it. With `SpacingMode` set to `fixed_mm`, frequency becomes an operating
  point only; in `lambda` mode frequency is no longer offered as a sweep variable
  at all.

### Changed

- Documentation now separates the stable surrogate pipeline from the beta
  antenna builder throughout, states the cloud-planner data egress explicitly,
  and describes the licence as source-available rather than open source.
- The README records what the builder actually is — recipe-seeded with
  compositional editing — rather than implying free-form text-to-CAD.

### Removed

- Benchmark result dumps, provider trajectory captures, development stage notes,
  one-off investigation reports and the stale open-defect register. Current
  product boundaries moved to [docs/LIMITATIONS.md](docs/LIMITATIONS.md);
  everything else remains in the commit history.
- Maintainer-only development launchers, which enabled local conversation
  logging.

### Known limitations

Exported models carry no field or farfield monitors, array row and column counts
are structural in CST, conductors export as PEC, and ports are simplified
discrete excitations. The full list is in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md). **Generated geometry is a starting
point to inspect and simulate, not a validated antenna.**
