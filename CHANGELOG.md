# Changelog

Notable changes to Antenna Surrogate Studio. Newest first.

Versions before `0.34.0-beta` predate this file; their history is in the commit
log.

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
