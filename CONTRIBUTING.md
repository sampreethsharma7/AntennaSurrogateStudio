# Contributing

Antenna Surrogate Studio is an author-maintained release. **External pull
requests and code contributions are not accepted.** The most useful things you
can send are bug reports, reproduction steps and tester feedback — contact
details are at the bottom of the [README](README.md).

Everything below applies to anyone working on the code directly, including the
author and any automated tooling.

## The GUI contract

The Studio ships a written description of its own interface,
[`snowbuddy/BLIND_GUI_READ.md`](snowbuddy/BLIND_GUI_READ.md), which records every
visible label, navigation path, dialog and user-facing state, pinned to the
SHA-256 of the UI source files it describes. A test enforces that the recorded
hashes match the source, so the description cannot silently drift away from the
application.

Any change that touches `studio/ui.py`, `studio/theme.py`, visible labels,
navigation, layout, dialogs, user-facing states or interaction behaviour must
update that file **in the same change**:

1. Correct every affected description in `BLIND_GUI_READ.md`.
2. Recompute the SHA-256 of each UI source file it records.
3. Replace the recorded hash values at the top of the file.
4. Run the suite: `python -m unittest discover -s tests`.

**Do not weaken or remove the GUI-contract test to get a change through.** If the
test fails, the description is out of date — fix the description. The test
existing is the only reason the description can be trusted.

Changes to SnowBuddy's identity, tone, scope or response rules must also update
[`snowbuddy/SNOWBUDDY_CHARACTER.md`](snowbuddy/SNOWBUDDY_CHARACTER.md).

## Tests

```sh
python -m unittest discover -s tests
```

The suite covers surrogate training, persistence, geometry, CAD construction,
CST export, excitation, engineering checks, UI behaviour and rollback safety.
Several GUI modules skip automatically where no display is available, so a run
that reports skips on a headless machine is expected; a run that reports
failures is not.

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the full suite on
**`windows-latest` with Python 3.12** for every push and pull request.

It runs on Windows rather than the cheaper Linux default on purpose. The GUI
tests gate on `os.name == "nt" or sys.platform == "darwin" or $DISPLAY`, so on a
stock Linux runner every Tk test would skip and the job would report a green run
with the entire UI coverage missing.

**Nothing is skipped in CI, and the job enforces that.** On a healthy Windows
machine the suite skips zero tests, so the workflow counts skips in the verbose
output and fails if there are any. A skipped test is invisible in an exit code,
and silently losing the GUI and VTK coverage is exactly the failure this job
exists to prevent. If some skip ever becomes genuinely unavoidable, add it to
that step deliberately, with a comment naming the missing software.

**No external service is required**, and none is configured:

| Dependency | Why CI does not need it |
| --- | --- |
| CST Studio Suite | `test_cst_native_history.py` patches `win32com.client.DispatchEx` and `pythoncom`; no COM server is contacted. The `.bas` adapter is pure text generation. |
| Ollama | `test_assistant.py` patches `urllib.request.urlopen` with a local fake. |
| Gemini / Groq / OpenRouter keys | `test_antenna_llm_planner.py` writes throwaway `.env` fixtures with dummy values and asserts the missing-key errors. No request leaves the runner. |
| A GPU vendor driver | `test_antenna_vtk_preview.py` only builds and inspects `vtkPolyData` and computes viewport scaling, with no render window at all. The builder-page tests do construct the preview widget, which creates an **offscreen** `vtkRenderWindow`; its assertions count actors rather than pixels, so they do not depend on a successful draw. See the note below. |

The complete verbose log is uploaded as the `test-output-windows-py312`
artifact on every run, including failures, so a red build can be diagnosed
without rerunning it.

**One thing to watch on the first run.** The antenna preview creates an
offscreen `vtkRenderWindow`, and GitHub's Windows images have no GPU vendor
driver. Actor-count assertions do not need a successful draw, so this is
expected to be fine, but it has not been proven on a hosted runner. If VTK
rendering turns out to be the thing that fails, the log will say so plainly, and
the fix is to provide a software OpenGL implementation on the runner (a Mesa
`opengl32.dll` alongside the interpreter) rather than to skip the tests.

## Documentation

User-facing documents are expected to describe what exists today:

- [`README.md`](README.md) — overview, install, scope of each half of the tool
- [`USER_MANUAL.md`](USER_MANUAL.md) — page-by-page operating instructions
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) — current product boundaries
- [`CHANGELOG.md`](CHANGELOG.md) — one entry per release

If a change adds a user-visible control, alters an exported artifact, or removes
a limitation, say so in the changelog and update the affected document in the
same change. Historical defect registers and development stage notes do not
belong in `docs/`; they live in the commit history.
