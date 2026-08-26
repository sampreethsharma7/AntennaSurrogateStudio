# Installing Antenna Surrogate Studio

This tester build runs directly from its source folder. It creates one private
Python environment at `.venv`; it does not install files elsewhere or require
you to activate that environment manually.

## Requirements

- A 64-bit Windows, macOS, or Linux computer.
- 64-bit Python 3.11, 3.12, or 3.13.
- Internet access during the first dependency installation.
- About 500 MB of free disk space for the source and private Python
  environment, excluding optional Ollama models and user projects.
- On Linux, Python Tk support (commonly the `python3-tk` package).
- Ollama is optional. The rest of the Studio, including SnowBuddy's built-in
  workflow guide, works without it.

CST, HFSS, a paid API, and a cloud account are not required to install or
launch the Studio. Solver data is exchanged through files when you choose to
use an external simulator.

Windows 10/11 is the primary tested platform. The macOS/Linux launch scripts
are provided, but this release was not freshly verified on those operating
systems.

## Windows: recommended setup

1. Clone the repository or download and extract its ZIP file.
2. Keep the extracted folder in a writable location such as Documents.
3. Double-click `Start Antenna Surrogate Studio.bat`.
4. Allow the first-run setup to create `.venv` and install the requirements.
   The Studio opens automatically when setup finishes.

Later launches reuse that same `.venv` and do not reinstall dependencies.
If Python is missing and Windows Package Manager (`winget`) is available, the
setup offers the standard Python 3.12 installation path automatically.

If the graphical launcher closes without opening the Studio, double-click
`run_studio.bat`. It keeps the diagnostic message visible.

Command-line cloning is optional:

```powershell
git clone https://github.com/sampreethsharma7/AntennaSurrogateStudio.git
Set-Location AntennaSurrogateStudio
```

## Windows: explicit command-line setup

Run these commands in PowerShell from the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Activation is optional because every command addresses the private environment
directly. Use `py -3.11` or `py -3.13` if that is the supported version already
installed on the computer.

## macOS and Linux

From a terminal in the repository folder:

```bash
bash start_studio.sh
```

The first launch runs `setup_macos_linux.sh`, creates `.venv`, installs the
requirements, and opens the Studio. If Tk is unavailable on Debian/Ubuntu,
install it first with your system package manager (commonly `python3-tk`). On
macOS, the Python installer from python.org includes Tk support.

## Optional local SnowBuddy model

SnowBuddy can use Ollama entirely on the local computer. Install Ollama from
<https://ollama.com/download>, start it, then open **SnowBuddy > Local model**
inside the Studio and choose one profile:

- `qwen3:1.7b` — about 1.4 GB; suitable for lower-resource machines.
- `qwen3:8b` — about 5.2 GB; recommended for machines with at least 16 GB RAM
  or a suitable GPU.

Ollama and a Qwen download are not required to launch the Studio. Without them,
SnowBuddy uses its built-in, project-aware workflow guidance.

## What setup creates

The source folder should contain `app.py`, `studio/`, `snowbuddy/`, and
`requirements.txt`. Setup adds only the ignored `.venv/` directory there.
Projects and Model Books are stored separately by default at:

```text
Documents/Antenna Surrogate Studio Library/projects/
```

Set `ANTENNA_STUDIO_LIBRARY` before launch to use another library location.

## Common fixes

- **Python not found or unsupported:** install 64-bit Python 3.11–3.13 and
  enable the Python launcher during installation.
- **GitHub says the repository was not found:** this tester repository is
  private. Ask the owner for repository access or use a ZIP supplied by them.
- **Tkinter/Tk error:** reinstall Python with Tcl/Tk enabled; on Linux install
  the distribution's Tk package.
- **Dependency installation failed:** confirm internet access, delete only the
  incomplete `.venv` folder, and run the setup again.
- **Windows Application Control blocked a dependency:** keep the extracted
  Studio in a writable Documents folder. On an organization-managed computer,
  an administrator may need to allow Python scientific-package DLLs.
- **Studio window did not appear on Windows:** run `run_studio.bat` and read the
  retained error message.
- **SnowBuddy says the local model is unavailable:** start Ollama and download
  a model in **Local model**, or continue with the built-in workflow guide.
- **Training feels slow:** begin with Linear Regression or a Medium search;
  XGBoost, Neural Network, Ensemble Auto High, and large datasets need more CPU
  time and memory.
