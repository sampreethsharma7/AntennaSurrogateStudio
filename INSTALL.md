# Installation Guide

This guide covers the first installation, normal launches, and common setup
problems. Antenna Surrogate Studio runs from its downloaded folder and creates
one private Python environment named `.venv` inside that folder.

## Before you begin

You need:

- A 64-bit Windows, macOS, or Linux computer.
- 64-bit Python 3.11, 3.12, or 3.13.
- Internet access during the first setup.
- Approximately 500 MB of free space for the application environment, plus
  space for your projects and any optional local AI model.

Windows 10 and 11 are the primary tested platforms. CST, HFSS, an API key, and
a cloud account are not required to install or launch the Studio.

## Windows: recommended installation

1. Download the repository ZIP and extract it to a writable location such as
   `Documents`, or clone it with Git.
2. Open the extracted `AntennaSurrogateStudio` folder.
3. Double-click **Start Antenna Surrogate Studio.bat**.
4. Wait while the first-run setup creates `.venv` and installs the requirements.
5. The Studio opens automatically.

If Python is not installed and Windows Package Manager is available, setup may
offer to install Python 3.12. Otherwise install 64-bit Python from
[python.org](https://www.python.org/downloads/) and run the launcher again.

### Launching after setup

Double-click **Start Antenna Surrogate Studio.bat** whenever you want to use the
Studio. It reuses the existing `.venv`; do not create or activate another
environment.

If the launcher closes before the Studio appears, run **run_studio.bat**. Its
window remains open so you can read the error message.

## Windows: manual command-line installation

From PowerShell in the repository folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Use `py -3.11` or `py -3.13` if that is the supported Python version already
installed on your computer. Environment activation is optional because these
commands address `.venv` directly.

## macOS and Linux

Open a terminal in the repository folder and run:

```bash
bash start_studio.sh
```

The first launch creates `.venv`, installs the requirements, and opens the
Studio. Later launches reuse the same environment.

If Linux reports that Tkinter is unavailable, install the Python Tk package for
your distribution. On Debian or Ubuntu this is commonly:

```bash
sudo apt install python3-tk
```

On macOS, the Python installer from python.org includes Tk support.

## Optional SnowBuddy local model

The Studio works without an external AI service. To enable SnowBuddy's local
conversational model:

1. Install [Ollama](https://ollama.com/download).
2. Start Ollama.
3. Open the Studio.
4. Open **SnowBuddy > Local model**.
5. Choose `qwen3:1.7b` or `qwen3:8b` and follow the download prompt.

The smaller model is intended for lower-resource computers. The 8B model is
recommended when the computer has approximately 16 GB RAM or more.

## Where projects are stored

The default project library is:

```text
Documents/Antenna Surrogate Studio Library/projects/
```

To use a different location, set the `ANTENNA_STUDIO_LIBRARY` environment
variable before launching the Studio.

To move a project to another computer, copy its complete project folder into a
location you can open from the Studio. The Python `.venv` does not need to be
copied; it belongs to the application folder, not to a project.

## Common setup problems

### Python was not found

Install 64-bit Python 3.11, 3.12, or 3.13 from python.org. On Windows, include
the Python launcher during installation.

### Dependency installation failed

Check your internet connection. If the first setup was interrupted, delete only
the incomplete `.venv` folder and start the launcher again. Do not delete your
project library.

### The Studio window did not appear

On Windows, run `run_studio.bat` and read the retained error. On macOS or Linux,
run `bash start_studio.sh` from a terminal and read the terminal output.

### Tkinter or Tk is missing

Reinstall Python with Tcl/Tk support. On Linux, install the distribution's
Python Tk package.

### Windows blocked a scientific-package file

Keep the extracted Studio folder in a writable location such as `Documents`.
On an organization-managed computer, an administrator may need to allow Python
scientific-package DLLs.

### SnowBuddy says the local model is unavailable

Start Ollama and confirm the selected model is installed, or continue using
SnowBuddy's built-in workflow guide without Ollama.

### Training is slow

Start with Linear Regression and Auto Medium. Larger datasets, Neural Network,
XGBoost, Ensemble AI Engine, and High searches require more time and memory.

## Removing the Studio

Delete the downloaded application folder to remove the code and `.venv`.
Projects are stored separately, so delete the project library only if you also
want to remove your projects, models, results, and local conversations.

## Support

- Author: **Sai Sampreeth Indharapu**
- Email: [sampreethsharma@gmail.com](mailto:sampreethsharma@gmail.com)
- LinkedIn: [Sai Sampreeth Indharapu, Ph.D.](https://www.linkedin.com/in/sai-sampreeth-indharapu-ph-d-a98802110/)
