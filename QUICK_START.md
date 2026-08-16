# Quick Start: Building the Executable

This guide explains how to create a standalone `.exe` for the PC Keyboard Piano
using PyInstaller and the included `Keyboard.spec` file.

Assumes Python and `pip` are installed.

### Step 1: Set up a virtual environment
Open a terminal in the project directory (the folder containing `Keyboard.py`).

```sh
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
You'll see `(venv)` at the start of the prompt when it's active.

### Step 2: Install dependencies
```sh
pip install -r requirements.txt
```
(`requirements.txt` already includes `pyinstaller`.)

### Step 3: Build from the spec file
`Keyboard.spec` is preconfigured with the entry point `Keyboard.py` and icon
`icon.ico`. From the project root:

```sh
pyinstaller Keyboard.spec
```

### Step 4: Locate and run
PyInstaller creates `build/` and `dist/`. Your app is in `dist/`:

* Open `dist/`.
* The bundled app folder is named **Pyano Keyboard**.
* Run the `.exe` inside it. Distribute the entire folder together.

The executable runs on compatible Windows machines without Python installed.
