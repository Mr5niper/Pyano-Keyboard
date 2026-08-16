# Quick Start: Building the Executable

You can build a standalone Windows .exe by double-clicking `BUILD_EXE.bat`. There
is nothing else to configure, and you do not need a `.spec` file. The script
generates the build settings on its own each time.

## What you need first

Python 3.13.12 installed, with the "py launcher" option enabled during
installation. The build script finds Python through the launcher, so Python does
not have to be on your PATH. If a different Python version is on PATH, that is
fine; the script still locates 3.13.12 through the launcher.

If you do not have the right version, the script opens the download page for you
and stops.

## How to build

1. Make sure these files are in the project folder:
   * `BUILD_EXE.bat`
   * `Keyboard.py`
   * `requirements.txt`
   * `icon.ico` (optional; used as the app icon if present)
   * `version.txt` (optional; embeds Windows version details if present)
2. Double-click `BUILD_EXE.bat`.

The script creates a virtual environment in `.venv`, installs the dependencies
and PyInstaller into it, and builds the executable. None of that gets committed
to git, because `.gitignore` excludes the virtual environment, the build folders,
and the generated `.spec`.

## Where the result goes

When the build finishes, your executable is at:

```
dist\Pyano Keyboard.exe
```

That single file is the whole app. You can copy it anywhere and run it on a
compatible Windows machine without Python installed.

## Rebuilding

Just run `BUILD_EXE.bat` again. It reuses the existing `.venv` if one is already
there and rebuilds the exe from scratch (`--clean --noconfirm`), so you always
get a fresh build.

## Renaming things

If you rename the script or want a different exe name, edit the four variables at
the top of `BUILD_EXE.bat` (`SCRIPT_NAME`, `EXE_NAME`, `ICON`, `VERSION_FILE`).
