@echo off
setlocal enabledelayedexpansion

:: ============================================================================
::  Pyano Keyboard  --  EXE Builder
::  Strictly requires Python 3.13.12
::  Works even when Python is NOT on PATH (uses the "py" launcher).
::
::  Put this .bat in the SAME folder as:
::     - Keyboard.py       (the app script)
::     - icon.ico          (window/exe icon)   [optional]
::     - version.txt       (exe version info)  [optional]
::     - requirements.txt  (dependencies)      [optional]
::  Then just double-click it.
::
::  This build does NOT use a .spec file. PyInstaller generates one from the
::  command-line flags below, so there is nothing extra to keep in the repo.
:: ============================================================================

:: ---- EDIT THESE IF YOU RENAME FILES --------------------------------------
set "SCRIPT_NAME=Keyboard.py"
set "EXE_NAME=Pyano Keyboard"
set "ICON=icon.ico"
set "VERSION_FILE=version.txt"
:: --------------------------------------------------------------------------

set "REQUIRED_VERSION=3.13.12"
set "DOWNLOAD_URL=https://www.python.org/downloads/release/python-31312/"

echo [INFO] Checking Python version...

:: --- Find an interpreter that is EXACTLY 3.13.12 --------------------------
:: Prefer the "py" launcher pinned to 3.13, then fall back to "python" on PATH.
:: The py launcher lives in C:\Windows and is reachable even when the
:: 'python' command on PATH is a different version.
set "PY_CMD="
set "VER_A="
set "VER_B="

for /f "tokens=2" %%I in ('py -3.13 --version 2^>nul') do set "VER_A=%%I"
if "!VER_A!"=="%REQUIRED_VERSION%" set "PY_CMD=py -3.13"

if not defined PY_CMD (
    for /f "tokens=2" %%I in ('python --version 2^>nul') do set "VER_B=%%I"
    if "!VER_B!"=="%REQUIRED_VERSION%" set "PY_CMD=python"
)

if not defined PY_CMD (
    set "CURRENT_VERSION=!VER_B!"
    if not defined CURRENT_VERSION set "CURRENT_VERSION=!VER_A!"
    if not defined CURRENT_VERSION set "CURRENT_VERSION=None"
    if "!CURRENT_VERSION!"=="" set "CURRENT_VERSION=None"
    goto :WrongVersion
)

echo [INFO] Python %REQUIRED_VERSION% detected via "!PY_CMD!". Starting build...
echo =======================================================

:: --- Locate the app script ------------------------------------------------
if not exist "%SCRIPT_NAME%" (
    echo [ERROR] "%SCRIPT_NAME%" not found next to this script.
    echo Put this .bat next to it, or edit SCRIPT_NAME at the top of this file.
    goto :error
)
echo [INFO] Using app script: "%SCRIPT_NAME%"

:: 1. Create Virtual Environment
echo [STEP 1/6] Creating virtual environment in '.venv'...
if not exist .venv (
    %PY_CMD% -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv creation failed. & goto :error )
) else (
    echo [INFO] '.venv' already exists. Skipping creation.
)

:: 2. Activate Virtual Environment
echo [STEP 2/6] Activating virtual environment...
call ".venv\Scripts\activate.bat"
if not defined VIRTUAL_ENV (
    echo [ERROR] Failed to activate the virtual environment.
    echo Make sure '.venv\Scripts\activate.bat' exists.
    goto :error
)

:: 3. Upgrade build tools
echo [STEP 3/6] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 ( echo [ERROR] Failed to upgrade build tools. & goto :error )

:: 4. Install Requirements
echo [STEP 4/6] Installing dependencies from requirements.txt...
if exist requirements.txt (
    python -m pip install -r requirements.txt
    if errorlevel 1 ( echo [ERROR] Failed to install dependencies. & goto :error )
) else (
    echo [INFO] No requirements.txt found. Skipping.
)

:: 5. Install PyInstaller
echo [STEP 5/6] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 ( echo [ERROR] PyInstaller install failed. & goto :error )

:: 6. Build the executable (no .spec; PyInstaller generates one from flags)
::    --onefile     one-file exe
::    --windowed    GUI app, no console window (Pyano opens its own window)
::    --clean       clear PyInstaller cache first
::    --noconfirm   overwrite a previous build without asking
::    --noupx       do not use UPX compression
::    --icon / --add-data icon.ico  app icon (added only if icon.ico exists)
::    --version-file version.txt     Windows file-details version (if present)
echo [STEP 6/6] Building executable...

set "HAVE_ICON=0"
set "HAVE_VER=0"
if exist "%ICON%" set "HAVE_ICON=1"
if exist "%VERSION_FILE%" set "HAVE_VER=1"

if "!HAVE_ICON!!HAVE_VER!"=="11" (
    echo [INFO] Building with icon and version info.
    pyinstaller --onefile --windowed --clean --noconfirm --noupx --add-data "%ICON%;." --icon "%ICON%" --version-file "%VERSION_FILE%" --name "%EXE_NAME%" "%SCRIPT_NAME%"
) else if "!HAVE_ICON!"=="1" (
    echo [INFO] %VERSION_FILE% missing - building without version metadata.
    pyinstaller --onefile --windowed --clean --noconfirm --noupx --add-data "%ICON%;." --icon "%ICON%" --name "%EXE_NAME%" "%SCRIPT_NAME%"
) else if "!HAVE_VER!"=="1" (
    echo [INFO] %ICON% missing - building without a custom icon.
    pyinstaller --onefile --windowed --clean --noconfirm --noupx --version-file "%VERSION_FILE%" --name "%EXE_NAME%" "%SCRIPT_NAME%"
) else (
    echo [INFO] %ICON% and %VERSION_FILE% missing - building bare.
    pyinstaller --onefile --windowed --clean --noconfirm --noupx --name "%EXE_NAME%" "%SCRIPT_NAME%"
)

if errorlevel 1 (
    echo =======================================================
    echo [ERROR] PyInstaller build failed. Scroll up for the error.
    goto :error
)

echo.
echo [SUCCESS] Build completed successfully.
echo Your executable is here:  dist\%EXE_NAME%.exe
goto :end

:WrongVersion
echo =======================================================
echo [ERROR] Incorrect Python Version!
echo.
echo You currently have: Python !CURRENT_VERSION!
echo This script requires exactly: Python %REQUIRED_VERSION%
echo.
echo Please download and install Python %REQUIRED_VERSION% from here:
echo %DOWNLOAD_URL%
echo.
echo During installation, enable the "py launcher" option (and optionally
echo "Add Python to PATH").
echo =======================================================
start "" "%DOWNLOAD_URL%"
goto :end

:error
echo.
echo [FAILURE] The build process failed. Please check the errors above.
echo.
pause
exit /b 1

:end
echo.
pause
endlocal
