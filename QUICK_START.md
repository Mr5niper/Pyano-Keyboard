# Quick Start: Building the Executable

This guide explains how to create a standalone `.exe` file for the PC Keyboard Piano application using PyInstaller and a pre-existing `.spec` file.

This assumes you have Python and `pip` installed on your system.

### Step 1: Set Up a Virtual Environment

First, create an isolated Python environment. This ensures that the dependencies for this project don't conflict with other Python projects on your system.

Open your terminal or command prompt in the project directory.

```sh
# Create the virtual environment folder named 'venv'
python -m venv venv

# Activate the environment on Windows
.\venv\Scripts\activate

# Or, activate the environment on macOS/Linux
source venv/bin/activate
```
You will know it's active when you see `(venv)` at the beginning of your command prompt line.

### Step 2: Install All Required Dependencies

With the virtual environment active, install PyInstaller and all the libraries the script needs to run. It is highly recommended to have these listed in a `requirements.txt` file.

```sh
# Install PyInstaller and other packages from your requirements file
pip install requirements.txt
```

### Step 3: Build the Executable from the `.spec` file

PyInstaller uses a `.spec` file to understand how to bundle your application. Since you already have this file, the process is straightforward.

Make sure your `Keyboard.spec` file is in the root of your project directory. Run the following command:

```sh
pyinstaller Keyboard.spec
```

PyInstaller will read the configuration from the spec file and begin the bundling process. You will see a lot of output in the console as it analyzes imports and packages the files.

### Step 4: Locate and Run Your Executable

Once PyInstaller finishes, it will have created two new folders: `build` and `dist`.

Your final, runnable application is located inside the `dist` folder.

*   Navigate into the `dist` directory.
*   You will find a folder inside (usually with the same name as your app).
*   Inside that folder is your `.exe` file, along with all the necessary supporting files.

You can now run this `.exe` on any compatible Windows machine, even if it doesn't have Python installed. The entire `dist/your_app_name` folder must be distributed together.
```
