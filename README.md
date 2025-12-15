# PC Keyboard Piano

<img width="1402" height="732" alt="image" src="https://github.com/user-attachments/assets/97841cee-43b5-43b6-b46f-bd02be917bde" />


This application is built with a focus on low-latency audio, a responsive user interface, and powerful recording/production tools, making it more than just a simple toy. It uses a custom synthesizer engine built with NumPy and is rendered using Pygame.

---

## Core Features

*   **Low-Latency Audio Engine**: Prioritizes modern audio drivers like **WASAPI** (Windows) and **CoreAudio** (macOS) to minimize input lag for a responsive playing experience.
*   **Custom Synthesizer**: Generates a warm, piano-like tone using a custom harmonic series and an ADSR envelope, rather than relying on static sample files.
*   **Responsive UI**: A clean, modern user interface with visual feedback for pressed keys.
*   **Full Keyboard & Mouse Support**: Play using the standard two-octave keyboard layout or by clicking the keys with your mouse.
*   **Sustain Pedal**: Use the `Tab` key to simulate a sustain pedal.
*   **Octave Shifting**: Easily shift octaves up and down to access the full MIDI range from C1 to C8.
*   **Built-in Reverb**: Add depth and ambiance to your sound with an adjustable stereo reverb effect.

## Recording & Production Suite

*   **Multi-Take Recording**: Record your performance. Stop and start recording to create multiple "takes".
*   **Overdubbing**: Enable overdub mode to layer new performances on top of previous takes within the same session.
*   **Background Rendering**: When you stop recording, the audio is rendered in a separate thread, keeping the UI fully responsive.
*   **Undo Last Take**: Made a mistake? Easily remove the last recorded take and try again.
*   **Count-in & Metronome**: A fully functional metronome with adjustable BPM and an optional count-in to ensure you start on beat.
*   **High-Quality Export**: Save your final composition as a **WAV** (lossless) or **MP3** (compressed) file.

---

## Controls

| Key(s)              | Action                                        | Category      |
| ------------------- | --------------------------------------------- | ------------- |
| `a,s,d,f,g,h,j,k,l,;,` | Play White Keys (F3 to B4)                    | Playing       |
| `w,e,r,y,u,o,p,[`    | Play Black Keys                               | Playing       |
| `LShift` / `RShift` | Octave Down                                   | Playing       |
| `Space`             | Octave Up                                     | Playing       |
| `Tab`               | Hold for Sustain                              | Playing       |
| `1`                 | Start / Stop Recording & Render               | Recording     |
| `2`                 | Save Rendered Audio as WAV                    | File          |
| `3`                 | Save Rendered Audio as MP3                    | File          |
| `4`                 | Preview Rendered Audio                        | Playback      |
| `5`                 | Stop Preview                                  | Playback      |
| `6`                 | Toggle Overdub Mode                           | Recording     |
| `7`                 | Undo Last Take                                | Recording     |
| `8`                 | Cancel In-Progress Render                     | Recording     |
| `9`                 | Clear All Takes (New Session)                 | Recording     |
| `0`                 | Toggle Metronome                              | Tools         |
| `F1` / `F2`         | Decrease / Increase Metronome BPM             | Tools         |
| `F3`                | Cycle Count-in Bars (0, 1, 2)                 | Tools         |
| `F4`                | Toggle Reverb                                 | Effects       |
| `F5` / `F6`         | Decrease / Increase Reverb Wetness            | Effects       |
| `F7` / `F8`         | Decrease / Increase Master Volume             | Audio         |
| `F9` / `F10`        | Decrease / Increase Octave                    | Playing       |
| `ESC`               | Quit the Application                          | Application   |

---

## Installation & Usage

### Prerequisites
*   Python 3.8+
*   `pip` (Python package installer)
*   (Optional for MP3 export) **FFmpeg**: Must be installed and accessible in your system's PATH.

### Steps
1.  **Clone the repository or save the script:**
    Save the provided Python script as `pc_piano.py`.

2.  **Create a `requirements.txt` file** in the same directory with the following content:
    ```
    pygame
    numpy
    soundfile
    pydub
    ```

3.  **Create and activate a virtual environment:**
    ```sh
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

4.  **Install the dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5.  **Run the application:**
    ```sh
    python pc_piano.py
    ```

## Advanced Configuration

You can fine-tune the audio buffer and frequency for your specific hardware by setting environment variables before running the script. Lower buffer sizes reduce latency but may cause audio crackling on some systems.

*   `PIANO_MIXER_BUFFER`: (e.g., 64, 128, 256, 512). Default is `64`.
*   `PIANO_MIXER_FREQ`: (e.g., 44100, 48000).

**Example (Windows Command Prompt):**
```cmd
set PIANO_MIXER_BUFFER=128
python pc_piano.py
```

**Example (macOS/Linux):**
```sh
PIANO_MIXER_BUFFER=128 python pc_piano.py
```
