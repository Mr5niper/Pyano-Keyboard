# PC Keyboard Piano

<img width="1402" height="732" alt="image" src="https://github.com/user-attachments/assets/97841cee-43b5-43b6-b46f-bd02be917bde" />

A playable piano you control from your computer keyboard. It uses a custom NumPy
synth engine and is drawn with Pygame, and it includes recording and export
tools alongside the basic playing.

## Core Features

* Low-latency audio. On Windows it prefers the WASAPI driver, and on macOS it
  prefers CoreAudio, to keep input lag down.
* A custom synthesizer that builds a warm, piano-like tone from a harmonic
  series and an ADSR envelope. It does not use recorded sample files.
* A clean interface with visual feedback when a key is pressed.
* Play with the computer keyboard or by clicking the keys with the mouse.
* Hold `Tab` for a sustain pedal.
* Shift octaves up and down to reach the full supported range (C1 to C8, MIDI
  notes 24 to 108).
* Adjustable stereo reverb.
* Remappable keys. Press `F12` to learn a custom layout and `F11` to reset to
  the default. Your layout is saved to `keymap.json`.
* Input isolation while the window is focused. See the section below for what
  this does and does not fix.

## Recording and Export

* Record multiple takes, and layer them with overdub mode.
* Rendering happens on a background thread, so the interface stays responsive
  after you stop recording.
* Undo the last take.
* A metronome with adjustable BPM and an optional count-in.
* Export to WAV, or to MP3 if you have pydub and FFmpeg installed.

## Controls

| Key(s)                         | Action                              | Category    |
| ------------------------------ | ----------------------------------- | ----------- |
| `a s d f g h j k l ; '`        | Play white keys (F3 to B4)          | Playing     |
| `w e r y u o p [`              | Play black keys                     | Playing     |
| `LShift` / `RShift`            | Octave down                         | Playing     |
| `Space`                        | Octave up                           | Playing     |
| `Tab`                          | Hold for sustain                    | Playing     |
| `1`                            | Start or stop recording and render  | Recording   |
| `2`                            | Save rendered audio as WAV          | File        |
| `3`                            | Save rendered audio as MP3          | File        |
| `4`                            | Preview rendered audio              | Playback    |
| `5`                            | Stop preview                        | Playback    |
| `6`                            | Toggle overdub mode                 | Recording   |
| `7`                            | Undo last take                      | Recording   |
| `8`                            | Cancel in-progress render           | Recording   |
| `9`                            | Clear all takes (new session)       | Recording   |
| `0`                            | Toggle metronome                    | Tools       |
| `F1` / `F2`                    | Decrease or increase metronome BPM  | Tools       |
| `F3`                           | Cycle count-in bars (0, 1, 2)       | Tools       |
| `F4`                           | Toggle reverb                       | Effects     |
| `F5` / `F6`                    | Decrease or increase reverb wetness | Effects     |
| `F7` / `F8`                    | Decrease or increase master volume  | Audio       |
| `F9` / `F10`                   | Decrease or increase octave         | Playing     |
| `F11`                          | Reset keymap to default             | Mapping     |
| `F12`                          | Learn a custom keymap               | Mapping     |
| `ESC`                          | Quit                                | Application |
| `` ` `` (grave)                | Toggle key diagnostic overlay       | Diagnostic  |
| `\` (backslash)                | Run the guided key test (8 steps)   | Diagnostic  |
| `=`                            | Toggle Raw Input vs SDL (Windows)   | Diagnostic  |

The number row, the function keys, `Tab`, `Shift`, `Space`, `Enter`,
`Backspace`, and `Esc` are used for controls, so they cannot be assigned to
notes.

## Input Isolation and Held-Key Issues

Read this if you have noticed that holding some keys stops other keys from
sounding.

### The held-key limit (hardware)

Most keyboards can only report a limited number of keys held at the same time,
and only in certain combinations. The keys are wired in a grid of rows and
columns, and keys that share a row or column cannot all be told apart when held
together, so some are dropped before they ever reach the computer. Which
combinations fail depends entirely on how your keyboard is wired, which is why
the pattern often looks like fixed pairs or groups of keys that will not sound
together.

This has been confirmed to be a hardware limit for the keyboard tested, not a
bug in the code. The app reads the keyboard two different ways: the normal path
(through SDL) and, on Windows, the Raw Input API, which reads key presses
straight from the HID layer below SDL. On a laptop built-in keyboard, a dropped
key was missing from both paths, which means the loss happened in the keyboard
or USB layer before any software this program can reach. The app's own key test
(see below) confirms it plays every key it actually receives.

Laptop built-in keyboards and membrane keyboards are the usual culprits; their
matrices are wired for typing, not for many simultaneous keys. A "gaming" or
"mechanical" label does not automatically make a keyboard immune, but it does
not doom it either. If a gaming keyboard drops keys, check its rollover mode
before concluding the hardware cannot do it (see below).

### What actually fixes it

* Use a keyboard with true **N-key rollover (NKRO)**. NKRO keyboards report
  every key independently, so held-key drops go away.
* **6-key rollover (6KRO)** helps but is not a full fix: it guarantees six
  arbitrary keys plus modifiers, which covers most chords but can still drop the
  seventh key.
* A "gaming" or "mechanical" label does not guarantee NKRO, and it does not rule
  it out. Many such keyboards ship in a 6KRO USB "boot" mode by default and only
  do full NKRO once you enable it. If yours has a companion app (for example
  Logitech G HUB) or an NKRO toggle (often an Fn key combination), turn NKRO on
  and run the key test again before deciding it is a hardware limit.
* Membrane and laptop keyboards generally cannot be fixed. Their matrices are
  wired for typing, not for many simultaneous keys.
* If you are stuck with the keyboard you have, you can re-finger chords to avoid
  the specific combinations it drops, or remap notes with `F12` onto keys that
  do not collide, once the key test tells you which keys conflict.

There is no software workaround for a hardware wiring limit. A key the keyboard
never sends cannot be recovered by any program. What the key test cannot tell
you is whether a keyboard's limit is permanent or just its current mode, so on a
keyboard that has an NKRO setting, enable it and retest before giving up on it.

### Diagnosing your keyboard

Press the backslash key (`\`) in the app to run the guided key test. It walks
you through several held-key combinations and writes `key_diagnostic_report.txt`
next to the program, listing exactly which keys were dropped in each
combination. That tells you which chords your keyboard cannot play. There is
also a live overlay on the grave key (`` ` ``) and a standalone probe,
`raw_input_test.py`; see TROUBLESHOOTING.md for details.

### Macros, OS key rules, and auto-repeat (software)

While the window is focused, the app turns off OS key auto-repeat, so holding a
key sends one note instead of a stream of repeats, and it grabs the keyboard so
global hotkeys, key remappers, and macro tools are less able to intercept your
keystrokes. Only the keyboard is grabbed. The mouse cursor is never confined and
can leave the window freely. The grab is released automatically when the window
loses focus or is minimized and taken again when you return, and all held notes
are released on focus loss so nothing sticks if you switch away mid-chord.

If a macro tool still steals keys, close it while you play. A running remapper
can consume keys before any application sees them, this one included.

### Raw Input (Windows)

On Windows the app reads the keyboard through the Raw Input API, which takes key
presses straight from the HID layer instead of the normal Windows message path
that SDL uses. It turns on automatically when available, and the info line shows
"Input: RAW" when it is active. Press `=` to switch between Raw and SDL. On other
systems, or if Raw Input cannot start, the app uses the standard path and the
toggle does nothing. Note that Raw Input does not cure the held-key limit
described above, because that loss happens below this layer too; it is the
better input path in general and is used to prove where a dropped key is lost.

## Installation and Usage


Requirements:

* Python 3.8 or newer
* pip
* FFmpeg on your PATH, if you want MP3 export

Steps:

1. Save the script as `Keyboard.py` (it is already in this repo).
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv

   # Windows
   .\venv\Scripts\activate

   # macOS or Linux
   source venv/bin/activate
   ```
3. Install the dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run it:
   ```sh
   python Keyboard.py
   ```

## Advanced Configuration

Set these environment variables before launching to tune the audio for your
hardware. A lower buffer reduces latency but can cause crackling on some
systems.

* `PIANO_MIXER_BUFFER`: one of 64, 128, 256, 512, 1024, 2048, 4096. The default
  is 64.
* `PIANO_MIXER_FREQ`: one of 22050, 44100, 48000, 96000.

Windows (Command Prompt):
```cmd
set PIANO_MIXER_BUFFER=128
python Keyboard.py
```

macOS or Linux:
```sh
PIANO_MIXER_BUFFER=128 python Keyboard.py
```

## Building an Executable

See `QUICK_START.md`. The build uses `Keyboard.spec`, with `Keyboard.py` as the
entry point and `icon.ico` as the icon.
