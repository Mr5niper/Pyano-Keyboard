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

The number row, the function keys, `Tab`, `Shift`, `Space`, `Enter`,
`Backspace`, and `Esc` are used for controls, so they cannot be assigned to
notes.

## Input Isolation and Held-Key Issues

Read this if you have noticed that holding some keys stops other keys from
sounding. There are two separate causes, and only one of them is something
software can fix.

### 1. Keyboard ghosting and rollover (hardware)

Most keyboards, especially membrane and laptop keyboards, can only register a
limited number of keys held at once, and only in certain combinations. The keys
are wired in a grid of rows and columns. When you hold two keys that sit on the
same row or column, a third key on a shared line can get blocked and never reach
the computer at all.

This is why the behavior looks strange. A held key blocks only some other keys,
not all of them, and the set of blocked keys depends on how your keyboard is
wired rather than on this program.

To check whether this is your problem, search for a "keyboard ghosting test"
online, then hold the same combination that fails in the piano. If keys drop in
the tester too, the limit is in your keyboard.

What actually helps:

* Use a keyboard rated for N-key rollover (NKRO) or at least 6-key rollover.
  Most gaming mechanical keyboards qualify.
* Choose fingerings that avoid the combinations your keyboard drops.

There is no software workaround for a hardware wiring limit. A key the keyboard
never sends cannot be recovered.

### 2. Macros, OS key rules, and auto-repeat (software)

While the window is focused, the app now does two things to keep other software
out of the way. It turns off OS key auto-repeat, so holding a key sends one
note instead of a stream of repeated presses that could crowd out other keys.
It also grabs the keyboard, which stops most global hotkeys, key remappers, and
macro tools from intercepting your keystrokes while you play.

The grab is released automatically when the window loses focus or is minimized,
and taken again when you come back, so it never interferes with the rest of your
system. All held notes are released on focus loss as well, so nothing sticks if
you switch away mid-chord.

If a macro tool still steals keys, close it while you play. A running remapper
can consume keys before any application sees them, this one included.

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
