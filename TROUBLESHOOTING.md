# Troubleshooting

## Holding some keys stops other keys from working

This is the most common report, and it is almost always keyboard ghosting,
which is a hardware limit rather than a bug in the app.

### Why it happens

Keyboards register presses through a grid of rows and columns. To keep the cost
down, most keyboards can only report a limited number of presses at once, and
only in certain combinations. When you hold two keys that share a row or column,
a third key on a shared line gets blocked before it ever reaches the computer.
That is why the blocking is partial and inconsistent. It depends entirely on
which keys share wiring on your particular keyboard.

### Quick self-test

1. Search online for "keyboard ghosting test" and open any of the rollover
   testers.
2. Hold the exact combination of keys that fails in the piano.
3. If the tester also fails to show one of the keys, the limit is your keyboard.
   The piano is receiving exactly what the keyboard sends.

### What actually fixes it

* Use a keyboard rated for N-key rollover (NKRO) or 6-key rollover. Most gaming
  mechanical keyboards can register many keys at once.
* Re-finger chords to avoid the combinations your keyboard drops.
* Remap notes with `F12` onto keys that do not share wiring, if you can find a
  set that works for the chords you play.

No software trick can recover a key the keyboard never sent.

## A macro tool or remapper is interfering

While the piano window is focused it turns off OS key auto-repeat and grabs the
keyboard, which bypasses most global hotkeys and remappers. If a macro utility
still gets in the way, it is intercepting keys at a lower level than a normal
application can reach. Close or disable that tool while you play.

The grab is released automatically when you switch away or minimize, so it will
not lock up the rest of your system.

## Notes get stuck or keep ringing

Switch away from the window and back, or press `Esc` to quit. The app releases
all held notes when it loses focus, so changing focus clears anything stuck. If
it keeps happening, it is usually a long reverb tail, so lower the reverb with
`F5`.

## No sound at all

The status bar reports if the mixer failed to start. Try a different buffer, for
example `set PIANO_MIXER_BUFFER=256` on Windows, then relaunch. Also make sure
no other program has taken exclusive control of the audio device.

## MP3 export fails and leaves a WAV behind

MP3 export needs both pydub and FFmpeg on your PATH. Install FFmpeg, or just
export WAV with `2`, which needs no extra tools.
