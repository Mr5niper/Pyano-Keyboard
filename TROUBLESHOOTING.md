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

The easiest test is the built-in guided key test. With the piano window focused,
press the backslash key (`\`) to start it. It walks you through eight steps. Each
step names a set of keys and asks you to hold them all down at once, let go, and
then click Next (or press Enter). Because the app records what each step asked
for, the result is unambiguous.

When you finish the last step, the app writes a file named
`key_diagnostic_report.txt` next to the program. Open it. For every step it lists
the keys the step asked for, the keys the app received, and the keys it actually
played, so a dropped key is traced to one of three layers:

* asked but never received: the key was lost before the app got it. That is
  keyboard ghosting (hardware) or the SDL/OS input layer. No change inside this
  program can recover it.
* received but never played: the app got the key and its own logic rejected it.
  That is a software bug, and the report prints the exact reason next to the key.
* received and played: that key worked.

The report also records the raw event order with a timestamp, scancode, and
modifier state for each key, the mixer channel state at each note, and a short
environment section (pygame and SDL versions, video driver, whether key-repeat
and the keyboard grab are active). The last line is a plain verdict that says
which layer is responsible.

There is also a live overlay you can toggle any time with the grave key (`` ` ``,
above Tab). It shows each key event as it arrives and which keys are held right
now, which is handy for spot checks outside the guided test.

You can confirm the same thing outside the app too. Search online for a "keyboard
ghosting test," then hold a failing combination there. If keys drop in the tester
as well, the limit is in your keyboard.

### Definitive test: the raw-input probe

If the in-app test shows keys as MISSING, there is one more test that says
whether the loss is in the SDL/message layer (fixable in software) or below it
(not fixable). The file `raw_input_test.py` reads the keyboard through the
Windows Raw Input API, before SDL or window-message translation. Run it with:

```
python raw_input_test.py
```

Hold the same keys that fail in the piano, in the same order. The console shows a
live "held now" list of every key raw input currently sees down, and it writes
`raw_input_report.txt` when you close it.

* If raw input shows every key you held, but the piano marked some MISSING, the
  loss is in the SDL/message layer and the app can be switched to raw input.
* If raw input also misses the same keys, they are lost below any software this
  program can reach (the keyboard controller or the USB/HID stack), and no code
  change recovers them.

### What actually fixes it

* Use a keyboard rated for N-key rollover (NKRO) or 6-key rollover. Most gaming
  mechanical keyboards can register many keys at once.
* Re-finger chords to avoid the combinations your keyboard drops.
* Remap notes with `F12` onto keys that do not share wiring, if you can find a
  set that works for the chords you play.

No software trick can recover a key the keyboard never sent.

## A macro tool or remapper is interfering

While the piano window is focused it turns off OS key auto-repeat and grabs the
keyboard, which bypasses most global hotkeys and remappers. Only the keyboard is
grabbed; the mouse cursor is never trapped and can leave the window at any time.
If a macro utility still gets in the way, it is intercepting keys at a lower
level than a normal application can reach. Close or disable that tool while you
play.

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
