# Troubleshooting

## Holding some keys stops other keys from working

If you hold certain combinations of keys and some of them do not sound, this is
almost always a hardware limit in the keyboard, not a bug in the app.

### Why it happens

Keyboards register presses through a grid of rows and columns. To keep the cost
down, most keyboards can only report a limited number of presses at once, and
only in certain combinations. Keys that share a row or column cannot all be told
apart when held together, so some are dropped before they ever reach the
computer. Which combinations fail depends entirely on how your keyboard is
wired, which is why it often looks like fixed pairs or groups of keys that will
not play together.

### Confirming where the loss happens

The app reads the keyboard two ways, and the key test shows which layer drops a
key so you are not guessing.

Press the backslash key (`\`) in the app to run the guided key test. It walks
you through several steps; each names a set of keys, asks you to hold them all
at once, then let go and click Next (or press Enter). Because the app records
what each step asked for, the result is unambiguous. When you finish, it writes
`key_diagnostic_report.txt` next to the program. For each step it lists the keys
asked for, the keys the app received, and the keys it played, classifying every
dropped key as one of:

* asked but never received: the key was lost before the app got it. On Windows,
  where the app also reads Raw Input from the HID layer, this means it was lost
  in the keyboard or USB layer, below any software the app can reach. No code
  change recovers it.
* received but never played: the app got the key and rejected it. That would be
  a software bug, and the report prints the exact reason.
* received and played: that key worked.

There is also a live overlay on the grave key (`` ` ``) that shows each key event
as it arrives and which keys are held now, for quick spot checks.

On Windows you can go one level deeper with `raw_input_test.py`:

```
python raw_input_test.py
```

It reads the keyboard through the Windows Raw Input API, below SDL, and writes
`raw_input_report.txt`. If a key you held is missing there too, it was lost in
the keyboard or USB layer and no software can recover it. (In testing on a laptop
built-in keyboard, the dropped keys were missing from Raw Input as well, which is
what confirms the limit is hardware on that keyboard.)

### What actually fixes it

* Use a keyboard with true **N-key rollover (NKRO)**. These report every key
  independently, so held-key drops go away.
* **6-key rollover (6KRO)** covers most chords but can still drop the seventh
  simultaneous key.
* A "gaming" or "mechanical" label does not guarantee NKRO and does not rule it
  out. Many such keyboards default to a 6KRO USB "boot" mode and only do full
  NKRO once you enable it. If yours has a companion app (for example Logitech G
  HUB) or an NKRO toggle (often an Fn combination), turn NKRO on and run the key
  test again before deciding the hardware is the limit.
* Laptop built-in and membrane keyboards generally cannot be fixed; their
  matrices are wired for typing, not for many simultaneous keys.
* Meanwhile, re-finger chords to avoid the combinations your keyboard drops, or
  remap notes with `F12` onto keys that do not collide, using the key test to
  see which keys conflict.

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
