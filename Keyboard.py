import logging
import os
import threading
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import time
from functools import wraps
import sys
import re
import shutil
import json
import numpy as np
import pygame
import soundfile as sf

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc_keyboard_piano")


# ============================================================================
# Windows Raw Input keyboard reader
# ----------------------------------------------------------------------------
# Reads key up/down straight from the HID layer via the Windows Raw Input API,
# bypassing SDL's keyboard message translation. This exists because on some
# machines SDL/pygame drop simultaneous keydowns (certain key groups cap out at
# a couple of concurrent keys). Raw Input reports each physical key
# independently, so it can recover keys SDL never delivers.
#
# It maps Windows Virtual-Key codes to the same pygame key constants the rest of
# the app uses, so raw input drives the existing note handlers unchanged.
# On non-Windows, or if setup fails, RawKeyboard.available stays False and the
# app uses the normal pygame path.
# ============================================================================
class RawKeyboard:
    # Windows Virtual-Key -> pygame key constant, for the keys the app cares
    # about (note keys + control keys used while playing).
    _VK_TO_PYGAME = {}

    def __init__(self):
        self.available = False
        self._events = []          # queued (is_down, pygame_key) for the app
        self._down = set()         # VKs currently held (dedupe auto-repeat)
        self._thread = None
        self._stop = False
        self._hwnd = None
        if not sys.platform.startswith("win"):
            return
        try:
            self._setup_maps()
            self._start()
            self.available = True
            logger.info("Raw Input keyboard active (bypassing SDL key path).")
        except Exception as e:
            logger.warning("Raw Input unavailable, using pygame keys: %s", e)
            self.available = False

    def _setup_maps(self):
        # Build VK -> pygame constant map. Letters: VK 'A'..'Z' == ord('A')..,
        # pygame K_a.. == ord('a').. so add 0x20. Punctuation uses OEM VKs.
        m = {}
        for c in range(ord('A'), ord('Z') + 1):
            m[c] = c + 0x20  # pygame.K_a etc.
        m.update({
            0x30 + i: pygame.K_0 + i for i in range(10)  # 0-9 top row
        })
        m.update({
            0xBA: pygame.K_SEMICOLON,     # ;
            0xDE: pygame.K_QUOTE,         # '
            0xDB: pygame.K_LEFTBRACKET,   # [
            0xDD: pygame.K_RIGHTBRACKET,  # ]
            0xBC: pygame.K_COMMA,
            0xBE: pygame.K_PERIOD,
            0xBF: pygame.K_SLASH,
            0xDC: pygame.K_BACKSLASH,
            0xC0: pygame.K_BACKQUOTE,
            0xBD: pygame.K_MINUS,
            0xBB: pygame.K_EQUALS,
            0x20: pygame.K_SPACE,
            0x09: pygame.K_TAB,
            0x0D: pygame.K_RETURN,
            0x1B: pygame.K_ESCAPE,
            0x08: pygame.K_BACKSPACE,
            0xA0: pygame.K_LSHIFT, 0xA1: pygame.K_RSHIFT, 0x10: pygame.K_LSHIFT,
            0x70: pygame.K_F1, 0x71: pygame.K_F2, 0x72: pygame.K_F3,
            0x73: pygame.K_F4, 0x74: pygame.K_F5, 0x75: pygame.K_F6,
            0x76: pygame.K_F7, 0x77: pygame.K_F8, 0x78: pygame.K_F9,
            0x79: pygame.K_F10, 0x7A: pygame.K_F11, 0x7B: pygame.K_F12,
        })
        self._VK_TO_PYGAME = m

    def _start(self):
        import ctypes
        import ctypes.wintypes as wt
        import threading

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        WM_INPUT = 0x00FF
        WM_DESTROY = 0x0002
        RID_INPUT = 0x10000003
        RIM_TYPEKEYBOARD = 1
        RIDEV_INPUTSINK = 0x00000100
        RI_KEY_BREAK = 0x01
        HWND_MESSAGE = wt.HWND(-3)

        class RAWINPUTDEVICE(ctypes.Structure):
            _fields_ = [("usUsagePage", wt.USHORT), ("usUsage", wt.USHORT),
                        ("dwFlags", wt.DWORD), ("hwndTarget", wt.HWND)]

        class RAWINPUTHEADER(ctypes.Structure):
            _fields_ = [("dwType", wt.DWORD), ("dwSize", wt.DWORD),
                        ("hDevice", wt.HANDLE), ("wParam", wt.WPARAM)]

        class RAWKEYBOARD(ctypes.Structure):
            _fields_ = [("MakeCode", wt.USHORT), ("Flags", wt.USHORT),
                        ("Reserved", wt.USHORT), ("VKey", wt.USHORT),
                        ("Message", wt.UINT), ("ExtraInformation", wt.ULONG)]

        class RAWINPUT(ctypes.Structure):
            _fields_ = [("header", RAWINPUTHEADER), ("keyboard", RAWKEYBOARD)]

        WNDPROCTYPE = ctypes.WINFUNCTYPE(ctypes.c_long, wt.HWND, wt.UINT,
                                         wt.WPARAM, wt.LPARAM)

        class WNDCLASS(ctypes.Structure):
            _fields_ = [("style", wt.UINT), ("lpfnWndProc", WNDPROCTYPE),
                        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                        ("hInstance", wt.HINSTANCE), ("hIcon", wt.HANDLE),
                        ("hCursor", wt.HANDLE), ("hbrBackground", wt.HANDLE),
                        ("lpszMenuName", wt.LPCWSTR), ("lpszClassName", wt.LPCWSTR)]

        def handle_input(lparam):
            size = wt.UINT(0)
            user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, None,
                                   ctypes.byref(size),
                                   ctypes.sizeof(RAWINPUTHEADER))
            if size.value == 0:
                return
            buf = ctypes.create_string_buffer(size.value)
            if user32.GetRawInputData(wt.HANDLE(lparam), RID_INPUT, buf,
                                      ctypes.byref(size),
                                      ctypes.sizeof(RAWINPUTHEADER)) != size.value:
                return
            ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
            if ri.header.dwType != RIM_TYPEKEYBOARD:
                return
            vk = ri.keyboard.VKey
            if vk in (0, 0xFF):
                return
            is_down = (ri.keyboard.Flags & RI_KEY_BREAK) == 0
            pk = self._VK_TO_PYGAME.get(vk)
            if pk is None:
                return
            if is_down:
                if vk in self._down:
                    return  # ignore hardware auto-repeat
                self._down.add(vk)
            else:
                self._down.discard(vk)
            self._events.append((is_down, pk))

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                try:
                    handle_input(lparam)
                except Exception:
                    pass
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc_ref = WNDPROCTYPE(wnd_proc)  # keep ref alive

        def run():
            hInstance = kernel32.GetModuleHandleW(None)
            cls = WNDCLASS()
            cls.lpfnWndProc = self._wndproc_ref
            cls.hInstance = hInstance
            cls.lpszClassName = "PyanoRawInput"
            user32.RegisterClassW(ctypes.byref(cls))
            hwnd = user32.CreateWindowExW(0, "PyanoRawInput", "PyanoRawInput",
                                          0, 0, 0, 0, 0, HWND_MESSAGE, None,
                                          hInstance, None)
            self._hwnd = hwnd
            rid = RAWINPUTDEVICE()
            rid.usUsagePage = 0x01
            rid.usUsage = 0x06
            rid.dwFlags = RIDEV_INPUTSINK
            rid.hwndTarget = hwnd
            user32.RegisterRawInputDevices(ctypes.byref(rid), 1,
                                           ctypes.sizeof(RAWINPUTDEVICE))
            msg = wt.MSG()
            while not self._stop and user32.GetMessageW(ctypes.byref(msg),
                                                        None, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))

        self._user32 = user32
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        # Give the message loop a moment to register.
        time.sleep(0.05)

    def poll(self):
        # Return and clear queued (is_down, pygame_key) events.
        evs = self._events
        self._events = []
        return evs

    def clear(self):
        # Forget queued events and held-key state (used when unfocused).
        self._events = []
        self._down.clear()

    def stop(self):
        self._stop = True
        try:
            if self._hwnd and sys.platform.startswith("win"):
                import ctypes
                ctypes.windll.user32.PostMessageW(self._hwnd, 0x0012, 0, 0)  # WM_QUIT
        except Exception:
            pass



# Setup Profiler (Fix 5)
def timed(threshold: float = 0.1):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = func(*args, **kwargs)
            dt = time.perf_counter() - t0
            if dt >= threshold:
                logger.info("%s took %.3fs", func.__name__, dt)
            return out
        return wrapper
    return deco

# Helper functions for environment variable validation (A2)
def get_mixer_buffer() -> int:
    try:
        value = int(os.environ.get("PIANO_MIXER_BUFFER", "64"))
        if value not in (64, 128, 256, 512, 1024, 2048, 4096):
            logger.warning(f"Invalid PIANO_MIXER_BUFFER={value}, using 64")
            return 64
        return value
    except ValueError:
        logger.warning("Invalid PIANO_MIXER_BUFFER value, using 64")
        return 64

def get_mixer_freq() -> Optional[int]:
    freq_str = os.environ.get("PIANO_MIXER_FREQ")
    if not freq_str:
        return None
    try:
        freq = int(freq_str)
        if freq in (22050, 44100, 48000, 96000):
            return freq
        logger.warning(f"Invalid PIANO_MIXER_FREQ={freq}, ignoring")
        return None
    except ValueError:
        logger.warning("Invalid PIANO_MIXER_FREQ value, ignoring")
        return None

# Optional MP3 export dependency
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except Exception:
    PYDUB_AVAILABLE = False

# -------------------------
# Audio/Display Settings
# -------------------------
SAMPLE_RATE = 44100
BIT_DEPTH = -16      # 16-bit signed
CHANNELS = 2
MIXER_BUFFER = get_mixer_buffer()
PREF_FREQ = get_mixer_freq()

# Set audio driver BEFORE any pygame init
if 'SDL_AUDIODRIVER' not in os.environ:
    if os.name == 'nt':
        os.environ['SDL_AUDIODRIVER'] = 'wasapi'
    elif sys.platform == 'darwin':
        os.environ['SDL_AUDIODRIVER'] = 'coreaudio'
    else:
        os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'

def init_mixer() -> Tuple[bool, Optional[int], Optional[int]]:
    freqs = [PREF_FREQ] if PREF_FREQ is not None else [44100]
    for freq in freqs:
        if freq is None:
            continue
        for buf in (64, 128, 256, 512):
            try:
                pygame.mixer.quit()
                try:
                    pygame.mixer.init(
                        frequency=freq,
                        size=BIT_DEPTH,
                        channels=CHANNELS,
                        buffer=buf,
                        allowedchanges=0
                    )
                except TypeError:
                    pygame.mixer.init(
                        frequency=freq,
                        size=BIT_DEPTH,
                        channels=CHANNELS,
                        buffer=buf
                    )
                pygame.mixer.set_num_channels(96)
                logger.info(f"Mixer initialized at {freq} Hz, buffer {buf}")
                return True, freq, buf
            except pygame.error as e:
                logger.warning(f"Mixer init failed at {freq}/{buf}: {e}")
                continue
    logger.error("Audio mixer failed to initialize. Sound is disabled.")
    return False, None, None

class AudioConfig:
    def __init__(self):
        self.ok: bool = False
        self.freq: Optional[int] = None
        self.buffer: Optional[int] = None
    def init(self):
        self.ok, self.freq, self.buffer = init_mixer()

pygame.init()
try:
    if pygame.mixer.get_init():
        pygame.mixer.quit()
except Exception:
    pass

AUDIO = AudioConfig()
AUDIO.init()
if not AUDIO.ok:
    print("Audio mixer failed to initialize. Sound is disabled.")

# -------------------------
# UI Constants
# -------------------------
WIDTH, HEIGHT = 1400, 700
BACKGROUND = (18, 18, 18)
WHITE = (250, 250, 250)
BLACK = (12, 12, 12)
DARK_GRAY = (45, 45, 45)
PRESSED_WHITE = (100, 180, 255)
PRESSED_BLACK = (60, 140, 220)
RED = (255, 70, 70)
GREEN = (50, 205, 50)
BLUE = (70, 130, 255)
INFO_COLOR = (160, 160, 160)

# -------------------------
# Functional Constants & Safety Limits
# -------------------------
DEFAULT_NOTE_VOLUME = 0.33
DEFAULT_FADE_OUT_MS = 220
DEFAULT_REVERB_WET = 0.15
MAX_KEY_CHANNELS = 8
MAX_TAKES = 64
MAX_RECORD_SECONDS = 900
MAX_RENDER_SECONDS = 600
MAX_TOTAL_SAMPLES = MAX_RENDER_SECONDS * SAMPLE_RATE

# Path safety setup
RECORDINGS_DIR = Path("recordings").resolve()
RECORDINGS_DIR.mkdir(exist_ok=True)
SAFE_FILENAME_RE = re.compile(r'^[\w\-. ]{1,128}$')
RESERVED_WIN = {'con', 'prn', 'aux', 'nul', 'com1', 'com2', 'com3', 'com4', 'lpt1', 'lpt2', 'lpt3'}

def safepath(filename: str) -> Path:
    filename = Path(filename).name
    if not SAFE_FILENAME_RE.match(filename):
        raise ValueError(f"Invalid characters in filename: {filename}")
    if filename.startswith('.'):
        raise ValueError("Hidden filenames are not allowed")
    if os.name == 'nt' and filename.split('.')[0].lower() in RESERVED_WIN:
        raise ValueError("Reserved filename on Windows")
    path = (RECORDINGS_DIR / filename).resolve()
    path.relative_to(RECORDINGS_DIR)
    return path

# Type Aliases
EventRecord = Tuple[float, str, Optional[int]]
NoteRecord = Tuple[int, float, float]

# -------------------------
# Musical Helpers
# -------------------------
A4_FREQ = 440.0
A4_MIDI = 69
def midi_to_freq(midi_note: int) -> float:
    return A4_FREQ * (2 ** ((midi_note - A4_MIDI) / 12))
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def midi_to_name(midi_note: int) -> str:
    octave = (midi_note // 12) - 1
    name = NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"

MIN_MIDI = 24
MAX_MIDI = 108

# -------------------------
# Piano span (F3..B4)
# -------------------------
WHITE_MIDIS = [53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71]
BLACK_MIDIS = [54, 56, 58, 61, 63, 66, 68, 70]
BASE_MIDIS = sorted(WHITE_MIDIS + BLACK_MIDIS)

# Default PC-key mapping (by keycodes, not characters)
DEFAULT_KC_TO_MIDI: Dict[int, int] = {
    pygame.K_a: 53, pygame.K_s: 55, pygame.K_d: 57, pygame.K_f: 59, pygame.K_g: 60,
    pygame.K_h: 62, pygame.K_j: 64, pygame.K_k: 65, pygame.K_l: 67, pygame.K_SEMICOLON: 69, pygame.K_QUOTE: 71,
    pygame.K_w: 54, pygame.K_e: 56, pygame.K_r: 58, pygame.K_y: 61, pygame.K_u: 63,
    pygame.K_o: 66, pygame.K_p: 68, pygame.K_LEFTBRACKET: 70,
}

KEYMAP_FILE = Path("keymap.json")

def load_user_keymap() -> Optional[Dict[int, int]]:
    if not KEYMAP_FILE.exists():
        return None
    try:
        raw = json.load(open(KEYMAP_FILE, "r", encoding="utf-8"))
        out = {}
        for k, v in raw.items():
            kc = int(k)
            midi = int(v)
            if midi in BASE_MIDIS:
                out[kc] = midi
        # Ensure 1:1 (no duplicated keycodes or weird midis)
        if not out:
            return None
        return out
    except Exception as e:
        logger.warning(f"Failed to load keymap.json: {e}")
        return None

def save_user_keymap(kc_to_midi: Dict[int, int]):
    try:
        data = {str(k): int(v) for k, v in kc_to_midi.items() if v in BASE_MIDIS}
        with open(KEYMAP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save keymap.json: {e}")

# -------------------------
# Synthesizer
# -------------------------
class Synth:
    def __init__(self, audio_config: AudioConfig, sample_rate=SAMPLE_RATE):
        self.audio_config = audio_config
        self.sample_rate = sample_rate
        self.realtime_seconds = 6.0
        self.realtime_release = 0.8
        self.cache = {}
    def _harmonic_series(self, freq, t):
        partials = [
            (1.00, 1.00), (2.00, 0.55), (3.00, 0.30),
            (4.05, 0.20), (5.00, 0.15), (6.10, 0.10),
            (7.00, 0.08), (8.00, 0.05), (10.00, 0.03),
        ]
        wave = np.zeros_like(t, dtype=np.float32)
        for mult, amp in partials:
            wave += (amp * np.sin(2 * np.pi * freq * mult * t)).astype(np.float32)
        return wave
    def _adsr_envelope(self, total_dur, sustain_level=0.65, attack=0.008, decay=0.18, release=0.35):
        sr = self.sample_rate
        N = int(total_dur * sr)
        env = np.zeros(N, dtype=np.float32)
        aN = max(1, int(attack * sr))
        dN = max(1, int(decay * sr))
        rN = max(1, int(release * sr))
        sN = max(0, N - (aN + dN + rN))
        env[:aN] = np.linspace(0.0, 1.0, aN, endpoint=False, dtype=np.float32)
        env[aN:aN + dN] = np.linspace(1.0, sustain_level, dN, endpoint=False, dtype=np.float32)
        env[aN + dN:aN + dN + sN] = sustain_level
        start = aN + dN + sN
        env[start:start + rN] = np.linspace(sustain_level, 0.0, rN, endpoint=True, dtype=np.float32)
        return env
    def render_note(self, freq, duration_sec, volume=DEFAULT_NOTE_VOLUME, release=0.35):
        total_dur = max(0.02, duration_sec + release)
        t = np.linspace(0, total_dur, int(self.sample_rate * total_dur), endpoint=False, dtype=np.float32)
        wave = self._harmonic_series(freq, t)
        body_decay = np.exp(-t * 1.5).astype(np.float32)
        env = self._adsr_envelope(total_dur, sustain_level=0.65, attack=0.008, decay=0.18, release=release)
        signal = np.tanh(wave * env * body_decay * 1.2) * volume
        return signal.astype(np.float32)
    def get_realtime_sound(self, midi_note, volume=DEFAULT_NOTE_VOLUME):
        if midi_note not in self.cache:
            if not self.audio_config.ok or midi_note < MIN_MIDI or midi_note > MAX_MIDI:
                self.cache[midi_note] = None
                return None
            freq = midi_to_freq(midi_note)
            mono = self.render_note(freq, 1.5, volume=volume, release=0.3)
            int16_mono = np.clip(mono, -1.0, 1.0)
            int16_mono = (int16_mono * 32767).astype(np.int16)
            stereo = np.column_stack((int16_mono, int16_mono))
            try:
                sound = pygame.sndarray.make_sound(stereo)
                for _ in range(3):
                    ch = sound.play()
                    if ch:
                        ch.set_volume(0, 0)
                        ch.stop()
                self.cache[midi_note] = sound
            except Exception as e:
                logger.warning(f"Failed to cache MIDI {midi_note}: {e}")
                self.cache[midi_note] = None
        return self.cache.get(midi_note, None)
    def clear_cache(self):
        try:
            for sound in self.cache.values():
                if sound:
                    try:
                        sound.stop()
                    except Exception:
                        pass
        finally:
            self.cache.clear()
    def __del__(self):
        try:
            self.clear_cache()
        except Exception:
            pass

# -------------------------
# Modular Audio Processor (Reverb & Stereoization)
# -------------------------
def apply_reverb_stereo(mono: np.ndarray, sr: int, wet: float) -> np.ndarray:
    wet = float(max(0.0, min(1.0, wet)))
    if mono is None or len(mono) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if wet <= 1e-6:
        return np.column_stack((mono, mono)).astype(np.float32)
    taps_ms_L = [18, 43, 71, 97, 131, 173, 211, 263, 311]
    taps_ms_R = [24, 39, 68, 103, 149, 189, 233, 281, 327]
    gains = np.array([0.50, 0.40, 0.34, 0.28, 0.23, 0.19, 0.16, 0.13, 0.11], dtype=np.float32)
    tail_samples = int(sr * 0.35)
    out_len = len(mono) + tail_samples
    outL = np.zeros(out_len, dtype=np.float32)
    outR = np.zeros(out_len, dtype=np.float32)
    for d_ms, g in zip(taps_ms_L, gains):
        d = int(sr * d_ms / 1000.0)
        outL[d:d+len(mono)] += mono * g
    for d_ms, g in zip(taps_ms_R, gains):
        d = int(sr * d_ms / 1000.0)
        outR[d:d+len(mono)] += mono * g
    peak_wet = max(np.max(np.abs(outL)), np.max(np.abs(outR)), 1e-6)
    outL = outL / peak_wet * 0.7
    outR = outR / peak_wet * 0.7
    dry = np.column_stack((mono, mono)).astype(np.float32)
    if out_len > len(mono):
        pad = out_len - len(mono)
        dry = np.pad(dry, ((0, pad), (0, 0)), mode='constant')
    stereo = np.empty((out_len, 2), dtype=np.float32)
    np.multiply((1.0 - wet), dry[:out_len, 0], out=stereo[:, 0])
    np.add(stereo[:, 0], wet * outL, out=stereo[:, 0])
    np.multiply((1.0 - wet), dry[:out_len, 1], out=stereo[:, 1])
    np.add(stereo[:, 1], wet * outR, out=stereo[:, 1])
    peak_final = max(np.max(np.abs(stereo)), 1e-6)
    if peak_final > 0.99:
        stereo = stereo / peak_final * 0.99
    return stereo.astype(np.float32)

# -------------------------
# Metronome
# -------------------------
METRO_EVENT = pygame.USEREVENT + 42
class Metronome:
    def __init__(self, audio_config: AudioConfig, bpm=100):
        self.audio_config = audio_config
        self.bpm = bpm
        self.enabled = False
        self.click_main, self.click_sub = self._make_clicks()
        self.beat_counter = 0
        self.beats_per_bar = 4
    def _make_clicks(self):
        if not self.audio_config.ok:
            return None, None
        dur = 0.05
        t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False, dtype=np.float32)
        main = np.sin(2 * np.pi * 3000.0 * t) * np.exp(-t * 40.0) * 0.8
        sub = np.sin(2 * np.pi * 1800.0 * t) * np.exp(-t * 35.0) * 0.6
        def mk(w):
            i16 = (np.clip(w, -1.0, 1.0) * 32767).astype(np.int16)
            stereo = np.column_stack((i16, i16))
            try:
                return pygame.sndarray.make_sound(stereo)
            except Exception:
                return None
        return mk(main), mk(sub)
    def start(self):
        if not self.audio_config.ok:
            return
        self.enabled = True
        ms = max(50, int(60000 / max(1, self.bpm)))
        pygame.time.set_timer(METRO_EVENT, ms)
    def stop(self):
        self.enabled = False
        pygame.time.set_timer(METRO_EVENT, 0)
        self.beat_counter = 0
    def toggle(self):
        if self.enabled:
            self.stop()
        else:
            self.start()
    def set_bpm(self, bpm: int):
        self.bpm = int(max(30, min(300, bpm)))
        if self.enabled:
            self.start()
    def click(self):
        if not self.enabled:
            return
        snd = self.click_main if (self.beat_counter % self.beats_per_bar == 0) else self.click_sub
        if snd:
            snd.play()
        self.beat_counter += 1

# -------------------------
# Piano Key UI + Playback
# -------------------------
class PianoKey:
    def __init__(self, x, y, width, height, base_midi, kb_label, is_black, synth: Synth):
        self.rect = pygame.Rect(x, y, width, height)
        self.base_midi = base_midi
        self.kb = kb_label  # display label (string)
        self.is_black = is_black
        self.is_pressed = False
        self.synth = synth
        self.channels = []
    def play(self, midi, volume=1.0):
        if pygame.mixer.get_init():
            self.channels = [ch for ch in self.channels if ch and ch.get_busy()]
        if len(self.channels) >= MAX_KEY_CHANNELS:
            try:
                oldest = self.channels.pop(0)
                oldest.stop()
            except Exception:
                pass
        sound = self.synth.get_realtime_sound(midi)
        if sound:
            ch = sound.play()
            if ch:
                ch.set_volume(volume, volume)
                self.channels.append(ch)
        self.is_pressed = True
    def release(self, fade_ms=15):
        if pygame.mixer.get_init():
            for ch in self.channels:
                if ch and ch.get_busy():
                    try:
                        ch.fadeout(fade_ms)
                    except Exception:
                        pass
        self.channels = []
        self.is_pressed = False
    def draw(self, surface, font, note_font, octave_shift):
        name_midi = self.base_midi + 12 * octave_shift
        name = midi_to_name(name_midi)
        if self.is_black:
            color = (80, 150, 255) if self.is_pressed else (28, 28, 32)
            border = (100, 170, 255) if self.is_pressed else (18, 18, 20)
            text_color = (240, 240, 245)
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, border, self.rect, 2)
            inner = self.rect.inflate(-4, -4)
            overlay = pygame.Surface((inner.width, inner.height), pygame.SRCALPHA)
            pygame.draw.rect(overlay, (0, 0, 0, 30), overlay.get_rect(), 1)
            surface.blit(overlay, inner.topleft)
        else:
            color = (120, 190, 255) if self.is_pressed else (252, 252, 254)
            border = (100, 170, 255) if self.is_pressed else (200, 200, 205)
            text_color = (30, 30, 35)
            pygame.draw.rect(surface, color, self.rect)
            pygame.draw.rect(surface, border, self.rect, 2)
            shine = pygame.Rect(self.rect.x + 4, self.rect.y + 4, self.rect.width - 8, 30)
            shine_surf = pygame.Surface((shine.width, shine.height), pygame.SRCALPHA)
            shine_surf.fill((255, 255, 255, 20))
            surface.blit(shine_surf, shine.topleft)
        label = font.render(self.kb, True, (240,240,245) if self.is_black else (30,30,35))
        surface.blit(label, label.get_rect(center=(self.rect.centerx, self.rect.bottom - 35)))
        nlabel = note_font.render(name, True, (240,240,245) if self.is_black else (30,30,35))
        surface.blit(nlabel, nlabel.get_rect(center=(self.rect.centerx, self.rect.bottom - 15)))

# -------------------------
# Recorder
# -------------------------
class Recorder:
    def __init__(self, synth: Synth, app):
        self.synth = synth
        self.app = app
        self.set_status = getattr(app, 'set_status', lambda *args, **kwargs: None)
        self._lock = threading.Lock()
        self.is_rendering = False
        self.rendered_audio = None
        self._preview_sound = None
        self._render_cancel_event = threading.Event()
        self._save_lock = threading.Lock()
        self.overdub_mode = False
        self.count_in_bars = 0
        self._count_in_until = 0
        self.takes = []
        self.reset_active()
    def _set_rendering(self, value: bool) -> None:
        with self._lock:
            self.is_rendering = value
    def _is_rendering(self) -> bool:
        with self._lock:
            return self.is_rendering
    def reset_active(self):
        self.is_recording = False
        self.start_ms = 0
        self.events = []
        self.current_pressed = set()
        self.sustain = False
    def reset_all(self):
        self.reset_active()
        with self._lock:
            self.takes = []
            self.rendered_audio = None
            self._preview_sound = None
    def toggle_overdub(self):
        self.overdub_mode = not self.overdub_mode
        self.set_status(
            f"Overdub {'ON' if self.overdub_mode else 'OFF'}",
            BLUE if self.overdub_mode else DARK_GRAY,
            1400
        )
    def undo_last_take(self):
        removed = False
        with self._lock:
            if self.takes:
                self.takes.pop()
                self.rendered_audio = None
                self._preview_sound = None
                removed = True
        self._set_rendering(False)
        if removed:
            self.set_status("Last take removed.", DARK_GRAY, 1200)
        else:
            self.set_status("No takes to undo.", DARK_GRAY, 1200)
    def cycle_count_in(self):
        self.count_in_bars = (self.count_in_bars + 1) % 3
        self.set_status(f"Count-in bars: {self.count_in_bars}", DARK_GRAY, 1200)
    def start(self):
        if not self.overdub_mode:
            with self._lock:
                self.takes = []
                self.rendered_audio = None
                self._preview_sound = None
        self.reset_active()
        if self.app.metronome.enabled and self.count_in_bars > 0:
            beat_ms = max(50, int(60000 / max(1, self.app.metronome.bpm)))
            self._count_in_until = pygame.time.get_ticks() + self.count_in_bars * self.app.metronome.beats_per_bar * beat_ms
            self.is_recording = False
            self.start_ms = 0
            self.set_status(f"Count-in {self.count_in_bars} bar(s)...", BLUE, self.count_in_bars * 2000 + 500)
        else:
            self.is_recording = True
            self.start_ms = pygame.time.get_ticks()
            self.set_status("🔴 Recording...", RED, 1200)
    def maybe_begin_after_count_in(self):
        if self.start_ms == 0 and self._count_in_until and pygame.time.get_ticks() >= self._count_in_until:
            self.is_recording = True
            self.start_ms = pygame.time.get_ticks()
            self._count_in_until = 0
            self.set_status("🔴 Recording...", RED, 1200)
    def time_sec(self):
        if self.start_ms == 0:
            return 0.0
        return (pygame.time.get_ticks() - self.start_ms) / 1000.0
    def _record_time_ok(self) -> bool:
        return self.time_sec() <= MAX_RECORD_SECONDS
    def note_on(self, midi_note):
        if not self.is_recording or not (MIN_MIDI <= midi_note <= MAX_MIDI):
            return
        if not self._record_time_ok():
            self.set_status("Max recording length reached. Stopping...", RED, 2500)
            self.stop_and_render_threaded()
            return
        self.events.append((self.time_sec(), 'on', midi_note))
        self.current_pressed.add(midi_note)
    def note_off(self, midi_note):
        if not self.is_recording or not (MIN_MIDI <= midi_note <= MAX_MIDI):
            return
        if not self._record_time_ok():
            return
        self.events.append((self.time_sec(), 'off', midi_note))
        self.current_pressed.discard(midi_note)
    def sustain_on(self):
        if self.is_recording and self._record_time_ok():
            self.sustain = True
            self.events.append((self.time_sec(), 'sus_on', None))
    def sustain_off(self):
        if self.is_recording and self._record_time_ok():
            self.sustain = False
            self.events.append((self.time_sec(), 'sus_off', None))
    def stop_and_render_threaded(self):
        if self._count_in_until and self.start_ms == 0:
            self._count_in_until = 0
            self.set_status("Recording canceled during count-in.", RED, 1500)
            return
        if self._is_rendering():
            self.set_status("Already rendering, please wait.", RED, 2000)
            return
        if not self.is_recording:
            self.set_status("Not recording.", DARK_GRAY, 1500)
            return
        self.is_recording = False
        take_events = list(self.events)
        with self._lock:
            self.takes.append(take_events)
            if len(self.takes) > MAX_TAKES:
                self.takes.pop(0)
        all_events = []
        for t in self.takes:
            all_events.extend(t)
        if not all_events:
            self.set_status("No events recorded.", DARK_GRAY, 2000)
            return
        self.set_status("⏳ Rendering audio in background...", BLUE, 99999)
        self._set_rendering(True)
        self._render_cancel_event.clear()
        threading.Thread(target=self._render_worker, args=(list(all_events),), daemon=True).start()
    def cancel_render(self):
        if self._is_rendering():
            self._render_cancel_event.set()
            self.set_status("Render cancel requested...", RED, 1500)
    def _process_events(self, events: List[EventRecord]) -> Tuple[List[NoteRecord], float]:
        events.sort(key=lambda e: e[0])
        end_time = events[-1][0] if events else 0.0
        notes = []
        active_notes = {}
        pressed = set()
        sustain_active = False
        for (t, ev, payload) in events:
            if self._render_cancel_event.is_set():
                raise KeyboardInterrupt
            if ev == 'on':
                midi = payload
                pressed.add(midi)
                active_notes.setdefault(midi, []).append((t, False))
            elif ev == 'off':
                midi = payload
                pressed.discard(midi)
                if midi in active_notes and active_notes[midi]:
                    st, sus_hold = active_notes[midi].pop(0)
                    if sustain_active:
                        active_notes[midi].insert(0, (st, True))
                    else:
                        notes.append((midi, st, t))
            elif ev == 'sus_on':
                sustain_active = True
                for midi, lst in active_notes.items():
                    active_notes[midi] = [(st, True) for st, _ in lst]
            elif ev == 'sus_off':
                sustain_active = False
                for midi, lst in list(active_notes.items()):
                    kept = []
                    for (st, sus_hold) in lst:
                        if sus_hold and midi not in pressed:
                            notes.append((midi, st, t))
                        else:
                            kept.append((st, sus_hold))
                    active_notes[midi] = kept
        for midi, lst in active_notes.items():
            for (st, _sus) in lst:
                notes.append((midi, st, end_time))
        if not notes:
            raise ValueError("No notes to mix.")
        return notes, end_time
    @timed(0.05)
    def _mix_notes(self, notes: List[NoteRecord]) -> np.ndarray:
        if not notes:
            return np.zeros(0, dtype=np.float32)
        release_tail = 0.35
        last_end = max(end for (_m, _s, end) in notes)
        last_end = min(last_end + release_tail + 0.05, float(MAX_RENDER_SECONDS))
        total_samples = min(int(last_end * SAMPLE_RATE), MAX_TOTAL_SAMPLES)
        mix = np.zeros(total_samples, dtype=np.float32)
        for (midi, st, end) in notes:
            if self._render_cancel_event.is_set():
                raise KeyboardInterrupt
            st = max(0.0, min(st, MAX_RENDER_SECONDS))
            end = max(st + 0.001, min(end, MAX_RENDER_SECONDS))
            dur = max(0.01, end - st)
            freq = midi_to_freq(midi)
            mono = self.synth.render_note(freq, dur, volume=DEFAULT_NOTE_VOLUME, release=release_tail)
            start_idx = int(st * SAMPLE_RATE)
            end_idx = min(start_idx + len(mono), total_samples)
            seg_len = end_idx - start_idx
            if seg_len > 0:
                mix[start_idx:end_idx] += mono[:seg_len]
        peak = np.max(np.abs(mix)) if len(mix) else 0.0
        if peak > 1e-6:
            mix = mix / peak * 0.92
        return mix
    def _render_worker(self, events: List[EventRecord]):
        try:
            notes, _end_time = self._process_events(events)
            mix = self._mix_notes(notes)
            with self._lock:
                self.rendered_audio = mix.copy()
                self._preview_sound = None
                self.set_status("✅ Rendering complete. Save (S/M) or Preview (P).", GREEN, 4000)
        except ValueError:
            with self._lock:
                self.rendered_audio = None
                self._preview_sound = None
            self.set_status("No playable notes finalized.", DARK_GRAY, 2000)
        except KeyboardInterrupt:
            with self._lock:
                self.rendered_audio = None
                self._preview_sound = None
            self.set_status("Rendering canceled.", RED, 2000)
        except Exception as e:
            with self._lock:
                self.rendered_audio = None
                self._preview_sound = None
            logger.exception("Rendering failed")
            self.set_status(f"Rendering failed: {e.__class__.__name__}", RED, 5000)
        finally:
            self._set_rendering(False)
    def _get_rendered_copy(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.rendered_audio is None:
                return None
            return self.rendered_audio.copy()
    def _get_rendered_readonly(self) -> Optional[np.ndarray]:
        with self._lock:
            if self.rendered_audio is None:
                return None
            arr = self.rendered_audio
        return arr.view()
    def _trigger_save_threaded(self, save_func, filename=None):
        if self._is_rendering():
            self.set_status("Wait: Rendering is still in progress.", BLUE, 2000)
            return
        with self._lock:
            ready = self.rendered_audio is not None and len(self.rendered_audio) > 0
        if not ready:
            self.set_status("Nothing to save. Record and render first.", DARK_GRAY, 2500)
            return
        if not self._save_lock.acquire(blocking=False):
            self.set_status("Save already in progress...", BLUE, 1500)
            return
        self.set_status("Saving file in background...", INFO_COLOR, 99999)
        def _runner():
            try:
                save_func(filename)
            finally:
                self._save_lock.release()
        threading.Thread(target=_runner, daemon=True).start()
    def save_wav(self):
        self._trigger_save_threaded(self._write_wav_file)
    def _write_wav_file(self, filename=None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"piano_recording_{timestamp}.wav"
        try:
            path = safepath(filename)
            mono = self._get_rendered_copy()
            if mono is None:
                self.set_status("Nothing to save.", RED, 2500)
                return None
            stereo = apply_reverb_stereo(
                mono, SAMPLE_RATE,
                self.app.reverb_wet if self.app.reverb_enabled else 0.0
            )
            estimated_bytes = int(stereo.shape[0]) * 2 * 2
            usage = shutil.disk_usage(RECORDINGS_DIR)
            if usage.free < estimated_bytes * 2:
                self.set_status("Insufficient disk space to save WAV.", RED, 5000)
                return None
            sf.write(path, stereo, SAMPLE_RATE, subtype='PCM_16')
            self.set_status(f"✅ Saved WAV: {path.name}", GREEN, 4000)
            return path
        except ValueError as e:
            self.set_status(f"Security error saving WAV: {e}", RED, 5000)
            logger.error(f"Security error: {e}")
            return None
        except sf.LibsndfileError as e:
            logger.error(f"Audio encoding error: {e}")
            self.set_status("Audio encoding error while saving WAV.", RED, 5000)
            return None
        except (OSError, IOError) as e:
            logger.error(f"File I/O error: {e}")
            self.set_status("File I/O error while saving WAV.", RED, 5000)
            return None
        except Exception as e:
            logger.exception("Error saving WAV file")
            self.set_status(f"Error saving WAV: {e.__class__.__name__}", RED, 5000)
            return None
    def save_mp3(self):
        if not PYDUB_AVAILABLE:
            self.set_status("MP3 requires pydub + ffmpeg. Saving WAV instead.", RED, 3500)
            return self._trigger_save_threaded(self._write_wav_file)
        self._trigger_save_threaded(self._write_mp3_file)
    def _write_mp3_file(self, filename=None):
        tmp_name = "temp_render_for_mp3.wav"
        tmp_path = self._write_wav_file(filename=tmp_name)
        if tmp_path is None or not os.path.exists(tmp_path):
            return
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mp3_filename = filename or f"piano_recording_{timestamp}.mp3"
            mp3_path = safepath(mp3_filename)
            audio = AudioSegment.from_wav(str(tmp_path))
            audio.export(mp3_path, format="mp3", bitrate="192k")
            self.set_status(f"✅ Saved MP3: {mp3_filename}", GREEN, 4000)
        except Exception:
            logger.exception("Error saving MP3 file")
            self.set_status(f"MP3 failed (FFmpeg issue?): WAV left: {tmp_name}", RED, 6000)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                logger.warning(f"Could not delete temporary WAV: {tmp_path}")
    def preview_rendered(self):
        if self._is_rendering():
            self.set_status("Still rendering... please wait.", BLUE, 2000)
            return
        if not self.app.audio_config.ok:
            self.set_status("Mixer not initialized; cannot preview.", RED, 3000)
            return
        mono = self._get_rendered_readonly()
        if mono is None or len(mono) == 0:
            self.set_status("Nothing to preview.", DARK_GRAY, 2000)
            return
        try:
            stereo = apply_reverb_stereo(mono, SAMPLE_RATE,
                                         self.app.reverb_wet if self.app.reverb_enabled else 0.0)
            if stereo.size == 0:
                self.set_status("Nothing to preview.", DARK_GRAY, 2000)
                return
            int16 = np.clip(stereo, -1.0, 1.0)
            int16 = (int16 * 32767).astype(np.int16)
            snd = pygame.sndarray.make_sound(int16)
            pygame.mixer.stop()
            snd.play()
            with self._lock:
                self._preview_sound = snd
            self.set_status("Preview playing. Press X to stop.", INFO_COLOR, 99999)
        except Exception as e:
            logger.exception("Preview failed")
            self.set_status(f"Preview error: {e.__class__.__name__}", RED, 4000)
    def stop_preview(self):
        try:
            pygame.mixer.stop()
            self.set_status("Preview stopped.", DARK_GRAY, 1500)
        except Exception:
            pass

# -------------------------
# Piano Application
# -------------------------
class PianoApp:
    RESERVED_KC = {
        # Number row (app controls)
        pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
        pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9,
        # Function keys
        pygame.K_F1, pygame.K_F2, pygame.K_F3, pygame.K_F4, pygame.K_F5, pygame.K_F6,
        pygame.K_F7, pygame.K_F8, pygame.K_F9, pygame.K_F10, pygame.K_F11, pygame.K_F12,
        # Modifiers / control
        pygame.K_TAB, pygame.K_LSHIFT, pygame.K_RSHIFT, pygame.K_SPACE, pygame.K_ESCAPE,
        pygame.K_RETURN, pygame.K_BACKSPACE,
    }

    def __init__(self, audio_config: AudioConfig):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Pyano Keyboard | Grandmaster Build")

        # --- Input isolation (prevents macros / OS key rules from interfering) ---
        # 1) Disable OS keyboard auto-repeat. Without this, the OS injects a stream
        #    of synthetic KEYDOWN events for a held key, which can look like extra
        #    presses and can starve/mask other keys. We want ONE keydown per press.
        try:
            pygame.key.set_repeat(0)  # 0 = off
        except Exception:
            pass
        # 2) Grab the KEYBOARD ONLY while the window is focused so global
        #    hotkeys, remappers, and macro tools are less able to intercept keys
        #    mid-play. The mouse is NOT confined - the cursor can leave freely.
        #    Released automatically when focus is lost (see _set_input_grab).
        self._input_grabbed = False
        self._set_input_grab(True)
        # Raw Input path: read the keyboard below SDL to recover simultaneous
        # keys SDL drops. When active, pygame key events are ignored (raw input
        # drives the note handlers instead) so keys are not counted twice.
        self.raw_kb = RawKeyboard()
        self._raw_supported = self.raw_kb.available
        self._eq_down = False   # debounce for the '=' raw/SDL toggle
        # --- Key diagnostic overlay (toggle with the ` / ~ grave key) ---
        # Shows, live, exactly which key events the app RECEIVES. Use it to tell
        # ghosting (app never sees the keydown) from a software bug (app sees it
        # but no note). Off by default; no effect on normal play.
        self._diag = False
        self._diag_log = []          # recent "(down|up) NAME" strings
        self._diag_down = set()      # keycodes currently held (note keys only)
        # --- Guided key diagnostic wizard (launch with the \ backslash key) ---
        # Walks through 4 prompted held-key combinations and writes a
        # self-labeling report to key_diagnostic_report.txt. Because the report
        # states which keys each step ASKED for, comparing asked-vs-received is
        # unambiguous. Off unless launched.
        self._wiz_active = False
        self._wiz_step = 0
        self._wiz_events = []        # per-step captured raw key events (dicts)
        self._wiz_accepts = []       # per-step note-on acceptance records
        self._wiz_results = []       # finished per-step summaries
        self._wiz_next_rect = None   # clickable Next button (set during draw)
        self._wiz_env = {}           # SDL / driver / mixer info captured at start
        # Each step: (human label, list of pygame keycodes to hold together)
        # Investigating an ORDER-dependent block: a "poisoned" key (f/g/h/j/;/')
        # appears to stop any key pressed AFTER it from registering, while keys
        # pressed BEFORE it are fine. These steps contrast press-order. The app
        # cannot enforce press order, so the on-screen prompt tells you the order
        # and you press them one at a time, left to right, holding each down.
        self._wiz_steps = [
            ("Press in order, hold each: S then D then F", [pygame.K_s, pygame.K_d, pygame.K_f]),
            ("Press in order, hold each: F then S then D", [pygame.K_f, pygame.K_s, pygame.K_d]),
            ("Press in order, hold each: S then D then G", [pygame.K_s, pygame.K_d, pygame.K_g]),
            ("Press in order, hold each: G then S then D", [pygame.K_g, pygame.K_s, pygame.K_d]),
            ("Press in order, hold each: A then H then J", [pygame.K_a, pygame.K_h, pygame.K_j]),
            ("Press in order, hold each: H then A then J", [pygame.K_h, pygame.K_a, pygame.K_j]),
            ("Press in order, hold each: A then ; then '", [pygame.K_a, pygame.K_SEMICOLON, pygame.K_QUOTE]),
            ("Press in order, hold each: ' then ; then A", [pygame.K_QUOTE, pygame.K_SEMICOLON, pygame.K_a]),
        ]
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.note_font = pygame.font.Font(None, 20)

        self.audio_config = audio_config
        self.synth = Synth(self.audio_config)
        self.recorder = Recorder(self.synth, self)
        self.keys: List[PianoKey] = []
        self.key_dict = {}  # not used for input anymore; display only
        self.midi_to_key: Dict[int, PianoKey] = {}  # MIDI note -> PianoKey UI
        self.kc_to_midi: Dict[int, int] = {}  # active mapping (loaded / learned)
        self.pressed_kc: Dict[int, Tuple[PianoKey, int]] = {}  # keycode -> (PianoKey, midi)

        self.mouse_notes_active = {}
        self.sustain = False
        self.master_volume = 0.9
        self.octave_shift = 0
        self.reverb_enabled = True
        self.reverb_wet = DEFAULT_REVERB_WET
        self.metronome = Metronome(self.audio_config, bpm=100)
        self.status_message = ""
        self.status_color = DARK_GRAY
        self.status_until = 0
        self.cache_total = len(BASE_MIDIS)
        self.cache_ready = 0

        # Mapping mode state
        self.mapping_mode = False
        self._map_sequence: List[int] = WHITE_MIDIS + BLACK_MIDIS
        self._map_index = 0
        self._temp_map: Dict[int, int] = {}

        self._create_piano_keys()

        # Load keymap (user) or fallback to default
        user_map = load_user_keymap()
        if user_map:
            self.apply_keymap(user_map)
            self.set_status("Loaded custom keymap (F11 reset, F12 remap).", BLUE, 2500)
        else:
            self.apply_keymap(DEFAULT_KC_TO_MIDI.copy())
            self.set_status("Using default keymap (F11 reset, F12 remap).", BLUE, 2000)

        if self.audio_config.ok:
            self.set_status("Loading sounds...", INFO_COLOR, 99999)
            self._warmup_cache_blocking()
            self.set_status("🎹 Ready to play!", GREEN, 2000)
        if self.audio_config.ok:
            for i in range(pygame.mixer.get_num_channels()):
                try:
                    ch = pygame.mixer.Channel(i)
                    dummy = self.synth.get_realtime_sound(60)
                    if dummy:
                        ch.play(dummy)
                        ch.set_volume(0, 0)
                        pygame.time.wait(1)
                        ch.stop()
                except:
                    pass

    def _warmup_cache_blocking(self):
        if not self.audio_config.ok:
            return
        for i, midi in enumerate(BASE_MIDIS):
            self.synth.get_realtime_sound(midi)
            self.cache_ready = i + 1
            self.screen.fill((18, 18, 18))
            progress = f"Loading sounds... {i+1}/{self.cache_total}"
            txt = self.font.render(progress, True, (100, 180, 255))
            self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            pygame.display.flip()
        for midi in BASE_MIDIS:
            sound = self.synth.cache.get(midi)
            if sound:
                ch = sound.play()
                if ch:
                    ch.set_volume(0, 0)
                    pygame.time.wait(15)
                    ch.stop()
        pygame.mixer.stop()
        pygame.time.wait(50)

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            try:
                self.raw_kb.stop()
            except Exception:
                pass
            try:
                self._set_input_grab(False)
            except Exception:
                pass
            try:
                self.synth.clear_cache()
            except Exception:
                pass
            pygame.quit()
        except Exception:
            pass
        return False

    def set_status(self, msg, color=DARK_GRAY, ms=2500):
        self.status_message = msg
        self.status_color = color
        self.status_until = pygame.time.get_ticks() + ms

    def _create_piano_keys(self):
        white_w = 90
        white_h = 380
        black_w = 58
        black_h = 240
        start_y = 180
        total_width = 11 * white_w
        start_x = (WIDTH - total_width) // 2
        self.keys.clear()
        self.midi_to_key.clear()
        white_positions = {}
        # White keys
        for i, midi in enumerate(WHITE_MIDIS):
            x = start_x + i * white_w
            key = PianoKey(x, start_y, white_w, white_h, midi, kb_label='', is_black=False, synth=self.synth)
            self.keys.append(key)
            white_positions[midi] = x
            self.midi_to_key[midi] = key
        # Black keys with placement based on neighbors
        black_specs = [
            (54, 53, 55),
            (56, 55, 57),
            (58, 57, 59),
            (61, 60, 62),
            (63, 62, 64),
            (66, 65, 67),
            (68, 67, 69),
            (70, 69, 71),
        ]
        for midi, left_midi, right_midi in black_specs:
            left_x = white_positions[left_midi]
            x = left_x + white_w - (black_w // 2)
            key = PianoKey(x, start_y, black_w, black_h, midi, kb_label='', is_black=True, synth=self.synth)
            self.keys.append(key)
            self.midi_to_key[midi] = key
        self.keys.sort(key=lambda k: (k.is_black, k.rect.x))
        self.min_base_midi = min(WHITE_MIDIS)
        self.max_base_midi = max(WHITE_MIDIS)

    def apply_keymap(self, kc_to_midi: Dict[int, int]):
        # Set active mapping and update labels on keys
        self.kc_to_midi = dict(kc_to_midi)
        # Invert: midi -> keycode (if multiple map to same midi, keep last)
        midi_to_kc = {}
        for kc, midi in self.kc_to_midi.items():
            if midi in BASE_MIDIS:
                midi_to_kc[midi] = kc
        # Update key labels
        for midi, key in self.midi_to_key.items():
            kc = midi_to_kc.get(midi)
            if kc is not None:
                name = pygame.key.name(kc).upper()
            else:
                name = ''
            key.kb = name

    def _set_input_grab(self, grab: bool):
        # Grab/release ONLY the keyboard so external key rules interfere less.
        # We deliberately do NOT call pygame.event.set_grab(), because that
        # confines the MOUSE to the window and would trap the cursor. On older
        # pygame builds that lack set_keyboard_grab we simply skip the grab
        # rather than trap the mouse; disabling key-repeat still helps there.
        if grab == self._input_grabbed:
            return
        try:
            if hasattr(pygame.event, "set_keyboard_grab"):
                pygame.event.set_keyboard_grab(grab)
        except Exception:
            pass
        self._input_grabbed = grab

    def _diag_note(self, is_down: bool, kc: int, event=None):
        # Record a raw key event for the diagnostic overlay and, if the guided
        # test is running, a rich record for the report. Runs for every key
        # event; cheap when the test/overlay are off.
        try:
            name = pygame.key.name(kc)
        except Exception:
            name = str(kc)
        if is_down:
            self._diag_down.add(kc)
        else:
            self._diag_down.discard(kc)
        self._diag_log.append(f"{'down' if is_down else 'up  '} {name}")
        if len(self._diag_log) > 12:
            self._diag_log = self._diag_log[-12:]
        # If the guided wizard is running, capture a detailed event record.
        if self._wiz_active:
            t = pygame.time.get_ticks()
            scancode = getattr(event, 'scancode', None) if event is not None else None
            mod = getattr(event, 'mod', None) if event is not None else None
            self._wiz_events.append({
                "t": t, "down": is_down, "kc": kc, "name": name,
                "scancode": scancode, "mod": mod,
            })

    # ----- Guided key diagnostic wizard -----
    def start_key_wizard(self):
        # Do not run during recording; it would pollute takes.
        if self.recorder.is_recording:
            self.set_status("Stop recording before the key test.", RED, 2200)
            return
        self._wiz_active = True
        self._wiz_step = 0
        self._wiz_events = []
        self._wiz_accepts = []
        self._wiz_results = []
        self._wiz_env = self._capture_env()
        self._flush_all_pressed()
        self.set_status("Key test started.", BLUE, 1500)

    def cancel_key_wizard(self):
        self._wiz_active = False
        self._wiz_events = []
        self._wiz_accepts = []
        self._wiz_results = []
        self.set_status("Key test cancelled.", DARK_GRAY, 1500)

    def _capture_env(self):
        # Snapshot the layers a keypress passes through, so the report can point
        # at where a drop is happening (SDL/driver vs mixer vs app logic).
        env = {}
        try:
            env["pygame"] = pygame.version.ver
        except Exception:
            env["pygame"] = "?"
        try:
            env["sdl"] = ".".join(str(x) for x in pygame.get_sdl_version())
        except Exception:
            env["sdl"] = "?"
        try:
            env["video_driver"] = pygame.display.get_driver()
        except Exception:
            env["video_driver"] = "?"
        for var in ("SDL_VIDEODRIVER", "SDL_HINT_WINDOWS_ENABLE_MESSAGELOOP"):
            env[var] = os.environ.get(var, "(unset)")
        try:
            env["key_repeat"] = str(pygame.key.get_repeat())
        except Exception:
            env["key_repeat"] = "?"
        try:
            env["keyboard_grabbed"] = str(bool(self._input_grabbed))
        except Exception:
            env["keyboard_grabbed"] = "?"
        try:
            if pygame.mixer.get_init():
                env["mixer_channels"] = str(pygame.mixer.get_num_channels())
                env["mixer_init"] = str(pygame.mixer.get_init())
            else:
                env["mixer_channels"] = "0"
                env["mixer_init"] = "(not initialized)"
        except Exception:
            env["mixer_channels"] = "?"
        env["platform"] = sys.platform
        return env

    def _wiz_advance(self):
        # Summarize the current step, then move on or finish.
        label, wanted = self._wiz_steps[self._wiz_step]
        wanted_names = [pygame.key.name(k) for k in wanted]
        downs = [e["name"] for e in self._wiz_events if e["down"]]
        received_down = []
        for n in downs:
            if n not in received_down:
                received_down.append(n)
        missing = [n for n in wanted_names if n not in received_down]
        extra = [n for n in received_down if n not in wanted_names]
        # Which received keys did note-on actually accept (play)?
        played = [a["name"] for a in self._wiz_accepts if a["outcome"].startswith("played")]
        rejected = [(a["name"], a["outcome"]) for a in self._wiz_accepts
                    if a["outcome"].startswith("REJECTED")]
        # received but never reached a "played" outcome = software-side loss
        not_played = [n for n in received_down if n not in played]
        self._wiz_results.append({
            "step": self._wiz_step + 1,
            "label": label,
            "asked": wanted_names,
            "received_down": received_down,
            "missing": missing,
            "extra": extra,
            "played": played,
            "not_played": not_played,
            "rejected": rejected,
            "raw": list(self._wiz_events),
            "accepts": list(self._wiz_accepts),
        })
        self._wiz_events = []
        self._wiz_accepts = []
        self._wiz_step += 1
        if self._wiz_step >= len(self._wiz_steps):
            path = self._write_wiz_report()
            self._wiz_active = False
            if path:
                self.set_status("Key test done. Report saved next to the app.", GREEN, 4000)
            else:
                self.set_status("Key test done, but the report could not be saved.", RED, 4000)

    def _write_wiz_report(self):
        import datetime
        lines = []
        lines.append("Pyano Keyboard - Key Diagnostic Report")
        lines.append("Generated: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("")
        lines.append("ENVIRONMENT (the layers a keypress passes through)")
        for k in ("platform", "pygame", "sdl", "video_driver", "SDL_VIDEODRIVER",
                  "key_repeat", "keyboard_grabbed", "mixer_init", "mixer_channels"):
            if k in self._wiz_env:
                lines.append(f"  {k:<16}: {self._wiz_env[k]}")
        lines.append("")
        lines.append("HOW TO READ THIS")
        lines.append("  Each step asked you to HOLD a set of keys together, then release.")
        lines.append("  'asked'    = what the step told you to press.")
        lines.append("  'received' = KEYDOWNs the app got from the keyboard (via SDL).")
        lines.append("  'played'   = of those, which the app turned into a note.")
        lines.append("  A key can drop at one of three layers, and this report tells")
        lines.append("  them apart:")
        lines.append("   1. asked but NOT received  -> lost before the app. Keyboard")
        lines.append("      ghosting (hardware) or the SDL/OS layer. No app fix recovers")
        lines.append("      it; a raw-input path or different keyboard is the only cure.")
        lines.append("   2. received but NOT played -> the app got the key but its own")
        lines.append("      logic rejected it. That IS a software bug; the reason is shown.")
        lines.append("   3. received and played     -> worked correctly.")
        lines.append("  'scancode' is the physical key id before layout translation;")
        lines.append("  keys that collide in a keyboard matrix often share nearby")
        lines.append("  scancodes. 'mod' is the modifier state at that instant.")
        lines.append("=" * 64)
        any_missing = False
        any_not_played = False
        for r in self._wiz_results:
            lines.append("")
            lines.append(f"STEP {r['step']}: hold  {r['label']}")
            lines.append(f"  asked    : {', '.join(r['asked'])}")
            lines.append(f"  received : {', '.join(r['received_down']) if r['received_down'] else '(none)'}")
            lines.append(f"  played   : {', '.join(r['played']) if r['played'] else '(none)'}")
            if r['missing']:
                any_missing = True
                lines.append(f"  MISSING (asked, not received): {', '.join(r['missing'])}")
                lines.append("           -> lost before the app (hardware/SDL layer)")
            else:
                lines.append("  MISSING (asked, not received): none")
            if r['not_played']:
                any_not_played = True
                lines.append(f"  RECEIVED BUT NOT PLAYED     : {', '.join(r['not_played'])}")
                lines.append("           -> app logic rejected these (software)")
            if r['rejected']:
                for nm, why in r['rejected']:
                    lines.append(f"             {nm}: {why}")
            if r['extra']:
                lines.append(f"  unexpected extra keys       : {', '.join(r['extra'])}")
            lines.append("  raw event order (t ms | dir | key | scancode | mod):")
            for e in r['raw']:
                sc = e.get('scancode')
                md = e.get('mod')
                sc = '?' if sc is None else sc
                md = '?' if md is None else md
                lines.append(f"     {e['t']:>8} ms  {'DOWN' if e['down'] else 'UP  '}  "
                             f"{e['name']:<10} sc={sc:<5} mod={md}")
            if r['accepts']:
                lines.append("  note-on outcomes (with mixer channel state):")
                for a in r['accepts']:
                    lines.append(f"     {a['t']:>8} ms  {a['name']:<10} {a['outcome']}"
                                 f"  [busy {a['busy_channels']}, free {a['free_channels']},"
                                 f" this-key {a['key_channels']}]")
        lines.append("")
        lines.append("=" * 64)
        lines.append("VERDICT")
        if any_missing and not any_not_played:
            lines.append("  Keys were asked for but never RECEIVED while others were held,")
            lines.append("  and everything the app received, it played. The loss is before")
            lines.append("  the app: keyboard ghosting (hardware) or the SDL/OS input layer.")
            lines.append("  Next step: try the raw-input path; if it also misses the keys,")
            lines.append("  the keyboard matrix is the limit and only different hardware")
            lines.append("  (NKRO/6KRO) fixes it.")
        elif any_not_played:
            lines.append("  At least one key was RECEIVED by the app but not turned into a")
            lines.append("  note. That is a software bug in this program. The per-key reason")
            lines.append("  is listed under each step ('RECEIVED BUT NOT PLAYED'). Send this")
            lines.append("  report back so the exact branch can be fixed.")
            if any_missing:
                lines.append("  NOTE: some keys were also never received (see MISSING). There")
                lines.append("  may be BOTH a hardware/SDL loss and a software issue.")
        else:
            lines.append("  Every asked key was received AND played in every step. Input is")
            lines.append("  working correctly in this test. If you still hear dropouts while")
            lines.append("  playing, capture them with the live overlay (grave key) and note")
            lines.append("  the exact combination, then send this report and that combination.")
        text = "\n".join(lines) + "\n"
        # Write next to the running app / script.
        try:
            base = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and sys.argv[0] else os.getcwd()
        except Exception:
            base = os.getcwd()
        path = os.path.join(base, "key_diagnostic_report.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info("Key diagnostic report written to %s", path)
            return path
        except Exception as e:
            logger.warning("Could not write key diagnostic report: %s", e)
            # Fallback to current working directory.
            try:
                alt = os.path.join(os.getcwd(), "key_diagnostic_report.txt")
                with open(alt, "w", encoding="utf-8") as f:
                    f.write(text)
                return alt
            except Exception:
                return None

    def _flush_all_pressed(self):
        # Release any pressed keyboard notes (by keycode)
        for kc in list(self.pressed_kc.keys()):
            self.handle_note_off_kc(kc)
        self.mouse_note_off_all()

    def _octave_limits(self):
        min_base = self.min_base_midi
        max_base = self.max_base_midi
        max_up = (MAX_MIDI - max_base) // 12
        max_down = (min_base - MIN_MIDI) // 12
        return -max_down, max_up

    def change_octave(self, delta):
        min_shift, max_shift = self._octave_limits()
        new_shift = max(min_shift, min(max_shift, self.octave_shift + delta))
        if new_shift != self.octave_shift:
            self.octave_shift = new_shift
            self.set_status(f"Octave: {self.octave_shift:+d}", DARK_GRAY, 900)
        else:
            if delta > 0:
                self.set_status(f"Max octave reached: +{max_shift}", RED, 900)
            else:
                self.set_status(f"Min octave reached: {min_shift}", RED, 900)

    def draw_ui(self):
        self.screen.fill((15, 15, 18))
        for i in range(140):
            shade = 22 + i // 8
            pygame.draw.line(self.screen, (shade, shade, shade + 2), (0, i), (WIDTH, i))
        for k in self.keys:
            if not k.is_black:
                k.draw(self.screen, self.font, self.note_font, self.octave_shift)
        for k in self.keys:
            if k.is_black:
                k.draw(self.screen, self.font, self.note_font, self.octave_shift)
        now = pygame.time.get_ticks()
        if now < self.status_until and self.status_message:
            txt = pygame.font.Font(None, 34).render(self.status_message, True, self.status_color)
            bg_rect = txt.get_rect(center=(WIDTH // 2, 25))
            bg_rect.inflate_ip(30, 10)
            pygame.draw.rect(self.screen, (25, 25, 28), bg_rect, border_radius=8)
            self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, 25)))
        rec_text = "● RECORDING" if self.recorder.is_recording else "READY"
        if self._raw_supported:
            inp = "RAW" if self.raw_kb.available else "SDL"
            info = f"{rec_text}  |  Vol {int(self.master_volume*100)}%  |  Octave {self.octave_shift:+d}  |  Input: {inp} (= to toggle)"
        else:
            info = f"{rec_text}  |  Vol {int(self.master_volume*100)}%  |  Octave {self.octave_shift:+d}"
        txt = self.small_font.render(info, True, (200, 200, 205))
        self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, 70)))
        hint = "1: Record  |  2: Save WAV  |  3: MP3  |  4: Preview  |  F11: Reset map  |  F12: Learn map"
        txt = self.small_font.render(hint, True, (120, 120, 125))
        self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, HEIGHT - 25)))
        if self.cache_ready < self.cache_total:
            bar_x, bar_y, bar_w, bar_h = 550, 100, 300, 10
            fill = int(bar_w * (self.cache_ready / self.cache_total))
            fill = max(0, min(bar_w, fill))
            pygame.draw.rect(self.screen, DARK_GRAY, (bar_x, bar_y, bar_w, bar_h), 1)
            pygame.draw.rect(self.screen, BLUE, (bar_x + 1, bar_y + 1, max(0, fill - 2), bar_h - 2))
            progress = f"Initializing sounds... {self.cache_ready}/{self.cache_total}"
            txt = self.small_font.render(progress, True, (100, 180, 255))
            self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, 100)))

        # Mapping overlay message
        if self.mapping_mode:
            midi = self._map_sequence[self._map_index] if self._map_index < len(self._map_sequence) else None
            prompt = f"Mapping mode: Press a key for {midi_to_name(midi)}" if midi else "Mapping complete"
            prompt += "  (Esc=cancel, Backspace=skip, Enter=finish)"
            txt = pygame.font.Font(None, 30).render(prompt, True, BLUE)
            self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, 120)))

        # Key diagnostic overlay (toggle with ` / ~ ). Shows exactly what the
        # app RECEIVES so you can tell hardware ghosting from a software bug.
        if self._diag:
            dfont = pygame.font.Font(None, 26)
            held_names = []
            for kc in sorted(self._diag_down):
                try:
                    held_names.append(pygame.key.name(kc))
                except Exception:
                    held_names.append(str(kc))
            lines = ["KEY DIAGNOSTIC  (` to toggle)",
                     f"held now ({len(held_names)}): {' '.join(held_names) if held_names else '-'}",
                     "recent events:"]
            lines += ["  " + s for s in self._diag_log[-8:]]
            panel_w, lh = 360, 24
            panel_h = 16 + lh * len(lines)
            panel = pygame.Surface((panel_w, panel_h))
            panel.set_alpha(230)
            panel.fill((14, 14, 18))
            self.screen.blit(panel, (10, 140))
            for i, ln in enumerate(lines):
                col = (120, 200, 255) if i == 0 else (210, 210, 215)
                self.screen.blit(dfont.render(ln, True, col), (20, 150 + i * lh))

        # Guided key-test wizard: a modal panel with the current prompt, a live
        # list of what the app has received this step, and a Next button.
        self._wiz_next_rect = None
        if self._wiz_active and self._wiz_step < len(self._wiz_steps):
            label, wanted = self._wiz_steps[self._wiz_step]
            # Dim the background.
            dim = pygame.Surface((WIDTH, HEIGHT))
            dim.set_alpha(200)
            dim.fill((8, 8, 10))
            self.screen.blit(dim, (0, 0))

            title_font = pygame.font.Font(None, 48)
            body_font = pygame.font.Font(None, 34)
            small = pygame.font.Font(None, 26)

            step_txt = f"KEY TEST  -  Step {self._wiz_step + 1} of {len(self._wiz_steps)}"
            self.screen.blit(title_font.render(step_txt, True, (120, 200, 255)),
                             (WIDTH // 2 - 220, 120))

            self.screen.blit(body_font.render("Hold ALL of these keys down at once:", True, (230, 230, 235)),
                             (WIDTH // 2 - 260, 190))
            self.screen.blit(title_font.render(label, True, GREEN),
                             (WIDTH // 2 - title_font.size(label)[0] // 2, 235))

            self.screen.blit(small.render("Then let go, and click Next (or press Enter).", True, (200, 200, 205)),
                             (WIDTH // 2 - 200, 300))

            # Live list of what the app has actually received this step.
            recd = []
            for e in self._wiz_events:
                if e["down"] and e["name"] not in recd:
                    recd.append(e["name"])
            wanted_names = [pygame.key.name(k) for k in wanted]
            self.screen.blit(small.render("received so far: " + (", ".join(recd) if recd else "(nothing yet)"),
                                          True, (170, 200, 170)), (WIDTH // 2 - 260, 345))

            # Next button.
            btn = pygame.Rect(WIDTH // 2 - 90, 400, 180, 56)
            pygame.draw.rect(self.screen, (40, 90, 200), btn, border_radius=10)
            pygame.draw.rect(self.screen, (120, 160, 255), btn, 2, border_radius=10)
            nlabel = "Next" if self._wiz_step < len(self._wiz_steps) - 1 else "Finish"
            ntxt = body_font.render(nlabel, True, (255, 255, 255))
            self.screen.blit(ntxt, ntxt.get_rect(center=btn.center))
            self._wiz_next_rect = btn

            self.screen.blit(small.render("Esc = cancel the test", True, (150, 150, 155)),
                             (WIDTH // 2 - 90, 475))

    # ----- Note handling by keycode -----
    def handle_note_on_kc(self, kc: int):
        midi = self.kc_to_midi.get(kc)
        if midi is None:
            self._wiz_accept(kc, "REJECTED: key not in current keymap")
            return
        if kc in self.pressed_kc:
            self._wiz_accept(kc, "ignored: already held (auto-repeat guard)")
            return
        key = self.midi_to_key.get(midi)
        if not key:
            self._wiz_accept(kc, "REJECTED: no piano key object for midi")
            return
        midi_used = midi + 12 * self.octave_shift
        if MIN_MIDI <= midi_used <= MAX_MIDI:
            self.pressed_kc[kc] = (key, midi_used)
            key.play(midi_used, volume=self.master_volume)
            self.recorder.note_on(midi_used)
            self._wiz_accept(kc, f"played {midi_to_name(midi_used)}", key=key)
        else:
            self.set_status(f"{midi_to_name(midi_used)} out of range.", RED, 1200)
            self._wiz_accept(kc, f"REJECTED: {midi_to_name(midi_used)} out of range")

    def _wiz_accept(self, kc, outcome, key=None):
        # During the guided test, record what the note-on logic did with a key
        # the app received, plus mixer state, so the report can show whether a
        # drop was hardware (never received) or software (received but rejected).
        if not self._wiz_active:
            return
        try:
            name = pygame.key.name(kc)
        except Exception:
            name = str(kc)
        busy = free = -1
        try:
            if pygame.mixer.get_init():
                total = pygame.mixer.get_num_channels()
                busy = sum(1 for i in range(total) if pygame.mixer.Channel(i).get_busy())
                free = total - busy
        except Exception:
            pass
        nchan = len(key.channels) if key is not None else None
        self._wiz_accepts.append({
            "t": pygame.time.get_ticks(), "name": name, "outcome": outcome,
            "busy_channels": busy, "free_channels": free, "key_channels": nchan,
        })

    def handle_note_off_kc(self, kc: int):
        item = self.pressed_kc.pop(kc, None)
        if not item:
            return
        key, midi_used = item
        self.recorder.note_off(midi_used)
        if not self.sustain:
            key.release(fade_ms=DEFAULT_FADE_OUT_MS)

    # ----- Mouse note helpers -----
    def mouse_note_on(self, key: PianoKey):
        if key in self.mouse_notes_active:
            return
        midi_used = key.base_midi + 12 * self.octave_shift
        if MIN_MIDI <= midi_used <= MAX_MIDI:
            self.mouse_notes_active[key] = midi_used
            key.play(midi_used, volume=self.master_volume)
            self.recorder.note_on(midi_used)
        else:
            self.set_status(f"{midi_to_name(midi_used)} out of range.", RED, 1200)

    def mouse_note_off_all(self):
        for key, midi_used in list(self.mouse_notes_active.items()):
            self.recorder.note_off(midi_used)
            if not self.sustain:
                key.release(fade_ms=DEFAULT_FADE_OUT_MS)
        self.mouse_notes_active.clear()

    def sustain_on(self):
        if not self.sustain:
            self.sustain = True
            self.recorder.sustain_on()
            self.set_status("Sustain ON", BLUE, 800)

    def sustain_off(self):
        if self.sustain:
            self.sustain = False
            self.recorder.sustain_off()
            held_keys = set(k for (k, _m) in self.pressed_kc.values()).union(set(self.mouse_notes_active.keys()))
            for key in self.keys:
                if key not in held_keys:
                    key.release(fade_ms=DEFAULT_FADE_OUT_MS)
            self.set_status("Sustain OFF", BLUE, 800)

    # ----- Mapping mode -----
    def start_mapping_mode(self):
        self.mapping_mode = True
        self._map_index = 0
        self._temp_map = {}
        self.set_status("Mapping mode started. Assign keys (F12).", BLUE, 2500)

    def cancel_mapping_mode(self):
        self.mapping_mode = False
        self._temp_map.clear()
        self.set_status("Mapping canceled.", RED, 2000)

    def finish_mapping_mode(self):
        self.mapping_mode = False
        if self._temp_map:
            self.apply_keymap(self._temp_map)
            save_user_keymap(self._temp_map)
            self.set_status("Custom keymap saved.", GREEN, 2500)
        else:
            self.set_status("No keys mapped.", DARK_GRAY, 1800)

    def handle_mapping_key(self, kc: int):
        # Reserved keys cannot be used for notes
        if kc in self.RESERVED_KC:
            self.set_status("That key is reserved. Choose a different key.", RED, 1400)
            return
        if self._map_index >= len(self._map_sequence):
            self.finish_mapping_mode()
            return
        midi = self._map_sequence[self._map_index]
        # Ensure unique keycode: if already assigned, we reassign
        # Also prevent multiple kc -> same midi by letting latest take precedence
        self._temp_map = {k: v for k, v in self._temp_map.items() if v != midi and k != kc}
        self._temp_map[kc] = midi
        self._map_index += 1
        if self._map_index >= len(self._map_sequence):
            self.finish_mapping_mode()

    def run(self):
        running = True
        self.set_status("Ready. F12 to learn a custom keymap. F11 resets to default.", DARK_GRAY, 3000)
        while running:
            self.recorder.maybe_begin_after_count_in()
            # If raw input is active, translate its events into synthetic pygame
            # key events so the existing handling below works unchanged. Real
            # SDL key events are then ignored (see the guard in the loop) to
            # avoid double-counting.
            #
            # Raw Input uses RIDEV_INPUTSINK, so it keeps delivering keys even
            # when this window is NOT focused. Gate on focus here so notes do
            # not play while another app is in the foreground. When unfocused,
            # drain and discard the queue and release any held notes once.
            if self.raw_kb.available:
                try:
                    focused = bool(pygame.key.get_focused())
                except Exception:
                    focused = True
                if focused:
                    for is_down, pk in self.raw_kb.poll():
                        evtype = pygame.KEYDOWN if is_down else pygame.KEYUP
                        try:
                            pygame.event.post(pygame.event.Event(
                                evtype, {"key": pk, "mod": 0, "unicode": "",
                                         "scancode": 0, "_raw": True}))
                        except Exception:
                            pass
                else:
                    self.raw_kb.clear()  # discard background keystrokes + state
                    if self.pressed_kc:
                        self._flush_all_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == METRO_EVENT:
                    self.metronome.click()
                    continue

                # Raw-input toggle ('='). Handled here, before the raw-active
                # SDL-suppression guard below, so it works no matter which path
                # is live. Debounced so one physical press flips once.
                if event.type == pygame.KEYDOWN and event.key == pygame.K_EQUALS \
                        and getattr(self, "_raw_supported", False):
                    if not self._eq_down:
                        self._eq_down = True
                        self.raw_kb.available = not self.raw_kb.available
                        self.raw_kb.clear()
                        self._flush_all_pressed()
                        self.set_status(
                            f"Raw Input {'ON' if self.raw_kb.available else 'OFF'} "
                            f"(SDL path {'off' if self.raw_kb.available else 'on'})",
                            BLUE, 2000)
                    continue
                if event.type == pygame.KEYUP and event.key == pygame.K_EQUALS \
                        and getattr(self, "_raw_supported", False):
                    self._eq_down = False
                    continue

                # When raw input is active, ignore SDL's own key events; only
                # our injected synthetic ones (marked _raw) drive notes.
                if self.raw_kb.available and event.type in (pygame.KEYDOWN, pygame.KEYUP) \
                        and not getattr(event, "_raw", False):
                    continue

                # --- Window focus handling ---
                # On focus LOSS: release all held notes (so nothing sticks) and
                # release the keyboard grab so the user can use other apps.
                # On focus GAIN: re-grab the keyboard so macros/hotkeys are
                # suppressed again while playing.
                _wfl = getattr(pygame, 'WINDOWFOCUSLOST', -1)
                _wfg = getattr(pygame, 'WINDOWFOCUSGAINED', -1)
                _wmin = getattr(pygame, 'WINDOWMINIMIZED', -1)
                if event.type in (_wfl, _wmin) or (
                    event.type == pygame.ACTIVEEVENT and getattr(event, 'state', 0) & 2 and getattr(event, 'gain', 1) == 0
                ):
                    self._flush_all_pressed()
                    self._set_input_grab(False)
                    continue
                if event.type == _wfg or (
                    event.type == pygame.ACTIVEEVENT and getattr(event, 'state', 0) & 2 and getattr(event, 'gain', 1) == 1
                ):
                    self._set_input_grab(True)
                    continue

                # --- Keyboard Input ---
                if event.type == pygame.KEYDOWN:
                    # Diagnostic: record the raw event the instant it arrives,
                    # before any note/control logic can consume or skip it.
                    if event.key != pygame.K_BACKQUOTE:
                        self._diag_note(True, event.key, event)
                    # Toggle the diagnostic overlay with the ` / ~ grave key.
                    if event.key == pygame.K_BACKQUOTE:
                        self._diag = not self._diag
                        continue

                    # Guided key-test wizard has priority over everything else.
                    if self._wiz_active:
                        if event.key == pygame.K_ESCAPE:
                            self.cancel_key_wizard()
                            continue
                        if event.key == pygame.K_RETURN:
                            # Enter also advances to the next step.
                            self._wiz_advance()
                            continue
                        # Let the asked-for keys still play so the user gets
                        # audible/visual feedback of what registered.
                        self.handle_note_on_kc(event.key)
                        continue

                    # Launch the guided key test with the \ backslash key.
                    if event.key == pygame.K_BACKSLASH:
                        self.start_key_wizard()
                        continue

                    # Mapping mode consumes keys first
                    if self.mapping_mode:
                        if event.key == pygame.K_ESCAPE:
                            self.cancel_mapping_mode()
                            continue
                        if event.key == pygame.K_RETURN:
                            self.finish_mapping_mode()
                            continue
                        if event.key == pygame.K_BACKSPACE:
                            # Skip current note
                            self._map_index = min(self._map_index + 1, len(self._map_sequence))
                            if self._map_index >= len(self._map_sequence):
                                self.finish_mapping_mode()
                            continue
                        self.handle_mapping_key(event.key)
                        continue

                    # === NUMBER ROW CONTROLS ===
                    if event.key == pygame.K_1:
                        if self.recorder.is_recording or self.recorder._count_in_until:
                            self.recorder.stop_and_render_threaded()
                        else:
                            self.recorder.start()
                        continue
                    if event.key == pygame.K_2:
                        if self.recorder.is_recording:
                            self.set_status("Stop recording first.", RED, 2200)
                        else:
                            self.recorder.save_wav()
                        continue
                    if event.key == pygame.K_3:
                        if self.recorder.is_recording:
                            self.set_status("Stop recording first.", RED, 2200)
                        else:
                            self.recorder.save_mp3()
                        continue
                    if event.key == pygame.K_4:
                        self.recorder.preview_rendered()
                        continue
                    if event.key == pygame.K_5:
                        self.recorder.stop_preview()
                        continue
                    if event.key == pygame.K_6:
                        self.recorder.toggle_overdub()
                        continue
                    if event.key == pygame.K_7:
                        self.recorder.undo_last_take()
                        continue
                    if event.key == pygame.K_8:
                        self.recorder.cancel_render()
                        continue
                    if event.key == pygame.K_9:
                        self.recorder.reset_all()
                        self.set_status("New session. Takes cleared.", DARK_GRAY, 1600)
                        continue
                    if event.key == pygame.K_0:
                        self.metronome.toggle()
                        self.set_status(f"Metronome {'ON' if self.metronome.enabled else 'OFF'} ({self.metronome.bpm} BPM)", DARK_GRAY, 1400)
                        continue

                    # === FUNCTION KEY CONTROLS ===
                    if event.key == pygame.K_F1:
                        self.metronome.set_bpm(self.metronome.bpm - 5)
                        self.set_status(f"BPM: {self.metronome.bpm}", DARK_GRAY, 900)
                        continue
                    if event.key == pygame.K_F2:
                        self.metronome.set_bpm(self.metronome.bpm + 5)
                        self.set_status(f"BPM: {self.metronome.bpm}", DARK_GRAY, 900)
                        continue
                    if event.key == pygame.K_F3:
                        self.recorder.cycle_count_in()
                        continue
                    if event.key == pygame.K_F4:
                        self.reverb_enabled = not self.reverb_enabled
                        self.set_status(f"Reverb {'ON' if self.reverb_enabled else 'OFF'}", DARK_GRAY, 1000)
                        continue
                    if event.key == pygame.K_F5:
                        self.reverb_wet = max(0.0, self.reverb_wet - 0.05)
                        self.set_status(f"Reverb: {int(self.reverb_wet*100)}%", DARK_GRAY, 900)
                        continue
                    if event.key == pygame.K_F6:
                        self.reverb_wet = min(1.0, self.reverb_wet + 0.05)
                        self.set_status(f"Reverb: {int(self.reverb_wet*100)}%", DARK_GRAY, 900)
                        continue
                    if event.key == pygame.K_F7:
                        self.master_volume = max(0.0, self.master_volume - 0.05)
                        self.set_status(f"Volume: {int(self.master_volume*100)}%", DARK_GRAY, 900)
                        continue
                    if event.key == pygame.K_F8:
                        self.master_volume = min(1.0, self.master_volume + 0.05)
                        self.set_status(f"Volume: {int(self.master_volume*100)}%", DARK_GRAY, 900)
                        continue
                    if event.key == pygame.K_F9:
                        self.change_octave(-1)
                        continue
                    if event.key == pygame.K_F10:
                        self.change_octave(1)
                        continue
                    if event.key == pygame.K_F11:
                        self.apply_keymap(DEFAULT_KC_TO_MIDI.copy())
                        save_user_keymap(DEFAULT_KC_TO_MIDI)
                        self.set_status("Keymap reset to default.", BLUE, 2000)
                        continue
                    if event.key == pygame.K_F12:
                        self.start_mapping_mode()
                        continue

                    # Sustain / octave modifiers
                    if event.key == pygame.K_TAB:
                        self.sustain_on()
                        continue
                    if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        self.change_octave(-1)
                        continue
                    if event.key == pygame.K_SPACE:
                        self.change_octave(1)
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        continue

                    # Notes via active keymap (pygame-only input path)
                    self.handle_note_on_kc(event.key)
                    continue

                elif event.type == pygame.KEYUP:
                    if event.key != pygame.K_BACKQUOTE:
                        self._diag_note(False, event.key, event)
                    if event.key == pygame.K_TAB:
                        self.sustain_off()
                    else:
                        self.handle_note_off_kc(event.key)

                # --- Mouse Input ---
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    # During the key test, clicks only operate the Next button.
                    if self._wiz_active:
                        if self._wiz_next_rect and self._wiz_next_rect.collidepoint(event.pos):
                            self._wiz_advance()
                        continue
                    key = self.find_key_at(event.pos)
                    if key:
                        self.mouse_note_on(key)
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    if self._wiz_active:
                        continue
                    key = self.find_key_at(event.pos)
                    if key:
                        self.mouse_note_on(key)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self._wiz_active:
                        continue
                    self.mouse_note_off_all()

            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)

    def find_key_at(self, pos):
        # Prefer black keys when overlapping
        for key in self.keys:
            if key.is_black and key.rect.collidepoint(pos):
                return key
        for key in self.keys:
            if not key.is_black and key.rect.collidepoint(pos):
                return key
        return None

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    def main():
        global AUDIO
        pygame.init()
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:
            pass
        audio_config = AudioConfig()
        audio_config.init()
        AUDIO = audio_config
        with PianoApp(audio_config) as app:
            app.run()
    main()
