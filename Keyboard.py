import logging
import os
import threading
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path
import time
from functools import wraps
import sys
import re
import shutil
import numpy as np
import pygame
import soundfile as sf

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc_keyboard_piano")

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

# CRITICAL: Set audio driver BEFORE any pygame init (Fix 1)
if 'SDL_AUDIODRIVER' not in os.environ:
    if os.name == 'nt':
        os.environ['SDL_AUDIODRIVER'] = 'wasapi'  # modern low latency
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
# Key Mapping (F3..B4)
# -------------------------
KEY_TO_MIDI = {
    # White keys A - ' (11)
    'a': 53, 's': 55, 'd': 57, 'f': 59, 'g': 60,
    'h': 62, 'j': 64, 'k': 65, 'l': 67, ';': 69, "'": 71,
    # Black keys W E R  Y U  O P [
    'w': 54, 'e': 56, 'r': 58, 'y': 61, 'u': 63, 'o': 66, 'p': 68, '[': 70,
}
BASE_MIDIS = sorted(set(KEY_TO_MIDI.values()))

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
    def __init__(self, x, y, width, height, base_midi, kb_key, is_black, synth: Synth):
        self.rect = pygame.Rect(x, y, width, height)
        self.base_midi = base_midi
        self.kb = kb_key.upper()
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
        label = font.render(self.kb, True, text_color)
        surface.blit(label, label.get_rect(center=(self.rect.centerx, self.rect.bottom - 35)))
        nlabel = note_font.render(name, True, text_color)
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
    def __init__(self, audio_config: AudioConfig):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("PC Keyboard Piano | Grandmaster Build")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.note_font = pygame.font.Font(None, 20)

        self.audio_config = audio_config
        self.synth = Synth(self.audio_config)
        self.recorder = Recorder(self.synth, self)
        self.keys = []
        self.key_dict = {}
        self.pressed_keys = {}
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

        self._create_piano_keys()

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
        white_key_notes = [53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71]
        white_chars = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', "'"]
        total_width = 11 * white_w
        start_x = (WIDTH - total_width) // 2
        white_positions = {}
        for i, (char, midi) in enumerate(zip(white_chars, white_key_notes)):
            x = start_x + i * white_w
            key = PianoKey(x, start_y, white_w, white_h, midi, char, is_black=False, synth=self.synth)
            self.keys.append(key)
            self.key_dict[char] = key
            white_positions[midi] = x
        black_mapping = {
            'w': (54, 53, 55),
            'e': (56, 55, 57),
            'r': (58, 57, 59),
            'y': (61, 60, 62),
            'u': (63, 62, 64),
            'o': (66, 65, 67),
            'p': (68, 67, 69),
            '[': (70, 69, 71),
        }
        for char, (midi, left_midi, right_midi) in black_mapping.items():
            left_x = white_positions[left_midi]
            x = left_x + white_w - (black_w // 2)
            key = PianoKey(x, start_y, black_w, black_h, midi, char, is_black=True, synth=self.synth)
            self.keys.append(key)
            self.key_dict[char] = key
        self.keys.sort(key=lambda k: (k.is_black, k.rect.x))
        self.min_base_midi = 53
        self.max_base_midi = 71

    def find_key_at(self, pos):
        for key in self.keys:
            if key.is_black and key.rect.collidepoint(pos):
                return key
        for key in self.keys:
            if not key.is_black and key.rect.collidepoint(pos):
                return key
        return None

    def _keycode_to_char(self, k):
        code_map = {
            pygame.K_a:'a', pygame.K_s:'s', pygame.K_d:'d', pygame.K_f:'f',
            pygame.K_g:'g', pygame.K_h:'h', pygame.K_j:'j', pygame.K_k:'k',
            pygame.K_l:'l', pygame.K_SEMICOLON:';', pygame.K_QUOTE:"'",
            pygame.K_w:'w', pygame.K_e:'e', pygame.K_r:'r',
            pygame.K_y:'y', pygame.K_u:'u',
            pygame.K_o:'o', pygame.K_p:'p', pygame.K_LEFTBRACKET:'[',
        }
        return code_map.get(k)

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
        if self.recorder.is_recording:
            rec_text = "● RECORDING"
        else:
            rec_text = "READY"
        info = f"{rec_text}  |  Vol {int(self.master_volume*100)}%  |  Octave {self.octave_shift:+d}"
        txt = self.small_font.render(info, True, (200, 200, 205))
        self.screen.blit(txt, txt.get_rect(center=(WIDTH // 2, 70)))
        hint = "1: Record  |  2: Save WAV  |  3: MP3  |  4: Preview  |  Tab: Sustain  |  Shift/Space: Octave"
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

    def handle_note_on(self, kb_char):
        if kb_char not in self.key_dict:
            return
        if kb_char in self.pressed_keys:
            return
        key = self.key_dict[kb_char]
        midi_used = key.base_midi + 12 * self.octave_shift
        if MIN_MIDI <= midi_used <= MAX_MIDI:
            self.pressed_keys[kb_char] = (key, midi_used)
            key.play(midi_used, volume=self.master_volume)
            self.recorder.note_on(midi_used)
        else:
            self.set_status(f"{midi_to_name(midi_used)} out of range.", RED, 1200)

    def handle_note_off(self, kb_char):
        if kb_char not in self.pressed_keys:
            return
        key, midi_used = self.pressed_keys.pop(kb_char)
        self.recorder.note_off(midi_used)
        if not self.sustain:
            key.release(fade_ms=DEFAULT_FADE_OUT_MS)

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
            held_keys = set(k for (k, _m) in self.pressed_keys.values()).union(set(self.mouse_notes_active.keys()))
            for key in self.keys:
                if key not in held_keys:
                    key.release(fade_ms=DEFAULT_FADE_OUT_MS)
            self.set_status("Sustain OFF", BLUE, 800)

    def _flush_all_pressed(self):
        # Safety: release everything on focus loss to prevent stuck notes
        for ch in list(self.pressed_keys.keys()):
            self.handle_note_off(ch)
        self.mouse_note_off_all()

    def run(self):
        running = True
        self.set_status("Ready. Play keys or click the keyboard!", DARK_GRAY, 2000)
        while running:
            self.recorder.maybe_begin_after_count_in()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == METRO_EVENT:
                    self.metronome.click()
                    continue

                # Focus-loss safety flush (covers both new and old pygame event models)
                if event.type == getattr(pygame, 'WINDOWFOCUSLOST', None) or (
                    event.type == pygame.ACTIVEEVENT and getattr(event, 'state', 0) & 2 and getattr(event, 'gain', 1) == 0
                ):
                    self._flush_all_pressed()
                    continue

                # --- Keyboard Input ---
                if event.type == pygame.KEYDOWN:
                    # App controls (number row)
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

                    # Function keys
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

                    # Notes (pygame-only input path)
                    ch = self._keycode_to_char(event.key)
                    if ch:
                        self.handle_note_on(ch)
                        continue

                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_TAB:
                        self.sustain_off()
                    else:
                        ch = self._keycode_to_char(event.key)
                        if ch:
                            self.handle_note_off(ch)

                # --- Mouse Input ---
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    key = self.find_key_at(event.pos)
                    if key:
                        self.mouse_note_on(key)
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    key = self.find_key_at(event.pos)
                    if key:
                        self.mouse_note_on(key)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.mouse_note_off_all()

            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(60)

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
