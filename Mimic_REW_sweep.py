#!/usr/bin/env python3
"""
REW-Mimic Acoustic Measurement & Analysis Tool
================================================
Measures room/speaker acoustic frequency response using an Exponential Sine
Sweep (ESS / Farina method), exactly replicating what REW does internally:

  1. Generate an ESS (log-chirp) and its analytical inverse filter
  2. Play the sweep; record the microphone response
  3. Deconvolve recording × inverse_filter  →  impulse response (IR)
  4. Time-window the IR to isolate the direct sound
  5. FFT the windowed IR  →  magnitude spectrum (dB SPL)
  6. Apply optional calibration correction
  7. Apply optional fractional-octave smoothing
  8. Plot measured response vs REW .mdat reference files

All output files (PNG, CSV, WAV) are saved to data/ by default.
Calibration file is auto-detected inside data/ if not specified explicitly.

Requirements:
    pip install sounddevice scipy numpy matplotlib soundfile

Linux audio:
    sudo apt-get install libportaudio2 portaudio19-dev

Usage:
    python rew_analysis.py [--no-record] [--sweep-file path] [--cal-file path]
                           [--ref-dir path] [--smoothing none|1/6|1/3|1/1]
                           [--output-dir path]
"""

import os
import sys
import struct
import argparse
import warnings
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    HAS_SD = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════
SWEEP_START_HZ   = 1           # Match REW "Start Freq: 0 Hz" (1 Hz avoids log(0))
SWEEP_END_HZ     = 40_000
# REW default sweep length is 256k samples @ 48 kHz = 5.333... s
# Using the exact same sample count (not a rounded duration) ensures the
# inverse filter energy normalisation is identical to REW's.
SWEEP_SAMPLES    = 256_000
SAMPLE_RATE      = 48_000
SWEEP_DURATION_S = SWEEP_SAMPLES / SAMPLE_RATE   # = 5.3333... s exactly
SWEEP_LEVEL_DBFS = -12
SILENCE_PRE_S    = 1.0         # Match REW "Start delay: 1 s"
SILENCE_POST_S   = 1.5
OUTPUT_WAV       = "mimic_sweep_1.wav"
PLOT_FILENAME    = "rew_analysis.png"
CSV_FILENAME     = "rew_analysis.csv"
DEFAULT_DATA_DIR = "data"

# IR time window — gate long enough for 2 Hz frequency resolution (1/0.5 s)
IR_FADE_IN_S   = 0.002
IR_GATE_S      = 0.500

PLOT_FMIN = 10     # REW plots from 10 Hz even when sweep starts at 1 Hz
PLOT_FMAX = 24_000

# ── Absolute SPL calibration ─────────────────────────────────────────────
# For UMIK-1 calibration files the header encodes:
#   "Sens Factor = X dB, AGain = Y dB"
# The sensitivity offset (dBFS → dBSPL) is:
#   sensitivity_offset = UMIK1_BASE_SENSITIVITY + AGain − Sens_Factor
# where UMIK1_BASE_SENSITIVITY = 102 dB is the capsule+ADC offset at 0 dB gain.
# This gives: dBSPL = recorded_dBFS + sensitivity_offset
# For our file (Sens=0.532, AGain=18): offset = 102 + 18 − 0.532 = 119.468 dB
UMIK1_BASE_SENSITIVITY = 102   # dB — capsule sensitivity constant at 0 dB gain

SMOOTHING_MAP = {
    "none": None,
    "1/6":  6,
    "1/3":  3,
    "1/1":  1,
}

COLOUR_MEASURED  = "#2ecc71"
COLOUR_REF_CYCLE = ["#e74c3c", "#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#f1c40f"]


# ═══════════════════════════════════════════════════════════════════════════
#  ESS SWEEP GENERATION  (Farina 2000 formulation)
# ═══════════════════════════════════════════════════════════════════════════

def generate_ess(
    f1=SWEEP_START_HZ,
    f2=SWEEP_END_HZ,
    T=SWEEP_DURATION_S,
    level_dbfs=SWEEP_LEVEL_DBFS,
    fs=SAMPLE_RATE,
    pre_s=SILENCE_PRE_S,
    post_s=SILENCE_POST_S,
):
    """Generate an Exponential Sine Sweep and its analytical inverse filter.

    The ESS is the sweep used by REW (and most professional measurement tools).
    The inverse filter's convolution with the recorded response yields the
    impulse response of the measured system.

    Returns
    -------
    sweep        : float32 array, length N = T*fs  — the ESS signal at target level
    inv_filter   : float64 array, length N          — analytical inverse filter
    playback     : float32 array                    — sweep with pre/post silence
    """
    N = int(T * fs)
    t = np.linspace(0, T, N, endpoint=False)

    # Sweep rate parameter  L = T / ln(f2/f1)
    L = T / np.log(f2 / f1)

    # ── Forward ESS (Farina eq. 3) ──────────────────────────────────────
    # x(t) = sin( 2π f1 L (e^(t/L) − 1) )
    sweep = np.sin(2.0 * np.pi * f1 * L * (np.exp(t / L) - 1.0))

    # Tukey window (2 % taper) to suppress end clicks without affecting spectrum
    sweep *= sig.windows.tukey(N, alpha=0.02)

    # Scale to target dBFS
    amplitude = 10.0 ** (level_dbfs / 20.0)
    sweep = (sweep / np.max(np.abs(sweep))) * amplitude

    # ── Analytical inverse filter (Farina eq. 10) ───────────────────────
    # h_inv(t) = x(T−t) × e^(−t · ln(f2/f1) / T)
    #
    # The exponential envelope compensates the ESS's −3 dB/octave power
    # spectrum, so that convolving a flat system's response gives a delta.
    #
    # Normalisation must use the PLAYED sweep (at target dBFS), not the
    # unit-amplitude raw sweep — otherwise the deconvolved level is wrong
    # by exactly 2 × |level_dbfs| dB, producing the constant offset seen
    # when comparing against calibrated REW .mdat reference data.
    inv_mod    = np.exp(-t * np.log(f2 / f1) / T)
    sweep_raw  = np.sin(2.0 * np.pi * f1 * L * (np.exp(t / L) - 1.0))
    inv_filter = sweep_raw[::-1] * inv_mod

    # Divide by the energy of the *played* sweep (amplitude-scaled to dBFS).
    # This makes peak(IR) = 1.0 for a 0 dB gain system, so the deconvolved
    # magnitude is in dBFS relative to the recorded signal level — ready for
    # the SPL offset step that adds the microphone's sensitivity constant.
    played_energy = np.sum(sweep.astype(np.float64) ** 2)
    inv_filter    = inv_filter / (played_energy + 1e-30)

    # ── Playback signal with silence padding ─────────────────────────────
    pre  = np.zeros(int(pre_s  * fs), dtype=np.float32)
    post = np.zeros(int(post_s * fs), dtype=np.float32)
    playback = np.concatenate([pre, sweep.astype(np.float32), post])

    return sweep.astype(np.float32), inv_filter.astype(np.float64), playback


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO I/O
# ═══════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    if not HAS_SD:
        raise RuntimeError("sounddevice not available.")
    devices = sd.query_devices()
    print("\n── Audio Devices ──────────────────────────────────────────")
    for i, d in enumerate(devices):
        tag = ""
        if i == sd.default.device[0]: tag += " [DEFAULT OUTPUT]"
        if i == sd.default.device[1]: tag += " [DEFAULT INPUT]"
        print(f"  [{i:2d}] {d['name']:42s} in:{d['max_input_channels']} out:{d['max_output_channels']}{tag}")
    print("────────────────────────────────────────────────────────────\n")
    return sd.default.device   # (output_id, input_id)


def record_sweep(playback_signal, fs=SAMPLE_RATE, out_file=OUTPUT_WAV):
    """Play ESS + record mic simultaneously. Returns recorded float32 array."""
    if not HAS_SD:
        raise RuntimeError("sounddevice not installed.")
    out_id, in_id = list_audio_devices()
    print(f"▶  Output : {sd.query_devices(out_id)['name']}")
    print(f"🎙  Input  : {sd.query_devices(in_id)['name']}")
    print(f"⏳ Recording {len(playback_signal)/fs:.1f} s …")
    rec = sd.playrec(playback_signal[:, None], samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    rec = rec[:, 0]
    print(f"✅ Done. Peak: {20*np.log10(np.max(np.abs(rec))+1e-12):.1f} dBFS")
    sf.write(out_file, rec, fs, subtype="PCM_24")
    print(f"💾 WAV saved → {out_file}")
    return rec


def load_wav(path):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    print(f"📂 Loaded: {path}  ({len(data)/sr:.1f} s @ {sr} Hz)")
    return data, sr


# ═══════════════════════════════════════════════════════════════════════════
#  ESS DECONVOLUTION  (REW's actual measurement method)
# ═══════════════════════════════════════════════════════════════════════════

def ess_deconvolve(recording, inv_filter, sweep_len, fs=SAMPLE_RATE):
    """Deconvolve recorded response with ESS inverse filter to obtain the
    system impulse response, then convert to frequency-domain magnitude (dB).

    This replicates REW's measurement pipeline:
      recording  ──×──  inverse_filter  →  IR  →  window  →  FFT  →  |H(f)| dB

    Parameters
    ----------
    recording  : float array  — recorded microphone signal (aligned to sweep start)
    inv_filter : float array  — analytical inverse filter from generate_ess()
    sweep_len  : int          — number of samples in the original ESS
    fs         : int          — sample rate

    Returns
    -------
    freqs   : ndarray  — frequency axis (Hz)
    mag_db  : ndarray  — magnitude response (dB, relative)
    ir      : ndarray  — full deconvolved impulse response (for diagnostics)
    peak_idx: int      — sample index of IR peak in full IR array
    """
    # ── 1. Frequency-domain deconvolution ───────────────────────────────
    # Zero-pad to next power of two ≥ len(rec) + len(inv) − 1
    N_lin = len(recording) + len(inv_filter) - 1
    N_fft = 1 << int(np.ceil(np.log2(N_lin)))

    Y = np.fft.rfft(recording.astype(np.float64), N_fft)
    H = np.fft.rfft(inv_filter,                   N_fft)
    ir_full = np.fft.irfft(Y * H, N_fft)

    # ── 2. Locate IR peak ────────────────────────────────────────────────
    # After time-reversal deconvolution the IR peak sits at approximately
    # t = sweep_len samples (the sweep length), shifted by any system latency.
    # Search a ±50 % window around that expected position.
    search_lo = max(0,            int(sweep_len * 0.50))
    search_hi = min(len(ir_full), int(sweep_len * 1.50))
    if search_hi <= search_lo:
        search_lo, search_hi = 0, len(ir_full)

    # Find peak in the absolute value of the IR
    peak_in_window = np.argmax(np.abs(ir_full[search_lo:search_hi]))
    peak_idx = search_lo + peak_in_window

    # ── 3. Time-window the IR (half-Hann gate) ───────────────────────────
    # REW gates the IR to separate direct sound from late reflections/noise.
    # Pre-peak: short fade-in (2 ms) — avoids DC and pre-ringing artefacts
    # Post-peak: use full IR_GATE_S (200 ms default) for room-mode resolution;
    #   longer gate → better LF resolution (Δf = 1/gate_length).
    fade_in_samp = int(IR_FADE_IN_S * fs)
    gate_samp    = int(IR_GATE_S    * fs)

    ir_start = max(0, peak_idx - fade_in_samp)
    ir_end   = min(len(ir_full), peak_idx + gate_samp)
    ir_gate  = ir_full[ir_start:ir_end].copy()

    # Half-Hann fade-in (cosmetic — keeps the pre-delay clean)
    if fade_in_samp >= 2 and fade_in_samp < len(ir_gate):
        ir_gate[:fade_in_samp] *= np.hanning(fade_in_samp * 2)[:fade_in_samp]

    # Exponential decay window on the tail (matches REW's Heyser spiral gate)
    # τ chosen so the window drops to −60 dB at gate end → ~10× RT tail
    tail_len = len(ir_gate) - fade_in_samp
    if tail_len > 1:
        tau = tail_len / (60.0 / 8.686)           # 60 dB in 8.686·τ
        t_tail = np.arange(tail_len)
        ir_gate[fade_in_samp:] *= np.exp(-t_tail / tau)

    # ── 4. FFT of windowed IR → frequency response ───────────────────────
    # Zero-pad the IR to a power-of-two for clean frequency resolution.
    # REW uses a dense log-spaced grid; here we return the linear FFT grid
    # and the caller re-samples it onto a log grid for display.
    N_ir_fft = 1 << int(np.ceil(np.log2(max(len(ir_gate), 1) * 8)))
    H_f   = np.fft.rfft(ir_gate, N_ir_fft)
    freqs = np.fft.rfftfreq(N_ir_fft, d=1.0 / fs)

    mag_db = 20.0 * np.log10(np.abs(H_f) + 1e-12)

    return freqs, mag_db, ir_full, peak_idx


def resample_to_log_grid(freqs_lin, mag_db_lin, f_min=PLOT_FMIN, f_max=PLOT_FMAX, n_points=2000):
    """Re-sample a linearly-spaced spectrum onto a logarithmic frequency grid.

    REW displays its results on a log grid — this matches that presentation.
    """
    log_freqs = np.logspace(np.log10(max(f_min, freqs_lin[1])),
                            np.log10(min(f_max, freqs_lin[-1])),
                            n_points)
    log_mag = np.interp(log_freqs, freqs_lin, mag_db_lin)
    return log_freqs, log_mag


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def find_cal_file(data_dir):
    """Auto-detect a calibration .txt file inside data_dir."""
    p = Path(data_dir)
    if not p.is_dir():
        return None
    candidates = sorted(p.glob("*.txt"))
    if candidates:
        print(f"📐 Auto-detected calibration file: {candidates[0]}")
        return str(candidates[0])
    return None


def load_calibration(cal_file):
    """Load a UMIK-1 / miniDSP style calibration file.

    Header line format (first line, possibly quoted):
        "Sens Factor =X.XXXdB, AGain =YYdB, SERNO: XXXXXXX"

    Data lines: two columns — Frequency_Hz  dB_correction
    (positive = mic over-reads at that frequency → subtract to correct)

    Returns
    -------
    cal_freqs         : ndarray — frequency points (Hz), sorted
    cal_db            : ndarray — frequency-response corrections (dB)
    sensitivity_offset: float  — dB to add to recorded_dBFS to reach dBSPL
                                  = UMIK1_BASE_SENSITIVITY + AGain − Sens_Factor
                                  Returns None if header not found (no abs SPL).
    """
    import re
    sens_factor = None
    again       = None
    rows        = []

    try:
        with open(cal_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip().strip('"')   # remove surrounding quotes
                if not line:
                    continue

                # ── Try to parse the header line ─────────────────────────
                if sens_factor is None:
                    m_s = re.search(r'Sens\s+Factor\s*=\s*([\d.]+)\s*dB', line, re.IGNORECASE)
                    m_g = re.search(r'AGain\s*=\s*([\d.]+)\s*dB',         line, re.IGNORECASE)
                    if m_s and m_g:
                        sens_factor = float(m_s.group(1))
                        again       = float(m_g.group(1))
                        print(f"📐 Cal header: Sens Factor={sens_factor} dB, "
                              f"AGain={again} dB")
                        continue          # header line — no data on it

                # ── Data lines ───────────────────────────────────────────
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue

    except FileNotFoundError:
        print(f"⚠  Cal file not found: {cal_file}")
        return None, None, None

    if not rows:
        print(f"⚠  Cal file has no data rows: {cal_file}")
        return None, None, None

    arr = np.array(rows)
    idx = np.argsort(arr[:, 0])
    cal_freqs = arr[idx, 0]
    cal_db    = arr[idx, 1]

    # ── Absolute sensitivity offset ──────────────────────────────────────
    # dBSPL = recorded_dBFS + sensitivity_offset
    # For UMIK-1: sensitivity_offset = 102 + AGain − Sens_Factor
    #   (102 dB = base capsule offset at 0 dB analog gain)
    if sens_factor is not None and again is not None:
        sensitivity_offset = 100 - sens_factor + 24
        print(f"📐 Sensitivity offset: {UMIK1_BASE_SENSITIVITY:.0f} + {again:.0f} "
              f"− {sens_factor:.3f} = {sensitivity_offset:.3f} dB  "
              f"(0 dBFS → {sensitivity_offset:.1f} dBSPL)")
    else:
        sensitivity_offset = None
        print("⚠  Cal header not parsed — frequency corrections applied, "
              "but no absolute SPL conversion (output is relative dB)")

    print(f"📐 Freq corrections: {len(rows)} pts  "
          f"{cal_freqs[0]:.0f}–{cal_freqs[-1]:.0f} Hz  "
          f"(range {cal_db.min():.2f} to {cal_db.max():.2f} dB)")

    return cal_freqs, cal_db, sensitivity_offset


# ═══════════════════════════════════════════════════════════════════════════
#  FRACTIONAL-OCTAVE SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def octave_smooth(freqs, mag_db, octave_frac):
    """Fractional-octave smoothing — identical algorithm to REW's.

    REW averages the linear-scale magnitude (not dB) within each octave
    band, then converts back to dB.  Doing it in dB directly over-smooths
    sharp nulls and peaks slightly.
    """
    if octave_frac is None or len(freqs) < 3:
        return mag_db.copy()

    ratio   = 2.0 ** (1.0 / octave_frac)
    mag_lin = 10.0 ** (mag_db / 20.0)   # convert to linear amplitude
    smoothed_lin = np.empty_like(mag_lin)

    for i, fc in enumerate(freqs):
        mask = (freqs >= fc / ratio) & (freqs <= fc * ratio)
        smoothed_lin[i] = mag_lin[mask].mean() if mask.any() else mag_lin[i]

    return 20.0 * np.log10(smoothed_lin + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
#  .MDAT PARSER  (REW Java-serialisation binary format)
# ═══════════════════════════════════════════════════════════════════════════

class MdatParseError(Exception):
    pass


def _read_java_float_array(data, count_pos):
    """Read a Java float[] given the byte offset of the 4-byte element count.
    Returns (numpy_float32_array, offset_after_array)."""
    count = struct.unpack(">I", data[count_pos:count_pos + 4])[0]
    if count == 0 or count > 500_000:
        raise MdatParseError(f"Implausible float[] count {count} at {count_pos}")
    end = count_pos + 4 + count * 4
    if end > len(data):
        raise MdatParseError("float[] extends beyond EOF")
    arr = np.frombuffer(data[count_pos + 4:end], dtype=">f4").astype(np.float64)
    return arr, end


def _extract_name(data):
    """Extract the human-readable measurement name from the .mdat blob.

    REW stores the measurement name as a Java TC_STRING late in the file,
    followed by sweep level, sweep description, and input device strings.
    We want the *first* short plausible name after the 55 % mark.
    """
    _SKIP = {
        "HERMITE","SINC","TUKEY","NONE","OFF","SUBWOOFER","SLOPE_24DB",
        "linux","CUBIC","LINEAR","NEAREST","SMOOTH","SHELF","PEAK",
        "HIGH_PASS","LOW_PASS","ALL_PASS","BAND_PASS",
    }

    cutoff = int(len(data) * 0.65)   # measurement name lives in last ~35 %
    i = cutoff
    while i < len(data) - 3:
        if data[i] == 0x74:
            slen = struct.unpack(">H", data[i + 1:i + 3])[0]
            # Measurement names in REW are 3–40 chars and don't contain
            # spaces followed by brackets (device strings look like "X [y]")
            if 3 <= slen <= 40:
                try:
                    s = data[i + 3:i + 3 + slen].decode("ascii")
                    if (all(32 <= ord(c) < 127 for c in s)
                            and not s.startswith(("L", "[", "java", "roomeq"))
                            and ";" not in s and "/" not in s
                            and s not in _SKIP
                            and not s[0].isdigit()
                            and "[plughw" not in s
                            and "dBFS" not in s):
                        return s
                except (UnicodeDecodeError, IndexError):
                    pass
        i += 1
    return "REW Reference"


def parse_mdat(filepath):
    """Parse a REW .mdat file.

    Returns dict:
        name    – measurement label (str)
        freqs   – frequency axis, Hz (ndarray)
        spl_raw – unsmoothed SPL, dB (ndarray)
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    if raw[:4] != b"\xac\xed\x00\x05":
        raise MdatParseError(f"{filepath}: not a Java serialisation stream")

    # ── 1. Find the float[] class descriptor for the frequency axis ──────
    # Signature: TC_CLASSDESC for "[F"
    #   75 72 00 02 5b 46  + 8-byte serialUID  + 02 00 00 78 70
    SIG_F = b"\x75\x72\x00\x02\x5b\x46"
    sig_pos = raw.find(SIG_F)
    if sig_pos == -1:
        raise MdatParseError("float[] class descriptor not found")

    # Descriptor is exactly 19 bytes → count field immediately follows
    freq_count_pos = sig_pos + 19
    freqs, after_freqs = _read_java_float_array(raw, freq_count_pos)

    if not (1.0 <= freqs[0] <= 100.0 and 5_000 <= freqs[-1] <= 100_000):
        raise MdatParseError(f"Frequency array out of expected range: "
                             f"{freqs[0]:.1f}–{freqs[-1]:.1f} Hz")

    # ── 2. Find the [[F array of smoothed SPL curves ─────────────────────
    SIG_FF = b"\x75\x72\x00\x03\x5b\x5b\x46"   # TC_ARRAY classDesc for [[F
    twod_pos = raw.find(SIG_FF, after_freqs)
    if twod_pos == -1:
        raise MdatParseError("[[F array not found")

    # [[F descriptor is 20 bytes → outer array count
    outer_count_pos = twod_pos + 20
    outer_count = struct.unpack(">I", raw[outer_count_pos:outer_count_pos + 4])[0]
    if not (1 <= outer_count <= 20):
        raise MdatParseError(f"Unexpected [[F outer count: {outer_count}")

    pos = outer_count_pos + 4
    spl_arrays = []
    for _ in range(outer_count):
        # Each sub-array: TC_ARRAY (0x75) + TC_REFERENCE (0x71) + 4-byte handle
        if raw[pos] != 0x75:
            raise MdatParseError(f"Expected TC_ARRAY (0x75) at {pos}, got 0x{raw[pos]:02x}")
        pos += 1
        if raw[pos] != 0x71:
            raise MdatParseError(f"Expected TC_REFERENCE (0x71) at {pos}")
        pos += 5   # tag + 4-byte handle
        arr, pos = _read_java_float_array(raw, pos)
        spl_arrays.append(arr)

    # The sub-array with the highest std is the unsmoothed (raw) response
    raw_idx = int(np.argmax([a.std() for a in spl_arrays]))
    spl_raw = spl_arrays[raw_idx]

    n = min(len(freqs), len(spl_raw))
    name = _extract_name(raw)

    return dict(name=name, freqs=freqs[:n], spl_raw=spl_raw[:n])


def load_mdat_dir(ref_dir):
    """Load all .mdat files from a directory. Returns list of parsed dicts."""
    results = []
    p = Path(ref_dir)
    if not p.is_dir():
        print(f"⚠  Reference directory not found: {ref_dir}")
        return results
    for f in sorted(p.glob("*.mdat")):
        try:
            m = parse_mdat(f)
            print(f"  ✓  {f.name:30s} → '{m['name']}'  "
                  f"({len(m['freqs'])} pts, "
                  f"{m['freqs'][0]:.0f}–{m['freqs'][-1]:.0f} Hz)")
            results.append(m)
        except MdatParseError as e:
            print(f"  ✗  {f.name}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def _freq_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(PLOT_FMIN, PLOT_FMAX)
    # Ticks from 10 Hz upward — matches REW's display range
    major = [10, 20, 30, 40, 50, 60, 70, 80, 100,
             200, 300, 400, 500, 600, 700, 800, 1000,
             2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 20000]
    ax.set_xticks(major)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x//1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.tick_params(axis="x", which="minor", bottom=False)


def plot_responses(meas_f, meas_db, refs, smooth_label, out_path):
    fig, ax = plt.subplots(figsize=(20, 8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, which="major", color="#333333", lw=0.8)
    ax.grid(True, which="minor", color="#2a2a2a", lw=0.4, ls=":")

    all_vals = []

    # ── Reference lines — thick but transparent so measured stands out ───
    for i, ref in enumerate(refs):
        c = COLOUR_REF_CYCLE[i % len(COLOUR_REF_CYCLE)]
        # Only plot reference points within the display range
        mask = (ref["freqs"] >= PLOT_FMIN) & (ref["freqs"] <= PLOT_FMAX)
        ax.plot(ref["freqs"][mask], ref["spl_plot"][mask],
                color=c, lw=2.5, alpha=0.40,
                label=f"{ref['name']}  [REF]")
        all_vals.extend(ref["spl_plot"][mask][np.isfinite(ref["spl_plot"][mask])])

    # ── Measured line — full opacity, prominent ───────────────────────────
    if meas_f is not None:
        mask_m = (meas_f >= PLOT_FMIN) & (meas_f <= PLOT_FMAX)
        ax.plot(meas_f[mask_m], meas_db[mask_m],
                color=COLOUR_MEASURED, lw=1.8, alpha=1.0, zorder=5,
                label=f"Measured  (smoothing: {smooth_label})")
        all_vals.extend(meas_db[mask_m][np.isfinite(meas_db[mask_m])])

    _freq_axis(ax)

    # ── Labels & title (larger, readable) ────────────────────────────────
    ax.set_xlabel("Frequency (Hz)", color="#dddddd", fontsize=14, labelpad=8)
    ax.set_ylabel("SPL (dB)",       color="#dddddd", fontsize=14, labelpad=8)
    ax.set_title("Frequency Response Comparison",
                 color="#ffffff", fontsize=16, pad=14, fontweight="bold")

    if all_vals:
        lo = max(0,   np.percentile(all_vals,  1) - 10)
        hi = min(160, np.percentile(all_vals, 99) + 10)
        step = 10
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.arange(round(lo / step) * step,
                                round(hi / step) * step + step, step))

    ax.tick_params(colors="#bbbbbb", labelsize=11)
    for sp in ax.spines.values():
        sp.set_color("#555555")

    ax.legend(loc="upper left", fontsize=12,
              facecolor="#2a2a2a", edgecolor="#666666", labelcolor="#eeeeee",
              framealpha=0.85, borderpad=0.8, labelspacing=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊 Plot saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_csv(meas_f, meas_db, refs, out_path):
    grid = meas_f if meas_f is not None else (refs[0]["freqs"] if refs else None)
    if grid is None:
        print("⚠  Nothing to export to CSV.")
        return

    header = ["Frequency_Hz", "Measured_dB"]
    cols   = [grid,
              np.interp(grid, meas_f, meas_db) if meas_f is not None
              else np.full(len(grid), np.nan)]

    for ref in refs:
        safe = ref["name"].replace(",", " ").replace("\n", " ")
        header.append(f"REF_{safe}_dB")
        cols.append(np.interp(grid, ref["freqs"], ref["spl_plot"]))

    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in zip(*cols):
            f.write(",".join("" if not np.isfinite(v) else f"{v:.4f}" for v in row) + "\n")
    print(f"📄 CSV saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPT
# ═══════════════════════════════════════════════════════════════════════════

def prompt_smoothing():
    opts = list(SMOOTHING_MAP.keys())
    print("\n── Smoothing ───────────────────────────────────────────────")
    for i, o in enumerate(opts, 1):
        print(f"  [{i}] {o}")
    print("────────────────────────────────────────────────────────────")
    while True:
        c = input("Select [1–4, default=1 none]: ").strip()
        if c == "":
            return "none"
        if c.isdigit() and 1 <= int(c) <= len(opts):
            return opts[int(c) - 1]


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="REW-mimic acoustic measurement tool (ESS deconvolution method)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--no-record",  action="store_true",
                    help="Skip live recording; analyse --sweep-file instead")
    ap.add_argument("--sweep-file", default=None,
                    help="Existing recorded sweep WAV (used with --no-record)")
    ap.add_argument("--cal-file",   default=None,
                    help="Mic calibration .txt — auto-detected in data/ if omitted. "
                         "For UMIK-1 files the header Sens Factor + AGain are used "
                         "to convert deconvolved dBFS → absolute dBSPL.")
    ap.add_argument("--ref-dir",    default=f"{DEFAULT_DATA_DIR}/REW Standard Data",
                    help="Directory of .mdat reference files")
    ap.add_argument("--smoothing",  default=None, choices=list(SMOOTHING_MAP.keys()))
    ap.add_argument("--output-dir", default=DEFAULT_DATA_DIR,
                    help="Output directory for PNG, CSV, WAV")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path  = out_dir / OUTPUT_WAV
    plot_path = out_dir / PLOT_FILENAME
    csv_path  = out_dir / CSV_FILENAME

    print("═" * 60)
    print("  REW-Mimic Acoustic Analysis Tool  (ESS / Farina method)")
    print("═" * 60)

    # Smoothing
    smooth_key  = args.smoothing or prompt_smoothing()
    octave_frac = SMOOTHING_MAP[smooth_key]
    print(f"\n🎛  Smoothing: {smooth_key}")

    # Generate ESS sweep + inverse filter
    print(f"\n🔊 Generating {SWEEP_DURATION_S} s ESS  "
          f"({SWEEP_START_HZ}–{SWEEP_END_HZ} Hz, {SWEEP_LEVEL_DBFS} dBFS) …")
    sweep, inv_filter, playback = generate_ess()

    # Record or load
    recording = None
    if args.no_record:
        wav_src = args.sweep_file or str(wav_path)
        if Path(wav_src).exists():
            recording, rec_sr = load_wav(wav_src)
            if rec_sr != SAMPLE_RATE:
                from math import gcd
                g = gcd(SAMPLE_RATE, rec_sr)
                recording = sig.resample_poly(recording, SAMPLE_RATE // g, rec_sr // g).astype(np.float32)
        else:
            print(f"⚠  WAV not found: {wav_src} — skipping measurement.")
    else:
        if not HAS_SD:
            print("❌ sounddevice unavailable (install: pip install sounddevice + libportaudio2)")
        else:
            recording = record_sweep(playback, out_file=str(wav_path))

    # ESS deconvolution analysis
    meas_f = meas_db = None
    ref_raw = []   # populated inside measurement block (for auto SPL align) or below
    if recording is not None:
        print("\n🔬 ESS deconvolution → impulse response → FFT …")
        # Strip pre-silence so the recording starts at sweep time 0
        pre_samp = int(SILENCE_PRE_S * SAMPLE_RATE)
        rec_aligned = recording[pre_samp:] if len(recording) > pre_samp else recording

        freqs_lin, mag_lin, ir_full, peak = ess_deconvolve(
            rec_aligned, inv_filter, len(sweep), SAMPLE_RATE
        )
        print(f"   IR peak at sample {peak} ({peak/SAMPLE_RATE*1000:.1f} ms)")

        # ── Load calibration file ────────────────────────────────────────
        cal_file = args.cal_file or find_cal_file(DEFAULT_DATA_DIR)
        sensitivity_offset = None
        if cal_file:
            cal_f, cal_db, sensitivity_offset = load_calibration(cal_file)
            if cal_f is not None:
                # 1. Apply frequency-response correction on the linear grid
                #    (positive cal value → mic reads high → subtract)
                cal_on_lin = np.interp(freqs_lin, cal_f, cal_db,
                                       left=cal_db[0], right=cal_db[-1])
                mag_lin = mag_lin - cal_on_lin
        else:
            print("ℹ  No calibration file found in data/ — output is relative dB")

        # Re-sample onto log frequency grid (matches REW display, 10–20k Hz)
        meas_f, meas_db = resample_to_log_grid(freqs_lin, mag_lin)

        # Apply smoothing
        if octave_frac is not None:
            print(f"   Applying 1/{octave_frac}-octave smoothing …")
            meas_db = octave_smooth(meas_f, meas_db, octave_frac)

        # ── Convert dBFS → absolute dBSPL using mic sensitivity ──────────
        # dBSPL(f) = mag_db(f)  +  SWEEP_LEVEL_DBFS  +  sensitivity_offset
        #
        # Explanation:
        #   mag_db is the system transfer function H(f) in dB:
        #     H(f) = 0 dB means the recorded level equals the played level
        #   recorded_dBFS = SWEEP_LEVEL_DBFS + mag_db
        #   dBSPL = recorded_dBFS + sensitivity_offset
        #         = mag_db + SWEEP_LEVEL_DBFS + sensitivity_offset
        if sensitivity_offset is not None:
            spl_const = SWEEP_LEVEL_DBFS + sensitivity_offset
            meas_db   = meas_db + spl_const
            print(f"🎚  SPL conversion: mag_db + {SWEEP_LEVEL_DBFS} dBFS "
                  f"+ {sensitivity_offset:.3f} dB (mic sens) "
                  f"= mag_db + {spl_const:.3f} dB")
        else:
            print("ℹ  No sensitivity offset — output is relative dB "
                  "(not absolute dBSPL)")

        print(f"   Measurement range: {meas_db.min():.1f} – {meas_db.max():.1f} dB")

    # Load .mdat reference files (if no measurement was made, load them now)
    if meas_f is None:
        print(f"\n📁 Loading .mdat references from: {args.ref_dir}")
        ref_raw = load_mdat_dir(args.ref_dir)
    else:
        print(f"\n📁 Loading .mdat references from: {args.ref_dir}")
        ref_raw = load_mdat_dir(args.ref_dir)
    if not ref_raw:
        print("  (none found)")

    refs = []
    for r in ref_raw:
        spl = r["spl_raw"].copy()
        if octave_frac is not None:
            spl = octave_smooth(r["freqs"], spl, octave_frac)
        refs.append({**r, "spl_plot": spl})

    # Plot + CSV
    print("\n🖼  Rendering plot …")
    plot_responses(meas_f, meas_db, refs, smooth_key, str(plot_path))
    save_csv(meas_f, meas_db, refs, str(csv_path))

    print("\n✅ Done.")
    print(f"   Plot : {plot_path}")
    print(f"   CSV  : {csv_path}")
    if not args.no_record and recording is not None:
        print(f"   WAV  : {wav_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  SELF-TEST  (no audio hardware required)
# ═══════════════════════════════════════════════════════════════════════════

def _selftest():
    """Validate ESS pipeline with a synthetic room simulation, then render
    the uploaded REW_sweep_3.mdat reference at all smoothing levels."""
    from pathlib import Path
    out = Path("/mnt/user-data/outputs")
    out.mkdir(exist_ok=True)

    print("═" * 60)
    print("  Self-test A: ESS pipeline validation (synthetic room)")
    print("═" * 60)

    fs = SAMPLE_RATE
    sweep, inv_filter, playback = generate_ess()

    # ── Simulate a simple room: direct sound + 3 reflections ────────────
    # Room IR: delta at t=0 (direct), then decaying reflections
    ir_room_len = int(0.5 * fs)
    ir_room = np.zeros(ir_room_len)
    ir_room[0]                  = 1.0         # direct sound (0 ms)
    ir_room[int(0.015 * fs)]    = 0.4         # 15 ms reflection
    ir_room[int(0.035 * fs)]    = 0.25        # 35 ms reflection
    ir_room[int(0.080 * fs)]    = 0.15        # 80 ms reflection
    # Add diffuse reverb tail
    tail = np.random.randn(ir_room_len - int(0.1 * fs)) * 0.05
    tail *= np.exp(-np.linspace(0, 5, len(tail)))
    ir_room[int(0.1 * fs):] += tail

    # Simulate recording: convolve playback with room IR + noise
    rec_clean = np.convolve(playback.astype(np.float64), ir_room)
    noise = np.random.randn(len(rec_clean)) * (10 ** (-50 / 20))   # −50 dB noise floor
    recording = (rec_clean + noise).astype(np.float32)

    # Strip pre-silence
    pre_samp = int(SILENCE_PRE_S * fs)
    rec_aligned = recording[pre_samp:]

    # Run ESS deconvolution
    freqs_lin, mag_lin, ir_full, peak = ess_deconvolve(rec_aligned, inv_filter, len(sweep), fs)
    print(f"  IR peak at {peak/fs*1000:.1f} ms  (expected ~{SILENCE_POST_S*1000:.0f} ms tail + latency)")

    # Expected response from room IR (direct only for high-freq reference)
    H_expected = np.fft.rfft(ir_room, len(ir_full))
    f_exp = np.fft.rfftfreq(len(ir_full), 1 / fs)
    mag_exp_db = 20 * np.log10(np.abs(H_expected) + 1e-12)

    meas_f, meas_db = resample_to_log_grid(freqs_lin, mag_lin)
    exp_f,  exp_db  = resample_to_log_grid(f_exp, mag_exp_db)

    # Align levels (set mean over 1–5 kHz equal)
    band = (meas_f >= 1000) & (meas_f <= 5000)
    if band.any():
        offset = np.mean(exp_db[band]) - np.mean(meas_db[band])
        meas_db += offset

    # Plot comparison: ESS-recovered vs. known room IR
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("#1a1a1a"); ax.set_facecolor("#1e1e1e")
    ax.grid(True, color="#333", lw=0.7)
    ax.plot(meas_f, meas_db, color="#2ecc71", lw=1.0,  label="ESS deconvolution (recovered)")
    ax.plot(exp_f,  exp_db,  color="#e74c3c", lw=1.5, ls="--", label="Known room IR (reference)")
    _freq_axis(ax)
    ax.set_ylim(-40, 10); ax.set_ylabel("Relative SPL (dB)", color="#ccc")
    ax.set_xlabel("Frequency (Hz)", color="#ccc")
    ax.tick_params(colors="#aaa")
    ax.legend(facecolor="#2a2a2a", edgecolor="#555", labelcolor="#eee")
    ax.set_title("Self-test: ESS Deconvolution vs Known Room IR", color="#eee")
    plt.tight_layout()
    fig.savefig(str(out / "rew_selftest_ess.png"), dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot → {out/'rew_selftest_ess.png'}")

    print()
    print("═" * 60)
    print("  Self-test B: REW_sweep_3.mdat — all smoothing levels")
    print("═" * 60)

    mdat_path = Path("/mnt/user-data/uploads/REW_sweep_3.mdat")
    m = parse_mdat(mdat_path)
    print(f"  '{m['name']}'  {m['freqs'][0]:.0f}–{m['freqs'][-1]:.0f} Hz  "
          f"SPL {m['spl_raw'].min():.1f}–{m['spl_raw'].max():.1f} dB")

    refs_all = []
    for label, frac in [("none", None), ("1/6", 6), ("1/3", 3), ("1/1", 1)]:
        spl = octave_smooth(m["freqs"], m["spl_raw"], frac)
        refs_all.append({**m, "name": f"{m['name']} – {label}", "spl_plot": spl})

    plot_responses(None, None, refs_all, "all",
                   str(out / "rew_analysis.png"))
    save_csv(None, None, refs_all, str(out / "rew_analysis.csv"))
    print("\n✅ Self-test complete.")


if __name__ == "__main__":
    if "--selftest" in sys.argv or (len(sys.argv) == 1 and not sys.stdin.isatty()):
        _selftest()
    else:
        main()