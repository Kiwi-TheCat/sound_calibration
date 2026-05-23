#!/usr/bin/env python3
"""
REW-Mimic Acoustic Measurement & Analysis Tool
================================================
Measures room/speaker acoustic frequency response using a log sweep,
then compares against REW .mdat reference files with optional smoothing.

Requirements:
    pip install sounddevice scipy numpy matplotlib soundfile

Usage:
    python rew_analysis.py [--no-record] [--sweep-file path] [--cal-file path]
                           [--ref-dir path] [--smoothing none|1/6|1/3|1/1]
                           [--output-dir path]

    --no-record     Skip live recording; load existing sweep WAV file instead
    --sweep-file    Path to existing WAV file (used with --no-record)
    --cal-file      Path to calibration .txt file (frequency, dB correction)
    --ref-dir       Directory containing .mdat reference files
    --smoothing     Smoothing amount: none, 1/6, 1/3, 1/1 (default: prompt)
    --output-dir    Where to save outputs (default: current directory)
"""

import os
import sys
import struct
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.signal as signal
import scipy.fft as fft
import soundfile as sf
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving PNG
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Try importing sounddevice (needed only for live recording) ──────────────
try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    HAS_SD = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════
SWEEP_START_HZ   = 20          # Start frequency of log sweep (Hz)
SWEEP_END_HZ     = 20_000      # End frequency of log sweep (Hz)
SWEEP_DURATION_S = 10          # Duration of sweep (seconds)
SWEEP_LEVEL_DBFS = -12         # Playback level in dBFS
SAMPLE_RATE      = 48_000      # Audio sample rate (Hz)
SILENCE_PRE_S    = 0.5         # Silence before sweep (seconds)
SILENCE_POST_S   = 1.0         # Silence after sweep (seconds)
OUTPUT_FILENAME  = "mimic_sweep_1.wav"
PLOT_FILENAME    = "rew_analysis.png"
CSV_FILENAME     = "rew_analysis.csv"

# Frequency axis limits for plot
PLOT_FMIN = 2
PLOT_FMAX = 24_000

# Smoothing octave fractions → window widths
SMOOTHING_MAP = {
    "none": None,
    "1/6":  6,
    "1/3":  3,
    "1/1":  1,
}

# ── .mdat SPL array index for "unsmoothed" reference ───────────────────────
MDAT_UNSMOOTHED_IDX = 1   # index 1 has highest variance (raw response)

# REW colour palette (approximate)
COLOUR_MEASURED  = "#2ecc71"   # green  – live measurement
COLOUR_REF_BASE  = "#e74c3c"   # red    – first reference
COLOUR_REF_EXTRA = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c"]


# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_log_sweep(
    start_hz=SWEEP_START_HZ,
    end_hz=SWEEP_END_HZ,
    duration_s=SWEEP_DURATION_S,
    level_dbfs=SWEEP_LEVEL_DBFS,
    sample_rate=SAMPLE_RATE,
    pre_silence_s=SILENCE_PRE_S,
    post_silence_s=SILENCE_POST_S,
):
    """Return (sweep_signal, full_signal_with_silence) as float32 arrays."""
    n_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)

    # Logarithmic frequency sweep (Chirp)
    sweep = signal.chirp(t, f0=start_hz, f1=end_hz, t1=duration_s, method="logarithmic")

    # Apply Tukey window (5 % taper) to avoid clicks
    window = signal.windows.tukey(n_samples, alpha=0.05)
    sweep *= window

    # Normalise to target dBFS
    amplitude = 10 ** (level_dbfs / 20.0)
    sweep = (sweep / np.max(np.abs(sweep))) * amplitude

    # Silence padding
    pre  = np.zeros(int(pre_silence_s  * sample_rate), dtype=np.float32)
    post = np.zeros(int(post_silence_s * sample_rate), dtype=np.float32)
    full = np.concatenate([pre, sweep.astype(np.float32), post])

    return sweep.astype(np.float32), full


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO I/O
# ═══════════════════════════════════════════════════════════════════════════

def list_audio_devices():
    """Print available audio devices; return (playback_id, capture_id) defaults."""
    if not HAS_SD:
        raise RuntimeError("sounddevice not installed – cannot list audio devices.")
    devices = sd.query_devices()
    print("\n── Available Audio Devices ────────────────────────────────")
    for i, d in enumerate(devices):
        marker = ""
        if i == sd.default.device[0]: marker += " [DEFAULT OUTPUT]"
        if i == sd.default.device[1]: marker += " [DEFAULT INPUT]"
        print(f"  [{i:2d}] {d['name']:40s}  in:{d['max_input_channels']}  out:{d['max_output_channels']}{marker}")
    print("──────────────────────────────────────────────────────────\n")
    return sd.default.device   # (output_id, input_id)


def record_sweep(sweep_signal, sample_rate=SAMPLE_RATE, out_file=OUTPUT_FILENAME):
    """Play sweep through default output while recording default input.
       Returns raw recorded samples as float32 numpy array."""
    if not HAS_SD:
        raise RuntimeError("sounddevice not installed – cannot record.")

    out_id, in_id = list_audio_devices()
    print(f"▶  Playback device : {sd.query_devices(out_id)['name']}")
    print(f"🎙  Capture device  : {sd.query_devices(in_id)['name']}")

    n_total = len(sweep_signal)
    print(f"\n⏳ Recording {n_total/sample_rate:.1f} s sweep at {SWEEP_LEVEL_DBFS} dBFS …")

    recording = sd.playrec(
        sweep_signal[:, None],   # ensure 2-D (samples × channels)
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    rec = recording[:, 0]
    print(f"✅ Recording complete. Peak level: {20*np.log10(np.max(np.abs(rec))+1e-12):.1f} dBFS")

    sf.write(out_file, rec, sample_rate, subtype="PCM_24")
    print(f"💾 Raw recording saved → {out_file}")
    return rec


def load_sweep_wav(path):
    """Load a previously recorded sweep WAV file."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]    # take first channel
    print(f"📂 Loaded sweep WAV: {path}  ({len(data)/sr:.1f} s @ {sr} Hz)")
    return data, sr


# ═══════════════════════════════════════════════════════════════════════════
#  FFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def fft_magnitude_db(
    recording,
    sweep_signal,
    sample_rate=SAMPLE_RATE,
    n_fft_segments=8,
):
    """Compute frequency response magnitude (dB) via averaged Welch-style FFT.

    Uses overlap-add of Hann-windowed segments, returns (freqs_Hz, magnitude_dB).
    """
    # Align recording length to sweep
    min_len = min(len(recording), len(sweep_signal))
    rec  = recording[:min_len]
    sw   = sweep_signal[:min_len]

    # Welch's method for averaging
    nperseg = len(rec) // n_fft_segments
    nperseg = max(nperseg, 4096)
    # Ensure power-of-two for efficiency
    nperseg = 2 ** int(np.log2(nperseg))

    freqs, Pxy = signal.csd(rec, sw, fs=sample_rate, nperseg=nperseg, window="hann", scaling="density")
    freqs, Pxx = signal.welch(sw, fs=sample_rate, nperseg=nperseg, window="hann", scaling="density")

    # Transfer function H = Pxy / Pxx  (avoids dividing by near-zero)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(Pxx > 0, Pxy / Pxx, 0)

    mag_db = 20 * np.log10(np.abs(H) + 1e-12)

    # Keep only positive frequencies above 1 Hz
    mask = freqs >= 1
    return freqs[mask], mag_db[mask]


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(cal_file):
    """Load a calibration .txt file with two columns: frequency (Hz), correction (dB).
    Lines starting with '*' or '#' are treated as comments.
    Returns interpolated (freqs, corrections) arrays."""
    rows = []
    try:
        with open(cal_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("*") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"⚠  Calibration file not found: {cal_file} – no correction applied.")
        return None, None

    if not rows:
        print(f"⚠  Calibration file is empty or unreadable: {cal_file}")
        return None, None

    cal_freqs  = np.array([r[0] for r in rows])
    cal_db     = np.array([r[1] for r in rows])
    sort_idx   = np.argsort(cal_freqs)
    print(f"📐 Calibration loaded: {len(cal_freqs)} points from {cal_freqs[sort_idx[0]]:.0f} – {cal_freqs[sort_idx[-1]]:.0f} Hz")
    return cal_freqs[sort_idx], cal_db[sort_idx]


def apply_calibration(freqs, mag_db, cal_freqs, cal_db):
    """Interpolate calibration curve onto measurement frequencies and subtract."""
    correction = np.interp(freqs, cal_freqs, cal_db, left=cal_db[0], right=cal_db[-1])
    return mag_db - correction


# ═══════════════════════════════════════════════════════════════════════════
#  SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def octave_smooth(freqs, mag_db, octave_frac):
    """Apply fractional-octave smoothing to a frequency response.

    Parameters
    ----------
    freqs : array of frequencies in Hz
    mag_db : array of magnitudes in dB (same length as freqs)
    octave_frac : integer N such that smoothing width = 1/N octave
                  (e.g. 6 → 1/6-octave,  1 → 1/1-octave)

    Returns
    -------
    smoothed_db : array same length as mag_db
    """
    if octave_frac is None or len(freqs) < 3:
        return mag_db.copy()

    ratio = 2 ** (1.0 / octave_frac)   # upper edge = freq * ratio
    smoothed = np.empty_like(mag_db)

    for i, fc in enumerate(freqs):
        f_lo = fc / ratio
        f_hi = fc * ratio
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        smoothed[i] = np.mean(mag_db[mask]) if mask.any() else mag_db[i]

    return smoothed


# ═══════════════════════════════════════════════════════════════════════════
#  .MDAT PARSER  (REW Java-serialisation format)
# ═══════════════════════════════════════════════════════════════════════════

class MdatParseError(Exception):
    pass


def _find_float_array_at(data, needle_pos):
    """Read a Java float[] starting at needle_pos (the 4-byte count field).

    Returns (numpy_array, next_pos).
    """
    count = struct.unpack(">I", data[needle_pos : needle_pos + 4])[0]
    if count == 0 or count > 200_000:
        raise MdatParseError(f"Implausible float array count {count} at {needle_pos}")
    end = needle_pos + 4 + count * 4
    if end > len(data):
        raise MdatParseError(f"Float array would exceed file size")
    floats = np.array(struct.unpack(f">{count}f", data[needle_pos + 4 : end]))
    return floats, end


def _extract_measurement_name(data):
    """Try to pull a human-readable measurement name out of the .mdat blob."""
    text = data.decode("latin-1", errors="replace")
    # REW stores the name as a short Java string in the instance data
    # Scan for TC_STRING (0x74) with plausible lengths 3–60
    for i in range(len(data) - 3):
        if data[i] == 0x74:
            slen = struct.unpack(">H", data[i + 1 : i + 3])[0]
            if 3 <= slen <= 60:
                candidate = data[i + 3 : i + 3 + slen].decode("latin-1", errors="replace")
                # Accept strings that look like measurement names
                if (
                    all(32 <= ord(c) < 127 for c in candidate)
                    and not candidate.startswith("L")
                    and not candidate.startswith("[")
                    and not "/" in candidate
                    and ";" not in candidate
                    and not candidate.startswith("java")
                ):
                    # Prefer strings found in the last third of the file
                    # (where instance data lives in REW files)
                    if i > len(data) * 0.6:
                        return candidate
    return "REW Reference"


def parse_mdat(filepath):
    """Parse a REW .mdat file and return a dict with:
        name    : str   – measurement label
        freqs   : ndarray(N)  – frequency axis (Hz)
        spl_raw : ndarray(N)  – unsmoothed SPL (dB)
    """
    with open(filepath, "rb") as f:
        data = f.read()

    if not data[:4] == b"\xac\xed\x00\x05":
        raise MdatParseError(f"{filepath}: not a Java serialisation stream")

    text = data.decode("latin-1", errors="replace")

    # ── 1. Locate the float frequency array ─────────────────────────────
    # Find the TC_ARRAY marker for [F class (already resolved in file)
    # Pattern: 75 72 00 02 5b 46 [8-byte UID] 02 00 00 78 70 [count] [floats…]
    FLOAT_ARRAY_SIG = b"\x75\x72\x00\x02\x5b\x46"
    sig_pos = data.find(FLOAT_ARRAY_SIG)
    if sig_pos == -1:
        raise MdatParseError(f"{filepath}: could not find float-array class descriptor")

    # The descriptor is 19 bytes: 6 (sig) + 8 (UID) + 1 (flags) + 2 (0x0000) + 1 (xp=TC_ENDBLOCKDATA) + 1 (TC_NULL)
    count_pos = sig_pos + 19
    freqs, spl_start = _find_float_array_at(data, count_pos)

    # Sanity-check: should be log-spaced frequencies in audio range
    if freqs[0] < 1 or freqs[-1] > 100_000:
        raise MdatParseError(f"{filepath}: frequency array values out of range")

    # ── 2. Locate the 10-element [[F smoothed-SPL outer array ───────────
    # After the freq array, there's: 70 70  (two TC_NULL refs)
    # then TC_ARRAY + [[F classDesc + 00 00 00 0a (10 elements)
    # Each element: 75 71 00 7e xx xx  00 00 04 xx  floats…
    TWOD_SIG = b"\x75\x72\x00\x03\x5b\x5b\x46"   # u r . . [ [ F
    twod_pos = data.find(TWOD_SIG, spl_start)
    if twod_pos == -1:
        raise MdatParseError(f"{filepath}: could not find [[F array")

    # Skip: 7 (sig) + 8 (UID) + 1 (flags) + 2 (0 fields) + 1 (TC_ENDBLOCKDATA) + 1 (TC_NULL) = 20 bytes
    outer_count_pos = twod_pos + 20
    outer_count = struct.unpack(">I", data[outer_count_pos : outer_count_pos + 4])[0]
    if outer_count < 1 or outer_count > 20:
        raise MdatParseError(f"{filepath}: unexpected outer array size {outer_count}")

    pos = outer_count_pos + 4
    spl_arrays = []
    for _ in range(outer_count):
        if data[pos] != 0x75:
            raise MdatParseError("Expected TC_ARRAY for sub-array")
        pos += 1
        if data[pos] != 0x71:
            raise MdatParseError("Expected TC_REFERENCE for sub-array class")
        pos += 5   # TC_REFERENCE + 4-byte handle
        arr, pos = _find_float_array_at(data, pos)
        spl_arrays.append(arr)

    # Array with highest std is the unsmoothed raw data
    stds      = [a.std() for a in spl_arrays]
    raw_idx   = int(np.argmax(stds))
    spl_raw   = spl_arrays[raw_idx]

    # Trim both to same length (should match, but be safe)
    n = min(len(freqs), len(spl_raw))
    name = _extract_measurement_name(data)

    return dict(name=name, freqs=freqs[:n], spl_raw=spl_raw[:n], all_spl=spl_arrays)


def load_mdat_directory(ref_dir):
    """Load all .mdat files from a directory. Returns list of parsed dicts."""
    results = []
    ref_path = Path(ref_dir)
    if not ref_path.is_dir():
        print(f"⚠  Reference directory not found: {ref_dir}")
        return results

    for mdat_file in sorted(ref_path.glob("*.mdat")):
        try:
            m = parse_mdat(mdat_file)
            print(f"  ✓  Loaded {mdat_file.name}  → '{m['name']}'  "
                  f"({len(m['freqs'])} pts, {m['freqs'][0]:.0f}–{m['freqs'][-1]:.0f} Hz)")
            results.append(m)
        except MdatParseError as e:
            print(f"  ✗  Skipped {mdat_file.name}: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def _format_freq_axis(ax):
    """Apply logarithmic x-axis formatting matching REW style."""
    ax.set_xscale("log")
    ax.set_xlim(PLOT_FMIN, PLOT_FMAX)

    major_ticks = [2, 3, 4, 5, 6, 7, 8, 10,
                   20, 30, 40, 50, 60, 70, 80, 100,
                   200, 300, 400, 500, 600, 700, 800, 1000,
                   2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000,
                   20000]
    ax.set_xticks(major_ticks)
    ax.get_xaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.tick_params(axis="x", which="minor", bottom=False)


def plot_responses(
    meas_freqs, meas_spl,
    reference_data,
    smoothing_label,
    output_path=PLOT_FILENAME,
    meas_label="Measured",
):
    """Create REW-style frequency response comparison plot."""
    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1e1e1e")

    # Grid
    ax.grid(True, which="major", color="#333333", linewidth=0.7, linestyle="-")
    ax.grid(True, which="minor", color="#2a2a2a", linewidth=0.4, linestyle=":")

    # ── Plot measured response ───────────────────────────────────────────
    if meas_freqs is not None and meas_spl is not None:
        ax.plot(meas_freqs, meas_spl,
                color=COLOUR_MEASURED, linewidth=0.9, alpha=0.95,
                label=f"{meas_label}  (smoothing: {smoothing_label})")

    # ── Plot reference data ──────────────────────────────────────────────
    ref_colours = [COLOUR_REF_BASE] + COLOUR_REF_EXTRA
    for idx, ref in enumerate(reference_data):
        colour = ref_colours[idx % len(ref_colours)]
        ax.plot(ref["freqs"], ref["spl_plot"],
                color=colour, linewidth=1.2, alpha=0.85,
                label=f"{ref['name']}  [REF]")

    # ── Axes ─────────────────────────────────────────────────────────────
    _format_freq_axis(ax)
    ax.set_xlabel("Frequency (Hz)", color="#cccccc", fontsize=11)
    ax.set_ylabel("SPL (dB)", color="#cccccc", fontsize=11)

    # Auto y-range with headroom
    all_values = []
    if meas_spl is not None:
        all_values.extend(meas_spl[np.isfinite(meas_spl)])
    for ref in reference_data:
        all_values.extend(ref["spl_plot"][np.isfinite(ref["spl_plot"])])

    if all_values:
        lo = max(0,   np.percentile(all_values,  1) - 10)
        hi = min(160, np.percentile(all_values, 99) + 10)
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.arange(round(lo, -1), round(hi + 10, -1), 10))

    ax.tick_params(colors="#aaaaaa", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    # ── Legend & title ───────────────────────────────────────────────────
    legend = ax.legend(loc="upper left", fontsize=9,
                       facecolor="#2a2a2a", edgecolor="#555555",
                       labelcolor="#eeeeee")

    ax.set_title("Frequency Response Comparison", color="#eeeeee", fontsize=13, pad=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊 Plot saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_csv(meas_freqs, meas_spl, reference_data, output_path=CSV_FILENAME):
    """Export all frequency responses to a single CSV file."""
    # Build a common frequency grid (use measurement freqs as primary)
    if meas_freqs is not None:
        grid = meas_freqs
    elif reference_data:
        grid = reference_data[0]["freqs"]
    else:
        print("⚠  Nothing to save to CSV.")
        return

    header = ["Frequency_Hz", "Measured_dB"]
    cols   = [grid]

    if meas_spl is not None:
        cols.append(np.interp(grid, meas_freqs, meas_spl))
    else:
        cols.append(np.full(len(grid), np.nan))

    for ref in reference_data:
        safe_name = ref["name"].replace(",", " ").replace("\n", " ")
        header.append(f"REF_{safe_name}_dB")
        cols.append(np.interp(grid, ref["freqs"], ref["spl_plot"]))

    rows = list(zip(*cols))
    with open(output_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(f"{v:.3f}" if np.isfinite(v) else "" for v in row) + "\n")

    print(f"📄 CSV saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  INTERACTIVE PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

def prompt_smoothing():
    """Ask the user to select smoothing level; returns key string."""
    options = list(SMOOTHING_MAP.keys())
    print("\n── Smoothing Options ───────────────────────────────────────")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    print("────────────────────────────────────────────────────────────")
    while True:
        choice = input("Select smoothing [1–4, default=1]: ").strip()
        if choice == "":
            return "none"
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("  Please enter 1, 2, 3, or 4.")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="REW-mimic acoustic measurement & analysis tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--no-record",   action="store_true",
                        help="Skip live recording; use --sweep-file instead")
    parser.add_argument("--sweep-file",  default=OUTPUT_FILENAME,
                        help="Existing recorded sweep WAV file")
    parser.add_argument("--cal-file",    default=None,
                        help="Microphone calibration .txt file (freq dB per line)")
    parser.add_argument("--ref-dir",     default="data/REW Standard Data",
                        help="Directory containing .mdat reference files")
    parser.add_argument("--smoothing",   default=None,
                        choices=list(SMOOTHING_MAP.keys()),
                        help="Smoothing amount (prompts if omitted)")
    parser.add_argument("--output-dir",  default=".",
                        help="Output directory for PNG and CSV files")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_path  = out_dir / OUTPUT_FILENAME
    plot_path = out_dir / PLOT_FILENAME
    csv_path  = out_dir / CSV_FILENAME

    print("═" * 60)
    print("  REW-Mimic Acoustic Analysis Tool")
    print("═" * 60)

    # ── Smoothing selection ──────────────────────────────────────────────
    smoothing_key = args.smoothing or prompt_smoothing()
    octave_frac   = SMOOTHING_MAP[smoothing_key]
    print(f"\n🎛  Smoothing: {smoothing_key}")

    # ── Generate log sweep ───────────────────────────────────────────────
    print(f"\n🔊 Generating {SWEEP_DURATION_S} s log sweep "
          f"({SWEEP_START_HZ}–{SWEEP_END_HZ} Hz, {SWEEP_LEVEL_DBFS} dBFS) …")
    sweep_signal, sweep_with_silence = generate_log_sweep()

    # ── Record or load ───────────────────────────────────────────────────
    meas_freqs = meas_spl = None

    if args.no_record:
        if Path(args.sweep_file).exists():
            recording, rec_sr = load_sweep_wav(args.sweep_file)
            # Resample if needed
            if rec_sr != SAMPLE_RATE:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(SAMPLE_RATE, rec_sr)
                recording = resample_poly(recording, SAMPLE_RATE // g, rec_sr // g)
        else:
            print(f"⚠  Sweep file not found: {args.sweep_file} – skipping measurement.")
            recording = None
    else:
        if not HAS_SD:
            print("❌ sounddevice not installed.  Install with:  pip install sounddevice")
            print("   Continuing without live recording.\n")
            recording = None
        else:
            recording = record_sweep(sweep_with_silence, out_file=str(wav_path))

    # ── FFT analysis ─────────────────────────────────────────────────────
    if recording is not None:
        print("\n🔬 Computing frequency response via cross-spectral analysis …")
        # Align recording to sweep (remove pre-silence)
        pre_samples = int(SILENCE_PRE_S * SAMPLE_RATE)
        rec_aligned = recording[pre_samples : pre_samples + len(sweep_signal)]

        raw_freqs, raw_db = fft_magnitude_db(rec_aligned, sweep_signal, SAMPLE_RATE)

        # ── Apply calibration ─────────────────────────────────────────
        if args.cal_file:
            cal_f, cal_d = load_calibration(args.cal_file)
            if cal_f is not None:
                raw_db = apply_calibration(raw_freqs, raw_db, cal_f, cal_d)
        else:
            print("ℹ  No calibration file supplied – using raw FFT magnitudes.")

        # Restrict to audio band for display
        mask       = (raw_freqs >= PLOT_FMIN) & (raw_freqs <= PLOT_FMAX)
        meas_freqs = raw_freqs[mask]
        meas_spl   = raw_db[mask]

        # ── Apply smoothing ───────────────────────────────────────────
        if octave_frac is not None:
            print(f"   Applying 1/{octave_frac}-octave smoothing …")
            meas_spl = octave_smooth(meas_freqs, meas_spl, octave_frac)

        print(f"   Measurement range: {meas_spl.min():.1f} – {meas_spl.max():.1f} dB")

    # ── Load .mdat reference files ────────────────────────────────────────
    print(f"\n📁 Loading reference .mdat files from: {args.ref_dir}")
    reference_raw = load_mdat_directory(args.ref_dir)

    # Also try the uploaded file directly
    upload_mdat = Path("/mnt/user-data/uploads/REW_sweep_3.mdat")
    if upload_mdat.exists() and not any(r["name"] == "REW_sweep 3" for r in reference_raw):
        try:
            m = parse_mdat(upload_mdat)
            print(f"  ✓  Using uploaded {upload_mdat.name}  → '{m['name']}'")
            reference_raw.insert(0, m)
        except MdatParseError as e:
            print(f"  ✗  Could not parse uploaded .mdat: {e}")

    if not reference_raw:
        print("  (No reference files found – plot will show measurement only)")

    # Apply smoothing to reference data too
    reference_data = []
    for ref in reference_raw:
        spl = ref["spl_raw"].copy()
        if octave_frac is not None:
            spl = octave_smooth(ref["freqs"], spl, octave_frac)
        reference_data.append({**ref, "spl_plot": spl})

    # ── Plot ─────────────────────────────────────────────────────────────
    print(f"\n🖼  Generating comparison plot …")
    plot_responses(
        meas_freqs, meas_spl,
        reference_data,
        smoothing_label=smoothing_key,
        output_path=str(plot_path),
    )

    # ── CSV ───────────────────────────────────────────────────────────────
    save_csv(meas_freqs, meas_spl, reference_data, output_path=str(csv_path))

    print("\n✅ Done.")
    print(f"   Plot : {plot_path}")
    print(f"   CSV  : {csv_path}")
    if not args.no_record:
        print(f"   WAV  : {wav_path}")


# ─── Self-test: parse the uploaded .mdat and render a standalone plot ───────

def _selftest():
    """Quick validation: parse the uploaded .mdat and produce rew_analysis.png."""
    from pathlib import Path

    mdat_path = Path("/mnt/user-data/uploads/REW_sweep_3.mdat")
    out_png   = Path("/mnt/user-data/outputs/rew_analysis.png")
    out_csv   = Path("/mnt/user-data/outputs/rew_analysis.csv")

    print("═" * 60)
    print("  Self-test: parsing REW_sweep_3.mdat …")
    print("═" * 60)

    m = parse_mdat(mdat_path)
    print(f"Measurement : '{m['name']}'")
    print(f"Frequencies : {m['freqs'][0]:.1f} – {m['freqs'][-1]:.1f} Hz  ({len(m['freqs'])} pts)")
    print(f"SPL range   : {m['spl_raw'].min():.1f} – {m['spl_raw'].max():.1f} dB")

    # Build reference list for each smoothing option
    smoothing_options = [
        ("none", None),
        ("1/6",  6),
        ("1/3",  3),
        ("1/1",  1),
    ]

    for label, frac in smoothing_options:
        spl = octave_smooth(m["freqs"], m["spl_raw"], frac)
        ref = [{**m, "spl_plot": spl}]
        png = Path("/mnt/user-data/outputs") / f"rew_{label.replace('/','_')}_smooth.png"
        plot_responses(None, None, ref,
                       smoothing_label=label,
                       output_path=str(png))

    # Also produce the main comparison with all smoothings overlaid
    # (using no-smoothing as the baseline)
    spl_none = octave_smooth(m["freqs"], m["spl_raw"], None)
    spl_1_6  = octave_smooth(m["freqs"], m["spl_raw"], 6)
    spl_1_3  = octave_smooth(m["freqs"], m["spl_raw"], 3)
    spl_1_1  = octave_smooth(m["freqs"], m["spl_raw"], 1)

    refs_all = [
        {**m, "name": f"{m['name']} – No smoothing",    "spl_plot": spl_none},
        {**m, "name": f"{m['name']} – 1/6 oct",         "spl_plot": spl_1_6},
        {**m, "name": f"{m['name']} – 1/3 oct",         "spl_plot": spl_1_3},
        {**m, "name": f"{m['name']} – 1/1 oct",         "spl_plot": spl_1_1},
    ]
    plot_responses(None, None, refs_all,
                   smoothing_label="all",
                   output_path=str(out_png))
    save_csv(None, None, refs_all, output_path=str(out_csv))

    print("\n✅ Self-test complete.")


if __name__ == "__main__":
    # If run with no arguments in a non-interactive context (e.g. CI/Docker),
    # run the self-test; otherwise run the full CLI tool.
    if len(sys.argv) == 1 and not sys.stdin.isatty():
        _selftest()
    else:
        # Check for explicit --selftest flag
        if "--selftest" in sys.argv:
            _selftest()
        else:
            main()
