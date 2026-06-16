#!/usr/bin/env python3
"""
Mimic_REW_analysis.py  —  Part B: deconvolution, analysis & comparison
======================================================================
Validated mimic of REW's analysis stage.

Run directly (after Mimic_REW_sweep.py):
    python3 Mimic_REW_analysis.py

What it does
------------
1. Loads the standard sweep automatically (sample_data/256k…mono.wav) as the
   signal that was played, and builds the analytical inverse filter from it.
2. Loads the three captured recordings from data/:
       captured_sweep_L.wav / _R.wav / _LR.wav
3. Loads the three REW-exported reference curves from sample_data/:
       rew_analyzed_L.txt / _R.txt / _LR.txt
4. Deconvolves each capture (ESS → gated IR → FFT), applies the UMIK-1
   calibration found in sample_data/ so the mimic is in absolute dBSPL, then
5. Plots three stacked subplots (one per channel), each overlaying the mimic
   response on the REW reference, and writes the figure + a combined CSV to
   data/.

The deconvolution / inverse-filter / smoothing math below is the validated
implementation and is left unchanged.

Requirements:  pip install scipy numpy matplotlib soundfile
"""

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import Mimic_REW_sweep as sweep_io

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
SWEEP_START_HZ       = 1
SWEEP_END_HZ         = 20_000
SAMPLE_RATE          = 48_000

# Paths (shared with the sweep module so they never drift apart)
SAMPLE_DATA_DIR      = sweep_io.SAMPLE_DATA_DIR
DATA_DIR             = sweep_io.DATA_DIR
STANDARD_SWEEP       = sweep_io.STANDARD_SWEEP
CHANNELS             = sweep_io.CHANNELS
CH_PRETTY            = sweep_io.CH_PRETTY

REW_REF_TXT          = {ch: SAMPLE_DATA_DIR / f"rew_analyzed_{ch}.txt" for ch in CHANNELS}
CAPTURE_WAV          = {ch: DATA_DIR / f"captured_sweep_{ch}.wav"      for ch in CHANNELS}

PLOT_FILENAME        = DATA_DIR / "rew_vs_mimic_3channel.png"
CSV_FILENAME         = DATA_DIR / "rew_vs_mimic_3channel.csv"

# Default smoothing applied to BOTH mimic and REW curves for a like-for-like
# overlay. None = raw. (1/6-octave matches the calibration stage.)
DEFAULT_SMOOTHING    = 6

# IR Window settings — matches REW "IR Windows" dialog
IR_REF_TIME_MS       = 0.0
IR_LEFT_WIDTH_MS     = 125.0
IR_RIGHT_WIDTH_MS    = 500.0
IR_LEFT_TUKEY_ALPHA  = 0.25
IR_RIGHT_TUKEY_ALPHA = 0.25

PLOT_FMIN            = 10
PLOT_FMAX            = 20_000

# UMIK-1 absolute SPL calibration constant
UMIK1_BASE_SENSITIVITY = 102   # dB at 0 dB gain

COLOUR_MEASURED = "#2ecc71"
COLOUR_REF      = "#e74c3c"


# ═══════════════════════════════════════════════════════════════════════════
#  UMIK CAL FILE LOOKUP  (peek for "Sens Factor"; skip rig-cal output files)
# ═══════════════════════════════════════════════════════════════════════════

def find_umik_cal_file(directory: str | Path = ".") -> Path | None:
    """Locate a UMIK calibration .txt in `directory`, ignoring rig-cal output."""
    p = Path(directory)
    if not p.is_dir():
        return None
    for c in sorted(p.glob("*.txt")):
        if "_mic_calibration" in c.name:        # skip our rig-cal output
            continue
        try:
            with open(c, encoding="utf-8", errors="replace") as f:
                first = f.readline().strip().strip('"')
            if re.search(r"Sens\s+Factor", first, re.IGNORECASE):
                return c
        except OSError:
            continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  INVERSE FILTER  (built from the standard sweep used during recording)
#  --- validated math, unchanged ---
# ═══════════════════════════════════════════════════════════════════════════

def build_inverse_filter(sweep_wav_path: Path,
                         fs: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Load the reference sweep and compute its analytical inverse filter.

    Returns (sweep, inv_filter) as float64 arrays.
    """
    sweep, sr = sf.read(str(sweep_wav_path), dtype="float32", always_2d=False)
    if sweep.ndim > 1:
        sweep = sweep[:, 0]
    assert sr == fs, f"Sweep sample rate {sr} ≠ expected {fs}"

    peak_amp = np.max(np.abs(sweep)) + 1e-12
    print(f"📁  Loaded sweep: {sweep_wav_path.name}  "
          f"({len(sweep)/fs:.4f} s  peak {20*np.log10(peak_amp):.1f} dBFS)")

    T   = len(sweep) / fs
    t   = np.arange(len(sweep)) / fs
    inv_mod = np.exp(-t * np.log(SWEEP_END_HZ / SWEEP_START_HZ) / T)

    # 1. Normalize a copy of the sweep to an ideal 0 dBFS (peak = 1.0).
    #    This prevents the inverse filter from compensating for quieter sweeps.
    norm_sweep = (sweep / peak_amp).astype(np.float64)

    # 2. Build the inverse filter using the 0 dBFS reference
    inv_filter = norm_sweep[::-1] * inv_mod

    # 3. Normalise so a *theoretical 0 dBFS sweep* ⊛ inv_filter peaks at 1.0
    N_lin = len(norm_sweep) + len(inv_filter) - 1
    N_fft = 1 << int(np.ceil(np.log2(N_lin)))
    test_ir = np.fft.irfft(
        np.fft.rfft(norm_sweep, N_fft) *
        np.fft.rfft(inv_filter, N_fft), N_fft)
    inv_filter /= np.max(np.abs(test_ir)) + 1e-30

    # Return the original, unmodified sweep, plus the fixed-gain inverse filter
    return sweep.astype(np.float64), inv_filter


# ═══════════════════════════════════════════════════════════════════════════
#  ESS DECONVOLUTION  --- validated math, unchanged ---
# ═══════════════════════════════════════════════════════════════════════════

def _build_rew_ir_window(left_samp, right_samp, alpha_l, alpha_r):
    w_left = np.ones(left_samp)
    tl = max(1, int(round(alpha_l * left_samp)))
    w_left[:tl] = 0.5 * (1.0 - np.cos(np.pi * np.arange(tl) / tl))

    w_right = np.ones(right_samp)
    tr = max(1, int(round(alpha_r * right_samp)))
    m = np.arange(tr)
    w_right[right_samp - tr:] = 0.5 * (1.0 + np.cos(np.pi * m / tr))

    return np.concatenate([w_left, [1.0], w_right])


def ess_deconvolve(recording: np.ndarray, inv_filter: np.ndarray,
                   sweep_len: int, fs: int = SAMPLE_RATE):
    """Deconvolve recording with inv_filter → gated IR → FFT magnitude (dB).

    Returns (freqs, mag_db, ir_full, peak_idx).
    """
    N_lin = len(recording) + len(inv_filter) - 1
    N_fft = 1 << int(np.ceil(np.log2(N_lin)))

    Y = np.fft.rfft(recording.astype(np.float64), N_fft)
    H = np.fft.rfft(inv_filter,                   N_fft)
    ir_full = np.fft.irfft(Y * H, N_fft)

    # Locate IR peak
    lo = max(0,            int(sweep_len * 0.50))
    hi = min(len(ir_full), int(sweep_len * 1.50))
    if hi <= lo:
        lo, hi = 0, len(ir_full)
    peak_idx = lo + int(np.argmax(np.abs(ir_full[lo:hi])))

    # REW asymmetric Tukey gate
    ref_idx    = peak_idx + int(round(IR_REF_TIME_MS   / 1000.0 * fs))
    left_samp  = int(round(IR_LEFT_WIDTH_MS  / 1000.0 * fs))
    right_samp = int(round(IR_RIGHT_WIDTH_MS / 1000.0 * fs))

    window    = _build_rew_ir_window(left_samp, right_samp,
                                     IR_LEFT_TUKEY_ALPHA, IR_RIGHT_TUKEY_ALPHA)
    seg_start = ref_idx - left_samp
    seg_end   = ref_idx + right_samp + 1
    ir_gate   = np.zeros(len(window))
    a = max(0, seg_start)
    b = min(len(ir_full), seg_end)
    ir_gate[a - seg_start : b - seg_start] = ir_full[a:b]
    ir_gate *= window

    N_ir_fft = 1 << int(np.ceil(np.log2(max(len(ir_gate), 1) * 4)))
    H_f   = np.fft.rfft(ir_gate, N_ir_fft)
    freqs = np.fft.rfftfreq(N_ir_fft, d=1.0 / fs)
    mag_db = 20.0 * np.log10(np.abs(H_f) + 1e-12)

    return freqs, mag_db, ir_full, peak_idx


def resample_to_log_grid(freqs_lin, mag_db_lin,
                         f_min=PLOT_FMIN, f_max=PLOT_FMAX, n_points=2000):
    log_freqs = np.logspace(
        np.log10(max(f_min, freqs_lin[1])),
        np.log10(min(f_max, freqs_lin[-1])),
        n_points,
    )
    return log_freqs, np.interp(log_freqs, freqs_lin, mag_db_lin)


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION  --- validated math, unchanged ---
# ═══════════════════════════════════════════════════════════════════════════

def load_calibration(cal_file: Path):
    """Parse a UMIK-1/miniDSP calibration file using exact REW anchor math.

    Returns (cal_freqs, cal_db, sensitivity_offset).
    sensitivity_offset is None if the header cannot be parsed.
    """
    sens_factor = None
    again       = None
    rows        = []

    try:
        with open(cal_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip().strip('"')
                if not line:
                    continue
                if sens_factor is None:
                    # FIXED REGEX: Added [+-]? to capture negative values
                    m_s = re.search(r'Sens\s+Factor\s*=\s*([+-]?[\d.]+)\s*dB', line, re.I)
                    m_g = re.search(r'AGain\s*=\s*([+-]?[\d.]+)\s*dB',         line, re.I)
                    if m_s and m_g:
                        sens_factor = float(m_s.group(1))
                        again       = float(m_g.group(1))
                        print(f"📐  Cal header: Sens Factor={sens_factor} dB, "
                              f"AGain={again} dB")
                        continue
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

    if sens_factor is not None:
        # REW-compatible UMIK-1 absolute SPL offset for this code path.
        #
        # REW author formula:
        #     offset = 94 + 24 - SensFactor + 6
        #
        # The +6 dB term depends on dBFS convention and FFT scaling.
        # In this code, white_noise_dbspl() already uses RMS bin power from Welch
        # and defines 0 dBFS using FULL_SCALE_RMS = 1/sqrt(2), i.e.
        # "full-scale sine RMS is 0 dBFS".
        #
        # Therefore one 3.0103 dB sine-RMS convention shift is already accounted for,
        # so we remove it from the REW +6 dB anchor.
        sensitivity_offset = 94.0 + 24.0 - sens_factor + 6.0 - 3.0103
        print(f"📐  Sensitivity offset: {sensitivity_offset:.3f} dB  "
              f"(0 dBFS → {sensitivity_offset:.1f} dBSPL)")
    else:
        sensitivity_offset = None
        print("⚠  Cal header not parsed — freq corrections applied, "
              "no absolute SPL conversion.")

    print(f"📐  Freq corrections: {len(rows)} pts  "
          f"{cal_freqs[0]:.0f}–{cal_freqs[-1]:.0f} Hz  "
          f"(range {cal_db.min():.2f}–{cal_db.max():.2f} dB)")

    return cal_freqs, cal_db, sensitivity_offset



# ═══════════════════════════════════════════════════════════════════════════
#  REW-EXPORTED TXT PARSER  --- validated math, unchanged ---
# ═══════════════════════════════════════════════════════════════════════════

def parse_rew_txt(txt_path: Path) -> dict:
    """Parse a REW-exported SPL text file (Freq(Hz)  SPL(dB), '*' headers).

    Returns dict(name, freqs, spl).
    """
    rows = []
    name = txt_path.stem   # fallback label

    with open(txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("#"):
                if line.lower().startswith("* measurement"):
                    name = line.split(":", 1)[-1].strip()
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue

    if not rows:
        raise ValueError(f"No numeric data found in {txt_path.name}")

    arr   = np.array(rows)
    idx   = np.argsort(arr[:, 0])
    freqs = arr[idx, 0]
    spl   = arr[idx, 1]

    print(f"📄  TXT '{txt_path.name}': '{name}'  "
          f"{freqs[0]:.0f}–{freqs[-1]:.0f} Hz  "
          f"SPL {spl.min():.1f}–{spl.max():.1f} dB  ({len(freqs)} pts)")

    return dict(name=name, freqs=freqs, spl=spl)


# ═══════════════════════════════════════════════════════════════════════════
#  FRACTIONAL-OCTAVE SMOOTHING  --- validated math, unchanged ---
# ═══════════════════════════════════════════════════════════════════════════

def octave_smooth(freqs, mag_db, octave_frac):
    if octave_frac is None or len(freqs) < 3:
        return mag_db.copy()
    ratio    = 2.0 ** (1.0 / octave_frac)
    mag_lin  = 10.0 ** (mag_db / 20.0)
    smoothed = np.empty_like(mag_lin)
    for i, fc in enumerate(freqs):
        mask = (freqs >= fc / ratio) & (freqs <= fc * ratio)
        smoothed[i] = mag_lin[mask].mean() if mask.any() else mag_lin[i]
    return 20.0 * np.log10(smoothed + 1e-12)


# ═══════════════════════════════════════════════════════════════════════════
#  CAPTURE ANALYSIS  (file → mimic SPL curve, reusing the validated pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_capture(capture_path: Path, sweep_path: Path,
                    octave_frac=None, cal=None) -> tuple[np.ndarray, np.ndarray]:
    """Full mimic pipeline for one recorded capture.

    `cal` is the tuple returned by load_calibration() (or None). When present
    the frequency correction is applied and the result is absolute dBSPL.
    Returns (log_freqs, mag_db).
    """
    recording, rec_sr = sf.read(str(capture_path), dtype="float32", always_2d=False)
    if recording.ndim > 1:
        recording = recording[:, 0]
    if rec_sr != SAMPLE_RATE:
        from math import gcd
        g = gcd(SAMPLE_RATE, rec_sr)
        recording = sig.resample_poly(
            recording, SAMPLE_RATE // g, rec_sr // g).astype(np.float32)
        print(f"   Resampled {capture_path.name} {rec_sr}→{SAMPLE_RATE} Hz")

    sweep, inv_filter = build_inverse_filter(sweep_path)
    f_lin, mag_lin, _ir, peak = ess_deconvolve(
        recording, inv_filter, len(sweep), SAMPLE_RATE)
    print(f"   {capture_path.name}: IR peak at "
          f"{peak} ({peak/SAMPLE_RATE*1000:.1f} ms)")

    # Frequency-response correction from UMIK cal (positive = mic over-reads)
    if cal is not None and cal[0] is not None:
        cal_f, cal_db, _ = cal
        mag_lin = mag_lin - np.interp(f_lin, cal_f, cal_db,
                                      left=cal_db[0], right=cal_db[-1])

    meas_f, meas_db = resample_to_log_grid(f_lin, mag_lin)
    if octave_frac is not None:
        meas_db = octave_smooth(meas_f, meas_db, octave_frac)

    # Absolute SPL conversion
    if cal is not None and cal[2] is not None:
        meas_db = meas_db + cal[2]

    return meas_f, meas_db


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def _freq_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(PLOT_FMIN, PLOT_FMAX)
    major = [10, 20, 30, 40, 50, 60, 70, 80, 100,
             200, 300, 400, 500, 600, 700, 800, 1000,
             2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 20000]
    ax.set_xticks(major)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, _: f"{int(x//1000)}k" if x >= 1000 else str(int(x))
        )
    )
    ax.tick_params(axis="x", which="minor", bottom=False)


def plot_three_channels(per_channel: dict, smooth_label: str, out_path: Path):
    """One figure, three stacked SPL-vs-frequency subplots (L, R, LR).

    per_channel[ch] = {meas_f, meas_db, ref_f, ref_db, ref_name}
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 15), sharex=True)
    fig.patch.set_facecolor("#1a1a1a")

    for ax, ch in zip(axes, CHANNELS):
        d = per_channel[ch]
        ax.set_facecolor("#1e1e1e")
        ax.grid(True, which="major", color="#333333", lw=0.8)
        ax.grid(True, which="minor", color="#2a2a2a", lw=0.4, ls=":")

        all_vals = []

        if d.get("ref_f") is not None:
            m = (d["ref_f"] >= PLOT_FMIN) & (d["ref_f"] <= PLOT_FMAX)
            ax.plot(d["ref_f"][m], d["ref_db"][m], color=COLOUR_REF,
                    lw=2.6, alpha=0.45, label=f"REW  ({d['ref_name']})")
            all_vals.extend(d["ref_db"][m][np.isfinite(d["ref_db"][m])])

        if d.get("meas_f") is not None:
            m = (d["meas_f"] >= PLOT_FMIN) & (d["meas_f"] <= PLOT_FMAX)
            ax.plot(d["meas_f"][m], d["meas_db"][m], color=COLOUR_MEASURED,
                    lw=1.8, zorder=5,
                    label=f"Mimic  (smoothing: {smooth_label})")
            all_vals.extend(d["meas_db"][m][np.isfinite(d["meas_db"][m])])

        _freq_axis(ax)
        ax.set_ylabel("SPL (dB)", color="#dddddd", fontsize=13, labelpad=8)
        ax.set_title(f"Channel {ch}  —  {CH_PRETTY[ch]}",
                     color="#ffffff", fontsize=14, pad=8, fontweight="bold")

        if all_vals:
            lo = max(0,   np.percentile(all_vals, 1) - 10)
            hi = min(160, np.percentile(all_vals, 99) + 10)
            ax.set_ylim(lo, hi)
            ax.set_yticks(np.arange(round(lo / 10) * 10,
                                    round(hi / 10) * 10 + 10, 10))

        ax.tick_params(colors="#bbbbbb", labelsize=10)
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.legend(loc="upper left", fontsize=11, facecolor="#2a2a2a",
                  edgecolor="#666666", labelcolor="#eeeeee", framealpha=0.85)

    axes[-1].set_xlabel("Frequency (Hz)", color="#dddddd", fontsize=13, labelpad=8)
    fig.suptitle("Frequency Response — Mimic vs REW (per channel)",
                 color="#ffffff", fontsize=17, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊  Plot saved → {out_path}")


def save_csv(per_channel: dict, out_path: Path):
    """Combined CSV on the mimic frequency grid (L's grid is the reference)."""
    grid = per_channel["L"]["meas_f"]
    if grid is None:
        print("⚠  Nothing to export to CSV.")
        return
    header = ["Frequency_Hz"]
    cols   = [grid]
    for ch in CHANNELS:
        d = per_channel[ch]
        header.append(f"Mimic_{ch}_dB")
        cols.append(np.interp(grid, d["meas_f"], d["meas_db"]))
        if d.get("ref_f") is not None:
            header.append(f"REW_{ch}_dB")
            cols.append(np.interp(grid, d["ref_f"], d["ref_db"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in zip(*cols):
            f.write(",".join("" if not np.isfinite(v) else f"{v:.4f}"
                             for v in row) + "\n")
    print(f"📄  CSV saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("═" * 62)
    print("  Mimic_REW_analysis.py — deconvolution & comparison")
    print("═" * 62)

    if not STANDARD_SWEEP.is_file():
        print(f"❌ Standard sweep not found: {STANDARD_SWEEP}")
        return 1

    octave_frac = DEFAULT_SMOOTHING
    smooth_label = "none" if octave_frac is None else f"1/{octave_frac}"
    print(f"🎛  Smoothing: {smooth_label}")

    # ── Calibration (so the mimic is in absolute dBSPL like REW) ─────────
    umik = find_umik_cal_file(SAMPLE_DATA_DIR)
    cal = None
    if umik is not None:
        print(f"📐  UMIK cal: {umik}")
        cal = load_calibration(umik)
    else:
        print("ℹ  No UMIK cal in sample_data/ — mimic stays in relative dB.")

    # ── Per-channel analysis ──────────────────────────────────────────────
    per_channel = {}
    for ch in CHANNELS:
        print(f"\n── Channel {ch} ({CH_PRETTY[ch]}) {'─'*30}")
        cap = CAPTURE_WAV[ch]
        if not cap.is_file():
            print(f"❌ Missing capture: {cap}  (run Mimic_REW_sweep.py first)")
            return 1
        meas_f, meas_db = analyze_capture(cap, STANDARD_SWEEP, octave_frac, cal)
        print(f"   Mimic range: {meas_db.min():.1f} – {meas_db.max():.1f} dB")

        ref_f = ref_db = None
        ref_name = ch
        ref_path = REW_REF_TXT[ch]
        if ref_path.is_file():
            ref = parse_rew_txt(ref_path)
            ref_f, ref_db, ref_name = ref["freqs"], ref["spl"], ref["name"]
            if octave_frac is not None:
                ref_db = octave_smooth(ref_f, ref_db, octave_frac)
        else:
            print(f"⚠  Missing REW reference: {ref_path}")

        per_channel[ch] = dict(meas_f=meas_f, meas_db=meas_db,
                               ref_f=ref_f, ref_db=ref_db, ref_name=ref_name)

    # ── Plot + CSV ────────────────────────────────────────────────────────
    print("\n🖼  Rendering comparison …")
    plot_three_channels(per_channel, smooth_label, PLOT_FILENAME)
    save_csv(per_channel, CSV_FILENAME)

    print("\n✅ Done.")
    print(f"   Plot : {PLOT_FILENAME}")
    print(f"   CSV  : {CSV_FILENAME}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)