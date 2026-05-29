#!/usr/bin/env python3
"""
validation.py — Compare rig speaker-calibration parquet with a REW reference .txt file.

Usage:
    python3 validation.py

The script will:
  1. List all .txt files in CWD and let you select one as the reference
  2. Auto-find the rig calibration .parquet file
  3. Plot both together
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Tunables ──────────────────────────────────────────────────────────────
TARGET_SPL_DB   = 78.0
TARGET_BAND_HZ  = (400.0, 10_000.0)
PLOT_FMIN       = 10
PLOT_FMAX       = 20_000

# Colours for the rig channels
CH_COLOURS = {"L": "#4fc3f7", "R": "#ef9a9a", "LR": "#a5d6a7"}
REF_COLOUR = "#ffd54f"   # amber for reference


# ═══════════════════════════════════════════════════════════════════════════
#  FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_txt_files(directory: Path = Path(".")) -> list[Path]:
    """Return list of .txt files in directory, sorted by name."""
    txt_files = sorted(directory.glob("*.txt"))
    return txt_files


def find_parquet(directory: Path = Path(".")) -> Path:
    """Find the most recent *_speaker_calibration_*.parquet in directory."""
    candidates = sorted(directory.glob("*_speaker_calibration_*.parquet"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No *_speaker_calibration_*.parquet found in CWD")
    return candidates[0]


def select_txt_file(txt_files: list[Path]) -> Path:
    """Prompt user to select a .txt file from a list."""
    if not txt_files:
        raise FileNotFoundError("No .txt files found in CWD")

    print("\n── Available .txt files ────────────────────────────────────")
    for i, f in enumerate(txt_files, 1):
        print(f"  [{i}] {f.name}")
    print("────────────────────────────────────────────────────────────")

    while True:
        try:
            choice = input(f"Select [1–{len(txt_files)}]: ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(txt_files):
                return txt_files[idx]
        except ValueError:
            pass
        print(f"Invalid choice. Enter a number between 1 and {len(txt_files)}.")


# ═══════════════════════════════════════════════════════════════════════════
#  REW TEXT EXPORT PARSER
# ═══════════════════════════════════════════════════════════════════════════

def parse_rew_txt(filepath):
    """Parse a REW text export file (.txt from REW's 'Export Measurement' → Text).

    Expected format:
      * Header lines (prefixed with *)
      * Measurement: <name>
      * ...
      * Freq(Hz) SPL(dB) Phase(degrees)
      <freq>  <spl>  <phase>
      ...

    Returns dict: {name, freqs, spl}
    """
    name = Path(filepath).stem
    freqs_list = []
    spl_list = []

    with open(filepath, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            # ── Extract measurement name from header ──────────────────────
            if line.startswith("* Measurement:"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    name = parts[1].strip().strip("*").strip()
                continue

            # ── Skip header / comment lines ───────────────────────────────
            if line.startswith("*") or not line:
                continue

            # ── Skip the column header line (Freq(Hz) SPL(dB) ...) ────────
            if "Freq(Hz)" in line or "SPL(dB)" in line:
                continue

            # ── Parse numeric data rows ──────────────────────────────────
            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                freq = float(parts[0])
                spl = float(parts[1])

                if freq > 0 and np.isfinite(freq) and np.isfinite(spl):
                    freqs_list.append(freq)
                    spl_list.append(spl)
            except (ValueError, IndexError):
                continue

    if not freqs_list:
        raise ValueError(f"No numeric data found in {filepath}")

    freqs = np.array(freqs_list, dtype=np.float64)
    spl = np.array(spl_list, dtype=np.float64)

    # Sort by frequency
    order = np.argsort(freqs)
    freqs = freqs[order]
    spl = spl[order]

    return {"name": name, "freqs": freqs, "spl": spl}


# ═══════════════════════════════════════════════════════════════════════════
#  PARQUET READER
# ═══════════════════════════════════════════════════════════════════════════

def load_parquet(path: Path) -> dict[str, dict]:
    """Load speaker-calibration parquet; return {channel: {freqs, spl}}."""
    df = pd.read_parquet(path)

    # Support both schema versions from calibrate_speaker.py:
    #   New:  frequency_Hz | L_dBSPL   | R_dBSPL   | LR_dBSPL
    #   Old:  frequency_Hz | L_before_* | R_before_* | LR_before_*
    freq_col = "frequency_Hz"
    if freq_col not in df.columns:
        raise ValueError(f"Expected column '{freq_col}' in {path.name}.")

    channels = {}
    for ch, candidates in {
        "L":  ["L_dBSPL",  "L_before_dBSPL"],
        "R":  ["R_dBSPL",  "R_before_dBSPL"],
        "LR": ["LR_dBSPL", "LR_before_dBSPL"],
    }.items():
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            continue
        channels[ch] = {
            "freqs": df[freq_col].to_numpy(dtype=float),
            "spl":   df[col].to_numpy(dtype=float),
        }

    if not channels:
        raise ValueError(f"Could not find any SPL columns in {path.name}.")

    return channels


# ═══════════════════════════════════════════════════════════════════════════
#  FREQUENCY AXIS SETUP
# ═══════════════════════════════════════════════════════════════════════════

def _freq_axis(ax):
    """Configure log frequency x-axis like REW."""
    ax.set_xscale("log")
    ax.set_xlim(PLOT_FMIN, PLOT_FMAX)
    major = [10, 20, 30, 40, 50, 60, 70, 80, 100,
             200, 300, 400, 500, 600, 700, 800, 1000,
             2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000, 20000]
    ax.set_xticks(major)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x//1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.tick_params(axis="x", which="minor", bottom=False)


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════════════════════════════════

def make_plot(cal_channels: dict, cal_label: str,
              ref_freqs: np.ndarray, ref_spl: np.ndarray, ref_label: str,
              png_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, which="major", color="#333", lw=0.8)
    ax.grid(True, which="minor", color="#262626", lw=0.4, ls=":")

    # ── Rig calibration channels ──────────────────────────────────────────
    ch_labels = {"L": "L  (rig cal)", "R": "R  (rig cal)", "LR": "L+R  (rig cal)"}
    for ch, data in sorted(cal_channels.items()):
        if ch != "L":  # Skip R and LR
            continue
        mask = (data["freqs"] >= PLOT_FMIN) & (data["freqs"] <= PLOT_FMAX)
        ax.plot(data["freqs"][mask], data["spl"][mask],
                color=CH_COLOURS.get(ch, "#999"), lw=1.8, alpha=0.85,
                label=ch_labels.get(ch, ch))

    # ── Reference measurement ─────────────────────────────────────────────
    mask_ref = (ref_freqs >= PLOT_FMIN) & (ref_freqs <= PLOT_FMAX)
    ax.plot(ref_freqs[mask_ref], ref_spl[mask_ref],
            color=REF_COLOUR, lw=2.4, ls="-",
            label=f"{ref_label}  (reference)")

    # ── Reference lines ───────────────────────────────────────────────────
    ax.axhline(TARGET_SPL_DB, color="#888", lw=1.0, ls="--", alpha=0.6,
               label=f"Target {TARGET_SPL_DB} dBSPL")
    ax.axvspan(TARGET_BAND_HZ[0], TARGET_BAND_HZ[1],
               alpha=0.05, color="#ffffff",
               label=f"Target band {TARGET_BAND_HZ[0]:.0f}–{TARGET_BAND_HZ[1]:.0f} Hz")

    # ── Axes, labels, legend ──────────────────────────────────────────────
    _freq_axis(ax)
    ax.set_xlabel("Frequency (Hz)", color="#ddd", fontsize=13)
    ax.set_ylabel("SPL (dB)",       color="#ddd", fontsize=13)
    ax.set_title(f"Validation — {cal_label}  vs  {ref_label}",
                 color="#fff", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="#aaa", labelsize=11)
    for sp in ax.spines.values():
        sp.set_color("#555")

    ax.legend(loc="lower center", ncol=5, fontsize=11,
              facecolor="#2a2a2a", edgecolor="#555", labelcolor="#eee",
              framealpha=0.85)

    fig.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"📊 Plot saved → {png_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  SUMMARY STATS
# ═══════════════════════════════════════════════════════════════════════════

def band_mean(freqs: np.ndarray, spl: np.ndarray,
              band: tuple[float, float]) -> float:
    m = (freqs >= band[0]) & (freqs <= band[1]) & np.isfinite(spl)
    return float(np.mean(spl[m])) if m.any() else float("nan")


def print_summary(cal_channels: dict,
                  ref_freqs: np.ndarray, ref_spl: np.ndarray,
                  ref_label: str) -> None:
    band = TARGET_BAND_HZ
    print(f"\n{'─'*65}")
    print(f"  Band mean SPL  {band[0]:.0f}–{band[1]:.0f} Hz")
    print(f"{'─'*65}")
    for ch in ["L", "R", "LR"]:
        if ch in cal_channels:
            data = cal_channels[ch]
            m = band_mean(data["freqs"], data["spl"], band)
            delta = m - TARGET_SPL_DB
            print(f"  Rig {ch:<4}        {m:7.2f} dB   "
                  f"(target {TARGET_SPL_DB} dB, Δ {delta:+.2f} dB)")
    m_ref = band_mean(ref_freqs, ref_spl, band)
    print(f"  REF ({ref_label[:20]:<20})  {m_ref:7.2f} dB")
    if "LR" in cal_channels:
        lr_mean = band_mean(cal_channels["LR"]["freqs"],
                            cal_channels["LR"]["spl"], band)
        diff = m_ref - lr_mean
        print(f"\n  ΔSPL  (ref − rig L+R):  {diff:+.2f} dB")
    print(f"{'─'*65}\n")

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
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("═" * 65)
    print("  Speaker Calibration Validation")
    print("═" * 65)

    # ── 1. Find and select reference .txt file ───────────────────────────
    try:
        txt_files = find_txt_files()
        ref_path = select_txt_file(txt_files)
        print(f"\n✓ Selected: {ref_path.name}")
    except Exception as e:
        print(f"\n❌ {e}")
        return 1

    # ── 2. Find calibration parquet ──────────────────────────────────────
    try:
        cal_path = find_parquet()
        print(f"✓ Found calibration: {cal_path.name}")
    except Exception as e:
        print(f"❌ {e}")
        return 1

    # ── 3. Load data ─────────────────────────────────────────────────────
    try:
        cal_channels = load_parquet(cal_path)
        ref_data = parse_rew_txt(ref_path)
        ref_freqs = ref_data["freqs"]
        ref_spl = ref_data["spl"]
        ref_label = ref_data["name"]

        ref_spl = octave_smooth(ref_freqs, ref_spl, 6)
        print(f"✓ Reference label: {ref_label}")
        print(f"✓ Data points: {len(ref_freqs)}")
        print(f"✓ Freq range: {ref_freqs[0]:.2f} – {ref_freqs[-1]:.0f} Hz")
    except Exception as e:
        print(f"\n❌ {e}")
        return 1

    # ── 4. Print band-mean summary ───────────────────────────────────────
    print_summary(cal_channels, ref_freqs, ref_spl, ref_label)

    # ── 5. Plot ──────────────────────────────────────────────────────────
    out_name = f"validation_{cal_path.stem}.png"
    png_path = Path(out_name)
    make_plot(
        cal_channels, cal_path.stem,
        ref_freqs, ref_spl, ref_label,
        png_path,
    )

    # Open in default viewer if available
    try:
        import subprocess, platform
        opener = {"Darwin": "open", "Windows": "start"}.get(platform.system(), "xdg-open")
        subprocess.Popen([opener, str(png_path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)