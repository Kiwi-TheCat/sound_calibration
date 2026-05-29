#!/usr/bin/env python3
"""
calibrate_speaker.py — Rig speaker measurement (script 4, no auto-adjust).

Manually executed at each rig with:
        python3 calibrate_speaker.py

Pre-requisites:
  • data/<rig_id>_mic_calibration.txt produced by calibrate_mic.py.
  • The standard sweep in sample_data/256k…mono.wav.

Workflow:
  1. Locate the rig mic-cal file in data/, parse rig_id from the filename
  2. Pick OUTPUT (speakers) + INPUT (rig mic) devices
  3. Sweep L / R / L+R, apply each channel's offset → dBSPL per channel
  4. Report mean dBSPL for L+R over the target band (informational only)
  5. Build a 4-col DataFrame:  freq, L_dBSPL, R_dBSPL, LR_dBSPL
  6. Save  data/<rig_id>_speaker_calibration_<YYYY_MM_DD_HH_MM>.parquet
  7. rsync that file to RSYNC_DEST (non-fatal if it fails)
  8. Plot the SPL curves to a sibling .png and open it in the default viewer
"""

from __future__ import annotations
import datetime as _dt
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sound_calibration_utility as scu

# ── Tunables ─────────────────────────────────────────────────────────────
TARGET_SPL_DB  = 78.0
TARGET_BAND_HZ = (400.0, 10_000.0)
RSYNC_DEST     = "ogma:speaker_calibration/"   # adjust per deployment


# ═══════════════════════════════════════════════════════════════════════════
#  RIG CAL FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_rig_mic_cal(directory: str | Path = scu.DATA_DIR) -> Optional[Path]:
    """Return newest *_mic_calibration.txt in `directory`, or None."""
    candidates = sorted(Path(directory).glob("*_mic_calibration.txt"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"⚠  Multiple *_mic_calibration.txt files found; "
              f"using newest: {candidates[-1].name}")
    return candidates[-1]


def rig_id_from_path(path: Path) -> str:
    """Pull rig_id from filename '<rig_id>_mic_calibration.txt'."""
    stem = path.name
    if not stem.endswith("_mic_calibration.txt"):
        raise ValueError(f"Unexpected cal filename: {path}")
    return stem[:-len("_mic_calibration.txt")]


def band_mean(freqs: np.ndarray, values: np.ndarray,
              band: tuple[float, float]) -> float:
    m = (freqs >= band[0]) & (freqs <= band[1]) & np.isfinite(values)
    return float(np.mean(values[m])) if m.any() else float("nan")


# ═══════════════════════════════════════════════════════════════════════════
#  MEASUREMENT WITH PER-CHANNEL OFFSETS
# ═══════════════════════════════════════════════════════════════════════════

def measure_with_rig_cal(output_idx: Optional[int],
                         input_idx: Optional[int],
                         rig_cal: dict) -> dict:
    """Run L/R/LR sweeps and apply each channel's offset.

    `rig_cal` is {'freqs', 'L', 'R', 'LR'}.
    Returns {ch: {'freqs', 'dbfs', 'spl'}} with dBSPL filled in per channel.
    """
    print(f"🔧 Building inverse filter from {scu.STANDARD_SWEEP.name} …")
    sweep, inv_filter = scu.build_inverse_filter(scu.STANDARD_SWEEP)

    results = {}
    for ch in scu.CHANNELS:
        print(f"\n   Channel {ch} ({scu.CH_PRETTY[ch]}):")
        f, dbfs, _ = scu.measure_channel(
            ch, sweep, inv_filter, output_idx, input_idx,
            mic_cal=None, smoothing=scu.SMOOTHING_OCT,
        )
        off_on_f = np.interp(f, rig_cal["freqs"], rig_cal[ch])
        results[ch] = {"freqs": f, "dbfs": dbfs, "spl": dbfs + off_on_f}

    for ch in results:
        print(f"  {ch} raw dBFS mean ({TARGET_BAND_HZ[0]:.0f}-{TARGET_BAND_HZ[1]:.0f}): "
              f"{band_mean(results[ch]['freqs'], results[ch]['dbfs'], TARGET_BAND_HZ):.2f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  RSYNC + PLOT
# ═══════════════════════════════════════════════════════════════════════════

def rsync_to_dest(local_path: Path, dest: str = RSYNC_DEST) -> bool:
    print(f"\n📤 rsync {local_path.name}  →  {dest}")
    r = subprocess.run(["rsync", "-avz", str(local_path), dest],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"   ⚠  rsync exit {r.returncode}")
        if r.stderr.strip():
            print("   stderr:\n      " + r.stderr.strip().replace("\n", "\n      "))
        return False
    print("   ✅ rsync ok")
    return True


def plot_results(results: dict, png_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, which="major", color="#333", lw=0.7)
    ax.grid(True, which="minor", color="#262626", lw=0.4, ls=":")

    for ch in scu.CHANNELS:
        ax.plot(results[ch]["freqs"], results[ch]["spl"],
                color=scu.CH_COLOUR[ch], lw=1.9,
                label=f"{ch}  ({scu.CH_PRETTY[ch]})")

    ax.axhline(TARGET_SPL_DB, color="#bbb", lw=1.0, ls="--", alpha=0.7,
               label=f"Target {TARGET_SPL_DB} dBSPL")
    ax.axvspan(TARGET_BAND_HZ[0], TARGET_BAND_HZ[1], alpha=0.07, color="#ffffff",
               label=f"Target band {TARGET_BAND_HZ[0]:.0f}–{TARGET_BAND_HZ[1]:.0f} Hz")

    scu.freq_axis(ax)
    ax.set_xlabel("Frequency (Hz)", color="#ddd", fontsize=12)
    ax.set_ylabel("SPL (dB)",       color="#ddd", fontsize=12)
    ax.set_title(title, color="#fff", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#555")
    ax.legend(loc="lower center", ncol=5, fontsize=10,
              facecolor="#2a2a2a", edgecolor="#555", labelcolor="#eee",
              framealpha=0.85)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("═" * 65)
    print("  Rig Speaker Measurement  (no auto-adjust)")
    print("═" * 65)

    # ── 1. Find rig-cal file in data/ ────────────────────────────────────
    cal_path = find_rig_mic_cal(scu.DATA_DIR)
    if cal_path is None:
        print(f"\n❌ No *_mic_calibration.txt found in {scu.DATA_DIR}/.")
        print("   Run calibrate_mic.py first to produce it.")
        return 1
    rig_id = rig_id_from_path(cal_path)
    print(f"📐 Rig mic cal: {cal_path.name}  (rig_id = {rig_id})")

    freqs_cal, off_L, off_R, off_LR, meta = scu.read_rig_mic_cal(cal_path)
    rig_cal = {"freqs": freqs_cal, "L": off_L, "R": off_R, "LR": off_LR}
    if meta:
        print(f"   Generated: {meta.get('Calibration date', '?')}")
        print(f"   Reference: {meta.get('Reference UMIK cal', '?')}")

    # ── 2. Pick output + input devices ───────────────────────────────────
    if not scu.HAS_SD:
        print("❌ sounddevice unavailable.")
        return 1
    scu.list_audio_devices()
    output_idx, output_name = scu.prompt_output_device(
        "Which device drives the SPEAKERS?")
    input_idx, input_name = scu.prompt_input_device(
        "Which device is the RIG mic?")
    print(f"🔊 Output: [{output_idx}] {output_name}")
    print(f"🎙  Input : [{input_idx}] {input_name}")

    # ── 3. Single measurement pass ───────────────────────────────────────
    print("\n" + "─" * 65)
    print("Measurement pass")
    print("─" * 65)
    results = measure_with_rig_cal(output_idx, input_idx, rig_cal)

    lr_mean = band_mean(results["LR"]["freqs"], results["LR"]["spl"], TARGET_BAND_HZ)
    print(f"\n📊 L+R mean SPL ({TARGET_BAND_HZ[0]:.0f}-{TARGET_BAND_HZ[1]:.0f} Hz): "
          f"{lr_mean:.3f} dB    (reference target {TARGET_SPL_DB} dB — not adjusted)")

    # ── 4. Build dataframe (common grid = results["LR"]["freqs"]) ───────
    grid = results["LR"]["freqs"]
    def _on_grid(ch):
        return np.interp(grid, results[ch]["freqs"], results[ch]["spl"])
    df = pd.DataFrame({
        "frequency_Hz": grid,
        "L_dBSPL":      _on_grid("L"),
        "R_dBSPL":      _on_grid("R"),
        "LR_dBSPL":     _on_grid("LR"),
    })

    # ── 5. Save parquet to data/ ─────────────────────────────────────────
    stamp = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
    base = f"{rig_id}_speaker_calibration_{stamp}"
    scu.DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = scu.DATA_DIR / f"{base}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Parquet saved → {parquet_path}")

    # ── 6. rsync (non-fatal) ─────────────────────────────────────────────
    rsync_to_dest(parquet_path)

    # ── 7. Plot + open in default viewer ─────────────────────────────────
    png_path = scu.DATA_DIR / f"{base}.png"
    plot_results(results, png_path,
                 f"Rig {rig_id} — speaker response  ({stamp})")
    print(f"📊 Plot → {png_path}")
    scu.open_file_default(png_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)