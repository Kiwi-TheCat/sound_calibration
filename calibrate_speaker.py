#!/usr/bin/env python3
"""
calibrate_speaker.py — Rig speaker calibration to a 78 dBSPL target.

Manually executed at each rig with:
        python3 calibrate_speaker.py

Pre-requisites:
  • <rig_id>_mic_calibration.txt produced by calibrate_mic.py must exist in
    the local directory.
  • The rig's mic must be the system default audio input (the script reads
    from sd.default.device[1] — sounddevice docs say the default is
    (input, output), so [1] is the output index and [0] is input; we
    therefore use [0] for the input we listen on).

Workflow:
  1. Locate the rig mic-cal file in CWD, parse rig_id from the filename
  2. Sweep L / R / L+R with the default input → dBSPL per channel (BEFORE)
  3. Compute mean dBSPL for L+R over 400 Hz–10 kHz
  4. If outside 78 ± 0.1 dB, scale the system volume by 10**((78−mean)/20)
     and re-sweep → dBSPL per channel (AFTER); otherwise AFTER = BEFORE
  5. Build a 7-col DataFrame:  freq, L_before, R_before, LR_before,
                                     L_after,  R_after,  LR_after  (all dBSPL)
  6. Save  <rig_id>_speaker_calibration_<YYYY_MM_DD_HH_MM>.parquet
  7. rsync that file to ogma:speaker_calibration/   (RSYNC_DEST below)
  8. Plot the AFTER curves to a sibling .png and open it in the default viewer
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

import Mimic_REW_sweep as rew
import rig_audio as ra

# ── Tunables ─────────────────────────────────────────────────────────────
TARGET_SPL_DB        = 78.0
TARGET_TOLERANCE_DB  = 0.1
TARGET_BAND_HZ       = (400.0, 10_000.0)
RSYNC_DEST           = "ogma:speaker_calibration/"   # adjust per deployment


# ═══════════════════════════════════════════════════════════════════════════
#  RIG CAL FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_rig_mic_cal(directory: str | Path = ".") -> Optional[Path]:
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


# ═══════════════════════════════════════════════════════════════════════════
#  MEASUREMENT WITH PER-CHANNEL OFFSETS
# ═══════════════════════════════════════════════════════════════════════════

def measure_with_rig_cal(input_idx: Optional[int], rig_cal: dict) -> dict:
    """Run L/R/LR sweeps with the default mic, apply each channel's offset.

    `rig_cal` is {'freqs': np.ndarray, 'L': ..., 'R': ..., 'LR': ...}.
    Returns {ch: {'freqs', 'dbfs', 'spl'}} with dBSPL filled in per channel.
    """
    print("🔊 Generating ESS sweep …")
    sweep, inv_filter, playback = rew.generate_ess()

    results = {}
    for ch in ra.CHANNELS:
        print(f"\n   Channel {ch} ({ra.CH_PRETTY[ch]}):")
        f, dbfs, _ = ra.measure_channel(
            ch, sweep, inv_filter, playback,
            input_idx, mic_cal=None, smoothing=ra.SMOOTHING_OCT,
        )
        off_on_f = np.interp(f, rig_cal["freqs"], rig_cal[ch])
        spl = dbfs + off_on_f
        results[ch] = {"freqs": f, "dbfs": dbfs, "spl": spl}
    return results


def band_mean(freqs: np.ndarray, values: np.ndarray,
              band: tuple[float, float]) -> float:
    m = (freqs >= band[0]) & (freqs <= band[1]) & np.isfinite(values)
    return float(np.mean(values[m])) if m.any() else float("nan")


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


def plot_after(results: dict, png_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1e1e1e")
    ax.grid(True, which="major", color="#333", lw=0.7)
    ax.grid(True, which="minor", color="#262626", lw=0.4, ls=":")

    for ch in ra.CHANNELS:
        f = results[ch]["freqs"]; y = results[ch]["spl"]
        ax.plot(f, y, color=ra.CH_COLOUR[ch], lw=1.9,
                label=f"{ch}  ({ra.CH_PRETTY[ch]})")

    # Target line + band shading
    ax.axhline(TARGET_SPL_DB, color="#bbb", lw=1.0, ls="--", alpha=0.7,
               label=f"Target {TARGET_SPL_DB} dBSPL")
    ax.axvspan(TARGET_BAND_HZ[0], TARGET_BAND_HZ[1],
               alpha=0.07, color="#ffffff",
               label=f"Target band {TARGET_BAND_HZ[0]:.0f}–{TARGET_BAND_HZ[1]:.0f} Hz")

    rew._freq_axis(ax)
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
    print("  Rig Speaker Calibration")
    print("═" * 65)

    # ── 1. Find rig-cal file ─────────────────────────────────────────────
    cal_path = find_rig_mic_cal(".")
    if cal_path is None:
        print("\n❌ No *_mic_calibration.txt found in CWD.")
        print("   Run calibrate_mic.py first to produce it.")
        return 1
    rig_id = rig_id_from_path(cal_path)
    print(f"📐 Rig mic cal: {cal_path.name}  (rig_id = {rig_id})")

    freqs_cal, off_L, off_R, off_LR, meta = ra.read_rig_mic_cal(cal_path)
    rig_cal = {"freqs": freqs_cal, "L": off_L, "R": off_R, "LR": off_LR}
    if meta:
        print(f"   Generated: {meta.get('Calibration date', '?')}")
        print(f"   Reference: {meta.get('Reference UMIK cal', '?')}")

    # ── 2. Default input device ──────────────────────────────────────────
    if not ra.HAS_SD:
        print("❌ sounddevice unavailable.")
        return 1
    import sounddevice as sd
    # sd.default.device is (input, output); [0] is input.
    in_idx = sd.default.device[0]
    if in_idx is None or in_idx < 0:
        print("❌ No default input device set. Configure the rig mic as default.")
        return 1
    print(f"🎙  Default input: [{in_idx}] {sd.query_devices(in_idx)['name']}")

    # ── 3. Pass 1 — BEFORE ──────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("Pass 1/?  —  initial measurement")
    print("─" * 65)
    before = measure_with_rig_cal(in_idx, rig_cal)

    lr_mean_before = band_mean(before["LR"]["freqs"], before["LR"]["spl"],
                               TARGET_BAND_HZ)
    print(f"\n📊 L+R mean SPL  ({TARGET_BAND_HZ[0]:.0f}-{TARGET_BAND_HZ[1]:.0f} Hz): "
          f"{lr_mean_before:.3f} dB    (target {TARGET_SPL_DB} ± {TARGET_TOLERANCE_DB})")

    # ── 4. Adjust volume if needed; Pass 2 — AFTER ───────────────────────
    volume_before = ra.get_system_volume()
    volume_after  = volume_before
    after         = before          # default if no adjustment needed

    if abs(lr_mean_before - TARGET_SPL_DB) <= TARGET_TOLERANCE_DB:
        print("✅ Already within tolerance — no volume change.")
    else:
        if volume_before is None:
            print("❌ Could not read system volume — cannot auto-adjust.")
            print("   Set the system volume manually and re-run.")
            return 1
        dV_db = TARGET_SPL_DB - lr_mean_before
        ratio = 10.0 ** (dV_db / 20.0)
        volume_after = volume_before * ratio
        print(f"🎚  Volume: {volume_before:.4f}  ×  10^({dV_db:+.2f}/20)  "
              f"=  {volume_after:.4f}")
        if not ra.set_system_volume(volume_after):
            print("❌ Failed to set system volume.")
            return 1
        # Confirm what the system actually accepted (may be clamped)
        v_now = ra.get_system_volume()
        if v_now is not None:
            print(f"   System volume now reads: {v_now:.4f}")
            volume_after = v_now

        print("\n" + "─" * 65)
        print("Pass 2/2  —  after volume adjustment")
        print("─" * 65)
        after = measure_with_rig_cal(in_idx, rig_cal)
        lr_mean_after = band_mean(after["LR"]["freqs"], after["LR"]["spl"],
                                  TARGET_BAND_HZ)
        err = lr_mean_after - TARGET_SPL_DB
        print(f"\n📊 L+R mean SPL after: {lr_mean_after:.3f} dB  "
              f"(error {err:+.3f} dB)")
        if abs(err) > TARGET_TOLERANCE_DB:
            print(f"   ⚠  Did not converge within ±{TARGET_TOLERANCE_DB} dB. "
                  f"Volume curve may be nonlinear; re-run if needed.")

    # ── 5. Build dataframe (common grid = before["LR"]["freqs"]) ────────
    grid = before["LR"]["freqs"]
    def _on_grid(d, ch): return np.interp(grid, d[ch]["freqs"], d[ch]["spl"])
    df = pd.DataFrame({
        "frequency_Hz":     grid,
        "L_before_dBSPL":   _on_grid(before, "L"),
        "R_before_dBSPL":   _on_grid(before, "R"),
        "LR_before_dBSPL":  _on_grid(before, "LR"),
        "L_after_dBSPL":    _on_grid(after,  "L"),
        "R_after_dBSPL":    _on_grid(after,  "R"),
        "LR_after_dBSPL":   _on_grid(after,  "LR"),
    })

    # ── 6. Save parquet  <rig_id>_speaker_calibration_<YYYY_MM_DD_HH_MM>.parquet ─
    stamp = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
    base = f"{rig_id}_speaker_calibration_{stamp}"
    parquet_path = Path(f"{base}.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Parquet saved → {parquet_path}")

    # ── 7. rsync ─────────────────────────────────────────────────────────
    rsync_to_dest(parquet_path)         # warning printed if it fails; non-fatal

    # ── 8. Plot AFTER + open in default viewer ───────────────────────────
    png_path = Path(f"{base}.png")
    plot_after(after, png_path,
               f"Rig {rig_id} — calibrated speaker response  ({stamp})")
    print(f"📊 Plot → {png_path}")
    ra.open_file_default(png_path)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)