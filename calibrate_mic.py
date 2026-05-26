#!/usr/bin/env python3
"""
calibrate_mic.py — Rig mic calibration.

Manually executed at each rig with:
        python3 calibrate_mic.py

Setup expected before running:
  • A calibrated USB mic (e.g. UMIK-1) placed in the inner box in front of
    the pokewall, between the two speakers.
  • The uncalibrated rig mic (Beige mic / camera mic) in its fixed location.
  • The UMIK calibration file (e.g. 7101790.txt) present in the local
    directory next to this script.

Workflow:
  1. Prompt for rig ID         (e.g. 373110)
  2. List input devices, prompt for which is the CALIBRATED mic
  3. Prompt for which is the UNCALIBRATED mic
  4. Sweep L / R / L+R through the CALIBRATED mic → dBSPL curves
  5. Sweep L / R / L+R through the UNCALIBRATED mic → raw dBFS curves
  6. For each channel:  offset(f) = SPL_ref(f) − dBFS_test(f)
  7. Write  <rig_id>_mic_calibration.txt   (4 columns:  f, off_L, off_R, off_LR)

The offset already absorbs SWEEP_LEVEL_DBFS, UMIK sensitivity, and the test
mic's frequency response — so the user of the file just does:
        SPL(f)  =  dBFS_raw(f) + offset(f)
"""

from __future__ import annotations
import datetime as _dt
import sys
from pathlib import Path

import numpy as np

import Mimic_REW_sweep as rew
import rig_audio as ra


def main() -> int:
    print("═" * 65)
    print("  Rig Mic Calibration")
    print("═" * 65)

    # ── 1. Rig ID ────────────────────────────────────────────────────────
    rig_id = input("\nEnter rig ID (e.g. 373110): ").strip()
    if not rig_id:
        print("❌ Rig ID is required.")
        return 1
    if not rig_id.replace("_", "").isalnum():
        print("❌ Rig ID must be alphanumeric (underscores allowed).")
        return 1

    # ── 2 & 3. Pick calibrated + uncalibrated mics ──────────────────────
    try:
        cal_idx, cal_name = ra.prompt_input_device(
            "Which device is the CALIBRATED USB mic?")
        test_idx, test_name = ra.prompt_input_device(
            "Which device is the UNCALIBRATED rig mic?")
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    if cal_idx == test_idx:
        print("❌ Calibrated and uncalibrated mics cannot be the same device.")
        return 1

    print("\nConfiguration:")
    print(f"   Rig ID            : {rig_id}")
    print(f"   Calibrated mic    : [{cal_idx}] {cal_name}")
    print(f"   Uncalibrated mic  : [{test_idx}] {test_name}")

    # ── 4. Locate UMIK calibration file ──────────────────────────────────
    umik_path = ra.find_umik_cal_file(".")
    if umik_path is None:
        print("\n❌ No UMIK .txt cal file found in the current directory.")
        print("   (Expected something like 7101790.txt with a "
              "'Sens Factor =...' first line.)")
        return 1
    print(f"\n📐 UMIK cal file: {umik_path}")
    ref_cal = rew.load_calibration(str(umik_path))
    if ref_cal[2] is None:
        print("❌ Could not parse Sens Factor / AGain header — aborting.")
        return 1

    # ── 5. Sweep with calibrated mic ─────────────────────────────────────
    print("\n" + "─" * 65)
    print("Phase 1/2 — sweeping through CALIBRATED mic …")
    print("─" * 65)
    ref_results = ra.measure_all_channels(cal_idx, mic_cal=ref_cal,
                                          smoothing=ra.SMOOTHING_OCT)

    # ── 6. Sweep with uncalibrated mic (no mic_cal — we want raw dBFS) ──
    print("\n" + "─" * 65)
    print("Phase 2/2 — sweeping through UNCALIBRATED mic …")
    print("─" * 65)
    test_results = ra.measure_all_channels(test_idx, mic_cal=None,
                                           smoothing=ra.SMOOTHING_OCT)

    # ── 7. Sanity check: same frequency grid ─────────────────────────────
    freqs = ref_results["LR"]["freqs"]
    for ch in ra.CHANNELS:
        if not np.allclose(ref_results[ch]["freqs"], freqs):
            print(f"❌ Frequency grids differ between channels — internal error.")
            return 1
        if not np.allclose(test_results[ch]["freqs"], freqs):
            print(f"❌ Frequency grids differ between mics — internal error.")
            return 1

    # ── 8. Compute per-channel per-frequency offsets ─────────────────────
    #     offset(f) = SPL_ref(f) − dBFS_test_raw(f)
    # The test sweep was run with mic_cal=None, so its `dbfs` is the raw
    # transfer-function magnitude (no sweep level, no sensitivity). The
    # offset therefore absorbs both, and applying it later gives absolute SPL.
    print("\n" + "─" * 65)
    print("Computing per-channel sensitivity offsets …")
    print("─" * 65)
    offsets = {}
    band = (freqs >= 250) & (freqs <= 4000)
    for ch in ra.CHANNELS:
        off = ref_results[ch]["spl"] - test_results[ch]["dbfs"]
        offsets[ch] = off
        mean_off = float(off[band].mean())
        print(f"   {ch:2s}: mean offset over 250–4000 Hz = {mean_off:+.3f} dB")

    # Cross-channel sanity (should be similar — speakers are at similar level)
    spread = max(float(offsets[ch][band].mean()) for ch in ra.CHANNELS) - \
             min(float(offsets[ch][band].mean()) for ch in ra.CHANNELS)
    print(f"   Cross-channel spread (flat band): {spread:.2f} dB")
    if spread > 3.0:
        print("   ⚠  Large spread — check both mics are firmly mounted and "
              "the room was quiet during the sweeps.")

    # ── 9. Write cal file ────────────────────────────────────────────────
    out_path = Path(f"{rig_id}_mic_calibration.txt")
    ra.write_rig_mic_cal(out_path, rig_id, freqs,
                         offsets["L"], offsets["R"], offsets["LR"],
                         ref_cal_file=str(umik_path))
    print(f"\n✅ Mic calibration written → {out_path}")
    print(f"   Calibration timestamp: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)