#!/usr/bin/env python3
"""
calibrate_mic.py — Rig mic calibration (script 3).

Manually executed at each rig with:
        python3 calibrate_mic.py

Setup expected before running:
  • A calibrated USB mic (e.g. UMIK-1) in the inner box in front of the
    pokewall, between the two speakers.
  • The uncalibrated rig mic (beige / camera mic) in its fixed location.
  • The UMIK calibration file in  sample_data/7101790.txt.
  • The standard sweep in  sample_data/256k…mono.wav.

Workflow:
  1. Prompt for rig ID                          (e.g. 373110)
  2. List devices; pick OUTPUT (speakers), CALIBRATED mic, UNCALIBRATED mic
  3. Sweep L / R / L+R through the CALIBRATED mic   → dBSPL curves
  4. Sweep L / R / L+R through the UNCALIBRATED mic → raw dBFS curves
  5. Per channel:  offset(f) = SPL_ref(f) − dBFS_test(f)
  6. Write  rig_mic_calibration_file/<rig_id>_mic_calibration.txt
      (f, off_L, off_R, off_LR)

The offset absorbs the sweep level, UMIK sensitivity, and the test mic's
frequency response, so downstream the user just does:
        SPL(f) = dBFS_raw(f) + offset(f)
"""

from __future__ import annotations
import datetime as _dt
import sys
from pathlib import Path

import numpy as np

import sound_calibration_utility as scu


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

    if not scu.HAS_SD:
        print("❌ sounddevice unavailable.")
        return 1

    # ── 2. Pick output + the two mics ────────────────────────────────────
    scu.list_audio_devices()
    try:
        output_idx, output_name = scu.prompt_output_device(
            "Which device drives the SPEAKERS?")
        cal_idx, cal_name = scu.prompt_input_device(
            "Which device is the CALIBRATED USB mic?")
        test_idx, test_name = scu.prompt_input_device(
            "Which device is the UNCALIBRATED rig mic?")
    except RuntimeError as e:
        print(f"❌ {e}")
        return 1
    if cal_idx == test_idx:
        print("❌ Calibrated and uncalibrated mics cannot be the same device.")
        return 1

    print("\nConfiguration:")
    print(f"   Rig ID            : {rig_id}")
    print(f"   Speakers (output) : [{output_idx}] {output_name}")
    print(f"   Calibrated mic    : [{cal_idx}] {cal_name}")
    print(f"   Uncalibrated mic  : [{test_idx}] {test_name}")

    # ── 3. Locate + load UMIK calibration file ───────────────────────────
    umik_path = (scu.DEFAULT_UMIK_CAL if scu.DEFAULT_UMIK_CAL.is_file()
                 else scu.find_umik_cal_file(scu.SAMPLE_DATA_DIR))
    if umik_path is None:
        print(f"\n❌ No UMIK .txt cal file found in {scu.SAMPLE_DATA_DIR}/.")
        print("   (Expected something like 7101790.txt with a "
              "'Sens Factor =…' first line.)")
        return 1
    print(f"\n📐 UMIK cal file: {umik_path}")
    ref_cal = scu.load_calibration(umik_path)
    if ref_cal[2] is None:
        print("❌ Could not parse Sens Factor / AGain header — aborting.")
        return 1

    # ── 4. Sweep with calibrated mic (mic_cal applied → dBSPL) ───────────
    print("\n" + "─" * 65)
    print("Phase 1/2 — sweeping through CALIBRATED mic …")
    print("─" * 65)
    ref_results = scu.measure_all_channels(output_idx, cal_idx,
                                           mic_cal=ref_cal,
                                           smoothing=scu.SMOOTHING_OCT)

    # ── 5. Sweep with uncalibrated mic (mic_cal=None → raw dBFS) ─────────
    print("\n" + "─" * 65)
    print("Phase 2/2 — sweeping through UNCALIBRATED mic …")
    print("─" * 65)
    test_results = scu.measure_all_channels(output_idx, test_idx,
                                            mic_cal=None,
                                            smoothing=scu.SMOOTHING_OCT)

    # ── 6. Sanity check: same frequency grid ─────────────────────────────
    freqs = ref_results["LR"]["freqs"]
    for ch in scu.CHANNELS:
        if not np.allclose(ref_results[ch]["freqs"], freqs):
            print("❌ Frequency grids differ between channels — internal error.")
            return 1
        if not np.allclose(test_results[ch]["freqs"], freqs):
            print("❌ Frequency grids differ between mics — internal error.")
            return 1

    # ── 7. Per-channel per-frequency offsets ─────────────────────────────
    #     offset(f) = SPL_ref(f) − dBFS_test_raw(f)
    print("\n" + "─" * 65)
    print("Computing per-channel sensitivity offsets …")
    print("─" * 65)
    offsets = {}
    band = (freqs >= 250) & (freqs <= 4000)
    for ch in scu.CHANNELS:
        off = ref_results[ch]["spl"] - test_results[ch]["dbfs"]
        offsets[ch] = off
        print(f"   {ch:2s}: mean offset over 250–4000 Hz = "
              f"{float(off[band].mean()):+.3f} dB")

    spread = (max(float(offsets[ch][band].mean()) for ch in scu.CHANNELS) -
              min(float(offsets[ch][band].mean()) for ch in scu.CHANNELS))
    print(f"   Cross-channel spread (flat band): {spread:.2f} dB")
    if spread > 3.0:
        print("   ⚠  Large spread — check both mics are firmly mounted and "
              "the room was quiet during the sweeps.")

    # ── 8. Write cal file to rig_mic_calibration_file/ ──────────────────
    out_dir = Path(__file__).resolve().parent / "rig_mic_calibration_file"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rig_id}_mic_calibration.txt"
    scu.write_rig_mic_cal(out_path, rig_id, freqs,
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