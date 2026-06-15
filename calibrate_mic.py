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
  6. White noise L / R / L+R through BOTH mics (reusing calibrate_speaker's
     generator):  WN_offset = UMIK_dBSPL − raw_dBFS  (one broadband scalar
     per channel)
  7. Write  rig_mic_calibration_file/<rig_id>_<timestamp>_mic_calibration.txt
      (per-freq f, off_L, off_R, off_LR  +  three WN_offset header lines).
      A timestamp is embedded so each run adds a NEW file rather than
      overwriting the previous calibration.

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

# White-noise reuse: pull the *exact* generator + playback/level helpers used
# by calibrate_speaker.py so the mic-cal white-noise bursts are identical
# (0.5 amplitude, independent L/R draws, same band-summed dBFS→dBSPL maths).
from calibrate_speaker import (
    play_and_record_white_noise,
    white_noise_dbspl,
    WN_DURATION_S,
    WN_WARMUP_S,
    TARGET_BAND_HZ,
)


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

    # Ensure EVERY mic input is at full level (and unmuted) before measuring,
    # so neither the calibrated nor the uncalibrated rig mic — which usually
    # aren't the system default source — sneaks through muted or attenuated.
    n_set = scu.set_all_input_volumes(1.0)
    if n_set:
        print(f"🔊 Set {n_set} mic input source(s) to 100% and unmuted them.")
    elif scu.set_system_input_volume(1.0):
        # Fallback: default source only (e.g. non-Linux / no pactl source list).
        print("🔊 Set default mic input level to 100% and unmuted it.")
    else:
        print("⚠  Could not set mic input level to 100%.")

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
    print("Phase 1/3 — sweeping through CALIBRATED mic …")
    print("─" * 65)
    ref_results = scu.measure_all_channels(output_idx, cal_idx,
                                           mic_cal=ref_cal,
                                           smoothing=scu.SMOOTHING_OCT)

    # ── 5. Sweep with uncalibrated mic (mic_cal=None → raw dBFS) ─────────
    print("\n" + "─" * 65)
    print("Phase 2/3 — sweeping through UNCALIBRATED mic …")
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

    # ── 7b. White-noise broadband offsets (both mics) ────────────────────
    #     For each channel, play+record white noise on the CALIBRATED mic
    #     (→ absolute dBSPL via the UMIK correction) and on the UNCALIBRATED
    #     mic (→ raw dBFS), then:
    #         WN_offset = UMIK_dBSPL − raw_dBFS
    #     One broadband (energy-summed) scalar per channel, mirroring the
    #     per-frequency relation  SPL = dBFS_raw + offset.
    print("\n" + "─" * 65)
    print("Phase 3/3 — white-noise bursts (calibrated then uncalibrated mic) …")
    print("─" * 65)

    fs = scu.SAMPLE_RATE
    # UMIK correction expressed as a dBFS→dBSPL offset per bin:
    #   dBSPL_bin = dBFS_bin − cal_db(f) + sensitivity   (matches measure_channel)
    cal_f, cal_db, cal_sens = ref_cal
    umik_off_f = np.asarray(cal_f, dtype=float)
    umik_off_v = -np.asarray(cal_db, dtype=float) + cal_sens
    # Zero offset → white_noise_dbspl returns plain band-summed dBFS.
    zero_off_f = np.array([0.0, fs / 2.0])
    zero_off_v = np.array([0.0, 0.0])
    warm = int(WN_WARMUP_S * fs)

    wn_offsets = {}
    for ch in scu.CHANNELS:
        print(f"\n   Channel {ch} ({scu.CH_PRETTY[ch]}):")

        print("     • calibrated mic (→ dBSPL) …")
        rec_cal = play_and_record_white_noise(
            ch, WN_DURATION_S, fs, output_idx, cal_idx)
        if rec_cal.size > warm:
            rec_cal = rec_cal[warm:]
        wn_dbspl = white_noise_dbspl(rec_cal, fs,
                                     umik_off_f, umik_off_v, TARGET_BAND_HZ)

        print("     • uncalibrated mic (→ dBFS) …")
        rec_test = play_and_record_white_noise(
            ch, WN_DURATION_S, fs, output_idx, test_idx)
        if rec_test.size > warm:
            rec_test = rec_test[warm:]
        wn_dbfs = white_noise_dbspl(rec_test, fs,
                                    zero_off_f, zero_off_v, TARGET_BAND_HZ)

        wn_offsets[ch] = wn_dbspl - wn_dbfs
        print(f"     UMIK {wn_dbspl:7.2f} dBSPL  −  raw {wn_dbfs:7.2f} dBFS  "
              f"→  WN offset {wn_offsets[ch]:+.3f} dB")

    # ── 8. Write cal file to rig_mic_calibration_file/ ──────────────────
    #     Timestamped filename so each run ADDS a new file rather than
    #     overwriting the previous calibration. calibrate_speaker.py discovers
    #     these via the '*_mic_calibration.txt' glob and strips the timestamp
    #     when recovering the rig_id.
    out_dir = Path(__file__).resolve().parent / "rig_mic_calibration_file"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_path = out_dir / f"{rig_id}_{stamp}_mic_calibration.txt"
    scu.write_rig_mic_cal(out_path, rig_id, freqs,
                          offsets["L"], offsets["R"], offsets["LR"],
                          ref_cal_file=str(umik_path),
                          wn_offsets=wn_offsets)
    print(f"\n✅ Mic calibration written → {out_path}")
    print(f"   White-noise offsets  L/R/LR: "
          f"{wn_offsets['L']:+.3f} / {wn_offsets['R']:+.3f} / "
          f"{wn_offsets['LR']:+.3f} dB")
    print(f"   Calibration timestamp: {_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)