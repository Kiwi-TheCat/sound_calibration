#!/usr/bin/env python3
"""
calibrate_speaker.py — Rig speaker measurement + system-volume calibration.

Manually executed once at each rig with:
        python3 calibrate_speaker.py

After launch the script asks for the audio devices **once**, runs a full
calibration cycle immediately, then stays resident and re-runs that cycle
automatically at 02:00 every day (no further user input required).

Pre-requisites:
  • rig_calibration_file/<rig_id>_mic_calibration.txt produced by calibrate_mic.py.
  • The standard sweep in sample_data/256k…mono.wav.
  • PulseAudio / PipeWire `pactl` available (Ubuntu) for volume control.
  • ~/.dbconf with a [client] section (host/user/passwd/port) for DB logging.

One calibration cycle:
  A) Sweep measurement with per-channel offsets (existing code, adapted from calibrate_mic.py):
     1. Build inverse filter, sweep L / R / L+R, apply per-channel offsets.
     2. Save data/<rig_id>_speaker_calibration_<stamp>.parquet, rsync, plot.
  B) White-noise volume calibration:
     1. Play + record white noise on L+R, L-only and R-only.
     2. Measure absolute dBSPL of each.
     3. Test #1  — stereo (L+R) must be 78 dBSPL ± 1 dB        → pass / fail
        L/R delta  — record (L − R) in dB; no pass/fail check
     4. Insert (rigid, stereo_78db_check, delta) into the DB.
     5. If failed: nudge the system volume so stereo lands on 78 dBSPL and
        repeat the tests. Up to 2 volume adjustments, then stop.
"""

from __future__ import annotations
import datetime as _dt
import re
import socket
import subprocess
import sys
import time
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
TARGET_BAND_HZ = (1.0, 24_000.0)
RSYNC_DEST     = "ogma:speaker_calibration/"   # adjust per deployment

# Directory holding the rig mic-calibration .txt files (written by
# calibrate_mic.py).  Lives in the project root alongside this script.
RIG_CAL_DIR    = Path(__file__).resolve().parent / "rig_mic_calibration_file"

# White-noise volume calibration
WN_DURATION_S      = 7.0     # length of each white-noise burst (seconds)
WN_WARMUP_S        = 1     # discard this much at the start of each recording
SPL_TOLERANCE_DB   = 1.0     # stereo level must be within ±this of 78 dBSPL
MAX_VOLUME_ADJUST  = 2       # number of volume nudges before giving up
VOLUME_CENTER_TOL_DB = 1   # leave stereo this close to 78 dBSPL (final volume)
# RMS of a full-scale (peak = 1.0) sine → defined as 0 dBFS.  This single
# constant fixes the dBFS→dBSPL convention; if a calibrated reference SLM
# reads a constant offset vs this script on first deployment, adjust here.
FULL_SCALE_RMS     = 1.0 / np.sqrt(2.0)

# Empirical absolute-anchor trim (dB) added to every white-noise SPL reading.
# The WN_offset values in the cal file are built from the UMIK measurement, so
# any fixed dBFS→dBSPL convention/sensitivity mismatch between the UMIK
# sensitivity and white_noise_dbspl's full-scale-sine reference shows up as a
# constant, channel-independent offset vs a calibrated SLM.  Measured value:
# set this to (SLM_dBSPL − script_dBSPL), averaged over channels.
#   e.g. SLM L/R/LR = 79.7/78.8/82.4, script = 75.68/74.94/78.40
#        → mean offset ≈ +3.96 dB
# Re-verify at a second volume to confirm it is level-independent; set back to
# 0.0 if the root cause is fixed in load_calibration's sensitivity handling.
#WN_SPL_TRIM_DB     = 3.96

# Daily auto-run
DAILY_RUN_HOUR     = 2       # 02:00 local time
DAILY_RUN_MINUTE   = 0

AMPLITUDE_RATIO_TO_MATLAB = 0.5  # MATLAB WhiteNoise_fm is ±0.5 peak, so matche that.
# ═══════════════════════════════════════════════════════════════════════════
#  RIG CAL FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_rig_mic_cal(directory: str | Path = RIG_CAL_DIR) -> Optional[Path]:
    """Return newest *_mic_calibration.txt in `directory`, or None."""
    candidates = sorted(Path(directory).glob("*_mic_calibration.txt"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"⚠  Multiple *_mic_calibration.txt files found; "
              f"using newest: {candidates[-1].name}")
    return candidates[-1]


def list_rig_mic_cals(directory: str | Path = RIG_CAL_DIR) -> list[Path]:
    """Return all *_mic_calibration.txt files in `directory`, newest first."""
    return sorted(Path(directory).glob("*_mic_calibration.txt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def prompt_rig_mic_cal(directory: str | Path = RIG_CAL_DIR) -> Optional[Path]:
    """Show the available rig mic-calibration files and let the user pick one.

    Always lists every *_mic_calibration.txt in `directory` (newest first)
    with an index — even when only one file is present — and reads a
    selection from stdin.  Pressing Enter takes the default (index 0, the
    newest file).  Returns the chosen Path, or None if the directory holds no
    calibration files.
    """
    candidates = list_rig_mic_cals(directory)
    if not candidates:
        return None

    print(f"\nRig mic-calibration files in {Path(directory)}/:")
    for i, p in enumerate(candidates):
        stamp = _dt.datetime.fromtimestamp(p.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M")
        marker = "  (newest)" if i == 0 else ""
        print(f"  [{i}] {p.name}   ({stamp}){marker}")

    while True:
        sel = input(f"Select rig mic cal file [0-{len(candidates) - 1}] "
                    f"(Enter = 0): ").strip()
        if sel == "":
            return candidates[0]
        try:
            idx = int(sel)
        except ValueError:
            print("  ⚠  Please enter a number.")
            continue
        if 0 <= idx < len(candidates):
            return candidates[idx]
        print(f"  ⚠  Out of range; pick 0-{len(candidates) - 1}.")


def rig_id_from_path(path: Path) -> str:
    """Pull rig_id from filename '<rig_id>[_<stamp>]_mic_calibration.txt'.

    calibrate_mic.py now writes non-overwriting, timestamped files named
    '<rig_id>_YYYY_MM_DD_HH_MM_SS_mic_calibration.txt', so an optional
    trailing timestamp is stripped here.  Legacy '<rig_id>_mic_calibration.txt'
    files (no stamp) are unaffected.
    """
    stem = path.name
    if not stem.endswith("_mic_calibration.txt"):
        raise ValueError(f"Unexpected cal filename: {path}")
    core = stem[:-len("_mic_calibration.txt")]
    # Strip a trailing YYYY_MM_DD_HH_MM[_SS] timestamp if present.
    core = re.sub(r"_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}(?:_\d{2})?$", "", core)
    return core


def band_mean(freqs: np.ndarray, values: np.ndarray,
              band: tuple[float, float]) -> float:
    m = (freqs >= band[0]) & (freqs <= band[1]) & np.isfinite(values)
    return float(np.mean(values[m])) if m.any() else float("nan")


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM IDENTITY  (rigid)
# ═══════════════════════════════════════════════════════════════════════════

def get_rig_id_from_system() -> Optional[int]:
    """Derive the numeric rigid from the hostname.

    The rigs are named like 'delab-373110' (prompt: delab@delab-373110:~/),
    so the rigid is the digit run in the hostname.
    """
    host = socket.gethostname()
    m = re.search(r"(\d{3,})", host)
    if not m:
        print(f"⚠  Could not parse rigid from hostname '{host}'.")
        return None
    return int(m.group(1))


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM VOLUME CONTROL  (Ubuntu / pactl)
# ═══════════════════════════════════════════════════════════════════════════

def get_system_volume_percent() -> Optional[float]:
    """Read the default sink volume as a percentage (e.g. 50.0)."""
    try:
        r = subprocess.run(
            r"pactl list sinks | awk '/^\t*Volume:/ {print $5; exit}'",
            shell=True, capture_output=True, text=True)
        token = r.stdout.strip()          # e.g. '50%'
        return float(token.rstrip("%")) if token else None
    except Exception as e:
        print(f"⚠  Could not read system volume %: {e}")
        return None


def get_system_volume_db() -> Optional[float]:
    """Read the default sink volume in dB (gain shown by pactl)."""
    try:
        r = subprocess.run(["pactl", "list", "sinks"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.startswith("Volume:"):
                # ...front-left: 32768 /  50% / -18.06 dB,   front-right: ...
                m = re.search(r"/\s*(-?\d+(?:\.\d+)?|-?inf)\s*dB", s)
                if m:
                    tok = m.group(1)
                    return float("-inf") if "inf" in tok else float(tok)
        return None
    except Exception as e:
        print(f"⚠  Could not read system volume (dB): {e}")
        return None


def set_system_volume_percent(percent: float) -> bool:
    """Set the default sink volume to an absolute percentage."""
    percent = max(0.0, percent)
    r = subprocess.run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent:.0f}%"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"⚠  set-sink-volume {percent:.0f}% failed: {r.stderr.strip()}")
        return False
    return True


def set_system_volume_db(db: float) -> bool:
    """Set the default sink volume to an absolute gain in dB.

    A gain change of N dB on the output moves the measured SPL by N dB, so
    dB is the correct (linear) domain for compensation — PulseAudio's % scale
    is not linear in dB.  PulseAudio clamps to its allowed range.
    """
    r = subprocess.run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{db:.2f}dB"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"⚠  set-sink-volume {db:.2f}dB failed: {r.stderr.strip()}")
        return False
    return True


def adjust_volume_for_target(measured_stereo_spl: float,
                             target_spl: float = TARGET_SPL_DB) -> bool:
    """Nudge the system volume (in %) so the next stereo measurement hits target.

    Uses the percent form of the sink-volume command:
        pactl set-sink-volume @DEFAULT_SINK@ <p>%
    delta = target − measured is the dB of SPL we still need to add.  PulseAudio's
    % volume is cubic (dB = 60·log10(%)), so adding `delta` dB means scaling the
    current % by 10**(delta/60); that converges toward `target_spl`.
    """
    if not np.isfinite(measured_stereo_spl):
        print("⚠  Stereo SPL not finite; skipping volume adjustment.")
        return False
    delta = target_spl - measured_stereo_spl
    cur_pct = get_system_volume_percent()
    if cur_pct is None:
        print("⚠  Current volume % unknown; skipping adjustment.")
        return False
    new_pct = cur_pct * (10.0 ** (delta / 60.0))
    new_pct = max(0.0, min(new_pct, 153.0))   # PulseAudio caps near 153 %
    print(f"🔧 Adjusting volume: measured {measured_stereo_spl:.2f} dBSPL, "
          f"target {target_spl:.1f} → {delta:+.2f} dB")
    print(f"   current: {cur_pct:.0f}%  →  new: {new_pct:.0f}%")
    return set_system_volume_percent(new_pct)


# ═══════════════════════════════════════════════════════════════════════════
#  WHITE NOISE  (generate / play+record / analyse)
# ═══════════════════════════════════════════════════════════════════════════

def _white_noise_fs() -> int:
    """Prefer a sample rate exported by scu, else fall back to 48 kHz."""
    for attr in ("FS", "SAMPLE_RATE", "SR", "DEFAULT_FS", "SAMPLERATE"):
        val = getattr(scu, attr, None)
        if val:
            return int(val)
    return 48_000


def generate_white_noise(duration_s: float, fs: int) -> np.ndarray:
    """Uniform white noise in [-1, 1], length round(duration_s * fs)."""
    n = int(round(duration_s * fs))
    # Scale amplitude to match MATLAB WhiteNoise_fm audio amplitude
    return AMPLITUDE_RATIO_TO_MATLAB * (np.random.rand(n) * 2 - 1).astype(np.float64)

"""Uniform white noise in [-1, 1], length round(duration_s * fs).

    Scaled to 0.5 RMS (≈ −10 dBFS) to leave headroom, which matches MATLAB's WhiteNoise_fm.
    """
def play_and_record_white_noise(channel: str, duration_s: float, fs: int,
                                output_idx: Optional[int],
                                input_idx: Optional[int]) -> np.ndarray:
    """Play white noise on `channel` (L / R / LR) and record the rig mic.

    Returns the mono recording as a 1-D float64 array.
    """
    import sounddevice as sd
    n = int(round(duration_s * fs))
    play = np.zeros((n, 2), dtype=np.float64)

    # Two INDEPENDENT white-noise draws — one per channel — so the L and R
    # feeds are uncorrelated.  For stereo (LR) the speakers radiate
    # independent signals whose powers add at the mic (≈ +3 dB over a single
    # channel for matched levels), so stereo measures louder than either
    # channel alone, and it mirrors real two-channel playback rather than a
    # single coherent mono feed driven into both channels.
    if channel in ("L", "LR"):
        play[:, 0] = generate_white_noise(duration_s, fs)
    if channel in ("R", "LR"):
        play[:, 1] = generate_white_noise(duration_s, fs)

    rec = sd.playrec(play, samplerate=fs, channels=1,
                     device=(input_idx, output_idx), dtype="float32")
    sd.wait()
    return np.asarray(rec, dtype=np.float64).reshape(-1)


def white_noise_dbspl(recording: np.ndarray, fs: int,
                      off_freqs: np.ndarray, off_vals: np.ndarray,
                      band: tuple[float, float] = TARGET_BAND_HZ) -> float:
    """Absolute band-limited SPL (dB) of a recorded white-noise burst.

    Per-bin recorded level (dBFS) + per-frequency mic offset (dBFS→dBSPL),
    energy-summed across `band` to give the total level a SLM would read.
    """
    from scipy.signal import welch
    recording = np.asarray(recording, dtype=np.float64).reshape(-1)
    if recording.size < 64:
        return float("nan")

    nperseg = int(min(recording.size, 8192))
    f, pxx = welch(recording, fs=fs, nperseg=nperseg, scaling="density")
    df = (f[1] - f[0]) if f.size > 1 else (fs / nperseg)

    rms_bin = np.sqrt(np.maximum(pxx * df, 0.0))           # RMS per bin
    with np.errstate(divide="ignore"):
        dbfs_bin = 20.0 * np.log10(rms_bin / FULL_SCALE_RMS + 1e-30)
    off = np.interp(f, off_freqs, off_vals)                # dBFS → dBSPL
    spl_bin = dbfs_bin + off

    sel = (f >= band[0]) & (f <= band[1]) & np.isfinite(spl_bin)
    if not sel.any():
        return float("nan")
    return float(10.0 * np.log10(np.sum(10.0 ** (spl_bin[sel] / 10.0))))


def measure_white_noise_spl(channel: str, rig_cal: dict,
                            output_idx: Optional[int],
                            input_idx: Optional[int],
                            fs: int, duration_s: float = WN_DURATION_S) -> float:
    """Play+record white noise on `channel` and return its absolute dBSPL.

    Preferred path (when the cal file carries broadband WN offsets):
        SPL = raw_dBFS_welch(rec) + WN_offset[channel]
    Both terms use the same Welch/full-scale-sine convention that
    calibrate_mic.py used to derive WN_offset, so the result is self-consistent
    for white noise (including the incoherent L+R addition on the LR channel)
    and matches an SLM/MATLAB ground-truth reading.

    Fallback path (legacy cal files with no WN offsets):
        apply the per-frequency sweep offsets per FFT bin.  This is anchored to
        the ESS-deconvolution's 0 dB, NOT the Welch convention, so it generally
        reads a near-constant offset low for white noise and mis-scales LR.
    """
    rec = play_and_record_white_noise(channel, duration_s, fs,
                                      output_idx, input_idx)
    warm = int(WN_WARMUP_S * fs)
    if rec.size > warm:
        rec = rec[warm:]

    wn = rig_cal.get("wn")
    if wn is not None and np.isfinite(wn.get(channel, np.nan)):
        # Broadband dBFS over the same band WN_offset was measured on, then
        # add the per-channel offset (volume-invariant dBFS→dBSPL conversion).
        zero_f = np.array([0.0, fs / 2.0])
        zero_v = np.array([0.0, 0.0])
        dbfs = white_noise_dbspl(rec, fs, zero_f, zero_v, TARGET_BAND_HZ)
        # Temporarily disable the global trim adjustment
        return dbfs + wn[channel]  # + WN_SPL_TRIM_DB

    # Legacy fallback: per-frequency sweep offsets (convention-mismatched).
    # Legacy fallback: per-frequency sweep offsets (convention-mismatched).
    # Temporarily disable the global trim adjustment
    return white_noise_dbspl(rec, fs, rig_cal["freqs"], rig_cal[channel],
                             TARGET_BAND_HZ)  # + WN_SPL_TRIM_DB


# ═══════════════════════════════════════════════════════════════════════════
#  PASS / FAIL CHECKS
# ═══════════════════════════════════════════════════════════════════════════

def check_stereo_78db(stereo_spl: float,
                      target: float = TARGET_SPL_DB,
                      tol: float = SPL_TOLERANCE_DB) -> bool:
    """True if stereo (L+R) white-noise level is within ±tol of target."""
    return bool(np.isfinite(stereo_spl) and abs(stereo_spl - target) <= tol)


def lr_delta_db(l_spl: float, r_spl: float) -> float:
    """Signed L/R level difference in dB (L − R); NaN if either is non-finite.

    Recorded as-is — there is no longer a pass/fail tolerance on L/R balance.
    """
    if not (np.isfinite(l_spl) and np.isfinite(r_spl)):
        return float("nan")
    return float(l_spl - r_spl)


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def insert_calibration_result(rigid: Optional[int],
                              stereo_78db_check: bool,
                              lr_balance_delta: float,
                              volume: Optional[float]) -> bool:
    """Insert one test outcome into test.sound_calibration. Non-fatal.

    `lr_balance_delta` is the signed L/R level difference (L − R) in dB; a
    non-finite value is stored as SQL NULL.  `volume` is the absolute system
    volume (%) the round was measured at; None is stored as SQL NULL.
    """
    if rigid is None:
        print("⚠  No rigid available; skipping DB insert.")
        return False
    try:
        import configparser
        from os.path import expanduser
        import pymysql

        config = configparser.ConfigParser()
        config.read(expanduser("~/.dbconf"))
        cfg = config["client"]

        con = pymysql.connect(
            host=cfg["host"],
            user=cfg["user"],
            password=cfg["passwd"],
            database="test",
            port=int(cfg.get("port", 3306)),
        )
        cur = con.cursor()
        sql = """
        INSERT INTO sound_calibration (
            rigid,
            stereo_78db_check,
            lr_balance_delta,
            volume
        )
        VALUES (%s, %s, %s, %s)
        """
        delta_val = float(lr_balance_delta) if np.isfinite(lr_balance_delta) else None
        vol_val = float(volume) if (volume is not None and np.isfinite(volume)) else None
        cur.execute(sql, (int(rigid),
                          bool(stereo_78db_check),
                          delta_val,
                          vol_val))
        con.commit()
        rowid = cur.lastrowid
        cur.close()
        con.close()
        delta_str = f"{delta_val:+.2f}" if delta_val is not None else "NULL"
        vol_str = f"{vol_val:.0f}%" if vol_val is not None else "NULL"
        print(f"   🗄  DB insert ok (row {rowid}): "
              f"rigid={rigid} stereo={stereo_78db_check} "
              f"delta={delta_str} volume={vol_str}")
        return True
    except Exception as e:
        print(f"   ⚠  DB insert failed (non-fatal): {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  VOLUME CALIBRATION  (white-noise tests + adjust/retry)
# ═══════════════════════════════════════════════════════════════════════════

def run_volume_calibration(rigid: Optional[int], rig_cal: dict,
                           output_idx: Optional[int], input_idx: Optional[int],
                           fs: int) -> bool:
    """Run the stereo white-noise test and leave the final volume at 78 dBSPL.

    Each round inserts (rigid, stereo_78db_check, delta) into the DB, where
    stereo pass = within ±SPL_TOLERANCE_DB of 78 and `delta` is the signed
    L/R level difference (L − R) in dB — recorded as data, with no pass/fail
    on it.  The system volume is then trimmed until stereo white-noise
    playback sits on 78 dBSPL (within VOLUME_CENTER_TOL_DB).  Up to
    MAX_VOLUME_ADJUST nudges are used; after that the closest achieved level
    is kept.  The L/R delta is logged but does not drive the adjustment —
    master volume scales both channels and cannot correct an imbalance.
    Returns True if the final round's stereo level passed.
    """
    print("\n" + "─" * 65)
    print("White-noise volume calibration")
    print("─" * 65)

    adjustments = 0
    while True:
        print(f"\n▶ Test round (adjustments so far: {adjustments})")
        # Absolute system volume these measurements are taken at (logged to DB).
        cur_volume = get_system_volume_percent()
        print("\n Measuring: Stereo (L+R)")
        stereo_spl = measure_white_noise_spl("LR", rig_cal,
                                            output_idx, input_idx, fs)

        print("\n Measuring: Left channel")
        l_spl = measure_white_noise_spl("L", rig_cal,
                                        output_idx, input_idx, fs)

        print("\n Measuring: Right channel")
        r_spl = measure_white_noise_spl("R", rig_cal,
                                        output_idx, input_idx, fs)

        stereo_pass = check_stereo_78db(stereo_spl)
        lr_delta = lr_delta_db(l_spl, r_spl)

        print(f"   Stereo (L+R) : {stereo_spl:6.2f} dBSPL   "
              f"(target {TARGET_SPL_DB:.0f} ±{SPL_TOLERANCE_DB:.0f})  → "
              f"{'PASS ✅' if stereo_pass else 'FAIL ❌'}")
        print(f"   L            : {l_spl:6.2f} dBSPL")
        print(f"   R            : {r_spl:6.2f} dBSPL")
        print(f"   delta (L − R): {lr_delta:+6.2f} dB        "
              f"(recorded to DB; no pass/fail)")

        # Log this round's outcome (stereo pass + signed L/R delta).
        insert_calibration_result(rigid, stereo_pass, lr_delta, cur_volume)

        if not np.isfinite(stereo_spl):
            print("⚠  Stereo SPL not finite; cannot set volume to target.")
            return False

        # Goal: leave the system volume so stereo playback == 78 dBSPL.
        if abs(stereo_spl - TARGET_SPL_DB) <= VOLUME_CENTER_TOL_DB:
            print(f"\n✅ Final volume set: stereo at {stereo_spl:.2f} dBSPL "
                  f"(target {TARGET_SPL_DB:.0f}).")
            print("ℹ  L/R delta is recorded only; master volume scales both "
                  "channels and cannot correct an imbalance (check routing / "
                  "per-speaker levels if it is large).")
            return stereo_pass

        if adjustments >= MAX_VOLUME_ADJUST:
            print(f"\n⏹  Stereo at {stereo_spl:.2f} dBSPL after "
                  f"{MAX_VOLUME_ADJUST} adjustment(s); closest to "
                  f"{TARGET_SPL_DB:.0f} achievable for now.")
            return stereo_pass

        # Trim volume so the next stereo measurement lands on 78 dBSPL.
        if adjust_volume_for_target(stereo_spl, TARGET_SPL_DB):
            adjustments += 1
            time.sleep(1.0)   # let the sink settle before re-measuring
        else:
            print("⚠  Volume adjustment did not apply; stopping retries.")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  SWEEP MEASUREMENT WITH PER-CHANNEL OFFSETS
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
#  ONE FULL CALIBRATION CYCLE
# ═══════════════════════════════════════════════════════════════════════════

def run_calibration_cycle(rig_id: str, rigid: Optional[int], rig_cal: dict,
                          output_idx: Optional[int], input_idx: Optional[int],
                          fs: int, interactive: bool = False) -> None:
    """Sweep measurement (+parquet/rsync/plot) then white-noise volume cal."""
    stamp = _dt.datetime.now().strftime("%Y_%m_%d_%H_%M")
    print("\n" + "═" * 65)
    print(f"  Calibration cycle @ {stamp}   (rig {rig_id}, rigid {rigid})")
    print("═" * 65)

    # ── A. Sweep measurement pass ────────────────────────────────────────
    print("\n" + "─" * 65)
    print("Sweep measurement pass")
    print("─" * 65)
    results = measure_with_rig_cal(output_idx, input_idx, rig_cal)

    lr_mean = band_mean(results["LR"]["freqs"], results["LR"]["spl"], TARGET_BAND_HZ)
    print(f"\n📊 L+R mean SPL ({TARGET_BAND_HZ[0]:.0f}-{TARGET_BAND_HZ[1]:.0f} Hz): "
          f"{lr_mean:.3f} dB    (reference target {TARGET_SPL_DB} dB — not adjusted)")

    # Build dataframe (common grid = results["LR"]["freqs"]).
    grid = results["LR"]["freqs"]
    def _on_grid(ch):
        return np.interp(grid, results[ch]["freqs"], results[ch]["spl"])
    df = pd.DataFrame({
        "frequency_Hz": grid,
        "L_dBSPL":      _on_grid("L"),
        "R_dBSPL":      _on_grid("R"),
        "LR_dBSPL":     _on_grid("LR"),
    })

    base = f"{rig_id}_speaker_calibration_{stamp}"
    scu.DATA_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = scu.DATA_DIR / f"{base}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"\n💾 Parquet saved → {parquet_path}")

    rsync_to_dest(parquet_path)

    png_path = scu.DATA_DIR / f"{base}.png"
    plot_results(results, png_path,
                 f"Rig {rig_id} — speaker response  ({stamp})")
    print(f"📊 Plot → {png_path}")
    if interactive:
        scu.open_file_default(png_path)   # only pop the viewer for a live run

    # ── B. White-noise volume calibration ────────────────────────────────
    run_volume_calibration(rigid, rig_cal, output_idx, input_idx, fs)


# ═══════════════════════════════════════════════════════════════════════════
#  SCHEDULING
# ═══════════════════════════════════════════════════════════════════════════

def seconds_until_daily(hour: int, minute: int) -> tuple[float, _dt.datetime]:
    """Seconds until the next `hour:minute`, plus that datetime."""
    now = _dt.datetime.now()
    nxt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if nxt <= now:
        nxt += _dt.timedelta(days=1)
    return (nxt - now).total_seconds(), nxt


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("═" * 65)
    print("  Rig Speaker Measurement + Volume Calibration")
    print("═" * 65)

    # ── 1. Select rig-cal file from rig_calibration_file/ ────────────────
    cal_path = prompt_rig_mic_cal(RIG_CAL_DIR)
    if cal_path is None:
        print(f"\n❌ No *_mic_calibration.txt found in {RIG_CAL_DIR}/.")
        print("   Run calibrate_mic.py first to produce it.")
        return 1
    rig_id = rig_id_from_path(cal_path)
    print(f"📐 Rig mic cal: {cal_path.name}  (rig_id = {rig_id})")

    freqs_cal, off_L, off_R, off_LR, meta = scu.read_rig_mic_cal(cal_path)
    rig_cal = {"freqs": freqs_cal, "L": off_L, "R": off_R, "LR": off_LR}

    # Broadband white-noise offsets (WN_offset_L/R/LR), if the cal file has
    # them. These give an SLM-accurate white-noise level via
    #   SPL = raw_dBFS_welch + WN_offset[ch]
    # and are preferred over the per-frequency sweep offsets, which are
    # anchored to a different (ESS-deconvolution) dBFS convention.
    wn = {}
    for ch in scu.CHANNELS:
        v = meta.get(f"WN_offset_{ch}")
        if v is not None:
            try:
                wn[ch] = float(v)
            except ValueError:
                pass
    rig_cal["wn"] = wn if wn else None

    if meta:
        print(f"   Generated: {meta.get('Calibration date', '?')}")
        print(f"   Reference: {meta.get('Reference UMIK cal', '?')}")
    if rig_cal["wn"]:
        print("   White-noise level path: broadband WN_offset "
              f"(L/R/LR = {wn.get('L', float('nan')):+.2f}/"
              f"{wn.get('R', float('nan')):+.2f}/"
              f"{wn.get('LR', float('nan')):+.2f} dB)")
    else:
        print("   ⚠  White-noise level path: legacy per-frequency offsets "
              "(no WN_offset_* in cal file — re-run calibrate_mic.py to add "
              "them; white-noise SPL may read low).")

    # rigid (for DB logging) comes from the system hostname.
    rigid = get_rig_id_from_system()
    print(f"🆔 rigid (from system): {rigid}")

    # ── 2. Pick output + input devices (ONCE) ────────────────────────────
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

    fs = _white_noise_fs()
    print(f"🎚  White-noise sample rate: {fs} Hz")

    # ── 3. Initial calibration cycle (interactive) ───────────────────────
    run_calibration_cycle(rig_id, rigid, rig_cal,
                          output_idx, input_idx, fs, interactive=True)

    # ── 4. Stay resident; re-run daily at 02:00 ──────────────────────────
    print("\n" + "═" * 65)
    print(f"  Idle. Next automatic calibration at "
          f"{DAILY_RUN_HOUR:02d}:{DAILY_RUN_MINUTE:02d} daily.  (Ctrl-C to quit)")
    print("═" * 65)
    while True:
        secs, nxt = seconds_until_daily(DAILY_RUN_HOUR, DAILY_RUN_MINUTE)
        print(f"\n⏳ Sleeping {secs / 3600:.2f} h until "
              f"{nxt:%Y-%m-%d %H:%M} …")
        time.sleep(secs)
        try:
            run_calibration_cycle(rig_id, rigid, rig_cal,
                                  output_idx, input_idx, fs, interactive=False)
        except Exception as e:
            # A bad nightly run must not kill the resident process.
            print(f"⚠  Calibration cycle errored (continuing): {e}")
        time.sleep(60)   # ensure we don't re-trigger within the same minute


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)