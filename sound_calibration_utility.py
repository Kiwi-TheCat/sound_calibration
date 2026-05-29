"""
sound_calibration_utility.py — shared helpers for the rig calibration scripts
=============================================================================
Common functions used by calibrate_mic.py (script 3) and calibrate_speaker.py
(script 4).  Builds on the two validated modules:

  • Mimic_REW_sweep.py     — device selection + L/R/LR sweep recording
  • Mimic_REW_analysis.py  — ESS deconvolution, smoothing, cal-file parsing

So that the calibration scripts only need to import THIS module, the most
useful pieces of those two modules are re-exported below.

Provides / re-exports:
  • Device listing & selection  (list_audio_devices, prompt_output/input_device)
  • Per-channel + all-channel ESS measurement (live recording → dBFS / dBSPL)
  • Rig mic-calibration .txt read/write
        format:   freq_Hz \t offset_L \t offset_R \t offset_LR
        relation: SPL(f) = dBFS_raw(f) + offset(f)
  • UMIK cal-file lookup + parser, log-frequency plot axis
  • Cross-platform system-volume read/write and default-app file opener
"""

from __future__ import annotations
import datetime as _dt
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

import Mimic_REW_sweep as sweep_io
import Mimic_REW_analysis as analysis

# ─── Re-exports (so scripts 3 & 4 import only this module) ────────────────
HAS_SD               = sweep_io.HAS_SD
SAMPLE_RATE          = sweep_io.SAMPLE_RATE
SAMPLE_DATA_DIR      = sweep_io.SAMPLE_DATA_DIR
DATA_DIR             = sweep_io.DATA_DIR
STANDARD_SWEEP       = sweep_io.STANDARD_SWEEP
CHANNELS             = sweep_io.CHANNELS
CH_PRETTY            = sweep_io.CH_PRETTY

list_audio_devices   = sweep_io.list_audio_devices
prompt_output_device = sweep_io.prompt_output_device
prompt_input_device  = sweep_io.prompt_input_device

build_inverse_filter = analysis.build_inverse_filter
load_calibration     = analysis.load_calibration
find_umik_cal_file   = analysis.find_umik_cal_file
freq_axis            = analysis._freq_axis

# ─── Local constants ──────────────────────────────────────────────────────
CH_COLOUR        = {"L": "#3498db", "R": "#e74c3c", "LR": "#2ecc71"}
SMOOTHING_OCT    = 6                                  # 1/6-octave smoothing
DEFAULT_UMIK_CAL = SAMPLE_DATA_DIR / "7101790.txt"   # UMIK-1 cal file


# ═══════════════════════════════════════════════════════════════════════════
#  MEASUREMENT PIPELINE  (live recording → IR → FFT → log-grid → smoothing)
# ═══════════════════════════════════════════════════════════════════════════

def measure_channel(channel: str,
                    sweep: np.ndarray,
                    inv_filter: np.ndarray,
                    output_idx: Optional[int],
                    input_idx: Optional[int],
                    mic_cal=None,
                    smoothing: Optional[int] = SMOOTHING_OCT
                    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Play `channel`, record, deconvolve → (freqs_log, dbfs_raw, spl_or_None).

    `mic_cal` is the tuple from load_calibration() (or None). When its
    sensitivity_offset is non-None an absolute-SPL array is also returned;
    otherwise only raw dBFS is meaningful.
    """
    rec = sweep_io.record_sweep_on_channel(sweep, channel, output_idx, input_idx)

    f_lin, mag_lin, _ir, _peak = analysis.ess_deconvolve(
        rec, inv_filter, len(sweep), SAMPLE_RATE)

    # Frequency-response correction from UMIK cal (positive = mic over-reads)
    if mic_cal is not None and mic_cal[0] is not None:
        cal_f, cal_db, _ = mic_cal
        mag_lin = mag_lin - np.interp(f_lin, cal_f, cal_db,
                                      left=cal_db[0], right=cal_db[-1])

    freqs, dbfs = analysis.resample_to_log_grid(f_lin, mag_lin)
    if smoothing is not None:
        dbfs = analysis.octave_smooth(freqs, dbfs, smoothing)

    spl = None
    if mic_cal is not None and mic_cal[2] is not None:
        spl = dbfs + mic_cal[2]

    return freqs, dbfs, spl


def measure_all_channels(output_idx: Optional[int],
                         input_idx: Optional[int],
                         mic_cal=None,
                         smoothing: Optional[int] = SMOOTHING_OCT,
                         sweep_path: Path = STANDARD_SWEEP) -> dict:
    """Run L, R, LR sweeps with one output + one input device.

    Returns {ch: {'freqs', 'dbfs', 'spl'}}.
    """
    if not Path(sweep_path).is_file():
        raise SystemExit(f"❌ Standard sweep not found: {sweep_path}")
    print(f"🔧 Building inverse filter from {Path(sweep_path).name} …")
    sweep, inv_filter = build_inverse_filter(Path(sweep_path))

    results = {}
    for ch in CHANNELS:
        print(f"\n   Channel {ch} ({CH_PRETTY[ch]}):")
        f, dbfs, spl = measure_channel(ch, sweep, inv_filter,
                                       output_idx, input_idx, mic_cal, smoothing)
        results[ch] = {"freqs": f, "dbfs": dbfs, "spl": spl}
    return results


# ═══════════════════════════════════════════════════════════════════════════
#  RIG MIC CALIBRATION FILE  (4-column txt: freq | off_L | off_R | off_LR)
# ═══════════════════════════════════════════════════════════════════════════

def write_rig_mic_cal(path: Path,
                      rig_id: str,
                      freqs: np.ndarray,
                      off_L:  np.ndarray,
                      off_R:  np.ndarray,
                      off_LR: np.ndarray,
                      ref_cal_file: str) -> None:
    """Write the per-channel per-frequency calibration for the rig's test mic.

    Relation when applied:  SPL(f) = dBFS_raw(f) + offset(f)
    (the offset absorbs the sweep level + UMIK sensitivity).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    with open(path, "w") as f:
        f.write("# Rig mic calibration file\n")
        f.write(f"# Rig ID: {rig_id}\n")
        f.write(f"# Calibration date: {now}\n")
        f.write(f"# Reference UMIK cal: {ref_cal_file}\n")
        f.write(f"# Smoothing: 1/{SMOOTHING_OCT}-octave\n")
        f.write("# Relation: SPL_dB(f) = dBFS_raw(f) + offset(f)\n")
        f.write("# Columns: frequency_Hz\toffset_L_dB\toffset_R_dB\toffset_LR_dB\n")
        for fr, lL, lR, lLR in zip(freqs, off_L, off_R, off_LR):
            if np.isfinite(lL) and np.isfinite(lR) and np.isfinite(lLR):
                f.write(f"{fr:.4f}\t{lL:.4f}\t{lR:.4f}\t{lLR:.4f}\n")


def read_rig_mic_cal(path: Path):
    """Inverse of write_rig_mic_cal(). Returns (freqs, offL, offR, offLR, meta)."""
    rows: list[list[float]] = []
    meta: dict = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if ":" in line:
                    k, _, v = line.lstrip("#").partition(":")
                    meta[k.strip()] = v.strip()
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    rows.append([float(p) for p in parts[:4]])
                except ValueError:
                    continue
    if not rows:
        raise ValueError(f"No data rows parsed from {path}")
    arr = np.asarray(rows)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], meta


# ═══════════════════════════════════════════════════════════════════════════
#  SYSTEM VOLUME  (Linux: wpctl > pactl;  macOS: osascript)
# ═══════════════════════════════════════════════════════════════════════════
#
# Both getter and setter speak in fractions, where 1.0 = 100% of the system's
# nominal max. Math elsewhere assumes the fraction scales linearly with output
# amplitude, which is exact for wpctl (PipeWire), close on pactl/macOS.

def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def get_system_volume() -> Optional[float]:
    sys_name = platform.system()
    if sys_name == "Darwin":
        r = _run(["osascript", "-e", "output volume of (get volume settings)"])
        try:
            return float(r.stdout.strip()) / 100.0
        except ValueError:
            return None
    if sys_name == "Linux":
        if shutil.which("wpctl"):
            r = _run(["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"])
            m = re.search(r"Volume:\s*([\d.]+)", r.stdout)
            if m:
                return float(m.group(1))
        if shutil.which("pactl"):
            r = _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
            m = re.search(r"(\d+)%", r.stdout)
            if m:
                return int(m.group(1)) / 100.0
    return None


def set_system_volume(fraction: float) -> bool:
    """Set system volume to `fraction` (1.0 = 100%). Returns True on success."""
    sys_name = platform.system()
    fraction = max(0.0, fraction)        # never set negative
    if sys_name == "Darwin":
        pct = max(0, min(100, int(round(fraction * 100))))   # macOS caps at 100
        if fraction > 1.0:
            print(f"   ⚠  Requested {fraction*100:.1f}% but macOS caps at 100%.")
        r = _run(["osascript", "-e", f"set volume output volume {pct}"])
        return r.returncode == 0
    if sys_name == "Linux":
        if shutil.which("wpctl"):
            v = min(fraction, 1.5)       # cap to avoid digital amplification damage
            r = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{v:.4f}"])
            return r.returncode == 0
        if shutil.which("pactl"):
            pct = max(0, min(150, int(round(fraction * 100))))
            r = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"])
            return r.returncode == 0
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  OPEN FILE IN OS DEFAULT VIEWER
# ═══════════════════════════════════════════════════════════════════════════

def open_file_default(path: str | Path) -> None:
    p = str(path)
    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            subprocess.Popen(["open", p])
        elif sys_name == "Linux":
            subprocess.Popen(["xdg-open", p],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys_name == "Windows":
            os.startfile(p)                 # noqa: pyright
    except Exception as e:                  # noqa: BLE001
        print(f"⚠  Could not open {p}: {e}")