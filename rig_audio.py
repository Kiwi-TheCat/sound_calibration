"""
Shared helpers used by calibrate_mic.py and calibrate_speaker.py.

Provides:
  • Audio device listing & interactive selection
  • Stereo playback routing for L / R / L+R sweeps
  • Per-channel ESS measurement pipeline (wraps Mimic_REW_sweep)
  • Rig mic-calibration .txt read/write
        format: freq_Hz \t offset_L \t offset_R \t offset_LR
        relation: SPL(f) = dBFS_raw(f) + offset(f)
  • Cross-platform system-volume read/write (Linux: wpctl→pactl, macOS: osascript)
  • Default-app file opener
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

import Mimic_REW_sweep as rew

# Importing sounddevice can fail on headless systems without PortAudio.
try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    sd = None        # type: ignore
    HAS_SD = False


CHANNELS  = ["L", "R", "LR"]
CH_PRETTY = {"L": "Left only", "R": "Right only", "LR": "Both (L+R)"}
CH_COLOUR = {"L": "#3498db", "R": "#e74c3c", "LR": "#2ecc71"}

SMOOTHING_OCT = 6   # 1/6-octave smoothing, per spec


# ═══════════════════════════════════════════════════════════════════════════
#  DEVICE LISTING / SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def list_input_devices() -> list[tuple[int, str]]:
    """Return [(sounddevice_index, name), ...] for devices with input channels."""
    if not HAS_SD:
        raise RuntimeError("sounddevice unavailable. "
                           "Install: pip install sounddevice "
                           "(and on Linux: apt-get install libportaudio2)")
    devs = sd.query_devices()
    return [(i, d['name']) for i, d in enumerate(devs)
            if d['max_input_channels'] > 0]


def prompt_input_device(prompt_text: str) -> tuple[int, str]:
    """Print numbered list of input devices, ask user to pick one. Returns (sd_idx, name)."""
    inputs = list_input_devices()
    if not inputs:
        raise RuntimeError("No audio input devices found.")
    print("\n── Available input devices ─────────────────────────────")
    for n, (idx, name) in enumerate(inputs, 1):
        print(f"  [{n}] {name}")
    print("─────────────────────────────────────────────────────────")
    while True:
        s = input(f"{prompt_text} (1-{len(inputs)}): ").strip()
        if s.isdigit() and 1 <= int(s) <= len(inputs):
            sel = inputs[int(s) - 1]
            print(f"   → selected: {sel[1]}")
            return sel
        print("   Invalid choice, try again.")


def _default_output_index() -> Optional[int]:
    """Default output device index, or None if not set.

    sounddevice.default.device is documented as (input, output).
    """
    if not HAS_SD:
        return None
    try:
        return sd.default.device[1]
    except (IndexError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  STEREO PLAYBACK / RECORDING
# ═══════════════════════════════════════════════════════════════════════════

def record_sweep_on_channel(playback_mono: np.ndarray,
                            channel: str,
                            input_idx: Optional[int],
                            fs: int = rew.SAMPLE_RATE) -> np.ndarray:
    """Play `playback_mono` on `channel` ∈ {L,R,LR}, record mono from `input_idx`.

    If input_idx is None, the system default input is used.
    """
    if channel not in ("L", "R", "LR"):
        raise ValueError(f"channel must be L, R or LR (got {channel!r})")
    if not HAS_SD:
        raise RuntimeError("sounddevice unavailable")

    stereo = np.zeros((len(playback_mono), 2), dtype=np.float32)
    if channel in ("L", "LR"):
        stereo[:, 0] = playback_mono
    if channel in ("R", "LR"):
        stereo[:, 1] = playback_mono

    out_idx = _default_output_index()
    device = (input_idx, out_idx)
    in_name  = sd.query_devices(input_idx)['name'] if input_idx is not None else "(default)"
    out_name = sd.query_devices(out_idx)['name']   if out_idx   is not None else "(default)"
    print(f"   ▶ {channel:2s}  in={in_name}  out={out_name}")

    # ── Buffered playback/record using sd.Stream (avoids ALSA underrun on Linux) ──
    # blocksize: Chunk size for processing. Larger (4096) reduces underrun but increases latency.
    # latency: 'high' adds buffering for robustness on resource-constrained systems.
    blocksize = 4096
    rec_buffer = np.zeros((len(stereo), 1), dtype=np.float32)
    playback_idx = [0]
    rec_idx = [0]

    def callback(indata, outdata, frames, time, status):
        """Stream callback: fill output, capture input."""
        if status:
            print(f"   ⚠  Stream status: {status}")
        
        # Output: fill with stereo signal
        end_idx = min(playback_idx[0] + frames, len(stereo))
        nframes = end_idx - playback_idx[0]
        outdata[:nframes, :] = stereo[playback_idx[0]:end_idx, :]
        if nframes < frames:
            outdata[nframes:, :] = 0
        playback_idx[0] = end_idx
        
        # Input: capture mono signal
        end_rec = min(rec_idx[0] + frames, len(rec_buffer))
        nrec = end_rec - rec_idx[0]
        rec_buffer[rec_idx[0]:end_rec, 0] = indata[:nrec, 0]
        rec_idx[0] = end_rec

    try:
        with sd.Stream(samplerate=fs, channels=(1, 2), dtype="float32",
                       device=device, blocksize=blocksize, latency="high",
                       callback=callback):
            # Keep stream alive until all samples are played/recorded
            while playback_idx[0] < len(stereo) or rec_idx[0] < len(stereo):
                sd.sleep(int(blocksize / fs * 1000))  # Sleep for one block duration
    except Exception as e:
        print(f"   ⚠  Stream error: {e}")
        raise

    rec = rec_buffer[:rec_idx[0], 0]
    
    # Safety check: ensure we captured audio
    if len(rec) == 0:
        print("   ⚠  No audio captured. Check device connections and levels.")
        return np.zeros(len(stereo), dtype=np.float32)
    
    peak = 20 * np.log10(np.max(np.abs(rec)) + 1e-12)
    print(f"        peak {peak:+.1f} dBFS")
    if peak > -1.0:
        print("        ⚠  near clip — reduce mic gain or sweep level")
    return rec
    

# ═══════════════════════════════════════════════════════════════════════════
#  MEASUREMENT PIPELINE  (ESS → IR → FFT → log-grid → smoothing)
# ═══════════════════════════════════════════════════════════════════════════

def measure_channel(channel: str,
                    sweep: np.ndarray,
                    inv_filter: np.ndarray,
                    playback: np.ndarray,
                    input_idx: Optional[int],
                    mic_cal=None,
                    smoothing: Optional[int] = SMOOTHING_OCT
                    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Play `channel`, record, deconvolve → (freqs_log, dbfs_raw, spl_or_None).

    Parameters
    ----------
    mic_cal : tuple as returned by rew.load_calibration(), or None
        When given AND its sensitivity_offset is non-None, an absolute-SPL
        array is also returned. When None, only raw dBFS is returned.
    """
    rec = record_sweep_on_channel(playback, channel, input_idx)

    pre = int(rew.SILENCE_PRE_S * rew.SAMPLE_RATE)
    rec_a = rec[pre:] if len(rec) > pre else rec

    f_lin, mag_lin, _ir, _peak = rew.ess_deconvolve(
        rec_a, inv_filter, len(sweep), rew.SAMPLE_RATE
    )

    # Frequency-response correction from UMIK cal file (positive = mic over-reads)
    if mic_cal is not None and mic_cal[0] is not None:
        cal_f, cal_db, _ = mic_cal
        mag_lin = mag_lin - np.interp(f_lin, cal_f, cal_db,
                                      left=cal_db[0], right=cal_db[-1])

    freqs, dbfs = rew.resample_to_log_grid(f_lin, mag_lin)
    if smoothing is not None:
        dbfs = rew.octave_smooth(freqs, dbfs, smoothing)

    spl = None
    if mic_cal is not None and mic_cal[2] is not None:
        spl = dbfs + rew.SWEEP_LEVEL_DBFS + mic_cal[2]

    return freqs, dbfs, spl


def measure_all_channels(input_idx: Optional[int],
                         mic_cal=None,
                         smoothing: Optional[int] = SMOOTHING_OCT) -> dict:
    """Run L, R, LR sweeps. Returns {ch: {'freqs','dbfs','spl'}}."""
    print("🔊 Generating ESS sweep …")
    sweep, inv_filter, playback = rew.generate_ess()
    results = {}
    for ch in CHANNELS:
        print(f"\n   Channel {ch} ({CH_PRETTY[ch]}):")
        f, dbfs, spl = measure_channel(ch, sweep, inv_filter, playback,
                                       input_idx, mic_cal, smoothing)
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

    Relation when applied:
            SPL(f) = dBFS_raw(f) + offset(f)
    where dBFS_raw is the deconvolved transfer-function magnitude (NOT including
    SWEEP_LEVEL_DBFS — the offset already absorbs it).
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


def read_rig_mic_cal(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Inverse of write_rig_mic_cal(). Returns (freqs, offL, offR, offLR, metadata)."""
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
#  UMIK CAL FILE LOOKUP  (skip rig-cal txt files, peek for "Sens Factor" header)
# ═══════════════════════════════════════════════════════════════════════════

def find_umik_cal_file(directory: str | Path = ".") -> Optional[Path]:
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
            # Output: "Volume: 0.50" (optionally "[MUTED]")
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
        pct = int(round(fraction * 100))
        pct = max(0, min(100, pct))      # macOS caps at 100
        if fraction > 1.0:
            print(f"   ⚠  Requested {fraction*100:.1f}% but macOS caps at 100%.")
        r = _run(["osascript", "-e", f"set volume output volume {pct}"])
        return r.returncode == 0
    if sys_name == "Linux":
        if shutil.which("wpctl"):
            # wpctl accepts a fraction directly (linear amplitude).
            # Cap at 1.5 to avoid digital amplification damage.
            v = min(fraction, 1.5)
            r = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{v:.4f}"])
            return r.returncode == 0
        if shutil.which("pactl"):
            pct = int(round(fraction * 100))
            pct = max(0, min(150, pct))
            r = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"])
            return r.returncode == 0
    return False


# ═══════════════════════════════════════════════════════════════════════════
#  OPEN FILE IN OS DEFAULT VIEWER  (so tester can inspect the PNG)
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