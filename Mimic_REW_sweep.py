#!/usr/bin/env python3
"""
Mimic_REW_sweep.py  —  Part A: multi-channel sweep playback & recording
=======================================================================
Validated mimic of REW's measurement-sweep recording stage.

Run directly:
    python3 Mimic_REW_sweep.py

What it does
------------
1. Loads the standard measurement sweep automatically from
       sample_data/256kMeasSweep_0_to_20000_-12_dBFS_48k_Float_mono.wav
2. Lists every audio device and asks for an OUTPUT (speaker) index and an
   INPUT (mic) index.
3. Plays the sweep three times — Left only, Right only, then both (L+R) —
   recording the microphone each time.
4. Writes three recordings to data/:
       captured_sweep_L.wav
       captured_sweep_R.wav
       captured_sweep_LR.wav

This module also exposes the device-selection and channel-recording helpers
(`prompt_output_device`, `prompt_input_device`, `record_sweep_on_channel`)
that the analysis and calibration stages build on.

Requirements:  pip install sounddevice soundfile numpy
Linux audio :  sudo apt-get install libportaudio2 portaudio19-dev
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Importing sounddevice can fail on headless systems without PortAudio.
try:
    import sounddevice as sd
    HAS_SD = True
except (ImportError, OSError):
    sd = None        # type: ignore
    HAS_SD = False


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  (single source of truth for paths used across the workflow)
# ═══════════════════════════════════════════════════════════════════════════
SAMPLE_RATE     = 48_000
SAMPLE_DATA_DIR = Path("sample_data")
DATA_DIR        = Path("data")
STANDARD_SWEEP  = SAMPLE_DATA_DIR / "256kMeasSweep_0_to_20000_-12_dBFS_48k_Float_mono.wav"

CHANNELS  = ["L", "R", "LR"]
CH_PRETTY = {"L": "Left only", "R": "Right only", "LR": "Both (L+R)"}


# ═══════════════════════════════════════════════════════════════════════════
#  WAV LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32. Returns (samples, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data[:, 0]
    peak_db = 20 * np.log10(np.max(np.abs(data)) + 1e-12)
    print(f"📂  Loaded {Path(path).name}  "
          f"({len(data)/sr:.4f} s @ {sr} Hz, peak {peak_db:.1f} dBFS)")
    return data, sr


# ═══════════════════════════════════════════════════════════════════════════
#  DEVICE LISTING / SELECTION  (raw sounddevice indices)
# ═══════════════════════════════════════════════════════════════════════════

def list_audio_devices() -> list:
    """Print every audio device with its sounddevice index and channel counts."""
    if not HAS_SD:
        raise RuntimeError("sounddevice unavailable. Install: pip install sounddevice "
                           "(and on Linux: apt-get install libportaudio2)")
    devices = sd.query_devices()
    try:
        default_in, default_out = sd.default.device
    except (TypeError, ValueError):
        default_in = default_out = None

    print("\n── Audio devices ──────────────────────────────────────────")
    for i, d in enumerate(devices):
        tag = ""
        if i == default_in:
            tag += " [default in]"
        if i == default_out:
            tag += " [default out]"
        print(f"  [{i:2d}] {d['name'][:42]:42s} "
              f"in:{d['max_input_channels']} out:{d['max_output_channels']}{tag}")
    print("───────────────────────────────────────────────────────────")
    return devices


def _prompt_device_index(prompt_text: str, capability: str) -> tuple[int, str]:
    """Ask the user for a device index that has the required capability.

    `capability` ∈ {'input', 'output'}.  Returns (index, name).
    """
    if not HAS_SD:
        raise RuntimeError("sounddevice unavailable")
    devices = sd.query_devices()
    key = "max_input_channels" if capability == "input" else "max_output_channels"
    while True:
        raw = input(f"{prompt_text} — enter device index: ").strip()
        if raw.isdigit():
            i = int(raw)
            if 0 <= i < len(devices) and devices[i][key] > 0:
                name = devices[i]["name"]
                print(f"   → selected [{i}] {name}")
                return i, name
            print(f"   Device {raw} has no {capability} channels (or is out of range).")
        else:
            print("   Please enter a numeric device index.")


def prompt_output_device(prompt_text: str = "Select OUTPUT (speaker) device"
                         ) -> tuple[int, str]:
    return _prompt_device_index(prompt_text, "output")


def prompt_input_device(prompt_text: str = "Select INPUT (mic) device"
                        ) -> tuple[int, str]:
    return _prompt_device_index(prompt_text, "input")


# ═══════════════════════════════════════════════════════════════════════════
#  PER-CHANNEL PLAYBACK / RECORDING
# ═══════════════════════════════════════════════════════════════════════════

def record_sweep_on_channel(playback_mono: np.ndarray,
                            channel: str,
                            output_idx: int | None,
                            input_idx: int | None,
                            fs: int = SAMPLE_RATE) -> np.ndarray:
    """Play `playback_mono` on `channel` ∈ {L, R, LR} through `output_idx`
    while recording mono from `input_idx`.  Returns the recorded float32 array.

    `None` for either index falls back to the system default device.
    """
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {CHANNELS} (got {channel!r})")
    if not HAS_SD:
        raise RuntimeError("sounddevice unavailable")

    stereo = np.zeros((len(playback_mono), 2), dtype=np.float32)
    if channel in ("L", "LR"):
        stereo[:, 0] = playback_mono
    if channel in ("R", "LR"):
        stereo[:, 1] = playback_mono

    in_name  = sd.query_devices(input_idx)["name"]  if input_idx  is not None else "(default)"
    out_name = sd.query_devices(output_idx)["name"] if output_idx is not None else "(default)"
    print(f"   ▶ {channel:2s}  out=[{output_idx}] {out_name}  in=[{input_idx}] {in_name}")

    # sounddevice.playrec device order is (input, output).
    rec = sd.playrec(stereo, samplerate=fs, channels=1, dtype="float32",
                     device=(input_idx, output_idx))
    sd.wait()
    rec = rec[:, 0]

    peak = 20 * np.log10(np.max(np.abs(rec)) + 1e-12)
    print(f"        captured peak {peak:+.1f} dBFS")
    if peak > -1.0:
        print("        ⚠  near clipping — reduce mic gain or output level")
    return rec


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("═" * 62)
    print("  Mimic_REW_sweep.py — multi-channel sweep recording")
    print("═" * 62)

    if not HAS_SD:
        print("❌ sounddevice unavailable — cannot record live audio.")
        return 1
    if not STANDARD_SWEEP.is_file():
        print(f"❌ Standard sweep not found: {STANDARD_SWEEP}")
        print("   Place the 48 kHz mono sweep there and re-run.")
        return 1

    # ── 1. Load the standard sweep automatically ─────────────────────────
    sweep, sr = load_wav(STANDARD_SWEEP)
    if sr != SAMPLE_RATE:
        print(f"❌ Sweep is {sr} Hz; expected {SAMPLE_RATE} Hz.")
        return 1

    # ── 2. Pick output + input devices ───────────────────────────────────
    list_audio_devices()
    output_idx, _ = prompt_output_device()
    input_idx,  _ = prompt_input_device()

    # ── 3. Sweep L / R / L+R, saving each capture ────────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "─" * 62)
    for ch in CHANNELS:
        print(f"\n🔊 Sweep {ch} ({CH_PRETTY[ch]}) …")
        rec = record_sweep_on_channel(sweep, ch, output_idx, input_idx)
        out_path = DATA_DIR / f"captured_sweep_{ch}.wav"
        sf.write(str(out_path), rec, SAMPLE_RATE, subtype="PCM_24")
        print(f"   💾 saved → {out_path}")

    print("\n✅ Recording complete.")
    print("   Next step: python3 Mimic_REW_analysis.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⏹  Interrupted.")
        sys.exit(130)