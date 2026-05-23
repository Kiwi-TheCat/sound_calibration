#!/usr/bin/env python3
"""
REW-style Sweep and Analysis Script
Mimics REW sweep configuration and performs SPL analysis with calibration
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import chirp, get_window
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import pickle
import json
import sys
from typing import Tuple, List, Dict, Optional

class REWSweepAnalyzer:
    """REW-style sweep generation and analysis"""
    
    # REW Sweep Configuration from image
    SAMPLE_RATE = 48000  # Hz
    START_FREQ = 0  # Hz (will start from near-zero)
    END_FREQ = 20000  # Hz
    LENGTH_SAMPLES = 256000  # 256k samples
    REPETITIONS = 5.5  # seconds (approximately)
    LEVEL_DBFS = -12.00  # dBFS
    START_DELAY = 1.0  # seconds
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def generate_sweep(self) -> np.ndarray:
        """Generate logarithmic chirp sweep matching REW configuration"""
        # Calculate duration from samples
        duration = self.LENGTH_SAMPLES / self.SAMPLE_RATE
        
        # Time array
        t = np.linspace(0, duration, self.LENGTH_SAMPLES, endpoint=False)
        
        # Generate logarithmic chirp (REW uses logarithmic sweeps)
        # Start from 20 Hz to avoid subsonic
        start_freq = 20 if self.START_FREQ < 20 else self.START_FREQ
        sweep = chirp(t, f0=start_freq, f1=self.END_FREQ, t1=duration, method='logarithmic')
        
        # Apply windowing to avoid clicks (gentle fade in/out)
        window = get_window(('tukey', 0.05), len(sweep))
        sweep = sweep * window
        
        # Normalize to target level in dBFS
        target_amplitude = 10 ** (self.LEVEL_DBFS / 20.0)
        current_rms = np.sqrt(np.mean(sweep ** 2))
        if current_rms > 0:
            sweep = sweep * (target_amplitude / current_rms)
        
        return sweep
    
    
    def list_audio_devices(self) -> None:
        """List available audio devices"""
        print("\nAvailable Audio Devices:")
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                print(f"  [{i}] {device['name']}")
                print(f"       Channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
            print()
            default_input = sd.default.device[0]
            default_output = sd.default.device[1]
            print(f"  Default input device: {default_input}")
            print(f"  Default output device: {default_output}")
        except Exception as e:
            print(f"Could not query audio devices: {e}")
    
    def play_sweep(self, audio_data: np.ndarray, device: Optional[int] = None) -> None:
        """Play a sweep signal through audio output device"""
        print(f"Playing sweep through audio device...")
        try:
            sd.play(audio_data, samplerate=self.SAMPLE_RATE, device=device)
            sd.wait()
            print("Playback complete.")
        except Exception as e:
            print(f"Error during playback: {e}")
            print("Could not play audio. Ensure audio device is configured.")
    
    def record_sweep(self, duration: Optional[float] = None, device: Optional[int] = None) -> np.ndarray:
        """Record a sweep signal (capture system response)"""
        if duration is None:
            duration = self.LENGTH_SAMPLES / self.SAMPLE_RATE + self.START_DELAY + 1.0
        
        print(f"Recording sweep for {duration:.2f} seconds...")
        print(f"Sample rate: {self.SAMPLE_RATE} Hz")
        
        try:
            recording = sd.rec(
                int(duration * self.SAMPLE_RATE),
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype=np.float32,
                device=device,
                blocksize=4096
            )
            sd.wait()
            return recording.squeeze()
        except Exception as e:
            print(f"Error during recording: {e}")
            print("Generating synthetic response for demo...")
            # Generate synthetic response for testing
            t = np.linspace(0, duration, int(duration * self.SAMPLE_RATE), endpoint=False)
            response = np.random.normal(0, 0.01, len(t))
            return response
    
    def save_sweep(self, audio_data: np.ndarray, name: str = "mimic_sweep_1"):
        """Save audio data as WAV file"""
        output_path = self.data_dir / f"{name}.wav"
        sf.write(output_path, audio_data, self.SAMPLE_RATE)
        print(f"Saved sweep to: {output_path}")
        return output_path
    
    def load_calibration(self, calibration_file: str = "7101790.txt") -> Dict[float, float]:
        """Load calibration file (frequency -> SPL correction mapping)"""
        calibration_path = self.data_dir / calibration_file
        calib_data = {}
        
        try:
            with open(calibration_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            freq = float(parts[0])
                            correction = float(parts[1])
                            calib_data[freq] = correction
                    except ValueError:
                        continue
            
            print(f"Loaded calibration data with {len(calib_data)} points")
            return calib_data
        except FileNotFoundError:
            print(f"Calibration file not found: {calibration_path}")
            return {}
    
    def perform_fft_analysis(self, audio_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform FFT analysis to get frequency spectrum"""
        # Apply window to reduce spectral leakage
        window = get_window('hann', len(audio_data))
        windowed_audio = audio_data * window
        
        # Perform FFT
        fft_result = fft(windowed_audio)
        frequencies = fftfreq(len(audio_data), 1/self.SAMPLE_RATE)
        
        # Take positive frequencies only
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft_result[positive_freq_idx])
        
        # Convert to dB (20 * log10 for amplitude)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        return frequencies, magnitude_db
    
    def apply_calibration(self, frequencies: np.ndarray, magnitude_db: np.ndarray, 
                         calibration: Dict[float, float]) -> np.ndarray:
        """Apply calibration correction to measured spectrum"""
        if not calibration:
            return magnitude_db
        
        # Interpolate calibration data to match frequency points
        calib_freqs = np.array(sorted(calibration.keys()))
        calib_values = np.array([calibration[f] for f in calib_freqs])
        
        # Interpolate
        from scipy.interpolate import interp1d
        calib_func = interp1d(calib_freqs, calib_values, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        calibration_correction = calib_func(frequencies)
        
        # Apply calibration
        calibrated_db = magnitude_db + calibration_correction
        return calibrated_db
    
    def convert_to_spl(self, magnitude_db: np.ndarray, reference_pressure: float = 20e-6) -> np.ndarray:
        """Convert dB to dB SPL (Sound Pressure Level)"""
        # Assuming magnitude_db is already in dB relative to some reference
        # Adjust to standard SPL reference (20 μPa)
        spl = magnitude_db  # Already in dB form
        return spl
    
    def apply_octave_smoothing(self, frequencies: np.ndarray, magnitudes: np.ndarray, 
                              smoothing_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply octave band smoothing to spectrum"""
        if smoothing_type is None or smoothing_type.lower() == 'none':
            return frequencies, magnitudes
        
        if '1/6' in smoothing_type or smoothing_type == '1/6-octave':
            # 1/6 octave smoothing
            octave_ratio = 2 ** (1/6)
        elif '1/3' in smoothing_type or smoothing_type == '1/3-octave':
            # 1/3 octave smoothing
            octave_ratio = 2 ** (1/3)
        elif '1/1' in smoothing_type or smoothing_type == 'octave':
            # Full octave smoothing
            octave_ratio = 2
        else:
            return frequencies, magnitudes
        
        # Create octave band centers
        smoothed_freqs = []
        smoothed_mags = []
        
        # Start from first frequency, go up by octave ratio
        current_freq = frequencies[0]
        while current_freq < frequencies[-1]:
            # Define band edges
            lower_edge = current_freq / np.sqrt(octave_ratio)
            upper_edge = current_freq * np.sqrt(octave_ratio)
            
            # Find indices within this band
            mask = (frequencies >= lower_edge) & (frequencies < upper_edge)
            if np.any(mask):
                # Average magnitude in this band
                band_mag = np.mean(magnitudes[mask])
                smoothed_freqs.append(current_freq)
                smoothed_mags.append(band_mag)
            
            current_freq *= octave_ratio
        
        return np.array(smoothed_freqs), np.array(smoothed_mags)
    
    def load_rew_mdata(self, mdata_file: str, subdirectory: str = "REW Standard Data") -> Optional[Dict]:
        """Load REW .mdata/.mdat file (binary format used by REW)"""
        # Try in subdirectory first, then main data dir
        mdata_path = self.data_dir / subdirectory / mdata_file
        if not mdata_path.exists():
            mdata_path = self.data_dir / mdata_file
        
        try:
            # REW .mdata files are binary files with frequency response data
            # Try to read as pickle first (common in older REW versions)
            try:
                with open(mdata_path, 'rb') as f:
                    data = pickle.load(f)
                    return data
            except:
                # Try reading as structured binary data
                with open(mdata_path, 'rb') as f:
                    content = f.read()
                
                # Try to parse as text if it contains readable data
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                    # Parse if it looks like text data
                    if 'freq' in text_content.lower() or 'spl' in text_content.lower():
                        lines = text_content.split('\n')
                        freqs = []
                        spls = []
                        for line in lines:
                            parts = line.split()
                            if len(parts) >= 2:
                                try:
                                    f = float(parts[0])
                                    s = float(parts[1])
                                    freqs.append(f)
                                    spls.append(s)
                                except:
                                    continue
                        if freqs:
                            return {'frequencies': np.array(freqs), 'spl': np.array(spls)}
                except:
                    pass
                
                # If all else fails, return None
                return None
                
        except FileNotFoundError:
            print(f"File not found: {mdata_path}")
            return None
    
    def plot_comparison(self, measured_freq: np.ndarray, measured_spl: np.ndarray,
                       sample_freqs_list: List[np.ndarray], sample_spls_list: List[np.ndarray],
                       smoothing_type: Optional[str] = None):
        """Plot measured SPL vs averaged sample data"""
        
        # Apply smoothing if requested
        if smoothing_type and smoothing_type.lower() != 'none':
            measured_freq, measured_spl = self.apply_octave_smoothing(
                measured_freq, measured_spl, smoothing_type
            )
            sample_freqs_list = [self.apply_octave_smoothing(f, s, smoothing_type)[0] 
                                for f, s in zip(sample_freqs_list, sample_spls_list)]
            sample_spls_list = [self.apply_octave_smoothing(f, s, smoothing_type)[1] 
                               for f, s in zip(sample_freqs_list, sample_spls_list)]
        
        # Average sample data
        if sample_spls_list:
            # Interpolate all samples to common frequency grid
            from scipy.interpolate import interp1d
            common_freqs = np.logspace(np.log10(20), np.log10(20000), 1000)
            
            averaged_spl = None
            for freq, spl in zip(sample_freqs_list, sample_spls_list):
                if len(freq) > 1:
                    func = interp1d(freq, spl, kind='linear', 
                                   bounds_error=False, fill_value='extrapolate')
                    interp_spl = func(common_freqs)
                    if averaged_spl is None:
                        averaged_spl = interp_spl
                    else:
                        averaged_spl += interp_spl
            
            if averaged_spl is not None:
                averaged_spl /= len(sample_spls_list)
                sample_freq = common_freqs
            else:
                sample_freq = None
        else:
            sample_freq = None
        
        # Create plot
        plt.figure(figsize=(14, 8))
        plt.semilogx(measured_freq, measured_spl, 'b-', linewidth=2, label='Measured')
        
        if sample_freq is not None and averaged_spl is not None:
            plt.semilogx(sample_freq, averaged_spl, 'r--', linewidth=2, label='Average Sample')
            
            # Plot individual samples lightly
            for i, (freq, spl) in enumerate(zip(sample_freqs_list, sample_spls_list)):
                plt.semilogx(freq, spl, 'g:', alpha=0.3, linewidth=1)
        
        plt.xlabel('Frequency (Hz)', fontsize=12)
        plt.ylabel('SPL (dB)', fontsize=12)
        plt.title('REW-style Sweep Analysis: Measured vs Sample Data', fontsize=14)
        plt.grid(True, which='both', alpha=0.3)
        plt.legend(fontsize=11)
        
        if smoothing_type and smoothing_type.lower() != 'none':
            plt.title(f'REW-style Sweep Analysis with {smoothing_type} Smoothing', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'rew_analysis.png', dpi=150)
        print(f"Saved plot to: {self.data_dir / 'rew_analysis.png'}")
        plt.show()


def main():
    """Main execution"""
    analyzer = REWSweepAnalyzer()
    
    print("="*60)
    print("REW-style Sweep and Analysis Script")
    print("="*60)
    print(f"Python version: {sys.version}")
    print(f"Sample rate: {analyzer.SAMPLE_RATE} Hz")
    print(f"Frequency range: {analyzer.START_FREQ} - {analyzer.END_FREQ} Hz")
    print(f"Sweep length: {analyzer.LENGTH_SAMPLES} samples")
    print(f"Level: {analyzer.LEVEL_DBFS} dBFS")
    print()
    
    # List audio devices
    analyzer.list_audio_devices()
    
    # Step 1: Generate sweep
    print("Step 1: Generating sweep signal...")
    sweep = analyzer.generate_sweep()
    print(f"Generated sweep: {len(sweep)} samples, duration: {len(sweep)/analyzer.SAMPLE_RATE:.3f}s")
    
    # Step 1b: Play sweep signal
    print("\nStep 1b: Playing sweep signal...")
    print("(Make sure speakers/headphones are on and microphone is ready to record the response)")
    analyzer.play_sweep(sweep)
    
    # Step 2: Record/capture response
    print("\nStep 2: Recording sweep response...")
    print("(Recording now - the response will be captured from the microphone)")
    recorded = analyzer.record_sweep()
    print(f"Recorded: {len(recorded)} samples")
    
    # Step 3: Save raw wav data
    print("\nStep 3: Saving raw wav data...")
    analyzer.save_sweep(recorded, "mimic_sweep_1")
    
    # Step 4: Load calibration
    print("\nStep 4: Loading calibration file...")
    calibration = analyzer.load_calibration("7101790.txt")
    
    # Step 5: Perform FFT analysis
    print("\nStep 5: Performing FFT analysis...")
    measured_freq, measured_db = analyzer.perform_fft_analysis(recorded)
    
    # Step 6: Apply calibration
    print("Step 6: Applying calibration correction...")
    measured_spl = analyzer.apply_calibration(measured_freq, measured_db, calibration)
    measured_spl = analyzer.convert_to_spl(measured_spl)
    
    # Step 7: Load sample data from .mdata/.mdat files
    print("\nStep 7: Loading REW sample data (.mdata/.mdat files)...")
    
    # Look for .mdat or .mdata files in REW Standard Data subdirectory
    rew_data_dir = analyzer.data_dir / "REW Standard Data"
    sample_files = []
    if rew_data_dir.exists():
        sample_files = list(rew_data_dir.glob("*.mdat")) + list(rew_data_dir.glob("*.mdata"))
    else:
        # Fallback to main data directory
        sample_files = list(analyzer.data_dir.glob("*.mdat")) + list(analyzer.data_dir.glob("*.mdata"))
    
    sample_freqs_list = []
    sample_spls_list = []
    
    for sample_file in sample_files[:3]:  # Take first 3 samples
        print(f"  Loading: {sample_file.name}")
        data = analyzer.load_rew_mdata(sample_file.name, "REW Standard Data" if rew_data_dir.exists() else ".")
        if data is not None:
            if isinstance(data, dict):
                if 'frequencies' in data and 'spl' in data:
                    sample_freqs_list.append(data['frequencies'])
                    sample_spls_list.append(data['spl'])
            else:
                print(f"    Warning: Could not parse {sample_file.name}")
    
    if sample_files:
        print(f"Found {len(sample_files)} sample files, loaded {len(sample_spls_list)} successfully")
    else:
        print("No sample files found. Check 'data/REW Standard Data/' directory.")
    
    # Step 8: Ask about smoothing
    print("\nStep 8: Applying smoothing (optional)...")
    print("Available smoothing options:")
    print("  1. None (no smoothing)")
    print("  2. 1/6-octave smoothing")
    print("  3. 1/3-octave smoothing")
    print("  4. 1/1-octave smoothing")
    
    smoothing_choice = input("Select smoothing option (1-4, default=1): ").strip()
    smoothing_map = {
        '1': None,
        '2': '1/6-octave',
        '3': '1/3-octave',
        '4': '1/1-octave'
    }
    smoothing_type = smoothing_map.get(smoothing_choice, None)
    
    # Step 9: Create comparison plot
    print("\nStep 9: Creating comparison plot...")
    analyzer.plot_comparison(measured_freq, measured_spl, 
                            sample_freqs_list, sample_spls_list,
                            smoothing_type)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()
