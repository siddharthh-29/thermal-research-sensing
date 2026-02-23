"""
Practical Usage Examples for Signal Filtering

This script demonstrates real-world applications of the SignalFilters class
with common scenarios you might encounter in signal processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_filters import SignalFilters


# ============================================================================
# EXAMPLE 1: ECG Signal Processing
# ============================================================================

def example_ecg_filtering():
    """
    Process an ECG signal with multiple filtering stages.
    
    Real ECG signals typically have:
    - Baseline wander (0.5-1 Hz) from respiration
    - Powerline interference (50/60 Hz)
    - High-frequency noise from muscle activity
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: ECG Signal Processing")
    print("="*70)
    
    # Simulate ECG signal (simplified)
    fs = 500  # 500 Hz sampling rate (typical for ECG)
    t = np.linspace(0, 5, 5*fs, endpoint=False)
    
    # Simplified ECG waveform (R-peaks at 1 Hz = 60 bpm)
    ecg_clean = np.zeros_like(t)
    for i in range(5):
        # R-peak
        peak_loc = int((i + 0.5) * fs)
        ecg_clean[peak_loc-10:peak_loc+10] = np.exp(-np.linspace(-2, 2, 20)**2) * 2
    
    # Add realistic artifacts
    baseline_wander = 0.3 * np.sin(2*np.pi*0.3*t)  # Respiration
    powerline = 0.15 * np.sin(2*np.pi*60*t)  # 60 Hz interference
    muscle_noise = 0.1 * np.random.randn(len(t))  # EMG noise
    
    ecg_noisy = ecg_clean + baseline_wander + powerline + muscle_noise
    
    # Initialize filters
    filters = SignalFilters(sampling_rate=fs)
    
    # Multi-stage filtering (typical ECG processing pipeline)
    print("\nProcessing pipeline:")
    print("  1. Highpass (0.5 Hz) - Remove baseline wander")
    ecg_step1 = filters.butterworth_highpass(ecg_noisy, cutoff=0.5, order=4)
    
    print("  2. Notch (60 Hz) - Remove powerline interference")
    ecg_step2 = filters.notch_filter_iir(ecg_step1, notch_freq=60, quality_factor=30)
    
    print("  3. Lowpass (40 Hz) - Remove high-frequency noise")
    ecg_clean_filtered = filters.butterworth_lowpass(ecg_step2, cutoff=40, order=4)
    
    print("  ✓ ECG signal cleaned!")
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, ecg_noisy, linewidth=0.8)
    plt.title('Raw ECG (with artifacts)', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 2)
    plt.plot(t, ecg_step1, linewidth=0.8)
    plt.title('After Highpass (baseline removed)', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 3)
    plt.plot(t, ecg_step2, linewidth=0.8)
    plt.title('After Notch (60 Hz removed)', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(4, 1, 4)
    plt.plot(t, ecg_clean_filtered, linewidth=0.8, color='green')
    plt.title('Final Clean ECG', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/example_ecg.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  → Saved: example_ecg.png")


# ============================================================================
# EXAMPLE 2: Audio Processing
# ============================================================================

def example_audio_processing():
    """
    Process an audio signal with FIR filters (linear phase is critical).
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Audio Processing (Linear Phase Required)")
    print("="*70)
    
    # Simulate audio signal
    fs = 44100  # CD quality
    duration = 0.5
    t = np.linspace(0, duration, int(duration*fs), endpoint=False)
    
    # Musical note (A4 = 440 Hz with harmonics)
    audio_clean = (np.sin(2*np.pi*440*t) +      # Fundamental
                   0.5*np.sin(2*np.pi*880*t) +   # 2nd harmonic
                   0.25*np.sin(2*np.pi*1320*t))  # 3rd harmonic
    
    # Add high-frequency hiss
    hiss = 0.3 * np.random.randn(len(t))
    audio_noisy = audio_clean + hiss
    
    # Use FIR filter to preserve phase relationships
    filters = SignalFilters(sampling_rate=fs)
    
    print("\nWhy FIR for audio:")
    print("  • Linear phase preserves timing between frequency components")
    print("  • No phase distortion = no 'smearing' of transients")
    print("  • Critical for music, speech, and hi-fi audio")
    
    print("\nApplying FIR lowpass filter...")
    audio_filtered = filters.fir_lowpass_windowed(
        audio_noisy, 
        cutoff=5000,  # Remove frequencies above 5 kHz
        numtaps=201,  # Long filter for sharp cutoff
        window='blackman'  # Low ripple
    )
    
    delay_ms = (201-1) / (2*fs) * 1000
    print(f"  ✓ Filtered with {delay_ms:.2f} ms constant delay")
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t[:1000], audio_noisy[:1000], linewidth=0.8, alpha=0.7)
    plt.title('Noisy Audio Signal', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t[:1000], audio_filtered[:1000], linewidth=0.8, color='green')
    plt.title('FIR Filtered (Linear Phase Preserved)', fontweight='bold')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Frequency spectrum comparison
    plt.subplot(3, 1, 3)
    freqs = np.fft.rfftfreq(len(audio_noisy), 1/fs)
    spectrum_noisy = np.abs(np.fft.rfft(audio_noisy))
    spectrum_filtered = np.abs(np.fft.rfft(audio_filtered))
    
    plt.plot(freqs[:5000], spectrum_noisy[:5000], alpha=0.5, label='Noisy', linewidth=0.8)
    plt.plot(freqs[:5000], spectrum_filtered[:5000], label='Filtered', linewidth=1.2)
    plt.axvline(5000, color='r', linestyle='--', alpha=0.5, label='Cutoff')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/example_audio.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  → Saved: example_audio.png")


# ============================================================================
# EXAMPLE 3: Sensor Data with Spikes
# ============================================================================

def example_sensor_data():
    """
    Process sensor data with occasional spikes/outliers.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Sensor Data with Spikes (Median Filter)")
    print("="*70)
    
    # Simulate temperature sensor
    fs = 10  # 10 Hz
    t = np.linspace(0, 60, 60*fs, endpoint=False)  # 1 minute
    
    # Slow temperature variation
    temp_true = 20 + 2*np.sin(2*np.pi*0.05*t) + 0.5*np.sin(2*np.pi*0.2*t)
    
    # Add small Gaussian noise
    temp_noisy = temp_true + np.random.normal(0, 0.1, len(t))
    
    # Add random spikes (sensor glitches)
    spike_indices = np.random.choice(len(t), 25)
    temp_noisy[spike_indices] += np.random.randn(25) * 3
    
    filters = SignalFilters(sampling_rate=fs)
    
    print("\nWhy Median Filter:")
    print("  • Preserves edges and sharp transitions")
    print("  • Removes spikes without smoothing the signal too much")
    print("  • Non-linear but very effective for outliers")
    
    # First median to remove spikes
    print("\nStep 1: Median filter to remove spikes")
    temp_despike = filters.median_filter(temp_noisy, kernel_size=5)
    
    # Then smooth with Savitzky-Golay
    print("Step 2: Savitzky-Golay for final smoothing")
    temp_smooth = filters.savitzky_golay_filter(temp_despike, window_length=11, polyorder=2)
    
    print("  ✓ Sensor data cleaned!")
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, temp_true, 'g-', alpha=0.5, label='True temperature', linewidth=2)
    plt.plot(t, temp_noisy, 'b.', markersize=2, alpha=0.5, label='Noisy with spikes')
    plt.title('Raw Sensor Data', fontweight='bold')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(t, temp_true, 'g-', alpha=0.5, label='True', linewidth=2)
    plt.plot(t, temp_despike, 'r-', label='After median filter', linewidth=1)
    plt.title('After Spike Removal', fontweight='bold')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(t, temp_true, 'g-', alpha=0.5, label='True', linewidth=2)
    plt.plot(t, temp_smooth, 'b-', label='Final (Median + SavGol)', linewidth=1)
    plt.title('Final Smoothed Result', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/example_sensor.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  → Saved: example_sensor.png")


# ============================================================================
# EXAMPLE 4: Real-time Control System
# ============================================================================

def example_realtime_control():
    """
    Real-time signal processing with minimal latency.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Real-time Control (Low-Latency Filtering)")
    print("="*70)
    
    # Simulate control system feedback
    fs = 1000  # 1 kHz
    t = np.linspace(0, 2, 2*fs, endpoint=False)
    
    # Position feedback with noise
    setpoint = np.ones_like(t)
    setpoint[500:] = 2  # Step change
    
    actual = setpoint + 0.2*np.random.randn(len(t))
    
    filters = SignalFilters(sampling_rate=fs)
    
    print("\nRequirements for real-time control:")
    print("  • CAUSAL filtering only (no future data)")
    print("  • Minimal delay/lag")
    print("  • Computational efficiency")
    
    print("\nComparing filtering approaches:")
    
    # 1. No filtering (too noisy)
    error_noisy = np.abs(actual - setpoint).mean()
    
    # 2. Low-order IIR (good balance)
    filtered_iir = filters.butterworth_lowpass(
        actual, 
        cutoff=50, 
        order=2,  # Low order = low delay
        zero_phase=False  # MUST be causal for real-time
    )
    error_iir = np.abs(filtered_iir - setpoint).mean()
    
    # 3. Exponential smoothing (minimal lag)
    filtered_exp = filters.exponential_smoothing(actual, alpha=0.3)
    error_exp = np.abs(filtered_exp - setpoint).mean()
    
    # 4. FIR would introduce too much delay
    # filtered_fir = filters.fir_lowpass_windowed(actual, cutoff=50, numtaps=51)
    # Delay = 25 ms (too much for fast control!)
    
    print(f"  • No filter: RMS error = {error_noisy:.3f}")
    print(f"  • IIR (order 2): RMS error = {error_iir:.3f}")
    print(f"  • Exponential: RMS error = {error_exp:.3f}")
    print(f"  • FIR would add ~25ms delay (not suitable)")
    
    print("\n  ✓ Recommendation: Use low-order IIR (causal) or exponential")
    
    # Plot
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, setpoint, 'k--', label='Setpoint', linewidth=2)
    plt.plot(t, actual, alpha=0.3, label='Noisy feedback', linewidth=0.5)
    plt.plot(t, filtered_iir, label='IIR filtered (causal)', linewidth=1.5)
    plt.plot(t, filtered_exp, label='Exponential smoothing', linewidth=1.5)
    plt.title('Real-time Position Control', fontweight='bold')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.4, 0.8)
    
    plt.subplot(2, 1, 2)
    plt.plot(t, setpoint, 'k--', label='Setpoint', linewidth=2)
    plt.plot(t, actual, alpha=0.3, label='Noisy', linewidth=0.5)
    plt.plot(t, filtered_iir, label='IIR (low delay)', linewidth=1.5)
    plt.title('Step Response (showing delay)', fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.45, 0.65)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/example_realtime.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  → Saved: example_realtime.png")


# ============================================================================
# EXAMPLE 5: Spectroscopy/Chromatography
# ============================================================================

def example_spectroscopy():
    """
    Process spectroscopy data with peak detection.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Spectroscopy (Savitzky-Golay for Peaks)")
    print("="*70)
    
    # Simulate spectrum with peaks
    x = np.linspace(0, 10, 1000)
    
    # Baseline
    baseline = 0.1 + 0.05*x
    
    # Several Gaussian peaks
    peaks = (np.exp(-((x-2)/0.2)**2) * 0.8 +
             np.exp(-((x-4.5)/0.15)**2) * 1.2 +
             np.exp(-((x-7)/0.25)**2) * 0.6 +
             np.exp(-((x-8.5)/0.1)**2) * 0.9)
    
    # Clean spectrum
    spectrum_clean = baseline + peaks
    
    # Add noise
    spectrum_noisy = spectrum_clean + np.random.normal(0, 0.05, len(x))
    
    fs = 100  # Pretend sampling rate
    filters = SignalFilters(sampling_rate=fs)
    
    print("\nWhy Savitzky-Golay for spectroscopy:")
    print("  • Preserves peak shapes and positions")
    print("  • Can compute derivatives for peak detection")
    print("  • Better than moving average for peak preservation")
    
    # Smooth the spectrum
    print("\nStep 1: Smooth with Savitzky-Golay")
    spectrum_smooth = filters.savitzky_golay_filter(
        spectrum_noisy,
        window_length=15,
        polyorder=2
    )
    
    # Compute first derivative for peak detection
    print("Step 2: Compute first derivative")
    derivative1 = filters.savitzky_golay_filter(
        spectrum_smooth,
        window_length=15,
        polyorder=2,
        deriv=1
    )
    
    # Find peaks (where derivative crosses zero)
    zero_crossings = np.where(np.diff(np.sign(derivative1)))[0]
    peak_indices = [i for i in zero_crossings if derivative1[i-1] > 0 and derivative1[i+1] < 0]
    
    print(f"  ✓ Found {len(peak_indices)} peaks")
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(x, spectrum_noisy, alpha=0.5, label='Noisy', linewidth=0.8)
    plt.plot(x, spectrum_smooth, label='Savitzky-Golay smoothed', linewidth=1.5)
    plt.plot(x[peak_indices], spectrum_smooth[peak_indices], 'r*', 
             markersize=15, label='Detected peaks')
    plt.title('Spectrum Smoothing and Peak Detection', fontweight='bold')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(x, derivative1, linewidth=1.2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.plot(x[peak_indices], derivative1[peak_indices], 'r*', markersize=15)
    plt.title('First Derivative (peaks at zero crossings)', fontweight='bold')
    plt.ylabel('dI/dx')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    # Compare with moving average
    spectrum_ma = filters.moving_average(spectrum_noisy, window_size=15)
    plt.plot(x, spectrum_clean, 'g-', alpha=0.7, label='True spectrum', linewidth=2)
    plt.plot(x, spectrum_ma, label='Moving average', linewidth=1.2, alpha=0.7)
    plt.plot(x, spectrum_smooth, label='Savitzky-Golay', linewidth=1.2)
    plt.title('Comparison: SavGol vs Moving Average (SavGol preserves peaks better)', 
              fontweight='bold')
    plt.xlabel('Wavelength / Retention Time')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/example_spectroscopy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("  → Saved: example_spectroscopy.png")


# ============================================================================
# Main execution
# ============================================================================

def main():
    """Run all practical examples."""
    print("\n" + "="*70)
    print("PRACTICAL SIGNAL FILTERING EXAMPLES")
    print("="*70)
    
    example_ecg_filtering()
    example_audio_processing()
    example_sensor_data()
    example_realtime_control()
    example_spectroscopy()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  • example_ecg.png - ECG processing pipeline")
    print("  • example_audio.png - Audio with FIR (linear phase)")
    print("  • example_sensor.png - Spike removal with median filter")
    print("  • example_realtime.png - Low-latency control filtering")
    print("  • example_spectroscopy.png - Peak preservation with SavGol")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
