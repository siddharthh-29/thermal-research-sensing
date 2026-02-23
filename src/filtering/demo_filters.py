"""
Visual Demonstration of Signal Processing Filters

This script creates comprehensive visualizations showing how different filters
behave with various types of signals and noise.
"""

import numpy as np
import matplotlib.pyplot as plt
from signal_filters import SignalFilters

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def create_test_signals(fs=1000, duration=2):
    """Create various test signals for filter demonstration."""
    t = np.linspace(0, duration, int(duration*fs), endpoint=False)
    
    # Signal 1: Multiple frequency components
    clean = (np.sin(2*np.pi*5*t) +           # 5 Hz
             0.5*np.sin(2*np.pi*15*t) +       # 15 Hz
             0.3*np.sin(2*np.pi*50*t))        # 50 Hz (powerline)
    
    # Signal 2: With Gaussian noise
    noisy = clean + np.random.normal(0, 0.15, len(t))
    
    # Signal 3: With impulse noise
    impulse_noise = noisy.copy()
    spike_indices = np.random.choice(len(t), 30)
    impulse_noise[spike_indices] += np.random.randn(30) * 1.5
    
    # Signal 4: With drift
    drift = noisy + 1 + 0.5*t
    
    return t, clean, noisy, impulse_noise, drift


def demo_lowpass_comparison(filters, noisy_signal, t, fs):
    """Compare different lowpass filter types."""
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(3, 2, 1)
    plt.plot(t[:500], noisy_signal[:500], alpha=0.7, linewidth=0.8)
    plt.title('Original Noisy Signal', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Butterworth
    plt.subplot(3, 2, 2)
    filtered = filters.butterworth_lowpass(noisy_signal, cutoff=20, order=4)
    plt.plot(t[:500], filtered[:500], linewidth=1.2, color='#2E86AB')
    plt.title('Butterworth Lowpass\n(Smooth, balanced)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Chebyshev I
    plt.subplot(3, 2, 3)
    filtered = filters.chebyshev_type1_lowpass(noisy_signal, cutoff=20, order=4, ripple_db=0.5)
    plt.plot(t[:500], filtered[:500], linewidth=1.2, color='#A23B72')
    plt.title('Chebyshev Type I\n(Steep roll-off, passband ripple)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Bessel
    plt.subplot(3, 2, 4)
    filtered = filters.bessel_lowpass(noisy_signal, cutoff=20, order=4, zero_phase=False)
    plt.plot(t[:500], filtered[:500], linewidth=1.2, color='#F18F01')
    plt.title('Bessel Lowpass\n(Best phase response)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # FIR
    plt.subplot(3, 2, 5)
    filtered = filters.fir_lowpass_windowed(noisy_signal, cutoff=20, numtaps=101)
    plt.plot(t[:500], filtered[:500], linewidth=1.2, color='#06A77D')
    plt.title('FIR Lowpass (Hamming)\n(Linear phase, delayed)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Elliptic
    plt.subplot(3, 2, 6)
    filtered = filters.elliptic_lowpass(noisy_signal, cutoff=20, order=4)
    plt.plot(t[:500], filtered[:500], linewidth=1.2, color='#C73E1D')
    plt.title('Elliptic Lowpass\n(Steepest, both ripples)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/lowpass_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Lowpass filter comparison saved")


def demo_filter_types(filters, signals, t, fs):
    """Demonstrate different filter types on the same signal."""
    _, clean, noisy, impulse_noise, drift = signals
    
    plt.figure(figsize=(15, 12))
    
    # Lowpass
    plt.subplot(4, 2, 1)
    plt.plot(t[:1000], noisy[:1000], alpha=0.5, label='Noisy', linewidth=0.8)
    filtered = filters.butterworth_lowpass(noisy, cutoff=20, order=4)
    plt.plot(t[:1000], filtered[:1000], label='Filtered', linewidth=1.5)
    plt.title('LOWPASS: Remove high-frequency noise', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highpass
    plt.subplot(4, 2, 2)
    plt.plot(t[:1000], drift[:1000], alpha=0.5, label='With drift', linewidth=0.8)
    filtered = filters.butterworth_highpass(drift, cutoff=3, order=4)
    plt.plot(t[:1000], filtered[:1000], label='Detrended', linewidth=1.5)
    plt.title('HIGHPASS: Remove DC and drift', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bandpass
    plt.subplot(4, 2, 3)
    plt.plot(t[:1000], noisy[:1000], alpha=0.5, label='Original', linewidth=0.8)
    filtered = filters.butterworth_bandpass(noisy, lowcut=45, highcut=55, order=4)
    plt.plot(t[:1000], filtered[:1000], label='50 Hz band', linewidth=1.5)
    plt.title('BANDPASS: Isolate 50 Hz component', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Notch
    plt.subplot(4, 2, 4)
    # Add strong 50 Hz interference
    interference = noisy + 0.8*np.sin(2*np.pi*50*t)
    plt.plot(t[:1000], interference[:1000], alpha=0.5, label='With 50 Hz', linewidth=0.8)
    filtered = filters.notch_filter_iir(interference, notch_freq=50, quality_factor=30)
    plt.plot(t[:1000], filtered[:1000], label='Notched', linewidth=1.5)
    plt.title('NOTCH: Remove 50 Hz interference', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Median filter
    plt.subplot(4, 2, 5)
    plt.plot(t[:500], impulse_noise[:500], alpha=0.5, label='With spikes', linewidth=0.8)
    filtered = filters.median_filter(impulse_noise, kernel_size=7)
    plt.plot(t[:500], filtered[:500], label='Median filtered', linewidth=1.5)
    plt.title('MEDIAN: Remove impulse noise', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Savitzky-Golay
    plt.subplot(4, 2, 6)
    plt.plot(t[:500], noisy[:500], alpha=0.5, label='Noisy', linewidth=0.8)
    filtered = filters.savitzky_golay_filter(noisy, window_length=25, polyorder=3)
    plt.plot(t[:500], filtered[:500], label='Savitzky-Golay', linewidth=1.5)
    plt.title('SAVITZKY-GOLAY: Smooth preserving peaks', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Moving average
    plt.subplot(4, 2, 7)
    plt.plot(t[:500], noisy[:500], alpha=0.5, label='Noisy', linewidth=0.8)
    filtered = filters.moving_average(noisy, window_size=15)
    plt.plot(t[:500], filtered[:500], label='Moving average', linewidth=1.5)
    plt.title('MOVING AVERAGE: Simple smoothing', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Exponential smoothing
    plt.subplot(4, 2, 8)
    plt.plot(t[:500], noisy[:500], alpha=0.5, label='Noisy', linewidth=0.8)
    filtered = filters.exponential_smoothing(noisy, alpha=0.3)
    plt.plot(t[:500], filtered[:500], label='Exponential (α=0.3)', linewidth=1.5)
    plt.title('EXPONENTIAL: Weighted smoothing', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/filter_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Filter types demonstration saved")


def demo_phase_response(filters, fs):
    """Demonstrate phase characteristics of different filters."""
    from scipy import signal as sp_signal
    
    plt.figure(figsize=(15, 8))
    
    # Design filters
    nyq = 0.5 * fs
    cutoff = 100
    normal_cutoff = cutoff / nyq
    
    b_butter, a_butter = sp_signal.butter(4, normal_cutoff, btype='low')
    b_bessel, a_bessel = sp_signal.bessel(4, normal_cutoff, btype='low', analog=False)
    b_fir = sp_signal.firwin(51, normal_cutoff)
    
    # Frequency response
    w_butter, h_butter = sp_signal.freqz(b_butter, a_butter, worN=2000)
    w_bessel, h_bessel = sp_signal.freqz(b_bessel, a_bessel, worN=2000)
    w_fir, h_fir = sp_signal.freqz(b_fir, 1.0, worN=2000)
    
    freq_butter = w_butter * fs / (2*np.pi)
    freq_bessel = w_bessel * fs / (2*np.pi)
    freq_fir = w_fir * fs / (2*np.pi)
    
    # Magnitude
    plt.subplot(2, 2, 1)
    plt.plot(freq_butter, 20*np.log10(abs(h_butter)), label='Butterworth', linewidth=2)
    plt.plot(freq_bessel, 20*np.log10(abs(h_bessel)), label='Bessel', linewidth=2)
    plt.plot(freq_fir, 20*np.log10(abs(h_fir)), label='FIR', linewidth=2)
    plt.axvline(cutoff, color='k', linestyle='--', alpha=0.5, label='Cutoff')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Magnitude Response', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 300)
    plt.ylim(-80, 5)
    
    # Phase
    plt.subplot(2, 2, 2)
    plt.plot(freq_butter, np.angle(h_butter), label='Butterworth', linewidth=2)
    plt.plot(freq_bessel, np.angle(h_bessel), label='Bessel', linewidth=2)
    plt.plot(freq_fir, np.angle(h_fir), label='FIR (linear!)', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')
    plt.title('Phase Response', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 300)
    
    # Group delay
    plt.subplot(2, 2, 3)
    _, gd_butter = sp_signal.group_delay((b_butter, a_butter), w=2000)
    _, gd_bessel = sp_signal.group_delay((b_bessel, a_bessel), w=2000)
    _, gd_fir = sp_signal.group_delay((b_fir, 1.0), w=2000)
    
    plt.plot(freq_butter, gd_butter/fs*1000, label='Butterworth', linewidth=2)
    plt.plot(freq_bessel, gd_bessel/fs*1000, label='Bessel (most linear)', linewidth=2)
    plt.plot(freq_fir, gd_fir/fs*1000, label='FIR (constant)', linewidth=2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Group Delay (ms)')
    plt.title('Group Delay (Phase Linearity)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 300)
    plt.ylim(0, 20)
    
    # Step response (shows phase impact)
    plt.subplot(2, 2, 4)
    step = np.zeros(500)
    step[100:] = 1
    
    step_butter = sp_signal.lfilter(b_butter, a_butter, step)
    step_bessel = sp_signal.lfilter(b_bessel, a_bessel, step)
    step_fir = sp_signal.lfilter(b_fir, 1.0, step)
    
    t_step = np.arange(500) / fs * 1000
    plt.plot(t_step, step, 'k--', alpha=0.5, label='Input step', linewidth=1)
    plt.plot(t_step, step_butter, label='Butterworth (overshoot)', linewidth=2)
    plt.plot(t_step, step_bessel, label='Bessel (no overshoot)', linewidth=2)
    plt.plot(t_step, step_fir, label='FIR (symmetric)', linewidth=2)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Step Response (Transient Behavior)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 150)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/phase_response.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Phase response comparison saved")


def demo_fir_vs_iir(filters, noisy_signal, t, fs):
    """Compare FIR and IIR characteristics."""
    plt.figure(figsize=(15, 10))
    
    # Same cutoff, same order
    cutoff = 20
    order = 4
    
    # IIR (Butterworth) - causal
    plt.subplot(3, 2, 1)
    iir_causal = filters.butterworth_lowpass(noisy_signal, cutoff, order, zero_phase=False)
    plt.plot(t[:500], noisy_signal[:500], alpha=0.4, label='Original', linewidth=0.8)
    plt.plot(t[:500], iir_causal[:500], label='IIR (causal)', linewidth=1.5)
    plt.title('IIR Butterworth - Causal (has delay)', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # IIR - zero phase
    plt.subplot(3, 2, 2)
    iir_zerophase = filters.butterworth_lowpass(noisy_signal, cutoff, order, zero_phase=True)
    plt.plot(t[:500], noisy_signal[:500], alpha=0.4, label='Original', linewidth=0.8)
    plt.plot(t[:500], iir_zerophase[:500], label='IIR (filtfilt)', linewidth=1.5)
    plt.title('IIR Butterworth - Zero Phase (not causal)', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # FIR - always has delay
    plt.subplot(3, 2, 3)
    fir_filtered = filters.fir_lowpass_windowed(noisy_signal, cutoff, numtaps=101)
    plt.plot(t[:500], noisy_signal[:500], alpha=0.4, label='Original', linewidth=0.8)
    plt.plot(t[:500], fir_filtered[:500], label='FIR (101 taps)', linewidth=1.5)
    delay_samples = 50  # (101-1)/2
    delay_ms = delay_samples / fs * 1000
    plt.title(f'FIR - Linear Phase (delay: {delay_ms:.1f} ms)', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Computational comparison
    plt.subplot(3, 2, 4)
    orders = [2, 4, 6, 8]
    fir_taps = [21, 51, 101, 201]
    
    plt.bar([x-0.2 for x in range(len(orders))], orders, width=0.4, 
            label='IIR coefficients', alpha=0.7)
    plt.bar([x+0.2 for x in range(len(orders))], fir_taps, width=0.4,
            label='FIR taps (equiv.)', alpha=0.7)
    plt.xticks(range(len(orders)), [f'Level {i+1}' for i in range(len(orders))])
    plt.ylabel('Number of Coefficients')
    plt.title('Computational Cost Comparison', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Stability
    plt.subplot(3, 2, 5)
    stability_data = [
        ['Filter Type', 'Always Stable?', 'Phase Type', 'Delay'],
        ['FIR', 'YES ✓', 'Linear', 'Constant'],
        ['IIR (causal)', 'Usually*', 'Non-linear', 'Variable'],
        ['IIR (filtfilt)', 'Usually*', 'Zero', 'None**']
    ]
    
    table_text = '\n'.join([f"{row[0]:15} {row[1]:15} {row[2]:15} {row[3]:15}" 
                           for row in stability_data])
    plt.text(0.1, 0.5, table_text, fontsize=10, family='monospace',
             verticalalignment='center')
    plt.text(0.1, 0.1, '* Can become unstable with high orders or narrow bands\n'
                       '** Not causal - uses future data', fontsize=8, style='italic')
    plt.axis('off')
    plt.title('Key Differences', fontsize=11, fontweight='bold')
    
    # When to use which
    plt.subplot(3, 2, 6)
    use_cases = """
    USE IIR WHEN:
    • Computational efficiency critical
    • Real-time processing needed
    • Some phase distortion OK
    • Memory limited
    
    USE FIR WHEN:
    • Linear phase REQUIRED
      (audio, images, communications)
    • Guaranteed stability needed
    • Arbitrary freq. response
    • Non-causal OK (offline)
    
    USE FILTFILT (zero-phase) WHEN:
    • Offline processing only
    • Zero phase distortion needed
    • Can use future data
    """
    plt.text(0.1, 0.5, use_cases, fontsize=9, family='monospace',
             verticalalignment='center')
    plt.axis('off')
    plt.title('Decision Guide', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/fir_vs_iir.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ FIR vs IIR comparison saved")


def demo_noise_types(filters, t, fs):
    """Demonstrate appropriate filters for different noise types."""
    clean = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Gaussian noise → Lowpass
    plt.subplot(3, 2, 1)
    gaussian_noise = clean + np.random.normal(0, 0.2, len(t))
    filtered = filters.butterworth_lowpass(gaussian_noise, cutoff=25, order=4)
    plt.plot(t[:300], gaussian_noise[:300], alpha=0.5, label='Noisy', linewidth=0.8)
    plt.plot(t[:300], filtered[:300], label='Lowpass filtered', linewidth=1.5)
    plt.title('GAUSSIAN NOISE → Lowpass Filter', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Impulse noise → Median
    plt.subplot(3, 2, 2)
    impulse = clean.copy()
    spike_idx = np.random.choice(len(t), 40)
    impulse[spike_idx] += np.random.randn(40) * 2
    filtered = filters.median_filter(impulse, kernel_size=7)
    plt.plot(t[:300], impulse[:300], alpha=0.5, label='With spikes', linewidth=0.8)
    plt.plot(t[:300], filtered[:300], label='Median filtered', linewidth=1.5)
    plt.title('IMPULSE NOISE → Median Filter', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Periodic interference → Notch
    plt.subplot(3, 2, 3)
    periodic = clean + 0.5*np.sin(2*np.pi*50*t)  # 50 Hz interference
    filtered = filters.notch_filter_iir(periodic, notch_freq=50, quality_factor=30)
    plt.plot(t[:300], periodic[:300], alpha=0.5, label='With 50 Hz', linewidth=0.8)
    plt.plot(t[:300], filtered[:300], label='Notch filtered', linewidth=1.5)
    plt.title('PERIODIC INTERFERENCE → Notch Filter', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Baseline drift → Highpass
    plt.subplot(3, 2, 4)
    drift = clean + 2 + 0.3*t
    filtered = filters.butterworth_highpass(drift, cutoff=1, order=4)
    plt.plot(t[:500], drift[:500], alpha=0.5, label='With drift', linewidth=0.8)
    plt.plot(t[:500], filtered[:500], label='Highpass filtered', linewidth=1.5)
    plt.title('BASELINE DRIFT → Highpass Filter', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Out-of-band noise → Bandpass
    plt.subplot(3, 2, 5)
    out_of_band = (clean + 
                   np.random.normal(0, 0.3, len(t)) +  # Wideband noise
                   2*np.sin(2*np.pi*80*t))  # High freq interference
    filtered = filters.butterworth_bandpass(out_of_band, lowcut=3, highcut=30, order=4)
    plt.plot(t[:300], out_of_band[:300], alpha=0.5, label='Noisy', linewidth=0.8)
    plt.plot(t[:300], filtered[:300], label='Bandpass filtered', linewidth=1.5)
    plt.title('OUT-OF-BAND NOISE → Bandpass Filter', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Mixed noise → Combination
    plt.subplot(3, 2, 6)
    mixed = (clean + 
             np.random.normal(0, 0.15, len(t)) +  # Gaussian
             0.5*np.sin(2*np.pi*50*t))  # 50 Hz
    spike_idx = np.random.choice(len(t), 20)
    mixed[spike_idx] += np.random.randn(20) * 1.5  # Spikes
    
    # Apply multiple filters
    step1 = filters.median_filter(mixed, kernel_size=5)  # Remove spikes
    step2 = filters.notch_filter_iir(step1, notch_freq=50)  # Remove 50 Hz
    step3 = filters.butterworth_lowpass(step2, cutoff=25, order=4)  # Smooth
    
    plt.plot(t[:300], mixed[:300], alpha=0.5, label='Mixed noise', linewidth=0.8)
    plt.plot(t[:300], step3[:300], label='Multi-stage filtered', linewidth=1.5)
    plt.title('MIXED NOISE → Median + Notch + Lowpass', fontsize=11, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/noise_types.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Noise types demonstration saved")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("COMPREHENSIVE FILTER DEMONSTRATIONS")
    print("="*70 + "\n")
    
    # Setup
    fs = 1000  # Sampling rate
    duration = 2
    signals = create_test_signals(fs, duration)
    t, clean, noisy, impulse_noise, drift = signals
    
    # Initialize filter class
    filters = SignalFilters(sampling_rate=fs)
    
    print("Generating visualizations...")
    print("-" * 70)
    
    # Generate all demonstrations
    demo_lowpass_comparison(filters, noisy, t, fs)
    demo_filter_types(filters, signals, t, fs)
    demo_phase_response(filters, fs)
    demo_fir_vs_iir(filters, noisy, t, fs)
    demo_noise_types(filters, t, fs)
    
    # Also generate the comparison from the main class
    filters.compare_filter_responses(cutoff=100, order=4)
    
    print("-" * 70)
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  1. lowpass_comparison.png - Compare different lowpass designs")
    print("  2. filter_types.png - All filter types on various signals")
    print("  3. phase_response.png - Phase characteristics analysis")
    print("  4. fir_vs_iir.png - FIR vs IIR detailed comparison")
    print("  5. noise_types.png - Matching filters to noise types")
    print("  6. filter_comparison.png - Frequency response comparison")
    
    print("\n" + "="*70)
    print("QUICK REFERENCE GUIDE")
    print("="*70)
    print("""
    FILTER SELECTION CHEAT SHEET:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ NOISE TYPE              → BEST FILTER                           │
    ├─────────────────────────────────────────────────────────────────┤
    │ High-frequency noise    → Lowpass (Butterworth/FIR)             │
    │ Spikes/outliers         → Median filter                         │
    │ Powerline (50/60 Hz)    → Notch filter                          │
    │ DC offset/drift         → Highpass                              │
    │ Out-of-band noise       → Bandpass                              │
    │ Mixed noise             → Multi-stage (Median→Notch→Lowpass)    │
    └─────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ REQUIREMENT             → FILTER CHOICE                         │
    ├─────────────────────────────────────────────────────────────────┤
    │ Linear phase CRITICAL   → FIR                                   │
    │ Preserve waveform shape → Bessel IIR                            │
    │ Sharp cutoff needed     → Elliptic/Chebyshev                    │
    │ General purpose         → Butterworth                           │
    │ Real-time, low latency  → Low-order IIR (causal)                │
    │ Offline, best quality   → FIR or IIR with filtfilt              │
    │ Minimal computation     → IIR (low order)                       │
    │ Guaranteed stability    → FIR                                   │
    └─────────────────────────────────────────────────────────────────┘
    
    PHASE CHARACTERISTICS:
    • FIR: Linear phase, constant delay = (numtaps-1)/(2*fs)
    • IIR causal: Non-linear phase, frequency-dependent delay
    • IIR filtfilt: Zero phase, but NOT causal (uses future data)
    • Bessel: Best phase linearity among IIR filters
    
    STABILITY:
    • FIR: Always stable (no feedback)
    • IIR: Usually stable, can fail with high orders or narrow bands
    
    COMPUTATIONAL COST:
    • IIR: Low (few coefficients, e.g., 5-10)
    • FIR: High (many taps, e.g., 50-500 for sharp filters)
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
