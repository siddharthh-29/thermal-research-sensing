# Comprehensive Signal Processing Filter Library

A complete Python library for digital signal processing filters with detailed explanations, comparisons, and practical examples.

## 📚 Contents

### Core Files

1. **`signal_filters.py`** - Main filter class with all implementations
2. **`demo_filters.py`** - Visual demonstrations and comparisons
3. **`practical_examples.py`** - Real-world application examples

## 🎯 Features

### IIR Filters (Infinite Impulse Response)
- ✅ **Butterworth** - Smooth, flat passband (general purpose)
- ✅ **Chebyshev Type I** - Steep roll-off, passband ripple
- ✅ **Chebyshev Type II** - Steep roll-off, flat passband, stopband ripple
- ✅ **Elliptic (Cauer)** - Steepest roll-off, both ripples
- ✅ **Bessel** - Best phase linearity, preserves waveform shape
- ✅ **Notch (IIR)** - Remove single frequency (e.g., 50/60 Hz powerline)

### FIR Filters (Finite Impulse Response)
- ✅ **Windowed FIR** - Linear phase, various windows (Hamming, Hann, Blackman, Kaiser)
- ✅ **Kaiser FIR** - Optimal design for given specifications
- ✅ Lowpass, Highpass, Bandpass variants

### Non-Linear Filters
- ✅ **Median Filter** - Spike/outlier removal, edge preservation
- ✅ **Savitzky-Golay** - Smoothing with peak preservation, derivative computation
- ✅ **Moving Average** - Simple smoothing
- ✅ **Exponential Smoothing** - Real-time, low-latency filtering

### Specialized Filters
- ✅ **Adaptive LMS** - Noise cancellation with reference signal
- ✅ **Wiener Filter** - Optimal for additive Gaussian noise

## 🚀 Quick Start

```python
from signal_filters import SignalFilters
import numpy as np

# Initialize with your sampling rate
fs = 1000  # 1 kHz
filters = SignalFilters(sampling_rate=fs)

# Create noisy signal
t = np.linspace(0, 1, fs)
signal = np.sin(2*np.pi*10*t) + 0.5*np.random.randn(fs)

# Apply lowpass filter
filtered = filters.butterworth_lowpass(signal, cutoff=20, order=4)

# Remove 60 Hz powerline noise
clean = filters.notch_filter_iir(signal, notch_freq=60, quality_factor=30)

# Remove spikes
despike = filters.median_filter(signal, kernel_size=5)
```

## 📊 Filter Selection Guide

### By Noise Type

| Noise Type | Best Filter | Why |
|------------|-------------|-----|
| High-frequency noise | Lowpass (Butterworth/FIR) | Removes frequencies above cutoff |
| Spikes/outliers | Median filter | Edge-preserving, non-linear |
| Powerline (50/60 Hz) | Notch filter | Removes single frequency |
| DC offset/drift | Highpass | Removes low frequencies |
| Out-of-band noise | Bandpass | Keeps only desired frequency range |
| Mixed noise | Multi-stage | Median → Notch → Lowpass |

### By Requirements

| Requirement | Filter Choice | Notes |
|-------------|---------------|-------|
| **Linear phase CRITICAL** | FIR | Audio, image processing |
| **Preserve waveform shape** | Bessel IIR | Best phase response among IIR |
| **Sharp cutoff needed** | Elliptic/Chebyshev | Steepest roll-off |
| **General purpose** | Butterworth | Balanced, no ripple |
| **Real-time, low latency** | Low-order IIR (causal) | Minimal delay |
| **Offline, best quality** | FIR or IIR with filtfilt | Zero phase or linear phase |
| **Minimal computation** | IIR (low order) | Few coefficients |
| **Guaranteed stability** | FIR | Always stable |

## 🔍 Key Concepts Explained

### Phase Response

**Linear Phase (FIR filters)**
- All frequencies delayed by same amount
- Constant group delay = (numtaps-1)/(2*fs) seconds
- Preserves signal shape
- Required for audio, images, communications

**Non-linear Phase (IIR filters)**
- Different frequencies delayed by different amounts
- Frequency-dependent group delay
- Can distort waveform shape
- Acceptable when only magnitude matters

**Zero Phase (filtfilt)**
- Filters signal forwards then backwards
- No phase distortion
- NOT causal (uses future data)
- Only for offline processing

### Stability

**FIR Filters**
- ✅ Always stable (no feedback)
- No poles outside unit circle
- Safe for all applications

**IIR Filters**
- ⚠️ Usually stable, can fail with:
  - Very high orders
  - Very narrow bands
  - Extreme cutoff frequencies
- Has poles that can cause instability

### Computational Cost

**IIR Filters**
- Low order (typically 2-8 coefficients)
- Fast computation
- Small memory footprint
- Ideal for embedded systems

**FIR Filters**
- High order (50-500+ taps for sharp filters)
- More computation per sample
- Larger memory requirements
- Worth it when phase linearity needed

## 📈 Filter Characteristics Comparison

### Magnitude Response (from steepest to slowest roll-off)
1. **Elliptic** - Steepest, but ripple in both bands
2. **Chebyshev I** - Very steep, passband ripple
3. **Chebyshev II** - Steep, stopband ripple
4. **Butterworth** - Moderate, no ripple
5. **Bessel** - Slowest, maximally linear phase

### Phase Linearity (best to worst)
1. **FIR** - Perfectly linear
2. **Bessel** - Best among IIR
3. **Butterworth** - Moderate
4. **Chebyshev II** - Poor
5. **Chebyshev I** - Worse
6. **Elliptic** - Worst

### Typical Use Cases

**Butterworth**
- General-purpose filtering
- When you need balanced performance
- ECG/EEG processing
- Biomedical signals

**Chebyshev**
- When sharp cutoff is more important than ripple
- Efficient filtering with lower order
- Anti-aliasing filters
- Telecommunications

**Elliptic**
- Minimum order for given specifications
- Computational efficiency critical
- Only magnitude matters
- Not suitable when phase is important

**Bessel**
- Pulse shaping
- Control systems
- Video processing
- When waveform shape must be preserved

**FIR**
- Audio processing
- Image processing
- Communications
- Any application requiring linear phase

**Median**
- Spike removal
- Salt-and-pepper noise
- Image noise reduction
- Outlier removal

**Savitzky-Golay**
- Spectroscopy
- Chromatography
- Peak detection
- Derivative computation

## 📝 Practical Examples

### Example 1: ECG Processing
```python
# Multi-stage ECG cleaning
ecg_clean = filters.butterworth_highpass(ecg_raw, cutoff=0.5, order=4)  # Remove baseline
ecg_clean = filters.notch_filter_iir(ecg_clean, notch_freq=60)  # Remove powerline
ecg_clean = filters.butterworth_lowpass(ecg_clean, cutoff=40, order=4)  # Smooth
```

### Example 2: Audio (Linear Phase Required)
```python
# Use FIR for audio to preserve phase
audio_clean = filters.fir_lowpass_windowed(
    audio, 
    cutoff=5000, 
    numtaps=201,  # Sharp cutoff
    window='blackman'  # Low ripple
)
```

### Example 3: Sensor with Spikes
```python
# Two-stage: median then smooth
sensor_clean = filters.median_filter(sensor_data, kernel_size=7)
sensor_clean = filters.savitzky_golay_filter(sensor_clean, window_length=11, polyorder=2)
```

### Example 4: Real-time Control
```python
# Low-order IIR for minimal latency
control_signal = filters.butterworth_lowpass(
    feedback, 
    cutoff=50, 
    order=2,  # Low order = low delay
    zero_phase=False  # MUST be causal
)
```

### Example 5: Spectroscopy
```python
# Preserve peaks with Savitzky-Golay
spectrum_smooth = filters.savitzky_golay_filter(
    spectrum, 
    window_length=15, 
    polyorder=2
)

# Get derivative for peak detection
derivative = filters.savitzky_golay_filter(
    spectrum_smooth, 
    window_length=15, 
    polyorder=2, 
    deriv=1
)
```

## 🎨 Generated Visualizations

Run the demo scripts to generate comprehensive visualizations:

```bash
python demo_filters.py
python practical_examples.py
```

This creates:
- `lowpass_comparison.png` - Compare different lowpass designs
- `filter_types.png` - All filter types on various signals  
- `phase_response.png` - Phase characteristics analysis
- `fir_vs_iir.png` - FIR vs IIR detailed comparison
- `noise_types.png` - Matching filters to noise types
- `filter_comparison.png` - Frequency response comparison
- `example_ecg.png` - ECG processing pipeline
- `example_audio.png` - Audio with FIR (linear phase)
- `example_sensor.png` - Spike removal with median filter
- `example_realtime.png` - Low-latency control filtering
- `example_spectroscopy.png` - Peak preservation with SavGol

## ⚠️ Common Pitfalls

### 1. Using filtfilt in Real-time Applications
```python
# ❌ WRONG - filtfilt is NOT causal!
filtered = filters.butterworth_lowpass(signal, cutoff=20, zero_phase=True)

# ✅ CORRECT - Use causal filtering
filtered = filters.butterworth_lowpass(signal, cutoff=20, zero_phase=False)
```

### 2. Too High Filter Order
```python
# ❌ WRONG - Can become unstable
filtered = filters.butterworth_lowpass(signal, cutoff=5, order=20)

# ✅ CORRECT - Use reasonable order
filtered = filters.butterworth_lowpass(signal, cutoff=5, order=4)
```

### 3. Narrow Bandpass with High Order
```python
# ❌ WRONG - Numerical issues likely
filtered = filters.butterworth_bandpass(signal, lowcut=49.9, highcut=50.1, order=8)

# ✅ CORRECT - Lower order or use notch filter instead
filtered = filters.notch_filter_iir(signal, notch_freq=50, quality_factor=30)
```

### 4. Using IIR When Phase Matters
```python
# ❌ WRONG - IIR will distort phase in audio
audio_clean = filters.butterworth_lowpass(audio, cutoff=5000, order=6)

# ✅ CORRECT - Use FIR for audio
audio_clean = filters.fir_lowpass_windowed(audio, cutoff=5000, numtaps=201)
```

## 🔧 Tips and Best Practices

1. **Start with Butterworth** - It's a good general-purpose choice
2. **Use filtfilt carefully** - Only for offline processing
3. **Keep IIR orders low** - Typically 2-6, rarely above 8
4. **FIR for audio** - Linear phase is critical
5. **Median for spikes** - Don't use linear filters for impulse noise
6. **Multi-stage for complex noise** - Median → Notch → Lowpass
7. **Test on real data** - Synthetic signals don't capture all issues
8. **Plot frequency response** - Use `compare_filter_responses()` method
9. **Check group delay** - Use `get_filter_delay()` for FIR filters
10. **Monitor edge effects** - Filters can cause artifacts at signal boundaries

## 📚 References

Each filter method includes detailed docstrings explaining:
- What it's good for
- Pros and cons
- Phase characteristics
- When to use vs avoid
- Typical applications
- Parameter guidance

Access these with:
```python
help(filters.butterworth_lowpass)
help(filters.median_filter)
# etc.
```

## 🎓 Learning Path

1. Start with `signal_filters.py` - Read the docstrings
2. Run `demo_filters.py` - See visual comparisons
3. Run `practical_examples.py` - See real-world applications
4. Experiment with your own data
5. Use the decision trees in the code comments

## 📊 Performance Notes

**IIR Filters**
- Time complexity: O(n) where n is signal length
- Memory: O(order)
- Typical order: 2-8

**FIR Filters**  
- Time complexity: O(n * numtaps)
- Memory: O(numtaps)
- Typical numtaps: 50-500

**Median Filter**
- Time complexity: O(n * kernel_size * log(kernel_size))
- Memory: O(kernel_size)
- Slower than linear filters

## 🤝 Contributing

Feel free to extend this library with:
- Additional filter types (Kalman, particle filters, etc.)
- More window functions
- Additional adaptive algorithms
- GPU acceleration
- Real-time streaming interfaces

## 📄 License

This code is provided as an educational resource. Use freely for learning and projects.

---

**Happy Filtering!** 🎵📊🔧
