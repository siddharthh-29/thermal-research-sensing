"""
Signal Processing Filter Library

Digital filters for physiological time-series processing, covering IIR and FIR
designs with notes on their frequency and phase characteristics.
"""

import numpy as np
from scipy import signal
from typing import Union, Tuple, Optional
import warnings


class SignalFilters:
    """
    A comprehensive collection of digital signal processing filters.
    
    This class implements various filtering techniques for time-series data,
    including IIR (Infinite Impulse Response) and FIR (Finite Impulse Response)
    filters, each with their own characteristics and applications.
    """
    
    def __init__(self, sampling_rate: float):
        """
        Initialize the filter class.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling frequency of the signal in Hz
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    # ============================================================================
    # IIR FILTERS (Infinite Impulse Response)
    # ============================================================================
    
    def butterworth_lowpass(self, data: np.ndarray, cutoff: float, order: int = 3,
                           zero_phase: bool = True) -> np.ndarray:
        """
        Butterworth lowpass filter - removes high-frequency components.
        
        WHAT IT'S GOOD FOR:
        - Smooth frequency response (maximally flat in passband)
        - General-purpose low-frequency signal extraction
        - Removing high-frequency noise
        - Audio processing, ECG/EEG filtering
        
        PROS:
        - Maximally flat frequency response in passband
        - No ripple in passband or stopband
        - Simple design, well-understood behavior
        - Good balance between steepness and phase response
        
        CONS:
        - Non-linear phase response (causes phase distortion)
        - Slower roll-off compared to Chebyshev or Elliptic
        - Can introduce ringing artifacts at sharp transitions
        
        PHASE CHARACTERISTICS:
        - Without zero_phase: Non-linear phase, causes group delay
        - With zero_phase (filtfilt): Zero phase distortion, but NOT causal
        
        Parameters:
        -----------
        data : np.ndarray
            Input signal
        cutoff : float
            Cutoff frequency in Hz
        order : int
            Filter order (higher = steeper roll-off, but more ringing)
        zero_phase : bool
            If True, uses filtfilt for zero phase distortion
            
        Returns:
        --------
        np.ndarray
            Filtered signal
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        
        if zero_phase:
            # Zero phase filtering - applies filter forwards and backwards
            # Doubles the effective filter order
            return signal.filtfilt(b, a, data)
        else:
            # Causal filtering - only uses past data (has group delay)
            return signal.lfilter(b, a, data)
    
    def butterworth_highpass(self, data: np.ndarray, cutoff: float, order: int = 4,
                            zero_phase: bool = True) -> np.ndarray:
        """
        Butterworth highpass filter - removes low-frequency components.
        
        WHAT IT'S GOOD FOR:
        - Removing DC offset and slow drifts
        - Baseline correction in biomedical signals
        - Removing low-frequency noise or trends
        - Detrending data
        
        PROS:
        - Smooth passband response
        - Effective DC removal
        - Preserves high-frequency signal content
        
        CONS:
        - Can amplify high-frequency noise
        - Non-linear phase (unless using filtfilt)
        - May cause edge effects at signal boundaries
        
        TYPICAL APPLICATIONS:
        - ECG baseline wander removal (0.5-1 Hz cutoff)
        - Audio high-pass filtering (rumble removal)
        - Removing measurement drift in sensors
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def butterworth_bandpass(self, data: np.ndarray, lowcut: float, highcut: float,
                            order: int = 4, zero_phase: bool = True) -> np.ndarray:
        """
        Butterworth bandpass filter - keeps frequencies in a specific range.
        
        WHAT IT'S GOOD FOR:
        - Extracting specific frequency bands (e.g., brain waves, voice)
        - Isolating oscillations or rhythms
        - Removing both high and low frequency noise
        
        PROS:
        - Flat passband response
        - Isolates frequency bands of interest
        - Versatile for many applications
        
        CONS:
        - Twice the phase distortion of single-sided filters
        - Can cause ringing near band edges
        - Narrow bands require higher orders (stability issues)
        
        TYPICAL APPLICATIONS:
        - EEG band extraction (alpha: 8-13 Hz, beta: 13-30 Hz)
        - Voice/music frequency isolation
        - Vibration analysis in specific frequency ranges
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def butterworth_bandstop(self, data: np.ndarray, lowcut: float, highcut: float,
                            order: int = 4, zero_phase: bool = True) -> np.ndarray:
        """
        Butterworth bandstop (notch) filter - removes a frequency band.
        
        WHAT IT'S GOOD FOR:
        - Removing powerline interference (50/60 Hz)
        - Eliminating specific interference frequencies
        - Removing narrow-band noise
        
        PROS:
        - Preserves signal outside the stopband
        - Effective for known interference frequencies
        - Minimal distortion to remaining signal
        
        CONS:
        - Can cause ringing if notch is too narrow
        - May not completely remove interference
        - Requires precise frequency knowledge
        
        TYPICAL APPLICATIONS:
        - 50/60 Hz powerline noise removal
        - Removing mechanical vibration frequencies
        - Eliminating carrier frequencies
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='bandstop')
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def chebyshev_type1_lowpass(self, data: np.ndarray, cutoff: float, 
                                ripple_db: float = 0.5, order: int = 4,
                                zero_phase: bool = True) -> np.ndarray:
        """
        Chebyshev Type I lowpass - steeper roll-off with passband ripple.
        
        WHAT IT'S GOOD FOR:
        - When you need sharper cutoff than Butterworth
        - Applications tolerating passband ripple
        - Efficient sharp filtering with lower order
        
        PROS:
        - Steeper roll-off than Butterworth for same order
        - More efficient (lower order needed)
        - Good for sharp frequency separation
        
        CONS:
        - Ripple in passband (magnitude variations)
        - More phase distortion than Butterworth
        - Less smooth frequency response
        - Can amplify frequencies near ripple peaks
        
        PHASE CHARACTERISTICS:
        - Worse phase linearity than Butterworth
        - More group delay variation
        - Not recommended when phase is critical
        
        Parameters:
        -----------
        ripple_db : float
            Maximum ripple allowed in passband (dB)
            Smaller values = less ripple but less steep roll-off
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.cheby1(order, ripple_db, normal_cutoff, btype='low')
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def chebyshev_type2_lowpass(self, data: np.ndarray, cutoff: float,
                                stopband_attenuation_db: float = 40, order: int = 4,
                                zero_phase: bool = True) -> np.ndarray:
        """
        Chebyshev Type II lowpass - steeper roll-off with stopband ripple.
        
        WHAT IT'S GOOD FOR:
        - When passband flatness is critical
        - Sharp cutoff without passband distortion
        - Better than Type I when signal integrity matters
        
        PROS:
        - Flat passband (no ripple in signal range)
        - Steeper roll-off than Butterworth
        - Better phase response than Type I
        
        CONS:
        - Ripple in stopband (less critical usually)
        - More complex design than Type I
        - Still has phase distortion
        
        TYPICAL APPLICATIONS:
        - Audio processing where passband quality matters
        - Biomedical signals requiring minimal distortion
        - Measurement systems with strict passband requirements
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.cheby2(order, stopband_attenuation_db, normal_cutoff, btype='low')
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def elliptic_lowpass(self, data: np.ndarray, cutoff: float,
                        passband_ripple_db: float = 0.5,
                        stopband_attenuation_db: float = 40,
                        order: int = 4, zero_phase: bool = True) -> np.ndarray:
        """
        Elliptic (Cauer) lowpass - steepest possible roll-off.
        
        WHAT IT'S GOOD FOR:
        - Minimum filter order for given specifications
        - Very sharp frequency transitions
        - Applications where order/complexity must be minimized
        
        PROS:
        - Steepest roll-off for given order (most efficient)
        - Smallest transition band possible
        - Lowest computational cost for sharp filtering
        
        CONS:
        - Ripple in BOTH passband AND stopband
        - Worst phase response of all IIR filters
        - Can cause significant signal distortion
        - Not suitable when phase linearity matters
        
        WHEN TO AVOID:
        - Audio applications (phase distortion audible)
        - When signal shape preservation is critical
        - Transient analysis (causes ringing)
        
        TYPICAL APPLICATIONS:
        - Anti-aliasing filters
        - Telecommunications (when only magnitude matters)
        - Data reduction where computational efficiency is critical
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.ellip(order, passband_ripple_db, stopband_attenuation_db,
                           normal_cutoff, btype='low')
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def bessel_lowpass(self, data: np.ndarray, cutoff: float, order: int = 4,
                      norm: str = 'phase', zero_phase: bool = False) -> np.ndarray:
        """
        Bessel lowpass - maximally linear phase response.
        
        WHAT IT'S GOOD FOR:
        - Preserving signal waveform shape
        - Pulse and step response applications
        - When phase linearity is MORE important than sharp cutoff
        
        PROS:
        - Best phase linearity of all IIR filters
        - Minimal overshoot and ringing
        - Excellent transient response
        - Preserves signal shape better than other IIR filters
        
        CONS:
        - Slowest roll-off (worst frequency selectivity)
        - Poor stopband attenuation
        - Requires higher orders for sharp filtering
        - Still not truly linear phase (use FIR for that)
        
        WHEN TO USE:
        - Video/image processing
        - Pulse shaping in communications
        - Control systems requiring good transient response
        - Any application where waveform shape matters
        
        NOTE: Bessel filters are often used WITHOUT zero_phase
              because their main advantage IS their good phase response.
              Using filtfilt negates this and makes them non-causal.
        
        Parameters:
        -----------
        norm : str
            'phase' for maximally flat group delay
            'mag' for maximally flat magnitude at DC
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        b, a = signal.bessel(order, normal_cutoff, btype='low', analog=False, norm=norm)
        
        if zero_phase:
            warnings.warn("Bessel filters are designed for good phase response. "
                        "Using filtfilt negates this advantage and makes filter non-causal.")
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    # ============================================================================
    # FIR FILTERS (Finite Impulse Response)
    # ============================================================================
    
    def fir_lowpass_windowed(self, data: np.ndarray, cutoff: float, 
                            numtaps: int = 101, window: str = 'hamming') -> np.ndarray:
        """
        FIR lowpass filter using window method.
        
        WHAT IT'S GOOD FOR:
        - When linear phase is CRITICAL
        - Audio processing (no phase distortion)
        - Image processing
        - Any application requiring exact phase relationships
        
        PROS:
        - ALWAYS stable (no feedback)
        - Exactly linear phase (symmetric delay)
        - No ringing or oscillation issues
        - Easy to design and implement
        - Can achieve arbitrary frequency responses
        
        CONS:
        - Requires MANY taps for sharp transitions (high computational cost)
        - Always introduces delay (group delay = (numtaps-1)/2 samples)
        - Larger memory requirements
        - Longer transient response
        
        DELAY CHARACTERISTICS:
        - Constant group delay across all frequencies
        - Delay = (numtaps - 1) / (2 * fs) seconds
        - For numtaps=101 and fs=1000Hz: delay = 50ms
        
        WINDOW COMPARISON:
        - 'hamming': Good general purpose, -53dB stopband
        - 'hann': Similar to Hamming, slightly wider transition
        - 'blackman': Better stopband (-74dB) but wider transition
        - 'kaiser': Adjustable trade-off (see fir_lowpass_kaiser)
        
        Parameters:
        -----------
        numtaps : int
            Filter length (odd number recommended)
            Larger = sharper cutoff but more delay and computation
        window : str
            Window function ('hamming', 'hann', 'blackman', 'bartlett')
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        
        # Design FIR filter
        fir_coeff = signal.firwin(numtaps, normal_cutoff, window=window)
        
        # Apply filter (FIR filters use lfilter, not filtfilt)
        filtered = signal.lfilter(fir_coeff, 1.0, data)
        
        return filtered
    
    def fir_lowpass_kaiser(self, data: np.ndarray, cutoff: float,
                          width: float, ripple_db: float = 60) -> np.ndarray:
        """
        FIR lowpass using Kaiser window - optimal FIR design.
        
        WHAT IT'S GOOD FOR:
        - Best FIR performance for given specifications
        - When you can specify transition bandwidth and stopband
        - Optimal trade-off between filter length and performance
        
        PROS:
        - Optimal window (minimizes order for given specs)
        - Linear phase like all FIR filters
        - Predictable stopband attenuation
        - Good control over design parameters
        
        DESIGN PARAMETERS:
        - width: Transition bandwidth (Hz) - smaller = more taps needed
        - ripple_db: Stopband attenuation - larger = more taps needed
        
        RULE OF THUMB:
        - Number of taps ≈ (ripple_db * fs) / (22 * width)
        - For 60dB rejection, 1kHz sampling, 50Hz transition: ~273 taps
        
        Parameters:
        -----------
        width : float
            Transition bandwidth in Hz
            Smaller = sharper but more taps (higher delay/cost)
        ripple_db : float
            Stopband attenuation in dB (typically 40-80)
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        normal_width = width / nyq
        
        # Calculate optimal number of taps
        numtaps, beta = signal.kaiserord(ripple_db, normal_width)
        numtaps |= 1  # Make odd
        
        # Design filter
        fir_coeff = signal.firwin(numtaps, normal_cutoff, window=('kaiser', beta))
        
        filtered = signal.lfilter(fir_coeff, 1.0, data)
        
        print(f"Kaiser filter designed with {numtaps} taps")
        print(f"Group delay: {(numtaps-1)/(2*self.fs)*1000:.2f} ms")
        
        return filtered
    
    def fir_bandpass(self, data: np.ndarray, lowcut: float, highcut: float,
                    numtaps: int = 101, window: str = 'hamming') -> np.ndarray:
        """
        FIR bandpass filter - linear phase band isolation.
        
        WHAT IT'S GOOD FOR:
        - Extracting frequency bands with no phase distortion
        - Multi-band processing where phase relationships matter
        - Feature extraction in signals (e.g., EEG rhythms)
        
        PROS:
        - Linear phase (critical for band analysis)
        - Stable for any band configuration
        - Symmetric impulse response
        
        CONS:
        - Narrow bands require MANY taps
        - Significant delay for sharp transitions
        - High computational cost
        
        TYPICAL APPLICATIONS:
        - EEG/MEG band extraction (must preserve phase)
        - Audio equalizers
        - Vibration analysis
        - Feature extraction for machine learning
        """
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        
        fir_coeff = signal.firwin(numtaps, [low, high], pass_zero=False, window=window)
        filtered = signal.lfilter(fir_coeff, 1.0, data)
        
        return filtered
    
    def fir_highpass(self, data: np.ndarray, cutoff: float,
                    numtaps: int = 101, window: str = 'hamming') -> np.ndarray:
        """
        FIR highpass filter - linear phase high-frequency extraction.
        
        WHAT IT'S GOOD FOR:
        - DC removal with no phase shift
        - Detrending while preserving phase relationships
        - High-frequency feature extraction
        
        ADVANTAGES OVER IIR HIGHPASS:
        - No phase distortion
        - Always stable
        - Better for derivative/edge detection
        
        TYPICAL APPLICATIONS:
        - Image edge detection
        - Audio rumble removal
        - Baseline correction in spectroscopy
        """
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        
        fir_coeff = signal.firwin(numtaps, normal_cutoff, pass_zero=False, window=window)
        filtered = signal.lfilter(fir_coeff, 1.0, data)
        
        return filtered
    
    # ============================================================================
    # NON-LINEAR FILTERS
    # ============================================================================
    
    def median_filter(self, data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Median filter - non-linear edge-preserving filter.
        
        WHAT IT'S GOOD FOR:
        - Removing impulse noise (salt-and-pepper, spikes)
        - Preserving edges and sharp transitions
        - Outlier removal
        - When signal has sudden jumps that must be preserved
        
        PROS:
        - Excellent for impulse/spike noise
        - Preserves edges (doesn't blur like linear filters)
        - No ringing artifacts
        - Robust to outliers
        
        CONS:
        - Non-linear (changes signal characteristics)
        - Can remove narrow pulses/features
        - Slower than linear filters
        - Not suitable for Gaussian noise
        - Distorts frequency content
        
        WHEN TO USE:
        - Image processing (removes spikes, keeps edges)
        - Sensor data with occasional outliers
        - Biomedical signals with artifact spikes
        
        WHEN TO AVOID:
        - When linear phase is needed
        - Signals with important narrow pulses
        - When frequency domain properties matter
        
        Parameters:
        -----------
        kernel_size : int
            Window size (odd number recommended)
            Larger = more smoothing but can remove features
        """
        from scipy.ndimage import median_filter as scipy_median
        
        return scipy_median(data, size=kernel_size, mode='reflect')
    
    def savitzky_golay_filter(self, data: np.ndarray, window_length: int = 11,
                             polyorder: int = 3, deriv: int = 0) -> np.ndarray:
        """
        Savitzky-Golay filter - polynomial smoothing with optional derivatives.
        
        WHAT IT'S GOOD FOR:
        - Smoothing while preserving peak shapes
        - Computing derivatives of noisy signals
        - Spectroscopy and chromatography
        - When local polynomial structure should be preserved
        
        PROS:
        - Preserves features better than moving average
        - Can compute derivatives directly
        - Maintains peak positions and heights well
        - Good for slowly varying signals
        
        CONS:
        - Assumes local polynomial behavior
        - Edge effects at boundaries
        - Not suitable for highly oscillatory signals
        - Can introduce artifacts if window is too large
        
        DERIVATIVE CALCULATION:
        - deriv=0: Smoothing only
        - deriv=1: First derivative (velocity, slope)
        - deriv=2: Second derivative (acceleration, curvature)
        
        TYPICAL APPLICATIONS:
        - Spectral peak detection
        - Smoothing calibration curves
        - Computing velocity from position data
        - Chemical/biological signal processing
        
        Parameters:
        -----------
        window_length : int
            Must be odd and > polyorder
        polyorder : int
            Polynomial order (1-5 typical, 2-3 most common)
        deriv : int
            Derivative order to compute (0 = just smoothing)
        """
        filtered = signal.savgol_filter(data, window_length, polyorder, deriv=deriv)
        
        return filtered
    
    def moving_average(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Simple moving average filter - basic smoothing.
        
        WHAT IT'S GOOD FOR:
        - Quick and simple noise reduction
        - Trend extraction
        - When computational efficiency is critical
        - Exploratory data analysis
        
        PROS:
        - Extremely simple and fast
        - Easy to understand and implement
        - Low computational cost
        - Good for quick visualization
        
        CONS:
        - Poor frequency response (gradual roll-off)
        - Causes lag (delay = window_size/2)
        - Reduces sharp features
        - Not optimal for most applications
        
        BETTER ALTERNATIVES:
        - For smoothing: Savitzky-Golay or FIR filter
        - For trend: Butterworth lowpass
        - For noise: Appropriate filter for noise type
        
        WHEN TO USE ANYWAY:
        - Quick data exploration
        - When simplicity matters
        - Real-time applications with minimal lag requirement
        """
        # Use uniform convolution
        weights = np.ones(window_size) / window_size
        filtered = np.convolve(data, weights, mode='same')
        
        return filtered
    
    def exponential_smoothing(self, data: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """
        Exponential moving average - weighted smoothing with decay.
        
        WHAT IT'S GOOD FOR:
        - Real-time processing (single-pass, causal)
        - When recent data should be weighted more
        - Low-latency smoothing
        - Adaptive filtering
        
        PROS:
        - Only requires one coefficient (alpha)
        - Minimal memory (only stores last value)
        - Naturally causal (no future data needed)
        - Adapts quickly to changes when alpha is large
        
        CONS:
        - Not linear phase
        - Frequency response not well-controlled
        - Can't achieve sharp frequency cutoffs
        - Lags behind rapid changes
        
        ALPHA PARAMETER:
        - alpha near 0: More smoothing, more lag
        - alpha near 1: Less smoothing, less lag
        - alpha = 0.3: Typical default
        - Equivalent cutoff ≈ alpha * fs / (2π)
        
        TYPICAL APPLICATIONS:
        - Financial time series
        - Real-time sensor smoothing
        - Control systems
        - Adaptive signal tracking
        """
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        
        for i in range(1, len(data)):
            filtered[i] = alpha * data[i] + (1 - alpha) * filtered[i-1]
        
        return filtered
    
    # ============================================================================
    # SPECIALIZED FILTERS
    # ============================================================================
    
    def notch_filter_iir(self, data: np.ndarray, notch_freq: float,
                        quality_factor: float = 30, zero_phase: bool = True) -> np.ndarray:
        """
        IIR notch filter - removes a single frequency.
        
        WHAT IT'S GOOD FOR:
        - Powerline interference removal (50/60 Hz)
        - Removing single-frequency interference
        - When you know the exact interference frequency
        
        PROS:
        - Very efficient (low order)
        - Narrow notch doesn't affect nearby frequencies
        - Minimal computational cost
        
        CONS:
        - Must know exact interference frequency
        - Narrow notch may not catch frequency drift
        - Can cause ringing if Q is too high
        
        QUALITY FACTOR (Q):
        - Higher Q = narrower notch
        - Q = 30: Common for powerline noise
        - Q = 10: Wider notch, more robust to frequency drift
        - Bandwidth ≈ notch_freq / Q
        
        POWERLINE NOISE REMOVAL:
        - 60 Hz (US): notch_freq=60, Q=30
        - 50 Hz (EU): notch_freq=50, Q=30
        - Apply harmonics too: 120 Hz, 180 Hz...
        """
        nyq = 0.5 * self.fs
        freq = notch_freq / nyq
        
        b, a = signal.iirnotch(freq, quality_factor, self.fs)
        
        if zero_phase:
            return signal.filtfilt(b, a, data)
        else:
            return signal.lfilter(b, a, data)
    
    def adaptive_filter_lms(self, data: np.ndarray, reference: np.ndarray,
                           filter_length: int = 32, mu: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adaptive LMS (Least Mean Squares) filter.
        
        WHAT IT'S GOOD FOR:
        - Noise cancellation when you have a reference signal
        - Echo cancellation
        - Interference removal with correlated reference
        - System identification
        
        PROS:
        - Adapts to changing conditions
        - Works when interference characteristics change
        - No need to know noise properties in advance
        - Can track time-varying systems
        
        CONS:
        - Requires reference signal correlated with noise
        - Convergence depends on step size (mu)
        - Can be unstable if mu is too large
        - Computational cost increases with filter length
        
        HOW IT WORKS:
        - Uses reference signal to predict and remove interference
        - Continuously adjusts filter coefficients
        - Minimizes mean square error
        
        PARAMETERS:
        - mu (step size): Trade-off between speed and stability
          - Too small: Slow convergence
          - Too large: Instability, oscillation
          - Typical: 0.001 to 0.1
        - filter_length: Longer = more complex systems, but slower
        
        TYPICAL APPLICATIONS:
        - Active noise cancellation (ANC) headphones
        - Echo cancellation in telecom
        - Fetal ECG extraction (mother's ECG as reference)
        - Removing motion artifacts with accelerometer reference
        
        Returns:
        --------
        filtered : np.ndarray
            Noise-reduced signal
        error : np.ndarray
            Adaptation error (can show convergence)
        """
        n_samples = len(data)
        w = np.zeros(filter_length)  # Filter weights
        y = np.zeros(n_samples)  # Output
        e = np.zeros(n_samples)  # Error
        
        for n in range(filter_length, n_samples):
            x = reference[n:n-filter_length:-1]  # Reference window
            y[n] = np.dot(w, x)  # Filter output
            e[n] = data[n] - y[n]  # Error
            w = w + mu * e[n] * x  # Update weights
        
        filtered = data - y  # Remove predicted interference
        
        return filtered, e
    
    def wiener_filter(self, data: np.ndarray, noise_power: Optional[float] = None) -> np.ndarray:
        """
        Wiener filter - optimal filter for additive noise (frequency domain).
        
        WHAT IT'S GOOD FOR:
        - When you can estimate noise power spectrum
        - Optimal MMSE (Minimum Mean Square Error) filtering
        - Image deblurring and denoising
        - Speech enhancement
        
        PROS:
        - Theoretically optimal for Gaussian noise
        - Works in frequency domain (efficient for long signals)
        - Can handle signal-dependent noise
        
        CONS:
        - Requires noise power estimation
        - Assumes stationary signal and noise
        - Can cause spectral distortion
        - Not real-time (requires full signal)
        
        HOW IT WORKS:
        - Estimates signal and noise power spectra
        - Applies optimal gain in frequency domain
        - Gain = Signal_power / (Signal_power + Noise_power)
        
        TYPICAL APPLICATIONS:
        - Image restoration
        - Audio enhancement
        - Radar/sonar signal processing
        - Any application with additive Gaussian noise
        
        Parameters:
        -----------
        noise_power : float, optional
            If None, estimated from signal
            If known, provide for better results
        """
        # Compute FFT
        signal_fft = np.fft.rfft(data)
        power_spectrum = np.abs(signal_fft) ** 2
        
        if noise_power is None:
            # Estimate noise from high frequencies (simple method)
            noise_power = np.mean(power_spectrum[-len(power_spectrum)//4:])
        
        # Wiener gain
        wiener_gain = power_spectrum / (power_spectrum + noise_power)
        
        # Apply filter
        filtered_fft = signal_fft * wiener_gain
        filtered = np.fft.irfft(filtered_fft, n=len(data))
        
        return filtered
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_filter_delay(self, filter_order: int, filter_type: str = 'fir') -> float:
        """
        Calculate filter group delay.
        
        For FIR filters: delay = (filter_order - 1) / (2 * fs)
        For IIR filters: delay is frequency-dependent (non-linear phase)
        
        Returns delay in seconds.
        """
        if filter_type.lower() == 'fir':
            return (filter_order - 1) / (2 * self.fs)
        else:
            warnings.warn("IIR filters have frequency-dependent delay. "
                         "Use filtfilt for zero-phase, or analyze group delay curve.")
            return None
    
    def compare_filter_responses(self, cutoff: float, order: int = 4):
        """
        Compare frequency responses of different filter types.
        
        Useful for choosing the right filter for your application.
        """
        import matplotlib.pyplot as plt
        
        nyq = 0.5 * self.fs
        normal_cutoff = cutoff / nyq
        
        # Design filters
        b_butter, a_butter = signal.butter(order, normal_cutoff, btype='low')
        b_cheby1, a_cheby1 = signal.cheby1(order, 0.5, normal_cutoff, btype='low')
        b_cheby2, a_cheby2 = signal.cheby2(order, 40, normal_cutoff, btype='low')
        b_ellip, a_ellip = signal.ellip(order, 0.5, 40, normal_cutoff, btype='low')
        b_bessel, a_bessel = signal.bessel(order, normal_cutoff, btype='low', analog=False)
        
        # Compute frequency responses
        w_butter, h_butter = signal.freqz(b_butter, a_butter, worN=2000)
        w_cheby1, h_cheby1 = signal.freqz(b_cheby1, a_cheby1, worN=2000)
        w_cheby2, h_cheby2 = signal.freqz(b_cheby2, a_cheby2, worN=2000)
        w_ellip, h_ellip = signal.freqz(b_ellip, a_ellip, worN=2000)
        w_bessel, h_bessel = signal.freqz(b_bessel, a_bessel, worN=2000)
        
        # Plot magnitude
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(w_butter * self.fs / (2*np.pi), 20 * np.log10(abs(h_butter)), label='Butterworth')
        plt.plot(w_cheby1 * self.fs / (2*np.pi), 20 * np.log10(abs(h_cheby1)), label='Chebyshev I')
        plt.plot(w_cheby2 * self.fs / (2*np.pi), 20 * np.log10(abs(h_cheby2)), label='Chebyshev II')
        plt.plot(w_ellip * self.fs / (2*np.pi), 20 * np.log10(abs(h_ellip)), label='Elliptic')
        plt.plot(w_bessel * self.fs / (2*np.pi), 20 * np.log10(abs(h_bessel)), label='Bessel')
        plt.axvline(cutoff, color='k', linestyle='--', alpha=0.3, label='Cutoff')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.title('Filter Magnitude Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-100, 5)
        
        # Plot phase
        plt.subplot(2, 1, 2)
        plt.plot(w_butter * self.fs / (2*np.pi), np.unwrap(np.angle(h_butter)), label='Butterworth')
        plt.plot(w_cheby1 * self.fs / (2*np.pi), np.unwrap(np.angle(h_cheby1)), label='Chebyshev I')
        plt.plot(w_cheby2 * self.fs / (2*np.pi), np.unwrap(np.angle(h_cheby2)), label='Chebyshev II')
        plt.plot(w_ellip * self.fs / (2*np.pi), np.unwrap(np.angle(h_ellip)), label='Elliptic')
        plt.plot(w_bessel * self.fs / (2*np.pi), np.unwrap(np.angle(h_bessel)), label='Bessel')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.title('Filter Phase Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('filter_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Filter comparison plot saved!")
        print("\nKey observations:")
        print("- Elliptic has steepest roll-off but worst phase")
        print("- Bessel has most linear phase but slowest roll-off")
        print("- Butterworth is middle ground (good general purpose)")
        print("- Chebyshev I has ripple in passband")
        print("- Chebyshev II has flat passband, ripple in stopband")


# ============================================================================
# EXAMPLE USAGE AND DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    """
    Demonstration of different filters and their characteristics.
    """
    
    # Create test signal: clean signal + noise
    fs = 1000  # 1 kHz sampling rate
    t = np.linspace(0, 2, 2*fs, endpoint=False)
    
    # Signal: combination of 5 Hz and 50 Hz
    clean_signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)
    
    # Add noise
    noise = np.random.normal(0, 0.2, len(t))
    noisy_signal = clean_signal + noise
    
    # Add some spikes (impulse noise)
    spike_indices = np.random.choice(len(t), 20)
    noisy_signal[spike_indices] += np.random.randn(20) * 2
    
    # Initialize filter class
    filters = SignalFilters(sampling_rate=fs)
    
    # Demonstrate different filters
    print("=" * 70)
    print("SIGNAL PROCESSING FILTER DEMONSTRATIONS")
    print("=" * 70)
    
    # 1. Lowpass filtering
    print("\n1. LOWPASS FILTERS (removing high-frequency noise)")
    print("-" * 70)
    
    butter_lp = filters.butterworth_lowpass(noisy_signal, cutoff=20, order=4)
    print("✓ Butterworth lowpass: Smooth, no ripple, moderate roll-off")
    
    cheby1_lp = filters.chebyshev_type1_lowpass(noisy_signal, cutoff=20, order=4)
    print("✓ Chebyshev I: Steeper roll-off, but has passband ripple")
    
    bessel_lp = filters.bessel_lowpass(noisy_signal, cutoff=20, order=4)
    print("✓ Bessel: Best phase response, preserves waveform shape")
    
    fir_lp = filters.fir_lowpass_windowed(noisy_signal, cutoff=20, numtaps=101)
    print("✓ FIR: Linear phase, but introduces delay and requires more taps")
    
    # 2. Highpass filtering
    print("\n2. HIGHPASS FILTER (removing DC and low frequencies)")
    print("-" * 70)
    
    # Add a DC offset and trend
    signal_with_drift = noisy_signal + 2 + 0.5*t
    butter_hp = filters.butterworth_highpass(signal_with_drift, cutoff=3, order=4)
    print("✓ Removed DC offset and slow drift")
    
    # 3. Bandpass filtering
    print("\n3. BANDPASS FILTER (isolating specific frequency range)")
    print("-" * 70)
    
    butter_bp = filters.butterworth_bandpass(noisy_signal, lowcut=45, highcut=55, order=4)
    print("✓ Isolated 50 Hz component (e.g., powerline interference detection)")
    
    # 4. Notch filtering
    print("\n4. NOTCH FILTER (removing 50 Hz powerline interference)")
    print("-" * 70)
    
    notch = filters.notch_filter_iir(noisy_signal, notch_freq=50, quality_factor=30)
    print("✓ Removed 50 Hz interference while preserving other frequencies")
    
    # 5. Median filtering
    print("\n5. MEDIAN FILTER (removing impulse noise/spikes)")
    print("-" * 70)
    
    median_filt = filters.median_filter(noisy_signal, kernel_size=5)
    print("✓ Removed spikes while preserving edges and transitions")
    
    # 6. Savitzky-Golay
    print("\n6. SAVITZKY-GOLAY FILTER (smoothing with feature preservation)")
    print("-" * 70)
    
    savgol = filters.savitzky_golay_filter(noisy_signal, window_length=21, polyorder=3)
    print("✓ Smoothed while preserving peak shapes")
    
    # Calculate derivative
    derivative = filters.savitzky_golay_filter(clean_signal, window_length=21, 
                                               polyorder=3, deriv=1)
    print("✓ Computed first derivative of signal")
    
    # 7. Filter comparison
    print("\n7. GENERATING FILTER COMPARISON PLOT")
    print("-" * 70)
    filters.compare_filter_responses(cutoff=100, order=4)
    
    print("\n" + "=" * 70)
    print("FILTER SELECTION GUIDE")
    print("=" * 70)
    print("""
    WHEN TO USE EACH FILTER TYPE:
    
    IIR FILTERS (Butterworth, Chebyshev, Elliptic, Bessel):
    ✓ When computational efficiency is important (low order)
    ✓ Real-time applications with limited resources
    ✓ When some phase distortion is acceptable
    ✗ Avoid when phase linearity is critical
    
    FIR FILTERS:
    ✓ When linear phase is REQUIRED (audio, image processing)
    ✓ When stability is critical (always stable)
    ✓ When arbitrary frequency responses are needed
    ✗ Avoid when computational resources are very limited
    ✗ Introduces delay proportional to filter length
    
    MEDIAN FILTER:
    ✓ Impulse noise, spikes, salt-and-pepper noise
    ✓ When edges must be preserved
    ✗ Not suitable for Gaussian noise
    ✗ Non-linear (changes signal characteristics)
    
    SAVITZKY-GOLAY:
    ✓ Smoothing spectra, chromatograms
    ✓ Computing derivatives of noisy signals
    ✓ When peak positions/heights must be preserved
    ✗ Not for highly oscillatory signals
    
    ADAPTIVE FILTERS:
    ✓ When you have a reference signal
    ✓ Echo/noise cancellation
    ✓ Time-varying interference
    ✗ Requires good reference signal
    
    QUICK DECISION TREE:
    1. Is linear phase critical? → Use FIR
    2. Is it impulse noise? → Use Median
    3. Need sharp cutoff with min resources? → Use Elliptic IIR
    4. Need to preserve waveform shape? → Use Bessel IIR
    5. General purpose filtering? → Use Butterworth IIR
    6. Computing derivatives? → Use Savitzky-Golay
    7. Have reference signal for noise? → Use Adaptive (LMS)
    """)
    
    print("\n" + "=" * 70)
    print("For detailed information on each filter, see the docstrings!")
    print("=" * 70)
