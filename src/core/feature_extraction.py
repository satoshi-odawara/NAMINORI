import numpy as np
from scipy import signal
from typing import Tuple

from src.core.models import TimeDomainFeatures, WindowFunction

def calculate_time_domain_features(data: np.ndarray) -> TimeDomainFeatures:
    """
    Calculates time-domain features from a signal.

    Args:
        data: Input signal (NumPy array).

    Returns:
        TimeDomainFeatures: Object containing calculated time-domain features.
    """
    rms = np.sqrt(np.mean(data**2))
    peak = np.max(np.abs(data))

    # Avoid division by zero if rms or mean(abs(data)) is zero
    kurtosis = np.mean((data / rms)**4) - 3 if rms > 0 else 0
    skewness = np.mean((data / rms)**3) if rms > 0 else 0
    crest_factor = peak / rms if rms > 0 else 0
    shape_factor = rms / np.mean(np.abs(data)) if np.mean(np.abs(data)) > 0 else 0

    return TimeDomainFeatures(
        rms=rms,
        peak=peak,
        kurtosis=kurtosis,
        skewness=skewness,
        crest_factor=crest_factor,
        shape_factor=shape_factor,
    )

def calculate_spectral_features(freq_hz: np.ndarray, magnitude: np.ndarray) -> dict[str, float]:
    """
    Calculates spectral shape features from a magnitude spectrum.

    Args:
        freq_hz: Frequency axis (Hz).
        magnitude: FFT magnitude spectrum.

    Returns:
        A dictionary containing spectral centroid, spread, and entropy.
    """
    # Normalize the magnitude spectrum to be a probability distribution
    mag_sum = np.sum(magnitude)
    if mag_sum == 0:
        return {'spectral_centroid': 0.0, 'spectral_spread': 0.0, 'spectral_entropy': 0.0}
    
    prob_dist = magnitude / mag_sum

    # Spectral Centroid
    centroid = np.sum(freq_hz * prob_dist)
    
    # Spectral Spread
    spread = np.sqrt(np.sum(((freq_hz - centroid)**2) * prob_dist))
    
    # Spectral Entropy
    # Use a small epsilon to avoid log(0)
    epsilon = 1e-12
    entropy = -np.sum(prob_dist * np.log2(prob_dist + epsilon))
    
    return {
        'spectral_centroid': centroid,
        'spectral_spread': spread,
        'spectral_entropy': entropy,
    }


def calculate_spectrogram(
    data: np.ndarray,
    fs_hz: int,
    window_type: WindowFunction,
    nperseg: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the spectrogram of a signal.

    Args:
        data: Input signal (NumPy array).
        fs_hz: Sampling frequency in Hz.
        window_type: Type of window function to apply.
        nperseg: Length of each segment for STFT. Default is 512.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - f: Array of sample frequencies.
            - t: Array of segment times.
            - Sxx: Spectrogram of data (power spectral density).
    """
    if window_type == WindowFunction.HANNING:
        window = 'hann'
    else:  # FLATTOP
        window = 'flattop'

    # Ensure nperseg is not longer than the data itself
    nperseg_actual = min(len(data), nperseg)
    
    f, t, Sxx = signal.spectrogram(data, fs_hz, window=window, nperseg=nperseg_actual, noverlap=nperseg_actual // 2)
    
    return f, t, Sxx

def calculate_fft_features(
    data: np.ndarray,
    fs_hz: int,
    window_type: WindowFunction
) -> Tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Performs FFT and calculates frequency-domain features including power bands
    and spectral shape features.

    Args:
        data: Input signal (NumPy array).
        fs_hz: Sampling frequency in Hz.
        window_type: Type of window function to apply (Hanning or Flat Top).

    Returns:
        Tuple[np.ndarray, np.ndarray, dict[str, float]]:
            - freq_hz: Frequency axis (Hz).
            - magnitude: FFT magnitude with amplitude correction.
            - all_features: Dictionary containing power bands and spectral shape features.
    """
    if window_type == WindowFunction.HANNING:
        window = np.hanning(len(data))
    else:  # FLATTOP
        window = signal.windows.flattop(len(data))

    data_windowed = data * window
    fft_result = np.fft.rfft(data_windowed)
    freq_hz = np.fft.rfftfreq(len(data_windowed), d=1/fs_hz)

    # Correct amplitude scaling
    magnitude = np.abs(fft_result) / np.sum(window)
    magnitude[1:] *= 2

    total_power = np.sum(magnitude**2)

    if total_power > 0:
        power_low = np.sum(magnitude[freq_hz < 1000]**2) / total_power
        power_mid = np.sum(magnitude[(freq_hz >= 1000) & (freq_hz < 5000)]**2) / total_power
        power_high = np.sum(magnitude[freq_hz >= 5000]**2) / total_power
    else:
        power_low, power_mid, power_high = 0.0, 0.0, 0.0

    power_bands = {
        'power_low': power_low,
        'power_mid': power_mid,
        'power_high': power_high
    }
    
    # Calculate spectral centroid, spread, and entropy
    spectral_shape_features = calculate_spectral_features(freq_hz, magnitude)

    # Calculate FFT Overall Level (Energy-based)
    # Use Parseval's theorem with window energy correction to match time-domain RMS
    mag_sq = np.abs(fft_result)**2
    
    # Scale factors for single-sided FFT power
    # mag_sq[0] (DC) and mag_sq[-1] (Nyquist, if N is even) are not doubled
    power_factors = np.full_like(mag_sq, 2.0)
    power_factors[0] = 1.0
    if len(data) % 2 == 0:
        power_factors[-1] = 1.0
    
    scaled_power = mag_sq * power_factors
    normalization = len(data) * np.sum(window**2)
    
    # Total Overall
    overall_level = np.sqrt(np.sum(scaled_power) / normalization)
    
    # Band-specific Overall
    overall_low = np.sqrt(np.sum(scaled_power[freq_hz < 1000]) / normalization)
    overall_high = np.sqrt(np.sum(scaled_power[freq_hz >= 1000]) / normalization)
    
    spectral_shape_features["overall_level"] = overall_level
    spectral_shape_features["overall_low"] = overall_low
    spectral_shape_features["overall_high"] = overall_high

    # Combine all frequency-domain features
    all_features = {**power_bands, **spectral_shape_features}

    return freq_hz, magnitude, all_features

