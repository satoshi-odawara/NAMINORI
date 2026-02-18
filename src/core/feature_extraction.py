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
        return {'centroid': 0.0, 'spread': 0.0, 'entropy': 0.0}
    
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
    
    # Calculate spectral shape features
    spectral_shape_features = calculate_spectral_features(freq_hz, magnitude)
    
    # Combine all frequency-domain features
    all_features = {**power_bands, **spectral_shape_features}

    return freq_hz, magnitude, all_features
