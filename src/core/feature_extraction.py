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

def calculate_fft_features(
    data: np.ndarray,
    fs_hz: int,
    window_type: WindowFunction
) -> Tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """
    Performs FFT and calculates frequency-domain features including power band contributions.

    Args:
        data: Input signal (NumPy array).
        fs_hz: Sampling frequency in Hz.
        window_type: Type of window function to apply (Hanning or Flat Top).

    Returns:
        Tuple[np.ndarray, np.ndarray, dict[str, float]]:
            - freq_hz: Frequency axis (Hz).
            - magnitude: FFT magnitude with amplitude correction.
            - power_bands: Dictionary containing power contributions of frequency bands.
    """
    if window_type == WindowFunction.HANNING:
        window = np.hanning(len(data))
        # This factor compensates for energy loss from the Hanning window.
        # For a pure sine wave with amplitude A, the FFT peak will be approx. A.
        amp_correction_factor = 2.0
    else:  # FLATTOP
        window = signal.windows.flattop(len(data))
        # The Flat Top window has a wider peak but is more accurate for amplitude measurements.
        # This specific factor is used to get the correct amplitude for a sine wave.
        amp_correction_factor = 4.18

    data_windowed = data * window
    fft_result = np.fft.rfft(data_windowed)
    freq_hz = np.fft.rfftfreq(len(data_windowed), d=1/fs_hz)
    magnitude = np.abs(fft_result) * amp_correction_factor / len(data_windowed)

    total_power = np.sum(magnitude**2)

    if total_power > 0:
        power_low = np.sum(magnitude[freq_hz < 1000]**2) / total_power
        power_mid = np.sum(magnitude[(freq_hz >= 1000) & (freq_hz < 5000)]**2) / total_power
        power_high = np.sum(magnitude[freq_hz >= 5000]**2) / total_power
    else:
        power_low, power_mid, power_high = 0.0, 0.0, 0.0

    power_bands = {
        'low': power_low,
        'mid': power_mid,
        'high': power_high
    }

    return freq_hz, magnitude, power_bands
