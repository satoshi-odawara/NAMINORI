import numpy as np
from scipy.io import wavfile
from scipy import signal
import hashlib
from typing import Tuple, Optional
from src.core.models import NoiseReductionFilterType # Added import

def load_wav_file(file_path: str) -> Tuple[int, np.ndarray, str]:
    """
    Reads a WAV file, normalizes the data, and calculates its SHA256 hash.

    Args:
        file_path: Path to the WAV file.

    Returns:
        Tuple[int, np.ndarray, str]:
            - fs_hz: Sampling frequency in Hz.
            - data_normalized: Normalized audio data (float array between -1.0 and 1.0).
            - file_hash: SHA256 hash of the file content.
    """
    with open(file_path, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    fs_hz, data_raw = wavfile.read(file_path)

    if data_raw.dtype == np.int16:
        data_normalized = data_raw / 32768.0
    elif data_raw.dtype == np.int32:
        data_normalized = data_raw / 2147483648.0
    elif data_raw.dtype == np.float32 or data_raw.dtype == np.float64:
        data_normalized = data_raw
    else:
        # Handle other integer types or raise an error
        # For simplicity, convert to float and assume maximum possible value for normalization
        if np.issubdtype(data_raw.dtype, np.integer):
            max_val = np.iinfo(data_raw.dtype).max
            data_normalized = data_raw / max_val if max_val != 0 else data_raw
        else:
            raise TypeError(f"Unsupported WAV data type: {data_raw.dtype}")
    
    return fs_hz, data_normalized.astype(np.float64), file_hash

def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Removes the DC (direct current) component from a signal.

    Args:
        data: Input signal (NumPy array).

    Returns:
        np.ndarray: Signal with DC component removed.
    """
    return data - np.mean(data)

def apply_butterworth_filter(
    data: np.ndarray,
    fs_hz: int,
    highpass_hz: Optional[float] = None,
    lowpass_hz: Optional[float] = None,
    order: int = 4
) -> np.ndarray:
    """
    Applies a Butterworth filter (high-pass, low-pass, or band-pass) to a signal.

    Args:
        data: Input signal (NumPy array).
        fs_hz: Sampling frequency in Hz.
        highpass_hz: Cutoff frequency for high-pass filter (Hz). If None, no HPF is applied.
        lowpass_hz: Cutoff frequency for low-pass filter (Hz). If None, no LPF is applied.
        order: Filter order.

    Returns:
        np.ndarray: Filtered signal.

    Raises:
        ValueError: If both highpass_hz and lowpass_hz are None, or if cutoff frequencies are invalid.
    """
    nyquist = 0.5 * fs_hz

    if highpass_hz is None and lowpass_hz is None:
        return data  # No filter specified

    if highpass_hz is not None and highpass_hz >= nyquist:
        raise ValueError(f"High-pass cutoff frequency ({highpass_hz} Hz) must be below Nyquist frequency ({nyquist} Hz).")
    if lowpass_hz is not None and lowpass_hz >= nyquist:
        raise ValueError(f"Low-pass cutoff frequency ({lowpass_hz} Hz) must be below Nyquist frequency ({nyquist} Hz).")
    if highpass_hz is not None and lowpass_hz is not None and highpass_hz >= lowpass_hz:
        raise ValueError("High-pass cutoff frequency must be less than low-pass cutoff frequency for band-pass filter.")

    if highpass_hz is not None and lowpass_hz is not None:
        # Band-pass filter
        freq_cutoff = [highpass_hz, lowpass_hz] # Use absolute frequencies
        btype = 'band'
    elif highpass_hz is not None:
        # High-pass filter
        freq_cutoff = highpass_hz # Use absolute frequency
        btype = 'high'
    else:
        # Low-pass filter
        freq_cutoff = lowpass_hz # Use absolute frequency
        btype = 'low'

    sos = signal.butter(order, freq_cutoff, btype=btype, fs=fs_hz, output='sos')
    return signal.sosfilt(sos, data)


def apply_noise_reduction_filter(
    data: np.ndarray,
    fs_hz: int,
    filter_type: NoiseReductionFilterType,
    notch_freq_hz: Optional[float] = None,
    notch_q_factor: Optional[float] = None,
    band_stop_low_hz: Optional[float] = None, # Added for BSF
    band_stop_high_hz: Optional[float] = None, # Added for BSF
    band_stop_order: int = 4 # Added for BSF
) -> np.ndarray:
    """
    Applies a noise reduction filter to the signal based on the specified type.

    Args:
        data: Input signal (NumPy array).
        fs_hz: Sampling frequency in Hz.
        filter_type: Type of noise reduction filter to apply.
        notch_freq_hz: Center frequency for the notch filter (Hz). Required if filter_type is NOTCH.
        notch_q_factor: Q-factor for the notch filter. Required if filter_type is NOTCH.
        band_stop_low_hz: Lower cutoff frequency for the band-stop filter (Hz). Required if filter_type is BAND_STOP.
        band_stop_high_hz: Upper cutoff frequency for the band-stop filter (Hz). Required if filter_type is BAND_STOP.
        band_stop_order: Order for the band-stop filter. Required if filter_type is BAND_STOP.

    Returns:
        np.ndarray: Filtered signal.

    Raises:
        ValueError: If required parameters for a specific filter type are missing or invalid.
    """
    if filter_type == NoiseReductionFilterType.NONE:
        return data

    elif filter_type == NoiseReductionFilterType.NOTCH:
        if notch_freq_hz is None or notch_q_factor is None:
            raise ValueError("notch_freq_hz and notch_q_factor are required for NOTCH filter.")
        
        nyquist = 0.5 * fs_hz
        if not (0 < notch_freq_hz < nyquist):
            raise ValueError(f"Notch frequency ({notch_freq_hz} Hz) must be between 0 and Nyquist frequency ({nyquist} Hz).")
        if notch_q_factor <= 0:
            raise ValueError("Notch Q-factor must be positive.")

        # Design notch filter
        b, a = signal.iirnotch(notch_freq_hz, notch_q_factor, fs_hz)
        # Apply filter forward and backward to avoid phase distortion
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data
    
    elif filter_type == NoiseReductionFilterType.BAND_STOP:
        if band_stop_low_hz is None or band_stop_high_hz is None:
            raise ValueError("band_stop_low_hz and band_stop_high_hz are required for BAND_STOP filter.")
        if not (0 < band_stop_low_hz < band_stop_high_hz < fs_hz / 2):
            raise ValueError(f"Band-stop frequencies ({band_stop_low_hz}-{band_stop_high_hz} Hz) must be between 0 and Nyquist frequency ({fs_hz / 2} Hz) and low_hz < high_hz.")

        # Design Butterworth band-stop filter
        sos = signal.butter(band_stop_order, [band_stop_low_hz, band_stop_high_hz], btype='bandstop', fs=fs_hz, output='sos')
        filtered_data = signal.sosfilt(sos, data)
        return filtered_data
    
    else:
        raise ValueError(f"Unsupported noise reduction filter type: {filter_type}")
