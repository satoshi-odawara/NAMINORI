import numpy as np
from scipy.io import wavfile
from scipy import signal
import hashlib
from typing import Tuple, Optional

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
    """
    nyquist = 0.5 * fs_hz

    # Filter bypassing if None or 0
    is_hpf = highpass_hz is not None and highpass_hz > 0
    is_lpf = lowpass_hz is not None and lowpass_hz > 0

    if not is_hpf and not is_lpf:
        return data

    # Validation for Physical Validity
    if highpass_hz is not None and highpass_hz >= nyquist:
        raise ValueError(f"High-pass cutoff frequency ({highpass_hz} Hz) must be below Nyquist frequency ({nyquist} Hz).")
    if lowpass_hz is not None and lowpass_hz >= nyquist:
        raise ValueError(f"Low-pass cutoff frequency ({lowpass_hz} Hz) must be below Nyquist frequency ({nyquist} Hz).")

    if is_hpf and is_lpf:
        if highpass_hz >= lowpass_hz:
            raise ValueError("High-pass cutoff frequency must be less than low-pass cutoff frequency for band-pass filter.")
        freq_cutoff = [highpass_hz, lowpass_hz]
        btype = 'band'
    elif is_hpf:
        freq_cutoff = highpass_hz
        btype = 'high'
    else: # is_lpf
        freq_cutoff = lowpass_hz
        btype = 'low'

    sos = signal.butter(order, freq_cutoff, btype=btype, fs=fs_hz, output='sos')
    return signal.sosfilt(sos, data)
