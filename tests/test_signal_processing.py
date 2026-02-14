import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib
from scipy import signal # Added for synthetic signal generation

from src.core import signal_processing
from src.core.models import NoiseReductionFilterType # Added for filter type

# Helper function to create a dummy WAV file for testing
@pytest.fixture
def dummy_wav_file(tmp_path):
    # Create a dummy WAV file with a simple sine wave
    fs = 44100  # sampling rate, Hz, must be integer
    duration = 1.0  # in seconds, may be float
    f = 440.0  # sine frequency, Hz, may be float
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)  # time vector
    amplitude = np.iinfo(np.int16).max * 0.5  # Half of max amplitude for int16
    data_int16 = (amplitude * np.sin(2. * np.pi * f * t)).astype(np.int16)

    test_file_path = tmp_path / "test_audio.wav"
    wavfile.write(test_file_path, fs, data_int16)
    return test_file_path, fs, data_int16

@pytest.fixture
def multi_tone_signal():
    fs = 1000  # Hz
    duration = 5.0  # seconds
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)

    # Signal components at 50 Hz, 150 Hz, and 250 Hz
    freq1, freq2, freq3 = 50, 150, 250
    signal1 = 1.0 * np.sin(2 * np.pi * freq1 * t)
    signal2 = 1.0 * np.sin(2 * np.pi * freq2 * t) # This one should be filtered
    signal3 = 1.0 * np.sin(2 * np.pi * freq3 * t)

    data = signal1 + signal2 + signal3
    data_normalized = data / np.max(np.abs(data))  # Normalize

    return data_normalized, fs, freq1, freq2, freq3


def test_load_wav_file(dummy_wav_file):
    file_path, expected_fs, expected_data_int16 = dummy_wav_file
    fs, data_normalized, file_hash = signal_processing.load_wav_file(str(file_path))

    assert fs == expected_fs
    assert isinstance(data_normalized, np.ndarray)
    assert data_normalized.dtype == np.float64
    assert np.all(data_normalized >= -1.0) and np.all(data_normalized <= 1.0)

    # Check normalization roughly matches expected (within some tolerance)
    expected_normalized = expected_data_int16 / 32768.0
    np.testing.assert_allclose(data_normalized, expected_normalized, atol=1e-5)

    # Verify file hash (read content directly for hash calculation)
    with open(file_path, 'rb') as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == expected_hash

def test_remove_dc_offset():
    # Test with a simple signal with a DC offset
    signal_with_dc = np.array([1, 2, 3, 4, 5], dtype=np.float64) + 10
    signal_no_dc = signal_processing.remove_dc_offset(signal_with_dc)
    
    # The mean of the DC-removed signal should be very close to zero
    np.testing.assert_allclose(np.mean(signal_no_dc), 0.0, atol=1e-9)
    # The shape and range of the signal (excluding DC) should be preserved
    np.testing.assert_allclose(signal_no_dc, np.array([-2, -1, 0, 1, 2], dtype=np.float64))

def test_apply_butterworth_filter_hpf():
    fs = 1000
    nyquist = fs / 2
    # Create a signal with low and high frequency components
    t = np.linspace(0, 1, fs, endpoint=False)
    low_freq_component = np.sin(2 * np.pi * 50 * t)   # 50 Hz
    high_freq_component = np.sin(2 * np.pi * 200 * t)  # 200 Hz
    data = low_freq_component + high_freq_component

    # Apply HPF at 100 Hz
    hpf_cutoff = 100
    filtered_data = signal_processing.apply_butterworth_filter(data, fs, highpass_hz=hpf_cutoff, lowpass_hz=None)

    # Check that low frequency component is attenuated and high frequency remains
    assert np.std(filtered_data) < np.std(data) # Should attenuate something
    assert np.std(filtered_data) > np.std(high_freq_component) * 0.8 # Should pass high freq significantly

    # Check edge case: HPF at Nyquist or above should raise ValueError
    with pytest.raises(ValueError, match=rf"High-pass cutoff frequency \((\d+\.\d+) Hz\) must be below Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_butterworth_filter(data, fs, highpass_hz=nyquist)
    with pytest.raises(ValueError, match=rf"High-pass cutoff frequency \((\d+\.\d+) Hz\) must be below Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_butterworth_filter(data, fs, highpass_hz=nyquist + 1)

def test_apply_butterworth_filter_lpf():
    fs = 1000
    nyquist = fs / 2
    # Create a signal with low and high frequency components
    t = np.linspace(0, 1, fs, endpoint=False)
    low_freq_component = np.sin(2 * np.pi * 50 * t)   # 50 Hz
    high_freq_component = np.sin(2 * np.pi * 200 * t)  # 200 Hz
    data = low_freq_component + high_freq_component

    # Apply LPF at 100 Hz
    lpf_cutoff = 100
    filtered_data = signal_processing.apply_butterworth_filter(data, fs, highpass_hz=None, lowpass_hz=lpf_cutoff)

    # Check that high frequency component is attenuated and low frequency remains
    assert np.std(filtered_data) < np.std(data) # Should attenuate something
    assert np.std(filtered_data) > np.std(low_freq_component) * 0.8 # Should pass low freq significantly

    # Check edge case: LPF at Nyquist or above should raise ValueError
    with pytest.raises(ValueError, match=rf"Low-pass cutoff frequency \((\d+\.\d+) Hz\) must be below Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_butterworth_filter(data, fs, lowpass_hz=nyquist)
    with pytest.raises(ValueError, match=rf"Low-pass cutoff frequency \((\d+\.\d+) Hz\) must be below Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_butterworth_filter(data, fs, lowpass_hz=nyquist + 1)

def test_apply_butterworth_filter_bpf():
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    low_freq = np.sin(2 * np.pi * 50 * t)
    mid_freq = np.sin(2 * np.pi * 150 * t)
    high_freq = np.sin(2 * np.pi * 300 * t)
    data = low_freq + mid_freq + high_freq

    # Apply BPF between 100 Hz and 200 Hz
    bpf_low = 100
    bpf_high = 200
    filtered_data = signal_processing.apply_butterworth_filter(data, fs, highpass_hz=bpf_low, lowpass_hz=bpf_high)

    # Expect mid_freq component to be dominant
    assert np.std(filtered_data) < np.std(data) # Should attenuate something
    assert np.std(filtered_data) > np.std(mid_freq) * 0.8 # Mid-freq should pass significantly

    # Check edge case: highpass_hz >= lowpass_hz for BPF
    with pytest.raises(ValueError, match="High-pass cutoff frequency must be less than low-pass cutoff frequency for band-pass filter."):
        signal_processing.apply_butterworth_filter(data, fs, highpass_hz=150, lowpass_hz=100)
    
    # Check no filter applied
    no_filter_data = signal_processing.apply_butterworth_filter(data, fs)
    np.testing.assert_allclose(no_filter_data, data)


# --- Tests for Noise Reduction Filters ---
def test_apply_noise_reduction_filter_none(multi_tone_signal):
    data, fs, *_ = multi_tone_signal
    filtered_data = signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NONE)
    np.testing.assert_allclose(filtered_data, data)

def test_apply_noise_reduction_filter_notch(multi_tone_signal):
    data, fs, freq1, notch_target_freq, freq3 = multi_tone_signal
    
    # Apply notch filter at the middle frequency (150 Hz)
    q_factor = 30.0
    filtered_data = signal_processing.apply_noise_reduction_filter(
        data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=notch_target_freq, notch_q_factor=q_factor
    )

    # Check FFT after filter
    fft_filtered = np.fft.fft(filtered_data)
    freqs_filtered = np.fft.fftfreq(len(filtered_data), d=1/fs)
    
    # Find indices for the three frequencies
    idx1 = np.argmin(np.abs(freqs_filtered - freq1))
    idx_notch = np.argmin(np.abs(freqs_filtered - notch_target_freq))
    idx3 = np.argmin(np.abs(freqs_filtered - freq3))

    # The magnitude at the notch frequency should be significantly reduced
    assert np.abs(fft_filtered[idx_notch]) < 0.1 * np.abs(fft_filtered[idx1])
    assert np.abs(fft_filtered[idx_notch]) < 0.1 * np.abs(fft_filtered[idx3])

    # The magnitudes at freq1 (50 Hz) and freq3 (250 Hz) should be largely preserved
    fft_original = np.fft.fft(data)
    orig_mag1 = np.abs(fft_original[np.argmin(np.abs(freqs_filtered - freq1))])
    orig_mag3 = np.abs(fft_original[np.argmin(np.abs(freqs_filtered - freq3))])

    assert np.isclose(np.abs(fft_filtered[idx1]), orig_mag1, rtol=0.1)
    assert np.isclose(np.abs(fft_filtered[idx3]), orig_mag3, rtol=0.1)


def test_apply_noise_reduction_filter_band_stop(multi_tone_signal):
    data, fs, freq1, freq2, freq3 = multi_tone_signal
    
    # Define band-stop filter from 100 Hz to 200 Hz to target freq2
    bs_low = 100.0
    bs_high = 200.0
    bs_order = 8

    # Apply band-stop filter
    filtered_data = signal_processing.apply_noise_reduction_filter(
        data, fs, NoiseReductionFilterType.BAND_STOP, 
        band_stop_low_hz=bs_low, band_stop_high_hz=bs_high, band_stop_order=bs_order
    )

    # Check FFT after filter
    fft_filtered = np.fft.fft(filtered_data)
    freqs_filtered = np.fft.fftfreq(len(filtered_data), d=1/fs)
    
    # Find indices for the three frequencies
    idx1 = np.argmin(np.abs(freqs_filtered - freq1))
    idx2 = np.argmin(np.abs(freqs_filtered - freq2))
    idx3 = np.argmin(np.abs(freqs_filtered - freq3))
    
    # The magnitude at freq2 (150 Hz) should be significantly reduced
    # Compare its magnitude to the magnitudes of the other two frequencies
    assert np.abs(fft_filtered[idx2]) < 0.1 * np.abs(fft_filtered[idx1])
    assert np.abs(fft_filtered[idx2]) < 0.1 * np.abs(fft_filtered[idx3])

    # The magnitudes at freq1 (50 Hz) and freq3 (250 Hz) should be largely preserved
    fft_original = np.fft.fft(data)
    orig_mag1 = np.abs(fft_original[np.argmin(np.abs(freqs_filtered - freq1))])
    orig_mag3 = np.abs(fft_original[np.argmin(np.abs(freqs_filtered - freq3))])

    assert np.isclose(np.abs(fft_filtered[idx1]), orig_mag1, rtol=0.1)
    assert np.isclose(np.abs(fft_filtered[idx3]), orig_mag3, rtol=0.1)

def test_apply_noise_reduction_filter_band_stop_invalid_params(multi_tone_signal):
    data, fs, *_ = multi_tone_signal
    nyquist = fs / 2

    # Missing band_stop_low_hz
    with pytest.raises(ValueError, match="band_stop_low_hz and band_stop_high_hz are required for BAND_STOP filter."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.BAND_STOP, band_stop_high_hz=150.0)

    # Missing band_stop_high_hz
    with pytest.raises(ValueError, match="band_stop_low_hz and band_stop_high_hz are required for BAND_STOP filter."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.BAND_STOP, band_stop_low_hz=100.0)

    # low_hz >= high_hz
    with pytest.raises(ValueError, match="must be between 0 and Nyquist frequency"):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.BAND_STOP, band_stop_low_hz=150.0, band_stop_high_hz=100.0)

    # Frequencies out of range
    with pytest.raises(ValueError, match="must be between 0 and Nyquist frequency"):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.BAND_STOP, band_stop_low_hz=-10.0, band_stop_high_hz=100.0)
    with pytest.raises(ValueError, match="must be between 0 and Nyquist frequency"):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.BAND_STOP, band_stop_low_hz=100.0, band_stop_high_hz=nyquist)
    with pytest.raises(ValueError, match="must be between 0 and Nyquist frequency"):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.BAND_STOP, band_stop_low_hz=100.0, band_stop_high_hz=nyquist + 1)
