import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib

from src.core import signal_processing

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
