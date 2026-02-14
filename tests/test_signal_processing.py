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

# New fixture for generating a sine wave with optional noise for filter tests
@pytest.fixture
def sine_with_noise_signal():
    fs = 1000 # Hz
    duration = 5.0 # seconds
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    
    # Main signal at 100 Hz
    main_freq = 100
    main_signal = 1.0 * np.sin(2 * np.pi * main_freq * t)
    
    # Noise components: 50 Hz (low), 60 Hz (notch target), 200 Hz (high)
    noise_50hz = 0.2 * np.sin(2 * np.pi * 50 * t)
    noise_60hz = 0.5 * np.sin(2 * np.pi * 60 * t) # Target for notch filter
    noise_200hz = 0.3 * np.sin(2 * np.pi * 200 * t)

    data = main_signal + noise_50hz + noise_60hz + noise_200hz
    data_normalized = data / np.max(np.abs(data)) # Normalize to -1 to 1 range

    return data_normalized, fs, main_freq, 50, 60, 200 # data, fs, main_freq, low_noise_freq, notch_freq, high_noise_freq


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
def test_apply_noise_reduction_filter_none(sine_with_noise_signal):
    data, fs, *_ = sine_with_noise_signal
    filtered_data = signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NONE)
    np.testing.assert_allclose(filtered_data, data)

def test_apply_noise_reduction_filter_notch(sine_with_noise_signal):
    data, fs, main_freq, low_noise_freq, notch_freq, high_noise_freq = sine_with_noise_signal
    
    # Check FFT before filter
    # fft_orig = np.fft.fft(data)
    # freqs_orig = np.fft.fftfreq(len(data), d=1/fs)
    # idx_notch_orig = np.argmin(np.abs(freqs_orig - notch_freq))
    # orig_notch_magnitude = np.abs(fft_orig[idx_notch_orig])

    # Apply notch filter at 60 Hz
    q_factor = 30.0 # Relatively narrow band
    filtered_data = signal_processing.apply_noise_reduction_filter(
        data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=notch_freq, notch_q_factor=q_factor
    )

    # Check FFT after filter
    fft_filtered = np.fft.fft(filtered_data)
    freqs_filtered = np.fft.fftfreq(len(filtered_data), d=1/fs)
    
    # Find indices for notch frequency and main signal frequency
    idx_notch = np.argmin(np.abs(freqs_filtered - notch_freq))
    idx_main = np.argmin(np.abs(freqs_filtered - main_freq))
    
    # The magnitude at notch frequency should be significantly reduced
    # Compare with a baseline (e.g., magnitude of main signal)
    # A simple threshold check
    assert np.abs(fft_filtered[idx_notch]) < 0.1 * np.abs(fft_filtered[idx_main]) # Notch magnitude should be small fraction of main signal

    # Check that main signal and other noise frequencies are largely preserved
    idx_low_noise = np.argmin(np.abs(freqs_filtered - low_noise_freq))
    idx_high_noise = np.argmin(np.abs(freqs_filtered - high_noise_freq))
    
    # These should not be attenuated as much as the notch frequency
    assert np.abs(fft_filtered[idx_main]) > 0.5 # Main signal should be strong
    assert np.abs(fft_filtered[idx_low_noise]) > 0.05 # Low noise should remain somewhat
    assert np.abs(fft_filtered[idx_high_noise]) > 0.05 # High noise should remain somewhat


def test_apply_noise_reduction_filter_notch_invalid_params(sine_with_noise_signal):
    data, fs, *_ = sine_with_noise_signal
    nyquist = fs / 2

    # Missing notch_freq_hz
    with pytest.raises(ValueError, match="notch_freq_hz and notch_q_factor are required for NOTCH filter."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_q_factor=30.0)

    # Missing notch_q_factor
    with pytest.raises(ValueError, match="notch_freq_hz and notch_q_factor are required for NOTCH filter."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=60.0)
    
    # notch_freq_hz out of range (<=0)
    with pytest.raises(ValueError, match=rf"Notch frequency \(([-]?\d+\.\d+) Hz\) must be between 0 and Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=0.0, notch_q_factor=30.0)
    with pytest.raises(ValueError, match=rf"Notch frequency \(([-]?\d+\.\d+) Hz\) must be between 0 and Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=-10.0, notch_q_factor=30.0)
    
    # notch_freq_hz out of range (>= nyquist)
    with pytest.raises(ValueError, match=rf"Notch frequency \((\d+\.\d+) Hz\) must be between 0 and Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=nyquist, notch_q_factor=30.0)
    with pytest.raises(ValueError, match=rf"Notch frequency \((\d+\.\d+) Hz\) must be between 0 and Nyquist frequency \((\d+\.\d+) Hz\)\."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=nyquist + 10, notch_q_factor=30.0)

    # notch_q_factor invalid (<=0)
    with pytest.raises(ValueError, match="Notch Q-factor must be positive."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=60.0, notch_q_factor=0.0)
    with pytest.raises(ValueError, match="Notch Q-factor must be positive."):
        signal_processing.apply_noise_reduction_filter(data, fs, NoiseReductionFilterType.NOTCH, notch_freq_hz=60.0, notch_q_factor=-1.0)
