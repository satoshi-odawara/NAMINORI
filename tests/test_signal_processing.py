import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib
from scipy import signal # Added for synthetic signal generation

from src.core import signal_processing
from src.core.plugins import PluginParameter
from src.plugins.noise_reduction.notch_filter import NotchFilterPlugin
from src.plugins.noise_reduction.band_stop_filter import BandStopFilterPlugin


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


# --- Tests for Plugin-based Noise Reduction Filters ---

# Fixture for a dummy signal to be used by plugins
@pytest.fixture
def dummy_nr_signal():
    fs = 1000
    duration = 1.0
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    # Signal with components at 50Hz, 150Hz, 250Hz, plus 60Hz hum
    data = (
        1.0 * np.sin(2 * np.pi * 50 * t) +       # To be preserved
        0.5 * np.sin(2 * np.pi * 60 * t) +       # Notch target
        1.0 * np.sin(2 * np.pi * 150 * t) +      # Band-stop target
        0.7 * np.sin(2 * np.pi * 250 * t)        # To be preserved
    )
    return data, fs

# --- NotchFilterPlugin Tests ---
def test_notch_filter_plugin_metadata():
    plugin = NotchFilterPlugin()
    assert plugin.get_name() == "notch_filter"
    assert plugin.get_display_name() == "Notch Filter"
    params = plugin.get_parameters()
    assert len(params) == 2
    assert params[0].name == "freq_hz"
    assert params[1].name == "q_factor"

def test_notch_filter_plugin_process_functionality(dummy_nr_signal):
    data, fs = dummy_nr_signal
    plugin = NotchFilterPlugin()
    notch_freq = 60.0
    q_factor = 30.0
    
    filtered_data = plugin.process(data, fs, freq_hz=notch_freq, q_factor=q_factor)
    
    fft_original = np.fft.fft(data)
    fft_filtered = np.fft.fft(filtered_data)
    freqs = np.fft.fftfreq(len(data), d=1/fs)
    
    idx_notch = np.argmin(np.abs(freqs - notch_freq))
    idx_50hz = np.argmin(np.abs(freqs - 50.0))
    idx_150hz = np.argmin(np.abs(freqs - 150.0))
    
    # Notch frequency should be significantly reduced
    assert np.abs(fft_filtered[idx_notch]) < 0.5 * np.abs(fft_original[idx_notch])
    
    # Other frequencies should be preserved
    np.testing.assert_allclose(np.abs(fft_filtered[idx_50hz]), np.abs(fft_original[idx_50hz]), rtol=0.1)
    np.testing.assert_allclose(np.abs(fft_filtered[idx_150hz]), np.abs(fft_original[idx_150hz]), rtol=0.1)

def test_notch_filter_plugin_process_invalid_params(dummy_nr_signal):
    data, fs = dummy_nr_signal
    plugin = NotchFilterPlugin()
    nyquist = fs / 2
    
    with pytest.raises(ValueError, match="Notch frequency .* must be between 0 and Nyquist frequency"):
        plugin.process(data, fs, freq_hz=nyquist, q_factor=30.0)
    with pytest.raises(ValueError, match="Notch Q-factor must be positive"):
        plugin.process(data, fs, freq_hz=60.0, q_factor=0.0)

# --- BandStopFilterPlugin Tests ---
def test_band_stop_filter_plugin_metadata():
    plugin = BandStopFilterPlugin()
    assert plugin.get_name() == "band_stop_filter"
    assert plugin.get_display_name() == "Band-Stop Filter"
    params = plugin.get_parameters()
    assert len(params) == 3
    assert params[0].name == "low_hz"
    assert params[1].name == "high_hz"
    assert params[2].name == "order"

def test_band_stop_filter_plugin_process_functionality(dummy_nr_signal):
    data, fs = dummy_nr_signal
    plugin = BandStopFilterPlugin()
    low_hz = 100.0
    high_hz = 200.0
    order = 8
    
    filtered_data = plugin.process(data, fs, low_hz=low_hz, high_hz=high_hz, order=order)
    
    fft_original = np.fft.fft(data)
    fft_filtered = np.fft.fft(filtered_data)
    freqs = np.fft.fftfreq(len(data), d=1/fs)
    
    idx_50hz = np.argmin(np.abs(freqs - 50.0))
    idx_150hz = np.argmin(np.abs(freqs - 150.0))
    idx_250hz = np.argmin(np.abs(freqs - 250.0))
    
    # Frequency within band-stop should be significantly reduced
    assert np.abs(fft_filtered[idx_150hz]) < 0.1 * np.abs(fft_original[idx_150hz])
    
    # Frequencies outside band-stop should be preserved
    np.testing.assert_allclose(np.abs(fft_filtered[idx_50hz]), np.abs(fft_original[idx_50hz]), rtol=0.1)
    np.testing.assert_allclose(np.abs(fft_filtered[idx_250hz]), np.abs(fft_original[idx_250hz]), rtol=0.1)

def test_band_stop_filter_plugin_process_invalid_params(dummy_nr_signal):
    data, fs = dummy_nr_signal
    plugin = BandStopFilterPlugin()
    nyquist = fs / 2
    
    with pytest.raises(ValueError, match="Band-stop frequencies .* must be between 0 and Nyquist"):
        plugin.process(data, fs, low_hz=100.0, high_hz=nyquist, order=4) # high_hz >= Nyquist
    with pytest.raises(ValueError, match="Band-stop frequencies .* must be between 0 and Nyquist"):
        plugin.process(data, fs, low_hz=150.0, high_hz=100.0, order=4) # low_hz >= high_hz
