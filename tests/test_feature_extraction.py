import pytest
import numpy as np
from src.core import feature_extraction
from src.core.models import VibrationFeatures, WindowFunction
from scipy import signal # Added for signal.square

def test_calculate_time_domain_features_sine_wave():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    amplitude = 1.0
    data = amplitude * np.sin(2 * np.pi * 50 * t) # 50 Hz sine wave

    features = feature_extraction.calculate_time_domain_features(data)

    # For a sine wave of amplitude A, RMS = A / sqrt(2), Peak = A
    np.testing.assert_allclose(features.rms, amplitude / np.sqrt(2), rtol=1e-3)
    np.testing.assert_allclose(features.peak, amplitude, rtol=1e-3)
    # Kurtosis for a sine wave is -1.5 (excess kurtosis is -1.5, normal is 3.0, so 1.5 - 3.0 = -1.5 for this definition)
    # Our definition of kurtosis is excess kurtosis
    np.testing.assert_allclose(features.kurtosis, -1.5, atol=0.1) # Allowing some tolerance due to discrete signal
    np.testing.assert_allclose(features.skewness, 0.0, atol=1e-3)
    np.testing.assert_allclose(features.crest_factor, np.sqrt(2), rtol=1e-3)
    np.testing.assert_allclose(features.shape_factor, (amplitude / np.sqrt(2)) / (2 * amplitude / np.pi), rtol=0.01) # Relax tolerance

def test_calculate_time_domain_features_square_wave():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    amplitude = 1.0
    data = amplitude * signal.square(2 * np.pi * 50 * t) # 50 Hz square wave

    features = feature_extraction.calculate_time_domain_features(data)

    # For an ideal square wave of amplitude A, RMS = A, Peak = A
    np.testing.assert_allclose(features.rms, amplitude, rtol=1e-3)
    np.testing.assert_allclose(features.peak, amplitude, rtol=1e-3)
    # Kurtosis for a square wave is 0 (excess kurtosis)
    np.testing.assert_allclose(features.kurtosis, -2.0, atol=0.5) # Square wave has kurtosis of 1, so 1 - 3 = -2
    np.testing.assert_allclose(features.skewness, 0.0, atol=0.05) # Relax tolerance
    np.testing.assert_allclose(features.crest_factor, 1.0, rtol=1e-3)
    np.testing.assert_allclose(features.shape_factor, 1.0, rtol=1e-3)

def test_calculate_time_domain_features_zero_signal():
    data = np.zeros(100)
    features = feature_extraction.calculate_time_domain_features(data)

    assert features.rms == 0.0
    assert features.peak == 0.0
    assert features.kurtosis == 0.0
    assert features.skewness == 0.0
    assert features.crest_factor == 0.0
    assert features.shape_factor == 0.0

def test_calculate_fft_features_sine_hanning():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    amplitude = 0.5
    frequency = 100
    data = amplitude * np.sin(2 * np.pi * frequency * t)

    freq_hz, magnitude, power_bands = feature_extraction.calculate_fft_features(data, fs, WindowFunction.HANNING)

    # The peak magnitude should be close to the original amplitude
    assert np.allclose(np.max(magnitude), amplitude, atol=0.05)
    # The frequency of the peak should be at the original frequency
    assert np.allclose(freq_hz[np.argmax(magnitude)], frequency, atol=1.0)

    # Check power band contributions
    assert power_bands['low'] > 0.9 and power_bands['mid'] < 0.1 and power_bands['high'] < 0.1

def test_calculate_fft_features_sine_flattop():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    amplitude = 0.5
    frequency = 100
    data = amplitude * np.sin(2 * np.pi * frequency * t)

    freq_hz, magnitude, power_bands = feature_extraction.calculate_fft_features(data, fs, WindowFunction.FLATTOP)

    # Flat Top window gives more accurate amplitude measurement
    assert np.allclose(np.max(magnitude), amplitude, atol=0.01)
    assert np.allclose(freq_hz[np.argmax(magnitude)], frequency, atol=1.0)
    # Check power band contributions
    assert power_bands['low'] > 0.9 and power_bands['mid'] < 0.1 and power_bands['high'] < 0.1

def test_calculate_fft_features_mixed_frequencies():
    fs = 15000  # Ensure Nyquist is high enough for the high frequency component
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    
    # Frequencies that fall into each band
    data_low = 0.5 * np.sin(2 * np.pi * 500 * t)    # Low band (< 1000)
    data_mid = 0.3 * np.sin(2 * np.pi * 2500 * t)   # Mid band (1000-5000)
    data_high = 0.2 * np.sin(2 * np.pi * 6000 * t)  # High band (>= 5000)
    
    data = data_low + data_mid + data_high

    _, _, power_bands = feature_extraction.calculate_fft_features(data, fs, WindowFunction.HANNING)
    
    # Check power band contributions for the mixed signal
    # Each band should have a significant portion of the power
    assert power_bands['low'] > 0.1
    assert power_bands['mid'] > 0.1
    assert power_bands['high'] > 0.1
    # The sum should be close to 1.0
    assert np.allclose(sum(power_bands.values()), 1.0)

def test_calculate_fft_features_zero_signal():
    fs = 1000
    data = np.zeros(100)
    
    freq_hz, magnitude, power_bands = feature_extraction.calculate_fft_features(data, fs, WindowFunction.HANNING)
    
    assert np.all(magnitude == 0.0)
    assert power_bands['low'] == 0.0
    assert power_bands['mid'] == 0.0
    assert power_bands['high'] == 0.0
