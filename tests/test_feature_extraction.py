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
    # Kurtosis for a sine wave is -1.5
    np.testing.assert_allclose(features.kurtosis, -1.5, atol=0.1)
    np.testing.assert_allclose(features.skewness, 0.0, atol=1e-3)
    np.testing.assert_allclose(features.crest_factor, np.sqrt(2), rtol=1e-3)
    np.testing.assert_allclose(features.shape_factor, np.pi / (2 * np.sqrt(2)), rtol=0.01)

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
    # Kurtosis for a square wave is -2.0
    np.testing.assert_allclose(features.kurtosis, -2.0, atol=0.05)
    np.testing.assert_allclose(features.skewness, 0.0, atol=0.05)
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

class TestSpectralFeatures:
    def test_spectral_features_simple_case(self):
        """Test with a simple, predictable spectrum."""
        freqs = np.array([10, 20, 30, 40])
        mags = np.array([0, 1, 1, 0]) # Symmetrical spectrum centered at 25 Hz
        
        features = feature_extraction.calculate_spectral_features(freqs, mags)
        
        assert np.isclose(features['spectral_centroid'], 25.0)
        assert features['spectral_spread'] > 0
        assert features['spectral_entropy'] > 0

    def test_spectral_features_zero_spectrum(self):
        """Test with an all-zero magnitude spectrum."""
        freqs = np.array([10, 20, 30, 40])
        mags = np.zeros(4)
        
        features = feature_extraction.calculate_spectral_features(freqs, mags)
        
        assert features['spectral_centroid'] == 0.0
        assert features['spectral_spread'] == 0.0
        assert features['spectral_entropy'] == 0.0

    def test_spectral_features_pure_tone(self):
        """Test with a single frequency peak (delta function)."""
        freqs = np.array([10, 20, 30, 40])
        mags = np.array([0, 0, 1, 0]) # Single peak at 30 Hz
        
        features = feature_extraction.calculate_spectral_features(freqs, mags)
        
        assert np.isclose(features['spectral_centroid'], 30.0)
        assert np.isclose(features['spectral_spread'], 0.0)
        # Entropy of a single-point distribution is 0
        assert np.isclose(features['spectral_entropy'], 0.0)

def test_calculate_fft_features_sine_hanning():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    amplitude = 0.5
    frequency = 100
    data = amplitude * np.sin(2 * np.pi * frequency * t)

    freq_hz, magnitude, all_features = feature_extraction.calculate_fft_features(data, fs, WindowFunction.HANNING)

    assert np.allclose(np.max(magnitude), amplitude, atol=0.05)
    assert np.allclose(freq_hz[np.argmax(magnitude)], frequency, atol=1.0)
    
    assert 'power_low' in all_features
    assert 'spectral_centroid' in all_features
    assert all_features['power_low'] > 0.9
    assert np.isclose(all_features['spectral_centroid'], frequency, atol=2.0)

def test_calculate_fft_features_sine_flattop():
    fs = 1000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    amplitude = 0.5
    frequency = 100
    data = amplitude * np.sin(2 * np.pi * frequency * t)

    freq_hz, magnitude, all_features = feature_extraction.calculate_fft_features(data, fs, WindowFunction.FLATTOP)

    assert np.allclose(np.max(magnitude), amplitude, atol=0.01)
    assert np.allclose(freq_hz[np.argmax(magnitude)], frequency, atol=1.0)
    
    assert all_features['power_low'] > 0.9
    assert np.isclose(all_features['spectral_centroid'], frequency, atol=5.0) # Flattop has wider peak

def test_calculate_fft_features_mixed_frequencies():
    fs = 15000
    duration = 1
    t = np.linspace(0, duration, fs, endpoint=False)
    
    data_low = 0.5 * np.sin(2 * np.pi * 500 * t)
    data_mid = 0.3 * np.sin(2 * np.pi * 2500 * t)
    data_high = 0.2 * np.sin(2 * np.pi * 6000 * t)
    
    data = data_low + data_mid + data_high

    _, _, all_features = feature_extraction.calculate_fft_features(data, fs, WindowFunction.HANNING)
    
    assert all_features['power_low'] > 0.1
    assert all_features['power_mid'] > 0.1
    assert all_features['power_high'] > 0.1
    assert np.allclose(sum(v for k, v in all_features.items() if 'power' in k), 1.0)
    assert all_features['spectral_centroid'] > 1000 # Should be weighted towards lower freqs with higher amplitude

def test_calculate_fft_features_zero_signal():
    fs = 1000
    data = np.zeros(100)
    
    freq_hz, magnitude, all_features = feature_extraction.calculate_fft_features(data, fs, WindowFunction.HANNING)
    
    assert np.all(magnitude == 0.0)
    assert all_features['power_low'] == 0.0
    assert all_features['spectral_centroid'] == 0.0
