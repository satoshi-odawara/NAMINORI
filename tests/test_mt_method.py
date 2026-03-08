import pytest
import numpy as np
from src.diagnostics.mt_method import MTSpace
from src.core.models import VibrationFeatures, AnalysisConfig, SignalQuantity, WindowFunction

# Helper function to create dummy VibrationFeatures
def create_vibration_features(
    rms: float = 1.0,
    peak: float = 2.0,
    kurtosis: float = 0.0,
    skewness: float = 0.0,
    crest_factor: float = 2.0,
    shape_factor: float = 1.5,
    power_low: float = 0.5,
    power_mid: float = 0.3,
    power_high: float = 0.2,
    overall_level: float = 1.0,
    overall_low: float = 0.8,
    overall_high: float = 0.6,
    spectral_centroid: float = 500.0,
    spectral_spread: float = 100.0,
    spectral_entropy: float = 0.5
) -> VibrationFeatures:
    return VibrationFeatures(
        rms=rms,
        peak=peak,
        kurtosis=kurtosis,
        skewness=skewness,
        crest_factor=crest_factor,
        shape_factor=shape_factor,
        power_low=power_low,
        power_mid=power_mid,
        power_high=power_high,
        overall_level=overall_level,
        overall_low=overall_low,
        overall_high=overall_high,
        spectral_centroid=spectral_centroid,
        spectral_spread=spectral_spread,
        spectral_entropy=spectral_entropy
    )

@pytest.fixture
def mock_signal_params():
    fs = 1000
    dummy_signal = np.random.rand(fs)
    config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=None, lowpass_hz=None, filter_order=4
    )
    return dummy_signal, fs, config

def test_mtspace_initialization():
    mt_space = MTSpace(min_samples=5, recommended_samples=10)
    assert mt_space.min_samples == 5
    assert mt_space.recommended_samples == 10
    assert len(mt_space.normal_samples_vectors) == 0
    assert mt_space.mean_vector is None
    assert mt_space.inverse_covariance_matrix is None
    assert mt_space.is_provisional is True

def test_add_normal_sample_insufficient(mock_signal_params):
    # Dim is 15, so 16+ samples needed for a valid space.
    mt_space = MTSpace(min_samples=20, recommended_samples=25)
    dummy_signal, fs, config = mock_signal_params
    
    # Add 2 samples
    mt_space.add_normal_sample(create_vibration_features(rms=1.1, peak=2.1), dummy_signal, fs, config)
    mt_space.add_normal_sample(create_vibration_features(rms=1.2, peak=2.2), dummy_signal, fs, config)

    assert len(mt_space.normal_samples_vectors) == 2
    assert mt_space.mean_vector is None
    assert mt_space.is_provisional is True
    assert "Insufficient" in mt_space.get_status()

def test_add_normal_sample_sufficient_provisional(mock_signal_params):
    # Dim is 15, so 16+ samples needed
    mt_space = MTSpace(min_samples=16, recommended_samples=25)
    dummy_signal, fs, config = mock_signal_params

    # Add 16 diverse samples to ensure non-singular matrix
    for i in range(16):
        mt_space.add_normal_sample(create_vibration_features(rms=1.0 + i*0.01), dummy_signal, fs, config)

    assert len(mt_space.normal_samples_vectors) == 16
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert mt_space.is_provisional is True
    assert "Provisional" in mt_space.get_status()

def test_add_normal_sample_established(mock_signal_params):
    mt_space = MTSpace(min_samples=16, recommended_samples=16)
    dummy_signal, fs, config = mock_signal_params

    for i in range(16):
        mt_space.add_normal_sample(create_vibration_features(rms=1.0 + i*0.01), dummy_signal, fs, config)

    assert len(mt_space.normal_samples_vectors) == 16
    assert mt_space.mean_vector is not None
    assert mt_space.is_provisional is False
    assert "Established" in mt_space.get_status()

def test_calculate_md_no_unit_space():
    mt_space = MTSpace(min_samples=16)
    features = create_vibration_features()
    md = mt_space.calculate_md(features)
    assert md == np.inf

def test_calculate_md_normal_sample(mock_signal_params):
    mt_space = MTSpace(min_samples=16, recommended_samples=16)
    dummy_signal, fs, config = mock_signal_params
    
    features_list = [create_vibration_features(rms=1.0 + i*0.01) for i in range(16)]
    for f in features_list:
        mt_space.add_normal_sample(f, dummy_signal, fs, config)
    
    # Calculate MD for the first sample
    md = mt_space.calculate_md(features_list[0])
    assert np.isfinite(md)
    assert md < 10.0 # Should be relatively small for a member of the set

def test_singular_covariance_matrix_regularization(mock_signal_params):
    mt_space = MTSpace(min_samples=16, recommended_samples=16)
    dummy_signal, fs, config = mock_signal_params
    
    # Add identical samples to force singularity
    f = create_vibration_features(rms=1.0)
    for _ in range(20):
        mt_space.add_normal_sample(f, dummy_signal, fs, config)

    # Should not raise LinAlgError due to regularization
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None

def test_mtspace_build_unit_space():
    mt_space = MTSpace(min_samples=16, recommended_samples=16)
    # Add 20 diverse samples
    features_list = [create_vibration_features(rms=1.0 + i*0.1, peak=2.0 + i*0.2) for i in range(20)]
    
    # Use the batch build method
    mt_space.build_unit_space(features_list)
    
    assert len(mt_space.normal_samples_vectors) == 20
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert mt_space.is_provisional is False
    
    md = mt_space.calculate_md(features_list[0])
    assert np.isfinite(md)
