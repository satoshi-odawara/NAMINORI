import pytest
import numpy as np
from src.diagnostics.mt_method import MTSpace
from src.core.models import VibrationFeatures

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
    power_high: float = 0.2
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
        power_high=power_high
    )

def test_mtspace_initialization():
    mt_space = MTSpace(min_samples=5, recommended_samples=10)
    assert mt_space.min_samples == 5
    assert mt_space.recommended_samples == 10
    assert len(mt_space.normal_samples_vectors) == 0
    assert mt_space.mean_vector is None
    assert mt_space.inverse_covariance_matrix is None
    assert mt_space.is_provisional is True

def test_add_normal_sample_insufficient():
    mt_space = MTSpace(min_samples=3, recommended_samples=5)
    
    # Add 2 samples (insufficient for min_samples=3)
    mt_space.add_normal_sample(create_vibration_features(rms=1.1, peak=2.1))
    mt_space.add_normal_sample(create_vibration_features(rms=1.2, peak=2.2))

    assert len(mt_space.normal_samples_vectors) == 2
    assert mt_space.mean_vector is None
    assert mt_space.inverse_covariance_matrix is None
    assert mt_space.is_provisional is True
    assert mt_space.get_status() == "Insufficient samples (2/3 min)"

def test_add_normal_sample_sufficient_provisional():
    mt_space = MTSpace(min_samples=3, recommended_samples=5)

    # Add 3 samples (sufficient for min_samples=3, but provisional for recommended_samples=5)
    mt_space.add_normal_sample(create_vibration_features(rms=1.1, peak=2.1, power_low=0.6))
    mt_space.add_normal_sample(create_vibration_features(rms=1.2, peak=2.2, power_low=0.7))
    mt_space.add_normal_sample(create_vibration_features(rms=1.3, peak=2.3, power_low=0.8))

    assert len(mt_space.normal_samples_vectors) == 3
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert mt_space.is_provisional is True
    assert mt_space.get_status() == "Provisional unit space (3/5 recommended)"

    # Check mean calculation (qualitative)
    expected_mean_rms = np.mean([1.1, 1.2, 1.3])
    np.testing.assert_allclose(mt_space.mean_vector[0], expected_mean_rms, rtol=1e-3)

def test_add_normal_sample_established():
    mt_space = MTSpace(min_samples=3, recommended_samples=3) # Set recommended_samples to min for easier test

    # Add 3 samples (sufficient for min_samples=3 and recommended_samples=3)
    mt_space.add_normal_sample(create_vibration_features(rms=1.1, peak=2.1, power_low=0.6))
    mt_space.add_normal_sample(create_vibration_features(rms=1.2, peak=2.2, power_low=0.7))
    mt_space.add_normal_sample(create_vibration_features(rms=1.3, peak=2.3, power_low=0.8))

    assert len(mt_space.normal_samples_vectors) == 3
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert mt_space.is_provisional is False
    assert mt_space.get_status() == "Established unit space (3 samples)"

def test_calculate_md_no_unit_space():
    mt_space = MTSpace(min_samples=3)
    features = create_vibration_features()
    md = mt_space.calculate_md(features)
    assert md == np.inf

def test_calculate_md_normal_sample():
    mt_space = MTSpace(min_samples=3, recommended_samples=3)
    normal_features = [
        create_vibration_features(rms=1.0, peak=2.0),
        create_vibration_features(rms=1.1, peak=2.1),
        create_vibration_features(rms=0.9, peak=1.9)
    ]
    for f in normal_features:
        mt_space.add_normal_sample(f)
    
    # Calculate MD for one of the normal samples (should be close to 0)
    md = mt_space.calculate_md(normal_features[0])
    np.testing.assert_allclose(md, 0.0, atol=1e-6) # MD for a point in its own space is 0

def test_calculate_md_anomalous_sample():
    mt_space = MTSpace(min_samples=3, recommended_samples=3)
    normal_features = [
        create_vibration_features(rms=1.0, peak=2.0, power_low=0.5),
        create_vibration_features(rms=1.1, peak=2.1, power_low=0.55),
        create_vibration_features(rms=0.9, peak=1.9, power_low=0.45)
    ]
    for f in normal_features:
        mt_space.add_normal_sample(f)

    # Create an anomalous sample (significantly different RMS)
    anomalous_features = create_vibration_features(rms=10.0, peak=20.0, power_low=0.1)
    md = mt_space.calculate_md(anomalous_features)
    assert md > 5.0 # Expect a significantly higher MD for anomaly

def test_singular_covariance_matrix_regularization():
    mt_space = MTSpace(min_samples=3, recommended_samples=3)
    # Create samples with very low variance in some dimensions to trigger singularity
    mt_space.add_normal_sample(create_vibration_features(rms=1.0, peak=2.0, kurtosis=0.5, skewness=0.1, crest_factor=2.0, shape_factor=1.5, power_low=0.5, power_mid=0.3, power_high=0.2))
    mt_space.add_normal_sample(create_vibration_features(rms=1.000001, peak=2.0, kurtosis=0.5, skewness=0.1, crest_factor=2.0, shape_factor=1.5, power_low=0.5, power_mid=0.3, power_high=0.2))
    mt_space.add_normal_sample(create_vibration_features(rms=1.000002, peak=2.0, kurtosis=0.5, skewness=0.1, crest_factor=2.0, shape_factor=1.5, power_low=0.5, power_mid=0.3, power_high=0.2))

    # Should not raise LinAlgError
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert mt_space.is_provisional is False
    assert mt_space.get_status() == "Established unit space (3 samples)"

    # Calculate MD for a new sample
    md = mt_space.calculate_md(create_vibration_features(rms=1.0, peak=2.0, kurtosis=0.5, skewness=0.1, crest_factor=2.0, shape_factor=1.5, power_low=0.5, power_mid=0.3, power_high=0.2))
    # With regularization, the MD of a sample identical to the mean won't be exactly zero,
    # but it should be a very small positive number.
    assert 0 < md < 0.1
