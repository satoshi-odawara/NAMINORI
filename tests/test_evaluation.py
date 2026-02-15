import pytest
import numpy as np
import sys
from pathlib import Path

# Temporarily add the src directory to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.core.evaluation import NoiseReductionEvaluation, perform_nr_evaluation
from src.core.models import TimeDomainFeatures
from src.core.feature_extraction import calculate_time_domain_features

# Fixture for dummy signals
@pytest.fixture
def dummy_signals():
    fs = 1000
    t = np.linspace(0, 1, fs, endpoint=False)
    # Signal with some noise
    signal_pre_nr = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(fs)
    # Signal with noise reduced (e.g., lower std dev, or specific frequency removed)
    signal_post_nr = 1.0 * np.sin(2 * np.pi * 50 * t) + 0.01 * np.random.randn(fs)
    return signal_pre_nr, signal_post_nr, fs

def test_noise_reduction_evaluation_dataclass(dummy_signals):
    signal_pre_nr, signal_post_nr, _ = dummy_signals
    
    features_before = calculate_time_domain_features(signal_pre_nr)
    features_after = calculate_time_domain_features(signal_post_nr)
    
    evaluation = NoiseReductionEvaluation(
        features_before=features_before,
        features_after=features_after,
        signal_pre_nr=signal_pre_nr,
        signal_post_nr=signal_post_nr
    )
    
    assert evaluation.features_before == features_before
    assert evaluation.features_after == features_after
    np.testing.assert_array_equal(evaluation.signal_pre_nr, signal_pre_nr)
    np.testing.assert_array_equal(evaluation.signal_post_nr, signal_post_nr)

def test_noise_reduction_evaluation_removed_signal_property(dummy_signals):
    signal_pre_nr, signal_post_nr, _ = dummy_signals
    
    features_before = calculate_time_domain_features(signal_pre_nr)
    features_after = calculate_time_domain_features(signal_post_nr)
    
    evaluation = NoiseReductionEvaluation(
        features_before=features_before,
        features_after=features_after,
        signal_pre_nr=signal_pre_nr,
        signal_post_nr=signal_post_nr
    )
    
    expected_removed_signal = signal_pre_nr - signal_post_nr
    np.testing.assert_array_almost_equal(evaluation.removed_signal, expected_removed_signal)

def test_perform_nr_evaluation_function(dummy_signals):
    signal_pre_nr, signal_post_nr, fs = dummy_signals
    
    evaluation = perform_nr_evaluation(signal_pre_nr, signal_post_nr)
    
    assert isinstance(evaluation, NoiseReductionEvaluation)
    assert isinstance(evaluation.features_before, TimeDomainFeatures)
    assert isinstance(evaluation.features_after, TimeDomainFeatures)
    np.testing.assert_array_equal(evaluation.signal_pre_nr, signal_pre_nr)
    np.testing.assert_array_equal(evaluation.signal_post_nr, signal_post_nr)

    # Basic check for feature difference
    assert evaluation.features_before.rms != evaluation.features_after.rms
    assert evaluation.features_before.peak != evaluation.features_after.peak
