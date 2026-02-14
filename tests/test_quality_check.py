import pytest
import numpy as np
from src.core import quality_check
from src.core.models import QualityMetrics

def test_calculate_quality_metrics_no_clipping():
    fs = 1000
    data = np.random.uniform(-0.5, 0.5, 1000) # No clipping
    rms = np.std(data) # Approximate RMS
    
    # Create a dummy magnitude array for SNR calculation
    # For a simple test, we can just create a flat spectrum or a single peak.
    # Let's create a magnitude array where noise floor is low.
    magnitude = np.zeros(501)
    magnitude[10] = 100 # A single strong frequency component

    metrics = quality_check.calculate_quality_metrics(data, fs, rms, magnitude)

    assert metrics.clipping_ratio == 0.0
    assert metrics.data_length_s == 1.0
    # SNR calculation is qualitative, so check if it's reasonably high
    assert metrics.snr_db > 20 # Expect good SNR for clean signal + low noise floor

def test_calculate_quality_metrics_with_clipping():
    fs = 1000
    data = np.concatenate((np.random.uniform(-0.5, 0.5, 900), np.full(100, 0.995))) # 10% clipping
    rms = np.std(data)
    magnitude = np.zeros(501)
    magnitude[10] = 100

    metrics = quality_check.calculate_quality_metrics(data, fs, rms, magnitude)

    np.testing.assert_allclose(metrics.clipping_ratio, 0.1, atol=1e-3)
    assert metrics.data_length_s == 1.0
    # SNR should still be reasonable if the signal part is good
    assert metrics.snr_db > 10 # Expect some degradation but not terrible

def test_calculate_quality_metrics_all_zeros():
    fs = 1000
    data = np.zeros(1000)
    rms = 0.0
    magnitude = np.zeros(501) # All zeros magnitude
    
    metrics = quality_check.calculate_quality_metrics(data, fs, rms, magnitude)

    assert metrics.clipping_ratio == 0.0
    assert metrics.data_length_s == 1.0
    # If rms is 0 and magnitude is 0, snr_db calculation handles division by zero.
    # The current implementation sets snr_db to 60 if noise_floor is 0.
    assert metrics.snr_db == 60.0

def test_get_confidence_score_high_confidence():
    metrics = QualityMetrics(clipping_ratio=0.0, snr_db=30.0, data_length_s=10.0)
    score = quality_check.get_confidence_score(metrics)
    # Expected scores: clip_score=100, snr_score=min(100, 30*5=150)=100, length_score=min(100, 10*10=100)=100
    # Average = (100+100+100)/3 = 100
    assert score == 100.0

def test_get_confidence_score_low_clipping():
    metrics = QualityMetrics(clipping_ratio=0.01, snr_db=30.0, data_length_s=10.0) # 1% clipping
    score = quality_check.get_confidence_score(metrics)
    # clip_score = max(0, 100 - 0.01 * 10000) = max(0, 100 - 100) = 0
    # snr_score=100, length_score=100
    # Average = (0+100+100)/3 = 66.66...
    np.testing.assert_allclose(score, 66.666, rtol=1e-3)

def test_get_confidence_score_low_snr():
    metrics = QualityMetrics(clipping_ratio=0.0, snr_db=10.0, data_length_s=10.0) # Low SNR (10dB)
    score = quality_check.get_confidence_score(metrics)
    # clip_score=100, snr_score=min(100, 10*5=50)=50, length_score=100
    # Average = (100+50+100)/3 = 83.33...
    np.testing.assert_allclose(score, 83.333, rtol=1e-3)

def test_get_confidence_score_short_data():
    metrics = QualityMetrics(clipping_ratio=0.0, snr_db=30.0, data_length_s=5.0) # Short data (5s)
    score = quality_check.get_confidence_score(metrics)
    # clip_score=100, snr_score=100, length_score=min(100, 5*10=50)=50
    # Average = (100+100+50)/3 = 83.33...
    np.testing.assert_allclose(score, 83.333, rtol=1e-3)

def test_get_confidence_score_all_low():
    metrics = QualityMetrics(clipping_ratio=0.01, snr_db=5.0, data_length_s=1.0) # All low
    score = quality_check.get_confidence_score(metrics)
    # clip_score=0, snr_score=min(100, 5*5=25)=25, length_score=min(100, 1*10=10)=10
    # Average = (0+25+10)/3 = 11.66...
    np.testing.assert_allclose(score, 11.666, rtol=1e-3)
