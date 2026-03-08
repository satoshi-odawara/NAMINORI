import pytest
import numpy as np
from src.core import quality_check
from src.core.models import QualityMetrics

def test_calculate_quality_metrics_no_clipping():
    fs = 1000
    data = np.random.uniform(-0.5, 0.5, 1000) # No clipping
    rms = np.std(data) # Approximate RMS
    
    magnitude = np.zeros(501)
    magnitude[10] = 100 # A single strong frequency component

    metrics = quality_check.calculate_quality_metrics(data, fs, rms, magnitude)

    assert metrics.clipping_ratio == 0.0
    assert metrics.data_length_s == 1.0
    assert metrics.snr_db > 20

def test_calculate_quality_metrics_with_clipping():
    fs = 1000
    data = np.concatenate((np.random.uniform(-0.5, 0.5, 900), np.full(100, 0.995))) # 10% clipping
    rms = np.std(data)
    magnitude = np.zeros(501)
    magnitude[10] = 100

    metrics = quality_check.calculate_quality_metrics(data, fs, rms, magnitude)

    np.testing.assert_allclose(metrics.clipping_ratio, 0.1, atol=1e-3)
    assert metrics.data_length_s == 1.0
    assert metrics.snr_db > 10

def test_calculate_quality_metrics_all_zeros():
    fs = 1000
    data = np.zeros(1000)
    rms = 0.0
    magnitude = np.zeros(501)
    
    metrics = quality_check.calculate_quality_metrics(data, fs, rms, magnitude)

    assert metrics.clipping_ratio == 0.0
    assert metrics.data_length_s == 1.0
    assert metrics.snr_db == 60.0

def test_get_confidence_score_high_confidence():
    metrics = QualityMetrics(clipping_ratio=0.0, snr_db=30.0, data_length_s=10.0)
    score, breakdown = quality_check.get_confidence_score(metrics)
    assert score == 100.0
    assert breakdown["飽和回避 (Clipping)"] == 100.0
    assert breakdown["ノイズ耐性 (SNR)"] == 100.0
    assert breakdown["データ量 (Length)"] == 100.0

def test_get_confidence_score_low_clipping():
    metrics = QualityMetrics(clipping_ratio=0.01, snr_db=30.0, data_length_s=10.0) # 1% clipping
    score, breakdown = quality_check.get_confidence_score(metrics)
    # NEW logic: clip_score = max(0, 100 - 0.01 * 2000) = 100 - 20 = 80
    # snr_score=100, length_score=100
    # Average = (80+100+100)/3 = 93.33...
    assert breakdown["飽和回避 (Clipping)"] == 80.0
    np.testing.assert_allclose(score, 93.333, rtol=1e-3)

def test_get_confidence_score_high_clipping():
    metrics = QualityMetrics(clipping_ratio=0.05, snr_db=30.0, data_length_s=10.0) # 5% clipping
    score, breakdown = quality_check.get_confidence_score(metrics)
    # clip_score = max(0, 100 - 0.05 * 2000) = 0
    assert breakdown["飽和回避 (Clipping)"] == 0.0
    np.testing.assert_allclose(score, 66.666, rtol=1e-3)

def test_get_confidence_score_low_snr():
    metrics = QualityMetrics(clipping_ratio=0.0, snr_db=10.0, data_length_s=10.0) # Low SNR (10dB)
    score, breakdown = quality_check.get_confidence_score(metrics)
    # snr_score=min(100, 10*5=50)=50
    assert breakdown["ノイズ耐性 (SNR)"] == 50.0
    np.testing.assert_allclose(score, 83.333, rtol=1e-3)

def test_get_confidence_score_short_data():
    metrics = QualityMetrics(clipping_ratio=0.0, snr_db=30.0, data_length_s=5.0) # Short data (5s)
    score, breakdown = quality_check.get_confidence_score(metrics)
    # length_score=min(100, 5*10=50)=50
    assert breakdown["データ量 (Length)"] == 50.0
    np.testing.assert_allclose(score, 83.333, rtol=1e-3)

def test_get_confidence_score_all_low():
    metrics = QualityMetrics(clipping_ratio=0.01, snr_db=5.0, data_length_s=1.0) # All low
    score, breakdown = quality_check.get_confidence_score(metrics)
    # clip_score=80, snr_score=25, length_score=10
    # Average = (80+25+10)/3 = 38.333...
    np.testing.assert_allclose(score, 38.333, rtol=1e-3)
