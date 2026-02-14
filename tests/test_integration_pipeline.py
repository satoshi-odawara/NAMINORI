import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib
from datetime import datetime
import json # Added for json.load

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics, AnalysisResult
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import save_audit_log

# Fixture to create a dummy sine wave WAV file
@pytest.fixture
def dummy_sine_wav(tmp_path):
    fs = 44100
    duration = 1.0
    frequency = 100.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data_int16 = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
    
    file_path = tmp_path / "sine_wave.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, data_int16

# Fixture to create a dummy noisy sine wave WAV file (anomalous-like)
@pytest.fixture
def dummy_noisy_wav(tmp_path):
    fs = 44100
    duration = 1.0
    frequency = 100.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data_int16 = (amplitude * np.sin(2. * np.pi * frequency * t) + 
                  np.random.normal(0, amplitude * 0.2, len(t))).astype(np.int16) # Add significant noise
    
    file_path = tmp_path / "noisy_sine_wave.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, data_int16

# Fixture for a set of normal WAV files for MT training
@pytest.fixture
def normal_wav_files_for_mt(tmp_path, dummy_sine_wav): # Added dummy_sine_wav as argument
    files = []
    fs = 44100
    duration = 1.0
    base_frequency = 100.0
    base_amplitude = np.iinfo(np.int16).max * 0.5

    # Ensure dummy_sine_wav is part of the normal training set
    files.append(dummy_sine_wav[0]) # Add the path of dummy_sine_wav

    for i in range(29): # Create 29 other normal samples (total 30 with dummy_sine_wav)
        # Very small frequency variation
        frequency = base_frequency + np.random.uniform(-0.005, 0.005) 
        # Very small amplitude variation
        amplitude = base_amplitude + np.random.uniform(-0.0005, 0.0005) * np.iinfo(np.int16).max 
        t = np.linspace(0., duration, int(fs * duration), endpoint=False)
        data_int16 = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
        
        file_path = tmp_path / f"normal_sine_{i}.wav"
        wavfile.write(file_path, fs, data_int16)
        files.append(file_path)
    return files, fs

def test_full_analysis_pipeline_integration(dummy_sine_wav, tmp_path): # Added tmp_path
    file_path, fs_expected, _ = dummy_sine_wav
    
    # 1. Load WAV file
    fs_hz, data_normalized, file_hash = load_wav_file(str(file_path))
    assert fs_hz == fs_expected
    assert data_normalized is not None
    assert file_hash is not None

    # Define Analysis Config
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=200,
        filter_order=4
    )

    # 2. Apply Signal Processing
    data_ac = remove_dc_offset(data_normalized)
    data_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    assert np.mean(data_filtered) < 1e-9 # DC component should be near zero

    # 3. Extract Features
    time_features = calculate_time_domain_features(data_filtered)
    # Check some basic properties (more rigorous checks are in unit tests)
    assert time_features.rms > 0
    assert time_features.peak > 0

    freq_hz, magnitude, power_low, power_mid, power_high = calculate_fft_features(
        data_filtered, fs_hz, analysis_config.window
    )
    assert len(freq_hz) > 0
    assert np.sum(magnitude) > 0
    assert 0 <= power_low <= 1
    assert 0 <= power_mid <= 1
    assert 0 <= power_high <= 1

    # 4. Calculate Quality Metrics
    rms_after_processing = np.sqrt(np.mean(data_filtered**2))
    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, rms_after_processing, magnitude)
    assert quality_metrics.clipping_ratio >= 0
    assert quality_metrics.snr_db > 0
    assert quality_metrics.data_length_s > 0

    confidence_score = get_confidence_score(quality_metrics)
    assert 0 <= confidence_score <= 100

    # 5. Construct AnalysisResult (for audit log)
    current_timestamp = datetime.now().isoformat()
    analysis_result = AnalysisResult(
        features=VibrationFeatures( # Reconstruct with power data
            rms=time_features.rms, peak=time_features.peak, kurtosis=time_features.kurtosis,
            skewness=time_features.skewness, crest_factor=time_features.crest_factor, shape_factor=time_features.shape_factor,
            power_low=power_low, power_mid=power_mid, power_high=power_high
        ),
        quality=quality_metrics,
        config=analysis_config,
        timestamp=current_timestamp,
        file_hash=file_hash,
        fs_hz=fs_hz # Added fs_hz
    )
    assert analysis_result.features.rms == time_features.rms
    assert analysis_result.quality.snr_db == quality_metrics.snr_db
    assert analysis_result.config.quantity == analysis_config.quantity

    # 6. Save Audit Log (test that it doesn't raise error)
    audit_log_path = str(tmp_path / "audit_log.json")
    save_audit_log(analysis_result, audit_log_path)
    assert os.path.exists(audit_log_path)
    with open(audit_log_path, 'r') as f:
        log_content = json.load(f)
    assert log_content['fs_hz'] == fs_hz

def test_mt_pipeline_integration(normal_wav_files_for_mt, dummy_sine_wav, dummy_noisy_wav, tmp_path): # Added tmp_path
    mt_space = MTSpace(min_samples=30, recommended_samples=30) # Increased samples
    normal_files, fs_normal_expected = normal_wav_files_for_mt
    
    # Define Analysis Config for MT training
    mt_train_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=200,
        filter_order=4
    )

    # 1. Train MT Space with normal samples
    for file_path in normal_files:
        fs_hz, data_normalized, _ = load_wav_file(str(file_path))
        data_ac = remove_dc_offset(data_normalized)
        data_filtered = apply_butterworth_filter(
            data_ac, fs_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
        )
        time_features = calculate_time_domain_features(data_filtered)
        _, _, power_low, power_mid, power_high = calculate_fft_features(
            data_filtered, fs_hz, mt_train_config.window
        )
        normal_features = VibrationFeatures(
            rms=time_features.rms, peak=time_features.peak, kurtosis=time_features.kurtosis,
            skewness=time_features.skewness, crest_factor=time_features.crest_factor, shape_factor=time_features.shape_factor,
            power_low=power_low, power_mid=power_mid, power_high=power_high
        )
        mt_space.add_normal_sample(normal_features)
    
    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert not mt_space.is_provisional # Should be established by 30 samples

    # 2. Test MD for a normal evaluation sample (one of the training files)
    # The dummy_sine_wav has the same base frequency and amplitude as the normal_wav_files_for_mt
    normal_eval_file, fs_eval, _ = dummy_sine_wav
    fs_eval_hz, data_eval_normalized, _ = load_wav_file(str(normal_eval_file))
    data_eval_ac = remove_dc_offset(data_eval_normalized)
    data_eval_filtered = apply_butterworth_filter(
        data_eval_ac, fs_eval_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
    )
    eval_time_features = calculate_time_domain_features(data_eval_filtered)
    _, _, eval_power_low, eval_power_mid, eval_power_high = calculate_fft_features(
        data_eval_filtered, fs_eval_hz, mt_train_config.window
    )
    normal_eval_features = VibrationFeatures(
        rms=eval_time_features.rms, peak=eval_time_features.peak, kurtosis=eval_time_features.kurtosis,
        skewness=eval_time_features.skewness, crest_factor=eval_time_features.crest_factor, shape_factor=eval_time_features.shape_factor,
        power_low=eval_power_low, power_mid=eval_power_mid, power_high=eval_power_high
    )
    md_normal = mt_space.calculate_md(normal_eval_features)
    assert md_normal < 5.0 # Expect low MD for a normal sample

    # 3. Test MD for an anomalous evaluation sample
    anomalous_eval_file, fs_anomalous, _ = dummy_noisy_wav
    fs_anomalous_hz, data_anomalous_normalized, _ = load_wav_file(str(anomalous_eval_file))
    data_anomalous_ac = remove_dc_offset(data_anomalous_normalized)
    data_anomalous_filtered = apply_butterworth_filter(
        data_anomalous_ac, fs_anomalous_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
    )
    anomalous_time_features = calculate_time_domain_features(data_anomalous_filtered)
    _, _, anomalous_power_low, anomalous_power_mid, anomalous_power_high = calculate_fft_features(
        data_anomalous_filtered, fs_anomalous_hz, mt_train_config.window
    )
    anomalous_eval_features = VibrationFeatures(
        rms=anomalous_time_features.rms, peak=anomalous_time_features.peak, kurtosis=anomalous_time_features.kurtosis,
        skewness=anomalous_time_features.skewness, crest_factor=anomalous_time_features.crest_factor, shape_factor=anomalous_time_features.shape_factor,
        power_low=anomalous_power_low, power_mid=anomalous_power_mid, power_high=anomalous_power_high
    )
    md_anomalous = mt_space.calculate_md(anomalous_eval_features)
    assert md_anomalous > 10.0 # Expect significantly higher MD for an anomalous sample
