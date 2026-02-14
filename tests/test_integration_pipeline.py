import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib
from datetime import datetime
import json

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, TimeDomainFeatures
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import save_audit_log, AnalysisResult


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
def normal_wav_files_for_mt(tmp_path, dummy_sine_wav):
    """
    Creates a set of normal samples for training the MT unit space.
    By generating 30 identical files, we create a zero-variance unit space.
    This is a deliberate choice to test that a sample identical to the training data
    results in a Mahalanobis Distance of nearly zero, validating the core calculation.
    """
    files = []
    fs = 44100
    _, _, data_int16 = dummy_sine_wav

    # Create 30 identical normal samples for a perfect unit space
    for i in range(30):
        file_path = tmp_path / f"normal_sine_{i}.wav"
        # Write the exact same data 30 times
        wavfile.write(file_path, fs, data_int16)
        files.append(file_path)
    return files, fs

def test_full_analysis_pipeline_integration(dummy_sine_wav, tmp_path):
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
    assert np.mean(data_filtered) < 1e-9

    # 3. Extract Features
    time_features = calculate_time_domain_features(data_filtered)
    assert time_features.rms > 0
    assert time_features.peak > 0

    freq_hz, magnitude, power_bands_dict = calculate_fft_features( # Renamed to power_bands_dict
        data_filtered, fs_hz, analysis_config.window
    )
    assert len(freq_hz) > 0
    assert np.sum(magnitude) > 0
    assert 0 <= power_bands_dict['low'] <= 1
    assert 0 <= power_bands_dict['mid'] <= 1
    assert 0 <= power_bands_dict['high'] <= 1

    # 4. Calculate Quality Metrics
    rms_after_processing = np.sqrt(np.mean(data_filtered**2))
    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, rms_after_processing, magnitude)
    assert quality_metrics.clipping_ratio >= 0
    assert quality_metrics.snr_db > 0
    assert quality_metrics.data_length_s > 0

    confidence_score = get_confidence_score(quality_metrics)
    assert 0 <= confidence_score <= 100

    # 5. Construct AnalysisResult for audit log
    vibration_features = VibrationFeatures(
        **time_features.__dict__,
        power_low=power_bands_dict['low'], # Use power_bands_dict
        power_mid=power_bands_dict['mid'], # Use power_bands_dict
        power_high=power_bands_dict['high'] # Use power_bands_dict
    )
    analysis_result = AnalysisResult(
        features=vibration_features,
        quality=quality_metrics,
        config=analysis_config,
        timestamp=datetime.now().isoformat(),
        file_hash=file_hash,
        fs_hz=fs_hz
    )
    assert analysis_result.features.rms == time_features.rms
    assert analysis_result.quality.snr_db == quality_metrics.snr_db

    # 6. Save Audit Log
    audit_log_path = str(tmp_path / "audit_log.json")
    save_audit_log(analysis_result, audit_log_path)
    assert os.path.exists(audit_log_path)
    with open(audit_log_path, 'r') as f:
        log_content = json.load(f)
    assert log_content['file_hash'] == file_hash


def test_mt_pipeline_integration(normal_wav_files_for_mt, dummy_sine_wav, dummy_noisy_wav):
    mt_space = MTSpace(min_samples=10, recommended_samples=30)
    normal_files, fs_normal_expected = normal_wav_files_for_mt

    mt_train_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=200,
        filter_order=4
    )

    # 1. Train MT Space
    for file_path in normal_files:
        fs_hz, data_normalized, _ = load_wav_file(str(file_path))
        data_ac = remove_dc_offset(data_normalized)
        data_filtered = apply_butterworth_filter(
            data_ac, fs_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
        )
        time_features = calculate_time_domain_features(data_filtered)
        _, _, power_bands_dict = calculate_fft_features( # Renamed to power_bands_dict
            data_filtered, fs_hz, mt_train_config.window
        )
        normal_features = VibrationFeatures(
            **time_features.__dict__,
            power_low=power_bands_dict['low'], # Use power_bands_dict
            power_mid=power_bands_dict['mid'], # Use power_bands_dict
            power_high=power_bands_dict['high'] # Use power_bands_dict
        )
        mt_space.add_normal_sample(normal_features)

    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert not mt_space.is_provisional

    # 2. Test MD for a normal sample
    normal_eval_file, _, _ = dummy_sine_wav
    fs_eval_hz, data_eval_normalized, _ = load_wav_file(str(normal_eval_file))
    data_eval_ac = remove_dc_offset(data_eval_normalized)
    data_eval_filtered = apply_butterworth_filter(
        data_eval_ac, fs_eval_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
    )
    eval_time_features = calculate_time_domain_features(data_eval_filtered)
    _, _, eval_power_bands_dict = calculate_fft_features( # Renamed to eval_power_bands_dict
        data_eval_filtered, fs_eval_hz, mt_train_config.window
    )
    normal_eval_features = VibrationFeatures(
        **eval_time_features.__dict__,
        power_low=eval_power_bands_dict['low'], # Use eval_power_bands_dict
        power_mid=eval_power_bands_dict['mid'], # Use eval_power_bands_dict
        power_high=eval_power_bands_dict['high'] # Use eval_power_bands_dict
    )
    md_normal = mt_space.calculate_md(normal_eval_features)
    assert md_normal < 3.0

    # 3. Test MD for an anomalous sample
    anomalous_eval_file, _, _ = dummy_noisy_wav
    fs_anomalous_hz, data_anomalous_normalized, _ = load_wav_file(str(anomalous_eval_file))
    data_anomalous_ac = remove_dc_offset(data_anomalous_normalized)
    data_anomalous_filtered = apply_butterworth_filter(
        data_anomalous_ac, fs_anomalous_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
    )
    anomalous_time_features = calculate_time_domain_features(data_anomalous_filtered)
    _, _, anomalous_power_bands_dict = calculate_fft_features( # Renamed to anomalous_power_bands_dict
        data_anomalous_filtered, fs_anomalous_hz, mt_train_config.window
    )
    anomalous_eval_features = VibrationFeatures(
        **anomalous_time_features.__dict__,
        power_low=anomalous_power_bands_dict['low'], # Use anomalous_power_bands_dict
        power_mid=anomalous_power_bands_dict['mid'], # Use anomalous_power_bands_dict
        power_high=anomalous_power_bands_dict['high'] # Use anomalous_power_bands_dict
    )
    md_anomalous = mt_space.calculate_md(anomalous_eval_features)
    assert md_anomalous > 5.0

# New test for ISSUE-008
def test_analysis_config_consistency(tmp_path):
    fs = 10000
    duration = 1
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    freq_low = 100
    freq_mid = 1000
    freq_high = 4000
    data_signal = (
        0.5 * np.sin(2 * np.pi * freq_low * t) +
        0.3 * np.sin(2 * np.pi * freq_mid * t) +
        0.2 * np.sin(2 * np.pi * freq_high * t)
    )

    amplitude_int16 = np.iinfo(np.int16).max * 0.5
    data_int16 = (data_signal * amplitude_int16).astype(np.int16)

    test_file_path = tmp_path / "multi_freq_signal.wav"
    wavfile.write(test_file_path, fs, data_int16)

    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=200,
        lowpass_hz=2000,
        filter_order=8
    )

    fs_hz, data_normalized, _ = load_wav_file(str(test_file_path))
    data_ac = remove_dc_offset(data_normalized)
    data_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )

    freqs, magnitudes, power_bands_dict = calculate_fft_features( # Renamed to power_bands_dict
        data_filtered, fs_hz, analysis_config.window
    )

    assert power_bands_dict['low'] < 0.2
    assert power_bands_dict['mid'] > 0.7
    assert power_bands_dict['high'] < 0.1

    peak_idx = np.argmax(magnitudes)
    assert np.allclose(freqs[peak_idx], freq_mid, atol=50)

# New test for `VibrationFeatures` instantiation
def test_vibration_features_instantiation_with_power_bands():
    dummy_time_features = TimeDomainFeatures(
        rms=0.1, peak=0.5, kurtosis=3.0, skewness=0.0, crest_factor=5.0, shape_factor=1.2
    )
    dummy_power_bands = {'low': 0.2, 'mid': 0.7, 'high': 0.1}

    # Ensure instantiation works correctly
    vf = VibrationFeatures(
        **dummy_time_features.__dict__,
        power_low=dummy_power_bands['low'],
        power_mid=dummy_power_bands['mid'],
        power_high=dummy_power_bands['high']
    )

    assert vf.rms == 0.1
    assert vf.peak == 0.5
    assert vf.kurtosis == 3.0
    assert vf.power_low == 0.2
    assert vf.power_mid == 0.7
    assert vf.power_high == 0.1
