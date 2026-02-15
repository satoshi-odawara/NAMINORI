import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib
from datetime import datetime
import json
from scipy import signal
from typing import Optional, Dict, Any

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, TimeDomainFeatures
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import AnalysisResult
from src.core.plugins import plugin_manager # Import plugin manager
from src.core.evaluation import perform_nr_evaluation, NoiseReductionEvaluation # Import evaluation components
from dataclasses import asdict # Added for asdict

# Ensure plugins are loaded for tests
plugin_manager.load_plugins()

# Fixture to create a dummy sine wave WAV file
@pytest.fixture
def dummy_sine_wav(tmp_path):
    fs = 44100
    duration = 1.0
    frequency = 100.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5  # Half of max amplitude for int16
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

@pytest.fixture
def sine_with_band_noise_wav(tmp_path):
    fs = 44100
    duration = 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    
    # Main signal outside the noise band
    main_signal = 0.5 * np.sin(2 * np.pi * 300 * t)
    
    # Generate band-limited noise (e.g., 100-200 Hz)
    noise_low_hz = 100.0
    noise_high_hz = 200.0
    # Generate white noise
    white_noise = np.random.normal(0, 1, len(t))
    # Filter the white noise to create band-limited noise
    sos = signal.butter(10, [noise_low_hz, noise_high_hz], btype='bandpass', fs=fs, output='sos')
    band_noise = signal.sosfilt(sos, white_noise) * 0.5

    data = main_signal + band_noise
    data_int16 = (data / np.max(np.abs(data)) * (np.iinfo(np.int16).max * 0.5)).astype(np.int16)

    file_path = tmp_path / "sine_with_band_noise.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, noise_low_hz, noise_high_hz


# Fixture to create a WAV file with a specific noise frequency for notch filter testing
@pytest.fixture
def sine_with_notch_noise_wav(tmp_path):
    fs = 44100
    duration = 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    
    # Main signal at 100 Hz
    main_signal = 0.5 * np.sin(2 * np.pi * 100 * t)
    # Strong 60 Hz noise to be notched
    noise_freq = 60.0
    noise_signal = 0.3 * np.sin(2 * np.pi * noise_freq * t)
    
    data = main_signal + noise_signal
    data_int16 = (data / np.max(np.abs(data)) * (np.iinfo(np.int16).max * 0.5)).astype(np.int16)

    file_path = tmp_path / "sine_with_notch_noise.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, noise_freq


# Helper function to run the full analysis pipeline with a specified plugin configuration
def run_full_analysis_pipeline(
    file_path: Path,
    fs_expected: int,
    plugin_name: Optional[str] = None,
    plugin_params: Optional[Dict[str, Any]] = None
) -> AnalysisResult:
    # 1. Load WAV file
    fs_hz, data_normalized, file_hash = load_wav_file(str(file_path))
    assert fs_hz == fs_expected
    assert data_normalized is not None
    assert file_hash is not None

    # Define Analysis Config with plugin settings
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=20000,
        filter_order=4,
        noise_reduction_plugin_name=plugin_name,
        noise_reduction_plugin_params=plugin_params
    )

    # 2. Apply Signal Processing
    processed_dc_removed = remove_dc_offset(data_normalized)
    
    signal_pre_nr = apply_butterworth_filter(
        processed_dc_removed, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    
    nr_eval_results = None
    processed_final = signal_pre_nr

    if plugin_name and plugin_params:
        selected_plugin = plugin_manager.get_plugin(plugin_name)
        if selected_plugin:
            signal_post_nr = selected_plugin.process(
                signal_pre_nr,
                fs_hz,
                **plugin_params
            )
            nr_eval_results = perform_nr_evaluation(signal_pre_nr, signal_post_nr)
            processed_final = signal_post_nr
        else:
            raise ValueError(f"Plugin '{plugin_name}' not found.")
    
    assert np.allclose(np.mean(processed_final), 0.0, atol=1e-3) # DC component should be removed
    # 3. Extract Features
    time_features = calculate_time_domain_features(processed_final)
    assert time_features.rms > 0
    assert time_features.peak > 0

    freq_hz, magnitude, power_bands_dict = calculate_fft_features(
        processed_final, fs_hz, analysis_config.window
    )
    assert len(freq_hz) > 0
    assert np.sum(magnitude) > 0
    assert 0 <= power_bands_dict['low'] <= 1
    assert 0 <= power_bands_dict['mid'] <= 1
    assert 0 <= power_bands_dict['high'] <= 1

    # 4. Calculate Quality Metrics
    rms_after_processing = np.sqrt(np.mean(processed_final**2))
    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, rms_after_processing, magnitude)
    assert quality_metrics.clipping_ratio >= 0
    assert quality_metrics.snr_db > -np.inf # SNR can be very low but not NaN
    assert quality_metrics.data_length_s > 0

    confidence_score = get_confidence_score(quality_metrics)
    assert 0 <= confidence_score <= 100

    # 5. Construct AnalysisResult for audit log
    vibration_features = VibrationFeatures(
        **asdict(time_features),
        power_low=power_bands_dict['low'],
        power_mid=power_bands_dict['mid'],
        power_high=power_bands_dict['high']
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

    return analysis_result


def test_full_pipeline_with_audit_log(tmp_path, dummy_sine_wav):
    file_path, fs_expected, _ = dummy_sine_wav
    analysis_result = run_full_analysis_pipeline(file_path, fs_expected)
    
    # 6. Save Audit Log
    audit_log_path = str(tmp_path / "audit_log.json")
    # For integration test, we simulate saving the log by converting to dict and back
    log_data = asdict(analysis_result)
    log_data['config']['quantity'] = analysis_result.config.quantity.value
    log_data['config']['window'] = analysis_result.config.window.value
    if analysis_result.config.noise_reduction_plugin_name:
        log_data['config']['noise_reduction_plugin_name'] = analysis_result.config.noise_reduction_plugin_name
        log_data['config']['noise_reduction_plugin_params'] = analysis_result.config.noise_reduction_plugin_params
    
    with open(audit_log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

    assert os.path.exists(audit_log_path)
    with open(audit_log_path, 'r') as f:
        log_content = json.load(f)
    assert log_content['file_hash'] == analysis_result.file_hash


def test_mt_pipeline_integration(normal_wav_files_for_mt, dummy_sine_wav, dummy_noisy_wav):
    mt_space = MTSpace(min_samples=10, recommended_samples=30)
    normal_files, fs_normal_expected = normal_wav_files_for_mt

    mt_train_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=20000,
        filter_order=4
    )

    # 1. Train MT Space
    for file_path in normal_files:
        fs_hz, data_normalized, _ = load_wav_file(str(file_path))
        processed_dc_removed = remove_dc_offset(data_normalized)
        processed_final = apply_butterworth_filter(
            processed_dc_removed, fs_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
        )
        
        time_features = calculate_time_domain_features(processed_final)
        _, _, power_bands_dict = calculate_fft_features(
            processed_final, fs_hz, mt_train_config.window
        )
        normal_features = VibrationFeatures(
            **asdict(time_features),
            power_low=power_bands_dict['low'],
            power_mid=power_bands_dict['mid'],
            power_high=power_bands_dict['high']
        )
        mt_space.add_normal_sample(normal_features)

    assert mt_space.mean_vector is not None
    assert mt_space.inverse_covariance_matrix is not None
    assert not mt_space.is_provisional

    # 2. Test MD for a normal sample
    normal_eval_file, fs_eval_expected, _ = dummy_sine_wav
    analysis_result_normal = run_full_analysis_pipeline(normal_eval_file, fs_eval_expected)
    md_normal = mt_space.calculate_md(analysis_result_normal.features)
    assert md_normal < 3.0

    # 3. Test MD for an anomalous sample
    anomalous_eval_file, fs_anomalous_expected, _ = dummy_noisy_wav
    analysis_result_anomalous = run_full_analysis_pipeline(anomalous_eval_file, fs_anomalous_expected)
    md_anomalous = mt_space.calculate_md(analysis_result_anomalous.features)
    assert md_anomalous > 5.0


def test_noise_reduction_pipeline_integration_notch(sine_with_notch_noise_wav):
    file_path, fs_expected, noise_freq = sine_with_notch_noise_wav
    
    plugin_name = "notch_filter"
    plugin_params = {"freq_hz": noise_freq, "q_factor": 30.0}

    analysis_result = run_full_analysis_pipeline(file_path, fs_expected, plugin_name, plugin_params)
    
    # Verify that the noise frequency is attenuated by comparing FFT magnitudes
    fs_hz, data_normalized, _ = load_wav_file(str(file_path))
    processed_dc_removed = remove_dc_offset(data_normalized)
    signal_pre_nr = apply_butterworth_filter(processed_dc_removed, fs_hz, analysis_result.config.highpass_hz, analysis_result.config.lowpass_hz, analysis_result.config.filter_order)

    _, magnitude_orig, _ = calculate_fft_features(signal_pre_nr, fs_hz, analysis_result.config.window)
    _, magnitude_nr, _ = calculate_fft_features(run_full_analysis_pipeline(file_path, fs_expected, plugin_name, plugin_params).features.to_vector(), fs_hz, analysis_result.config.window) # Need access to processed_final for FFT
    _, magnitude_nr_from_result, _ = calculate_fft_features(
        # This is a bit indirect, but we can't easily get the processed_final signal from AnalysisResult
        # Re-running the pipeline to get the processed_final for FFT comparison for now.
        # A more direct approach would be to return processed_final from run_full_analysis_pipeline.
        plugin_manager.get_plugin(plugin_name).process(signal_pre_nr, fs_hz, **plugin_params),
        fs_hz,
        analysis_result.config.window
    )

    freqs_temp, _, _ = calculate_fft_features(signal_pre_nr, fs_hz, analysis_result.config.window)

    idx_noise_orig = np.argmin(np.abs(freqs_temp - noise_freq))
    
    # Check that noise magnitude is significantly reduced after NR filter
    assert magnitude_nr_from_result[idx_noise_orig] < 0.5 * magnitude_orig[idx_noise_orig]

    # Check that main signal is preserved (100 Hz in this fixture)
    main_freq_val = 100.0
    idx_main_orig = np.argmin(np.abs(freqs_temp - main_freq_val))
    assert np.isclose(magnitude_nr_from_result[idx_main_orig], magnitude_orig[idx_main_orig], rtol=0.2)


def test_band_stop_filter_pipeline_integration(sine_with_band_noise_wav):
    file_path, fs_expected, noise_low, noise_high = sine_with_band_noise_wav

    plugin_name = "band_stop_filter"
    plugin_params = {"low_hz": noise_low, "high_hz": noise_high, "order": 8}

    analysis_result = run_full_analysis_pipeline(file_path, fs_expected, plugin_name, plugin_params)

    # Verify that the noise band is attenuated by comparing FFT magnitudes
    fs_hz, data_normalized, _ = load_wav_file(str(file_path))
    processed_dc_removed = remove_dc_offset(data_normalized)
    signal_pre_nr = apply_butterworth_filter(processed_dc_removed, fs_hz, analysis_result.config.highpass_hz, analysis_result.config.lowpass_hz, analysis_result.config.filter_order)

    _, magnitude_orig, _ = calculate_fft_features(signal_pre_nr, fs_hz, analysis_result.config.window)
    _, magnitude_nr_from_result, _ = calculate_fft_features(
        plugin_manager.get_plugin(plugin_name).process(signal_pre_nr, fs_hz, **plugin_params),
        fs_hz,
        analysis_result.config.window
    )

    freqs_temp, _, _ = calculate_fft_features(signal_pre_nr, fs_hz, analysis_result.config.window)

    def energy_in_band(freqs, mags, low, high):
        band_indices = np.where((freqs >= low) & (freqs <= high))
        return np.sum(mags[band_indices]**2)

    energy_orig_band = energy_in_band(freqs_temp, magnitude_orig, noise_low, noise_high)
    energy_nr_band = energy_in_band(freqs_temp, magnitude_nr_from_result, noise_low, noise_high)
    
    # Assert that energy in the filtered band is significantly reduced
    assert energy_nr_band < 0.2 * energy_orig_band

    # Verify that the main signal outside the band is preserved (300 Hz in this fixture)
    main_signal_freq = 300.0
    idx_main_orig = np.argmin(np.abs(freqs_temp - main_signal_freq))
    assert np.isclose(magnitude_nr_from_result[idx_main_orig], magnitude_orig[idx_main_orig], rtol=0.2)
