import pytest
import numpy as np
from scipy.io import wavfile
import os
import json
from dataclasses import asdict

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, TimeDomainFeatures, NoiseReductionFilterType
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter, apply_noise_reduction_filter
from scipy import signal
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics
from src.utils.audit_log import AnalysisResult

# Use the same dummy sine wave fixture from integration tests for consistency
@pytest.fixture
def dummy_sine_wav_for_regression(tmp_path):
    fs = 44100
    duration = 1.0
    frequency = 100.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data_int16 = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)

    file_path = tmp_path / "sine_wave_regression.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, data_int16

def get_current_analysis_result(wav_file_path: str) -> dict:
    """Helper function to run the full analysis and return the result as a dict."""
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=200,
        filter_order=4
    )

    fs_hz, data_normalized, file_hash = load_wav_file(str(wav_file_path))
    data_ac = remove_dc_offset(data_normalized)
    data_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    data_nr_filtered = apply_noise_reduction_filter(
        data_filtered, fs_hz, analysis_config.noise_reduction_type,
        analysis_config.notch_freq_hz, analysis_config.notch_q_factor,
        analysis_config.band_stop_low_hz, analysis_config.band_stop_high_hz, analysis_config.band_stop_order
    )

    time_features = calculate_time_domain_features(data_nr_filtered)
    freq_hz, magnitude, power_bands = calculate_fft_features(data_nr_filtered, fs_hz, analysis_config.window)

    vibration_features = VibrationFeatures(
        **asdict(time_features),
        power_low=power_bands['low'],
        power_mid=power_bands['mid'],
        power_high=power_bands['high']
    )

    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, time_features.rms, magnitude)

    result = AnalysisResult(
        features=vibration_features,
        quality=quality_metrics,
        config=analysis_config,
        timestamp="dummy_timestamp",
        file_hash="dummy_hash",
        fs_hz=fs_hz
    )
    
    # Convert to dict and manually handle Enums for JSON serialization
    result_dict = asdict(result)
    result_dict['config']['quantity'] = result.config.quantity.value
    result_dict['config']['window'] = result.config.window.value
    result_dict['config']['noise_reduction_type'] = result.config.noise_reduction_type.value
    
    return result_dict



# Fixture and test for band_stop filter regression
@pytest.fixture
def sine_with_band_noise_wav_for_regression(tmp_path):
    np.random.seed(0) # Ensure deterministic noise
    fs = 44100
    duration = 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    main_signal = 0.5 * np.sin(2 * np.pi * 300 * t)
    
    noise_low_hz, noise_high_hz = 100.0, 200.0
    white_noise = np.random.normal(0, 1, len(t))
    sos = signal.butter(10, [noise_low_hz, noise_high_hz], btype='bandpass', fs=fs, output='sos')
    band_noise = signal.sosfilt(sos, white_noise) * 0.5

    data = main_signal + band_noise
    data_int16 = (data / np.max(np.abs(data)) * (np.iinfo(np.int16).max * 0.5)).astype(np.int16)
    file_path = tmp_path / "sine_with_band_noise_regression.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path

def get_current_analysis_result_band_stop(wav_file_path: str) -> dict:
    """Helper function to run analysis with a band-stop filter."""
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        noise_reduction_type=NoiseReductionFilterType.BAND_STOP,
        band_stop_low_hz=100.0,
        band_stop_high_hz=200.0,
        band_stop_order=8
    )

    fs_hz, data_normalized, file_hash = load_wav_file(str(wav_file_path))
    data_ac = remove_dc_offset(data_normalized)
    data_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    data_nr_filtered = apply_noise_reduction_filter(
        data_filtered, fs_hz, analysis_config.noise_reduction_type,
        analysis_config.notch_freq_hz, analysis_config.notch_q_factor,
        analysis_config.band_stop_low_hz, analysis_config.band_stop_high_hz, analysis_config.band_stop_order
    )
    
    time_features = calculate_time_domain_features(data_nr_filtered)
    freq_hz, magnitude, power_bands = calculate_fft_features(data_nr_filtered, fs_hz, analysis_config.window)
    vibration_features = VibrationFeatures(
        **asdict(time_features),
        power_low=power_bands['low'],
        power_mid=power_bands['mid'],
        power_high=power_bands['high']
    )
    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, time_features.rms, magnitude)

    result = AnalysisResult(
        features=vibration_features, quality=quality_metrics, config=analysis_config,
        timestamp="dummy_timestamp", file_hash="dummy_hash", fs_hz=fs_hz
    )
    
    result_dict = asdict(result)
    result_dict['config']['quantity'] = result.config.quantity.value
    result_dict['config']['window'] = result.config.window.value
    result_dict['config']['noise_reduction_type'] = result.config.noise_reduction_type.value
    return result_dict

def test_band_stop_regression(sine_with_band_noise_wav_for_regression):
    golden_data_path = "tests/golden_data/band_stop_filter_result.json"
    wav_file_path = sine_with_band_noise_wav_for_regression

    current_result = get_current_analysis_result_band_stop(str(wav_file_path))
    current_result.pop('timestamp', None)
    current_result.pop('file_hash', None)

    if not os.path.exists(golden_data_path):
        os.makedirs(os.path.dirname(golden_data_path), exist_ok=True)
        with open(golden_data_path, "w") as f:
            json.dump(current_result, f, indent=2, sort_keys=True)
        pytest.skip("Golden data file for band_stop created. Re-run tests for regression check.")
    else:
        with open(golden_data_path, "r") as f:
            golden_result = json.load(f)
        
        for key, value in current_result['features'].items():
            assert np.allclose(value, golden_result['features'][key], rtol=1e-5, atol=1e-8), f"Feature '{key}' regression failed"
        for key, value in current_result['quality'].items():
            assert np.allclose(value, golden_result['quality'][key], rtol=1e-5, atol=1e-8), f"Quality metric '{key}' regression failed"
        
        # Remove float values from config for direct comparison
        current_config_filtered = {k: v for k, v in current_result['config'].items() if not isinstance(v, float)}
        golden_config_filtered = {k: v for k, v in golden_result['config'].items() if not isinstance(v, float)}
        assert current_config_filtered == golden_config_filtered, "Config regression failed"

        # Compare float values in config separately with tolerance
        current_config_floats = {k: v for k, v in current_result['config'].items() if isinstance(v, float)}
        golden_config_floats = {k: v for k, v in golden_result['config'].items() if isinstance(v, float)}
        for key, value in current_config_floats.items():
            assert np.allclose(value, golden_config_floats.get(key, None)), f"Config float '{key}' regression failed"
