import pytest
import numpy as np
from scipy.io import wavfile
import os
import json
from dataclasses import asdict

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, TimeDomainFeatures
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
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

    time_features = calculate_time_domain_features(data_filtered)
    freq_hz, magnitude, power_bands = calculate_fft_features(data_filtered, fs_hz, analysis_config.window)

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
    
    return result_dict


def test_feature_regression(dummy_sine_wav_for_regression):
    """
    This test compares the current analysis result with a 'golden' version.
    If the golden file doesn't exist, it creates one.
    If it exists, it compares the current result against it.
    """
    golden_data_path = "tests/golden_data/dummy_sine_wave_result.json"
    wav_file_path, _, _ = dummy_sine_wav_for_regression

    current_result = get_current_analysis_result(str(wav_file_path))

    # Exclude non-deterministic or irrelevant fields from comparison
    current_result.pop('timestamp', None)
    current_result.pop('file_hash', None)
    
    if not os.path.exists(golden_data_path):
        # First run: create the golden data file
        os.makedirs(os.path.dirname(golden_data_path), exist_ok=True)
        with open(golden_data_path, "w") as f:
            json.dump(current_result, f, indent=2, sort_keys=True)
        pytest.skip("Golden data file created. Re-run tests to perform regression check.")
    else:
        # Subsequent runs: compare with the golden data
        with open(golden_data_path, "r") as f:
            golden_result = json.load(f)

        # Compare features with a tolerance
        for key, value in current_result['features'].items():
            assert np.allclose(value, golden_result['features'][key], rtol=1e-5, atol=1e-8), f"Feature '{key}' regression failed. Got {value}, expected {golden_result['features'][key]}"

        # Compare quality metrics
        for key, value in current_result['quality'].items():
            assert np.allclose(value, golden_result['quality'][key], rtol=1e-5, atol=1e-8), f"Quality metric '{key}' regression failed. Got {value}, expected {golden_result['quality'][key]}"

        # Compare config
        for key, value in current_result['config'].items():
            assert value == golden_result['config'][key], f"Config '{key}' regression failed. Got {value}, expected {golden_result['config'][key]}"
