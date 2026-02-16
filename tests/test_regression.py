import pytest
import numpy as np
from scipy.io import wavfile
import os
import json
from dataclasses import asdict
from typing import Optional, Dict, Any

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, TimeDomainFeatures
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics
from src.core.plugins import plugin_manager
from src.core.evaluation import NoiseReductionEvaluation, perform_nr_evaluation
from src.diagnostics.mt_method import MTSpace # Added import for MTSpace
from src.utils.audit_log import AnalysisResult
from scipy import signal

# Ensure plugins are loaded for tests
plugin_manager.load_plugins()

# Fixture for a dummy sine wave WAV file without specific noise for generic tests
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
    return file_path

# Fixture for a sine wave with notch noise
@pytest.fixture
def sine_with_notch_noise_wav_for_regression(tmp_path):
    np.random.seed(0) # Ensure deterministic noise
    fs = 44100
    duration = 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    main_signal = 0.5 * np.sin(2 * np.pi * 300 * t)
    
    # Add strong 60 Hz hum for notch filter to remove
    notch_noise = 0.8 * np.sin(2 * np.pi * 60 * t)
    
    data = main_signal + notch_noise + 0.1 * np.random.randn(len(t)) # Add some broadband noise
    data_int16 = (data / np.max(np.abs(data)) * (np.iinfo(np.int16).max * 0.5)).astype(np.int16)
    file_path = tmp_path / "sine_with_notch_noise_regression.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path

# Fixture for a sine wave with band noise (e.g., to be removed by band-stop filter)
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

# Helper to run the analysis pipeline with a specified plugin
def get_current_analysis_result_with_plugin(
    wav_file_path: str, 
    plugin_name: Optional[str] = None, 
    plugin_params: Optional[Dict[str, Any]] = None,
    p_noise_avg: Optional[np.ndarray] = None # New parameter for spectral subtraction
) -> Dict[str, Any]:
    """
    Helper function to run the full analysis using the plugin system
    and return the result along with evaluation data as a dict.
    """
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=5000, # A bit higher LPF for general purpose
        filter_order=4,
        noise_reduction_plugin_name=plugin_name,
        noise_reduction_plugin_params=plugin_params
    )

    fs_hz, data_normalized, file_hash = load_wav_file(str(wav_file_path))
    processed_dc_removed = remove_dc_offset(data_normalized)
    
    # Apply standard Butterworth filter first
    signal_pre_nr = apply_butterworth_filter(
        processed_dc_removed, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    
    nr_eval_results = None
    processed_final = signal_pre_nr

    if plugin_name and plugin_params:
        selected_plugin = plugin_manager.get_plugin(plugin_name)
        if selected_plugin:
            if plugin_name == "spectral_subtraction":
                if p_noise_avg is None:
                    raise ValueError("p_noise_avg must be provided for spectral_subtraction plugin.")
                signal_post_nr = selected_plugin.process(
                    signal_pre_nr,
                    fs_hz,
                    p_noise_avg=p_noise_avg, # Pass the noise profile
                    **plugin_params
                )
            else:
                signal_post_nr = selected_plugin.process(
                    signal_pre_nr,
                    fs_hz,
                    **plugin_params
                )
            nr_eval_results = perform_nr_evaluation(signal_pre_nr, signal_post_nr)
            processed_final = signal_post_nr
        else:
            raise ValueError(f"Plugin '{plugin_name}' not found.")

    time_features = calculate_time_domain_features(processed_final)
    freq_hz, magnitude, power_bands = calculate_fft_features(processed_final, fs_hz, analysis_config.window)

    vibration_features = VibrationFeatures(
        **asdict(time_features),
        power_low=power_bands['low'],
        power_mid=power_bands['mid'],
        power_high=power_bands['high']
    )

    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, time_features.rms, magnitude)

    # Prepare AnalysisResult
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

    # Include NR evaluation results if available
    if nr_eval_results:
        result_dict['nr_evaluation'] = {
            'features_before': asdict(nr_eval_results.features_before),
            'features_after': asdict(nr_eval_results.features_after),
        }
    
    return result_dict

# Generic test to generate golden data if it doesn't exist, otherwise compares
def _run_regression_test(current_result: Dict[str, Any], golden_data_path: str):
    current_result.pop('timestamp', None)
    current_result.pop('file_hash', None)

    if not os.path.exists(golden_data_path):
        os.makedirs(os.path.dirname(golden_data_path), exist_ok=True)
        with open(golden_data_path, "w") as f:
            json.dump(current_result, f, indent=2, sort_keys=True)
        pytest.fail(f"Golden data file '{golden_data_path}' created. Please review and re-run tests for regression check.")
    else:
        with open(golden_data_path, "r") as f:
            golden_result = json.load(f)
        
        # Compare main features
        for key, value in current_result['features'].items():
            assert np.allclose(value, golden_result['features'][key], rtol=1e-5, atol=1e-8), f"Feature '{key}' regression failed"
        
        # Compare quality metrics
        for key, value in current_result['quality'].items():
            assert np.allclose(value, golden_result['quality'][key], rtol=1e-5, atol=1e-8), f"Quality metric '{key}' regression failed"
        
        # Compare NR evaluation features if present
        if 'nr_evaluation' in current_result and 'nr_evaluation' in golden_result:
            for stage in ['features_before', 'features_after']:
                for key, value in current_result['nr_evaluation'][stage].items():
                    assert np.allclose(value, golden_result['nr_evaluation'][stage][key], rtol=1e-5, atol=1e-8), \
                        f"NR Evaluation {stage} feature '{key}' regression failed"

        # Compare config parameters (handle floats separately)
        for config_key in current_result['config']:
            if isinstance(current_result['config'][config_key], float):
                assert np.allclose(current_result['config'][config_key], golden_result['config'][config_key], rtol=1e-5, atol=1e-8), \
                    f"Config float '{config_key}' regression failed"
            else:
                assert current_result['config'][config_key] == golden_result['config'][config_key], \
                    f"Config parameter '{config_key}' regression failed"


# --- Regression Tests ---

# Test for Notch Filter Plugin
def test_notch_filter_plugin_regression(sine_with_notch_noise_wav_for_regression):
    golden_data_path = "tests/golden_data/notch_filter_plugin_result.json"
    wav_file_path = sine_with_notch_noise_wav_for_regression

    plugin_name = "notch_filter"
    plugin_params = {"freq_hz": 60.0, "q_factor": 30.0}

    current_result = get_current_analysis_result_with_plugin(
        str(wav_file_path), plugin_name, plugin_params
    )
    _run_regression_test(current_result, golden_data_path)

# Test for Band-Stop Filter Plugin
def test_band_stop_filter_plugin_regression(sine_with_band_noise_wav_for_regression):
    golden_data_path = "tests/golden_data/band_stop_filter_plugin_result.json" # New golden file name
    wav_file_path = sine_with_band_noise_wav_for_regression

    plugin_name = "band_stop_filter"
    plugin_params = {"low_hz": 100.0, "high_hz": 200.0, "order": 8}

    current_result = get_current_analysis_result_with_plugin(
        str(wav_file_path), plugin_name, plugin_params
    )
    _run_regression_test(current_result, golden_data_path)

# Fixture for a sine wave with broadband noise for spectral subtraction regression
@pytest.fixture
def sine_with_broadband_noise_wav_for_regression(tmp_path):
    np.random.seed(1) # Ensure deterministic noise (different seed from others to avoid collision)
    fs = 44100
    duration = 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    
    main_signal = 0.5 * np.sin(2 * np.pi * 100 * t) # Main signal
    broadband_noise = 0.3 * np.random.randn(len(t)) # Broadband noise component

    data = main_signal + broadband_noise
    data_int16 = (data / np.max(np.abs(data)) * (np.iinfo(np.int16).max * 0.5)).astype(np.int16)

    file_path = tmp_path / "sine_with_broadband_noise_regression.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path

# Test for Spectral Subtraction Plugin
def test_spectral_subtraction_regression(sine_with_broadband_noise_wav_for_regression):
    golden_data_path = "tests/golden_data/spectral_subtraction_plugin_result.json"
    wav_file_path = sine_with_broadband_noise_wav_for_regression
    
    fs_hz, _, _ = load_wav_file(str(wav_file_path))
    N = int(fs_hz * 2.0) # Data length used for fixture (2 seconds)
    
    # Simulate a learned noise power spectrum (flat for white noise)
    # This P_noise_avg must match the expected output length of FFT from a signal of length N
    # For a purely random broadband noise, its power spectrum would be relatively flat.
    # We create a constant power spectrum for simplicity in this test.
    mock_noise_amplitude = 0.3 # Matches broadband_noise amplitude in fixture
    mock_noise_power_avg = np.full(N, (mock_noise_amplitude**2) * N / 2) # Simplified flat power spectrum
                                                                          # This is a rough approximation, actual value depends on exact FFT scaling
    
    # Create a mock MTSpace and set its noise_power_spectrum_avg
    mock_mt_space = MTSpace()
    mock_mt_space.noise_power_spectrum_avg = mock_noise_power_avg
    
    # Need to pass this mock_mt_space or its noise_power_spectrum_avg to get_current_analysis_result_with_plugin
    # This implies modifying run_full_analysis_pipeline to accept p_noise_avg as a direct arg
    
    plugin_name = "spectral_subtraction"
    plugin_params = {"alpha": 2.0, "floor": 0.01, "post_filter_cutoff_hz": 0.0}

    current_result = get_current_analysis_result_with_plugin(
        str(wav_file_path), plugin_name, plugin_params, p_noise_avg=mock_mt_space.noise_power_spectrum_avg
    )
    _run_regression_test(current_result, golden_data_path)
