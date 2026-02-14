import pytest
import numpy as np
from scipy.io import wavfile
import os
import hashlib
from datetime import datetime
import json
from scipy import signal

from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, TimeDomainFeatures, NoiseReductionFilterType
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter, apply_noise_reduction_filter
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


    # 1. Load WAV file
    fs_hz, data_normalized, file_hash = load_wav_file(str(file_path))
    assert fs_hz == fs_expected
    assert data_normalized is not None
    assert file_hash is not None

    # Define Analysis Config (without noise reduction for this test)
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=50,
        lowpass_hz=200,
        filter_order=4,
        noise_reduction_type=NoiseReductionFilterType.NONE # Explicitly none
    )

    # 2. Apply Signal Processing
    data_ac = remove_dc_offset(data_normalized)
    data_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    # No noise reduction filter applied in this particular test
    data_nr_filtered = apply_noise_reduction_filter(
        data_filtered,
        fs_hz,
        analysis_config.noise_reduction_type,
        analysis_config.notch_freq_hz,
        analysis_config.notch_q_factor,
        analysis_config.band_stop_low_hz,
        analysis_config.band_stop_high_hz,
        analysis_config.band_stop_order
    )
    assert np.mean(data_nr_filtered) < 1e-9

    # 3. Extract Features
    time_features = calculate_time_domain_features(data_nr_filtered)
    assert time_features.rms > 0
    assert time_features.peak > 0

    freq_hz, magnitude, power_bands_dict = calculate_fft_features(
        data_nr_filtered, fs_hz, analysis_config.window
    )
    assert len(freq_hz) > 0
    assert np.sum(magnitude) > 0
    assert 0 <= power_bands_dict['low'] <= 1
    assert 0 <= power_bands_dict['mid'] <= 1
    assert 0 <= power_bands_dict['high'] <= 1

    # 4. Calculate Quality Metrics
    rms_after_processing = np.sqrt(np.mean(data_nr_filtered**2))
    quality_metrics = calculate_quality_metrics(data_normalized, fs_hz, rms_after_processing, magnitude)
    assert quality_metrics.clipping_ratio >= 0
    assert quality_metrics.snr_db > 0
    assert quality_metrics.data_length_s > 0

    confidence_score = get_confidence_score(quality_metrics)
    assert 0 <= confidence_score <= 100

    # 5. Construct AnalysisResult for audit log
    vibration_features = VibrationFeatures(
        **time_features.__dict__,
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
        filter_order=4,
        noise_reduction_type=NoiseReductionFilterType.NONE # Explicitly none
    )

    # 1. Train MT Space
    for file_path in normal_files:
        fs_hz, data_normalized, _ = load_wav_file(str(file_path))
        data_ac = remove_dc_offset(data_normalized)
        data_filtered = apply_butterworth_filter(
            data_ac, fs_hz, mt_train_config.highpass_hz, mt_train_config.lowpass_hz, mt_train_config.filter_order
        )
        data_filtered = apply_noise_reduction_filter(
            data_filtered, fs_hz, mt_train_config.noise_reduction_type,
            mt_train_config.notch_freq_hz, mt_train_config.notch_q_factor,
            mt_train_config.band_stop_low_hz, mt_train_config.band_stop_high_hz, mt_train_config.band_stop_order
        )
        time_features = calculate_time_domain_features(data_filtered)
        _, _, power_bands_dict = calculate_fft_features(
            data_filtered, fs_hz, mt_train_config.window
        )
        normal_features = VibrationFeatures(
            **time_features.__dict__,
            power_low=power_bands_dict['low'],
            power_mid=power_bands_dict['mid'],
            power_high=power_bands_dict['high']
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
    data_eval_filtered = apply_noise_reduction_filter(
        data_eval_filtered, fs_eval_hz, mt_train_config.noise_reduction_type,
        mt_train_config.notch_freq_hz, mt_train_config.notch_q_factor,
        mt_train_config.band_stop_low_hz, mt_train_config.band_stop_high_hz, mt_train_config.band_stop_order
    )
    eval_time_features = calculate_time_domain_features(data_eval_filtered)
    _, _, eval_power_bands_dict = calculate_fft_features(
        data_eval_filtered, fs_eval_hz, mt_train_config.window
    )
    normal_eval_features = VibrationFeatures(
        **eval_time_features.__dict__,
        power_low=eval_power_bands_dict['low'],
        power_mid=eval_power_bands_dict['mid'],
        power_high=eval_power_bands_dict['high']
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
    data_anomalous_filtered = apply_noise_reduction_filter(
        data_anomalous_filtered, fs_anomalous_hz, mt_train_config.noise_reduction_type,
        mt_train_config.notch_freq_hz, mt_train_config.notch_q_factor,
        mt_train_config.band_stop_low_hz, mt_train_config.band_stop_high_hz, mt_train_config.band_stop_order
    )
    anomalous_time_features = calculate_time_domain_features(data_anomalous_filtered)
    _, _, anomalous_power_bands_dict = calculate_fft_features(
        data_anomalous_filtered, fs_anomalous_hz, mt_train_config.window
    )
    anomalous_eval_features = VibrationFeatures(
        **anomalous_time_features.__dict__,
        power_low=anomalous_power_bands_dict['low'],
        power_mid=anomalous_power_bands_dict['mid'],
        power_high=anomalous_power_bands_dict['high']
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
        filter_order=8,
        noise_reduction_type=NoiseReductionFilterType.NONE # Explicitly none
    )

    fs_hz, data_normalized, _ = load_wav_file(str(test_file_path))
    data_ac = remove_dc_offset(data_normalized)
    data_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    data_nr_filtered = apply_noise_reduction_filter(
        data_filtered, fs_hz, analysis_config.noise_reduction_type,
        analysis_config.notch_freq_hz, analysis_config.notch_q_factor,
        analysis_config.band_stop_low_hz, analysis_config.band_stop_high_hz, analysis_config.band_stop_order
    )

    freqs, magnitudes, power_bands_dict = calculate_fft_features( # Renamed to power_bands_dict
        data_nr_filtered, fs_hz, analysis_config.window
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

# New integration test for noise reduction filter
def test_noise_reduction_pipeline_integration(sine_with_notch_noise_wav, tmp_path):
    file_path, fs_expected, noise_freq = sine_with_notch_noise_wav
    
    # 1. Load WAV file
    fs_hz, data_normalized, file_hash = load_wav_file(str(file_path))
    assert fs_hz == fs_expected

    # Define Analysis Config with Notch filter for 60 Hz noise
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=None, # No Butterworth filter for simplicity in this test
        lowpass_hz=None,
        filter_order=4,
        noise_reduction_type=NoiseReductionFilterType.NOTCH,
        notch_freq_hz=noise_freq,
        notch_q_factor=30.0 # Relatively narrow Q factor
    )

    # 2. Apply Signal Processing
    data_ac = remove_dc_offset(data_normalized)
    data_butter_filtered = apply_butterworth_filter(
        data_ac, fs_hz, analysis_config.highpass_hz, analysis_config.lowpass_hz, analysis_config.filter_order
    )
    data_nr_filtered = apply_noise_reduction_filter(
        data_butter_filtered,
        fs_hz,
        analysis_config.noise_reduction_type,
        analysis_config.notch_freq_hz,
        analysis_config.notch_q_factor,
        analysis_config.band_stop_low_hz,
        analysis_config.band_stop_high_hz,
        analysis_config.band_stop_order
    )

    # 3. Extract Features and check noise reduction effect
    freq_hz_orig, magnitude_orig, _ = calculate_fft_features(data_butter_filtered, fs_hz, analysis_config.window)
    freq_hz_nr, magnitude_nr, _ = calculate_fft_features(data_nr_filtered, fs_hz, analysis_config.window)

    # Find the magnitude at the noise frequency before and after filtering
    idx_noise_orig = np.argmin(np.abs(freq_hz_orig - noise_freq))
    idx_noise_nr = np.argmin(np.abs(freq_hz_nr - noise_freq))
    
    # Check that noise magnitude is significantly reduced after NR filter
    # Allow some tolerance for side effects of filtering
    assert magnitude_nr[idx_noise_nr] < 0.5 * magnitude_orig[idx_noise_orig] # Noise should be at least 50% reduced

    # Ensure main signal is still present (e.g., peak at 100 Hz)
    main_freq_val = 100.0
    idx_main_orig = np.argmin(np.abs(freq_hz_orig - main_freq_val))
    idx_main_nr = np.argmin(np.abs(freq_hz_nr - main_freq_val))

    np.testing.assert_allclose(magnitude_nr[idx_main_nr], magnitude_orig[idx_main_orig], rtol=0.2) # Main signal should be largely preserved (within 20% tolerance)

def test_band_stop_filter_pipeline_integration(sine_with_band_noise_wav):
    file_path, fs_hz, noise_low, noise_high = sine_with_band_noise_wav

    # 1. Load WAV file
    fs_loaded, data_normalized, _ = load_wav_file(str(file_path))
    assert fs_hz == fs_loaded

    # 2. Define Analysis Config with BAND_STOP filter
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        noise_reduction_type=NoiseReductionFilterType.BAND_STOP,
        band_stop_low_hz=noise_low,
        band_stop_high_hz=noise_high,
        band_stop_order=8
    )

    # 3. Apply Signal Processing
    data_ac = remove_dc_offset(data_normalized)
    data_nr_filtered = apply_noise_reduction_filter(
        data_ac,
        fs_hz,
        analysis_config.noise_reduction_type,
        band_stop_low_hz=analysis_config.band_stop_low_hz,
        band_stop_high_hz=analysis_config.band_stop_high_hz,
        band_stop_order=analysis_config.band_stop_order
    )

    # 4. Extract Features and check noise reduction effect
    freqs_orig, mags_orig, _ = calculate_fft_features(data_ac, fs_hz, analysis_config.window)
    freqs_nr, mags_nr, _ = calculate_fft_features(data_nr_filtered, fs_hz, analysis_config.window)

    # Helper to calculate energy in a band
    def energy_in_band(freqs, mags, low, high):
        band_indices = np.where((freqs >= low) & (freqs <= high))
        return np.sum(mags[band_indices]**2)

    # 5. Compare energy in the stop-band before and after
    energy_orig_band = energy_in_band(freqs_orig, mags_orig, noise_low, noise_high)
    energy_nr_band = energy_in_band(freqs_nr, mags_nr, noise_low, noise_high)
    
    # Assert that energy in the filtered band is significantly reduced
    assert energy_nr_band < 0.2 * energy_orig_band

    # 6. Verify that the main signal outside the band is preserved
    main_signal_freq = 300.0
    idx_main_orig = np.argmin(np.abs(freqs_orig - main_signal_freq))
    idx_main_nr = np.argmin(np.abs(freqs_nr - main_signal_freq))
    
    # Check that the peak magnitude of the main signal is not excessively attenuated
    assert np.isclose(mags_nr[idx_main_nr], mags_orig[idx_main_orig], rtol=0.2)