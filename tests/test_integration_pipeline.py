import pytest
import numpy as np
from scipy.io import wavfile
import os
from pathlib import Path
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
from src.core.plugins import plugin_manager
from src.core.evaluation import perform_nr_evaluation, NoiseReductionEvaluation
from src.utils.m2d_parser import parse_m2d_data, load_m2d_file
from dataclasses import asdict

# Ensure plugins are loaded for tests
plugin_manager.load_plugins()

@pytest.fixture
def dummy_sine_wav(tmp_path):
    fs, duration, frequency = 44100, 1.0, 100.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data_int16 = (amplitude * np.sin(2. * np.pi * frequency * t)).astype(np.int16)
    file_path = tmp_path / "sine_wave.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, data_int16

@pytest.fixture
def dummy_noisy_wav(tmp_path):
    fs, duration, frequency = 44100, 1.0, 100.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data_int16 = (amplitude * np.sin(2. * np.pi * frequency * t) +
                  np.random.normal(0, amplitude * 0.2, len(t))).astype(np.int16)
    file_path = tmp_path / "noisy_sine_wave.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, data_int16

@pytest.fixture
def normal_wav_files_for_mt(tmp_path, dummy_sine_wav):
    files, fs, data_int16 = [], dummy_sine_wav[1], dummy_sine_wav[2]
    np.random.seed(42)
    for i in range(25):
        file_path = tmp_path / f"normal_sine_{i}.wav"
        noisy_data = data_int16 + np.random.normal(0, 500, len(data_int16)).astype(np.int16)
        wavfile.write(file_path, fs, noisy_data)
        files.append(file_path)
    return files, fs

@pytest.fixture
def sine_with_band_noise_wav(tmp_path):
    fs, duration = 44100, 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    sos = signal.butter(10, [100.0, 200.0], btype='bandpass', fs=fs, output='sos')
    data = 0.5 * np.sin(2 * np.pi * 300 * t) + signal.sosfilt(sos, np.random.normal(0, 1, len(t))) * 0.5
    data_int16 = (data / np.max(np.abs(data)) * 16000).astype(np.int16)
    file_path = tmp_path / "sine_with_band_noise.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, 100.0, 200.0

@pytest.fixture
def sine_with_notch_noise_wav(tmp_path):
    fs, duration = 44100, 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 60 * t)
    data_int16 = (data / np.max(np.abs(data)) * 16000).astype(np.int16)
    file_path = tmp_path / "sine_with_notch_noise.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs, 60.0

@pytest.fixture
def sine_with_broadband_noise_wav(tmp_path):
    fs, duration = 44100, 2.0
    t = np.linspace(0., duration, int(fs * duration), endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.random.randn(len(t))
    data_int16 = (data / np.max(np.abs(data)) * 16000).astype(np.int16)
    file_path = tmp_path / "sine_with_broadband_noise.wav"
    wavfile.write(file_path, fs, data_int16)
    return file_path, fs

def run_full_analysis_pipeline(*, file_path, fs_expected, plugin_name=None, plugin_params=None, p_noise_avg=None):
    fs_hz, data_norm, file_hash = load_wav_file(str(file_path))
    analysis_config = AnalysisConfig(SignalQuantity.ACCEL, WindowFunction.HANNING, 50, 20000, 4, plugin_name, plugin_params)
    sig_pre = apply_butterworth_filter(remove_dc_offset(data_norm), fs_hz, 50, 20000, 4)
    sig_final = sig_pre
    if plugin_name and plugin_params:
        plugin = plugin_manager.get_plugin(plugin_name)
        if plugin_name == "spectral_subtraction":
            sig_final = plugin.process(sig_pre, fs_hz, p_noise_avg=p_noise_avg, **plugin_params)
        else:
            sig_final = plugin.process(sig_pre, fs_hz, **plugin_params)
    t_f = calculate_time_domain_features(sig_final)
    _, mag, f_f_dict = calculate_fft_features(sig_final, fs_hz, WindowFunction.HANNING)
    qual = calculate_quality_metrics(data_norm, fs_hz, t_f.rms, mag)
    v_f = VibrationFeatures(**asdict(t_f), **f_f_dict)
    return AnalysisResult(v_f, qual, analysis_config, "dummy", file_hash, fs_hz)



def test_full_pipeline_with_audit_log(dummy_sine_wav):
    res = run_full_analysis_pipeline(file_path=dummy_sine_wav[0], fs_expected=44100)
    assert res.features.rms > 0
def test_mt_pipeline_integration(normal_wav_files_for_mt, dummy_sine_wav, dummy_noisy_wav):
    mt_space = MTSpace(min_samples=16, recommended_samples=20)
    conf = AnalysisConfig(SignalQuantity.ACCEL, WindowFunction.HANNING, 50, 20000, 4)
    normal_files, fs_normal = normal_wav_files_for_mt
    for f_path in normal_files:
        fs, d, _ = load_wav_file(str(f_path))
        p = apply_butterworth_filter(remove_dc_offset(d), fs, 50, 20000, 4)
        t_f = calculate_time_domain_features(p)
        _, mags, f_f_d = calculate_fft_features(p, fs, WindowFunction.HANNING)
        mt_space.add_normal_sample(VibrationFeatures(**asdict(t_f), **f_f_d), mags)

    # Keyword arguments used here
    md_normal = mt_space.calculate_md(run_full_analysis_pipeline(file_path=dummy_sine_wav[0], fs_expected=44100).features)
    md_noisy = mt_space.calculate_md(run_full_analysis_pipeline(file_path=dummy_noisy_wav[0], fs_expected=44100).features)
    assert np.isfinite(md_normal)
    assert md_noisy > md_normal

def test_noise_reduction_pipeline_integration_notch(sine_with_notch_noise_wav):
    res = run_full_analysis_pipeline(file_path=sine_with_notch_noise_wav[0], fs_expected=44100, plugin_name="notch_filter", plugin_params={"freq_hz": 60.0, "q_factor": 30.0})
    assert res.features.rms > 0

def test_band_stop_filter_pipeline_integration(sine_with_band_noise_wav):
    res = run_full_analysis_pipeline(file_path=sine_with_band_noise_wav[0], fs_expected=44100, plugin_name="band_stop_filter", plugin_params={"low_hz": 100.0, "high_hz": 200.0, "order": 8})
    assert res.features.rms > 0

def test_spectral_subtraction_pipeline_integration(sine_with_broadband_noise_wav):
    file_path, fs = sine_with_broadband_noise_wav
    # 1. Load data
    fs_hz, data_norm, _ = load_wav_file(str(file_path))
    
    # 2. Prepare noise profile (Must match FFT size of data_norm)
    # Using the exact same length ensures p_noise_avg has correct bins
    noise_data = np.random.normal(0, 0.1, len(data_norm))
    noise_dc_removed = remove_dc_offset(noise_data)
    
    # Extract noise power spectrum
    _, _, f_feat_noise = calculate_fft_features(noise_dc_removed, fs_hz, WindowFunction.HANNING)
    mt = MTSpace()
    # Mocking noise_power_spectrum_avg directly for reliability in test
    mt.noise_power_spectrum_avg = np.abs(np.fft.rfft(noise_dc_removed))**2 / len(noise_dc_removed)

    # 3. Process with plugin
    plugin = plugin_manager.get_plugin("spectral_subtraction")
    processed = plugin.process(data_norm, fs_hz, p_noise_avg=mt.noise_power_spectrum_avg, alpha=2.0, floor=0.01)
    
    # 4. Verify
    assert len(processed) == len(data_norm)
    assert np.std(processed) < np.std(data_norm) # Noise should be reduced


def test_m2d_pipeline_integration():
    """Verify end-to-end signal processing and feature extraction with real M2D file."""
    m2d_file = Path("data/condition-chacher/normal.m2d")
    if not m2d_file.exists():
        pytest.skip("normal.m2d not found.")

    # 1. Load M2D binary data
    fs_hz, data_raw, file_hash = load_m2d_file(m2d_file)
    assert fs_hz == 50000.0
    assert len(data_raw) == 430080

    # 2. Preprocess: DC offset removal and optional filtering
    data_dc = remove_dc_offset(data_raw)
    data_filtered = apply_butterworth_filter(
        data_dc, fs_hz, highpass_hz=50, lowpass_hz=10000, order=4,
        hpf_enabled=True, lpf_enabled=True
    )
    assert len(data_filtered) == len(data_raw)

    # 3. Feature Extraction
    t_feat = calculate_time_domain_features(data_filtered)
    freqs, mags, f_feat = calculate_fft_features(data_filtered, fs_hz, WindowFunction.HANNING)

    v_feat = VibrationFeatures(**asdict(t_feat), **f_feat)
    feature_vector = v_feat.to_vector()

    # Must match the 23-dimension standard defined in GEMINI.md / IMPLEMENTATION_RULES.md
    assert len(feature_vector) == 23
    assert v_feat.rms > 0
    assert v_feat.peak > 0
    assert v_feat.kurtosis is not None
    assert len(freqs) == len(mags)

    # 4. Quality check
    q_metrics = calculate_quality_metrics(data_filtered, int(fs_hz), v_feat.rms, mags)
    conf_score, conf_breakdown = get_confidence_score(q_metrics)
    assert 0.0 <= conf_score <= 100.0
    assert "飽和回避 (Clipping)" in conf_breakdown

