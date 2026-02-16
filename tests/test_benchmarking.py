import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures
from src.core.benchmarking import run_benchmark_test, BenchmarkConfig, MTConfig
from src.diagnostics.mt_method import MTSpace
from dataclasses import asdict
import shutil

# Fixture for a dummy benchmark dataset
@pytest.fixture
def dummy_benchmark_dataset(tmp_path):
    """
    Creates a dummy benchmark dataset structure with normal/anomaly WAV files.
    - train/normal
    - test/normal
    - test/anomaly
    """
    dataset_root = tmp_path / "dummy_benchmark"
    (dataset_root / "train" / "normal").mkdir(parents=True)
    (dataset_root / "test" / "normal").mkdir(parents=True)
    (dataset_root / "test" / "anomaly").mkdir(parents=True)

    # Create dummy WAV files
    # For simplicity, these are just empty files for now.
    # We'll mock load_wav_file to return actual data.
    for i in range(5):
        (dataset_root / "train" / "normal" / f"train_normal_{i}.wav").touch()
    for i in range(3):
        (dataset_root / "test" / "normal" / f"test_normal_{i}.wav").touch()
        (dataset_root / "test" / "anomaly" / f"test_anomaly_{i}.wav").touch()
    
    return dataset_root

# Mock functions for dependencies
@pytest.fixture
def mock_load_wav_file():
    with patch('src.core.benchmarking.load_wav_file') as mock:
        # Default mock behavior: return a sine wave
        def _mock_load(file_path_str):
            fs = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            data = 0.5 * np.sin(2 * np.pi * 1000 * t) # 1kHz sine wave
            file_hash = "mock_hash_" + Path(file_path_str).name
            return fs, data, file_hash
        mock.side_effect = _mock_load
        yield mock

@pytest.fixture
def mock_signal_processing():
    with patch('src.core.benchmarking.remove_dc_offset', side_effect=lambda x: x) as mock_dc:
        with patch('src.core.benchmarking.apply_butterworth_filter', side_effect=lambda data, fs, hpf, lpf, order: data) as mock_filter:
            yield mock_dc, mock_filter

@pytest.fixture
def mock_feature_extraction():
    with patch('src.core.benchmarking.calculate_time_domain_features') as mock_time:
        mock_time.return_value = VibrationFeatures(rms=0.1, peak=0.2, kurtosis=3.0, skewness=0.0, crest_factor=2.0, shape_factor=1.0, power_low=0.3, power_mid=0.4, power_high=0.3)
        with patch('src.core.benchmarking.calculate_fft_features') as mock_fft:
            mock_fft.return_value = (np.array([10, 20]), np.array([0.5, 0.5]), {'low': 0.3, 'mid': 0.4, 'high': 0.3})
            yield mock_time, mock_fft

@pytest.fixture
def mock_mt_space():
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = np.array([0.1, 0.2, 3.0, 0.0, 2.0, 1.0, 0.3, 0.4, 0.3])
        instance.inverse_covariance_matrix = np.eye(9)
        instance.noise_power_spectrum_avg = np.zeros(100) # Mock noise profile
        
        # Mock calculate_md to return different values for normal/anomaly
        def mock_calculate_md(features):
            if "anomaly" in features.to_vector(): # Simple check for mock anomaly
                return 5.0 # High MD for anomaly
            return 1.0 # Low MD for normal
        instance.calculate_md.side_effect = lambda features: 5.0 if "anomaly" in features.to_vector() else 1.0
        # A more robust mock for features would be needed here, or adjust the simple check
        
        yield instance

@pytest.fixture
def mock_nr_plugin_manager():
    with patch('src.core.benchmarking.plugin_manager') as mock_pm:
        mock_plugin = MagicMock()
        mock_plugin.get_name.return_value = "mock_nr_plugin"
        mock_plugin.process.side_effect = lambda data, fs, **kwargs: data * 0.5 # Simple reduction
        mock_pm.get_plugin.return_value = mock_plugin
        yield mock_pm

@pytest.fixture
def mock_nr_evaluation():
    with patch('src.core.benchmarking.perform_nr_evaluation') as mock_eval:
        mock_eval.return_value = MagicMock(
            features_before=VibrationFeatures(rms=0.2, peak=0.4, kurtosis=3, skewness=0, crest_factor=2, shape_factor=1, power_low=0.1, power_mid=0.2, power_high=0.1),
            features_after=VibrationFeatures(rms=0.1, peak=0.2, kurtosis=3, skewness=0, crest_factor=2, shape_factor=1, power_low=0.05, power_mid=0.1, power_high=0.05),
            signal_pre_nr=np.array([1,2,3]),
            signal_post_nr=np.array([0.5,1,1.5]),
            removed_signal=np.array([0.5,1,1.5])
        )
        yield mock_eval

def test_run_benchmark_test_basic(
    dummy_benchmark_dataset, 
    mock_load_wav_file, 
    mock_signal_processing, 
    mock_feature_extraction, 
    mock_mt_space
):
    """
    Test basic functionality of run_benchmark_test without NR plugin.
    """
    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(
            quantity=SignalQuantity.ACCEL,
            window=WindowFunction.HANNING,
            highpass_hz=10.0,
            lowpass_hz=2000.0,
            filter_order=4
        ),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )

    result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

    assert result is not None
    assert result.total_files == 6 # 3 test normal + 3 test anomaly
    assert result.processed_files == 6
    assert isinstance(result.accuracy, float)
    assert isinstance(result.classification_report, str)
    assert len(result.file_results) == 6

    # Verify MTSpace was trained
    assert mock_mt_space.add_normal_sample.call_count == 5 # 5 train normal files

    # Verify calculate_md was called for all test files
    assert mock_mt_space.calculate_md.call_count == 6

    # Verify classification metrics (mock_calculate_md always returns 1.0 for normal, 5.0 for anomaly)
    # 3 normal files (MD=1.0) -> predicted normal
    # 3 anomaly files (MD=5.0) -> predicted anomaly
    assert result.accuracy == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1_score == 1.0

def test_run_benchmark_test_with_nr_plugin(
    dummy_benchmark_dataset,
    mock_load_wav_file,
    mock_signal_processing,
    mock_feature_extraction,
    mock_mt_space,
    mock_nr_plugin_manager,
    mock_nr_evaluation
):
    """
    Test functionality with a mocked NR plugin.
    """
    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(
            quantity=SignalQuantity.ACCEL,
            window=WindowFunction.HANNING,
            highpass_hz=10.0,
            lowpass_hz=2000.0,
            filter_order=4
        ),
        mt_config=MTConfig(anomaly_threshold=3.0),
        nr_plugin_config={"name": "mock_nr_plugin", "params": {"param1": 1.0}}
    )

    result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

    assert result is not None
    assert result.total_files == 6
    assert len(result.file_results) == 6

    # Verify NR plugin was called for all test files
    assert mock_nr_plugin_manager.get_plugin.call_count >= 1 # Should be called once for setup, then for each file
    assert mock_nr_plugin_manager.get_plugin.return_value.process.call_count == 6

    # Verify NR evaluation was performed
    assert mock_nr_evaluation.call_count == 6
    for file_res in result.file_results:
        assert file_res.nr_evaluation is not None
    
    # Check NR performance metrics
    assert result.nr_performance_metrics is not None
    assert 'avg_rms_reduction_pct' in result.nr_performance_metrics
    assert result.nr_performance_metrics['avg_rms_reduction_pct'] == pytest.approx(50.0) # From mock_nr_evaluation

def test_run_benchmark_test_no_training_files(dummy_benchmark_dataset):
    """
    Test scenario where no normal training files are found.
    """
    shutil.rmtree(dummy_benchmark_dataset / "train" / "normal") # Remove training files
    (dummy_benchmark_dataset / "train" / "normal").mkdir(parents=True)

    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )

    with pytest.raises(ValueError, match="No normal training files found"):
        run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

def test_run_benchmark_test_no_test_files(dummy_benchmark_dataset):
    """
    Test scenario where no test files are found.
    """
    shutil.rmtree(dummy_benchmark_dataset / "test") # Remove test files
    (dummy_benchmark_dataset / "test").mkdir(parents=True)

    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )

    with pytest.raises(ValueError, match="No test files found"):
        run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

def test_run_benchmark_test_mt_space_not_established(
    dummy_benchmark_dataset,
    mock_load_wav_file,
    mock_signal_processing,
    mock_feature_extraction
):
    """
    Test scenario where MTSpace mean_vector is None after training (e.g., not enough samples).
    """
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = None # Simulate MTSpace not established
        instance.add_normal_sample.return_value = None # Ensure add_normal_sample does not change mean_vector
        instance.noise_power_spectrum_avg = None # Mock noise profile

        benchmark_config = BenchmarkConfig(
            dataset_name="dummy_benchmark",
            analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
            mt_config=MTConfig(anomaly_threshold=3.0)
        )

        with pytest.raises(RuntimeError, match="MT Space could not be established"):
            run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

