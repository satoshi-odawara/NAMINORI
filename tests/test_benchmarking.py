import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, asdict
from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures
from src.core.benchmarking import run_benchmark_test, BenchmarkConfig, MTConfig
from src.diagnostics.mt_method import MTSpace
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
    for i in range(5):
        (dataset_root / "train" / "normal" / f"train_normal_{i}.wav").touch()
    for i in range(3):
        (dataset_root / "test" / "normal" / f"test_normal_{i}.wav").touch()
        (dataset_root / "test" / "anomaly" / f"test_anomaly_{i}.wav").touch()
    
    return dataset_root

@pytest.fixture
def dcase_like_dataset(tmp_path):
    """
    Creates a DCASE-like dataset structure with multiple machine IDs.
    - id_00/
        - train/normal/ (2 files)
        - test/normal/ (1 file)
        - test/anomaly/ (1 file)
    - id_02/
        - train/normal/ (3 files)
        - test/normal/ (2 files)
        - test/anomaly/ (2 files)
    """
    dataset_root = tmp_path / "dcase_like_benchmark"
    
    # --- Machine ID 00 ---
    id_00_dir = dataset_root / "id_00"
    (id_00_dir / "train" / "normal").mkdir(parents=True)
    (id_00_dir / "test" / "normal").mkdir(parents=True)
    (id_00_dir / "test" / "anomaly").mkdir(parents=True)
    for i in range(2):
        (id_00_dir / "train" / "normal" / f"train_normal_00_{i}.wav").touch()
    (id_00_dir / "test" / "normal" / "test_normal_00_0.wav").touch()
    (id_00_dir / "test" / "anomaly" / "test_anomaly_00_0.wav").touch()

    # --- Machine ID 02 ---
    id_02_dir = dataset_root / "id_02"
    (id_02_dir / "train" / "normal").mkdir(parents=True)
    (id_02_dir / "test" / "normal").mkdir(parents=True)
    (id_02_dir / "test" / "anomaly").mkdir(parents=True)
    for i in range(3):
        (id_02_dir / "train" / "normal" / f"train_normal_02_{i}.wav").touch()
    for i in range(2):
        (id_02_dir / "test" / "normal" / f"test_normal_02_{i}.wav").touch()
        (id_02_dir / "test" / "anomaly" / f"test_anomaly_02_{i}.wav").touch()

    return dataset_root

# Mock functions for dependencies
@pytest.fixture
def mock_load_wav_file():
    with patch('src.core.benchmarking.load_wav_file') as mock:
        def _mock_load(file_path_str):
            fs = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            path_obj = Path(file_path_str)
            # Make anomaly data different so feature extraction mock can distinguish
            if "anomaly" in path_obj.name:
                data = 0.8 * np.sin(2 * np.pi * 1000 * t) 
            else:
                data = 0.5 * np.sin(2 * np.pi * 1000 * t)
            file_hash = "mock_hash_" + path_obj.name
            return fs, data, file_hash
        mock.side_effect = _mock_load
        yield mock

@pytest.fixture
def mock_signal_processing():
    with patch('src.core.benchmarking.remove_dc_offset', side_effect=lambda x: x), \
         patch('src.core.benchmarking.apply_butterworth_filter', side_effect=lambda data, *args, **kwargs: data):
        yield

@pytest.fixture
def mock_feature_extraction():
    # Define a temporary dataclass that mirrors what calculate_time_domain_features
    # is expected to return (i.e., just the time-domain features).
    @dataclass
    class TimeDomainFeatures:
        rms: float
        peak: float
        kurtosis: float
        skewness: float
        crest_factor: float
        shape_factor: float

    # This mock will now return different features for normal vs anomaly
    def mock_calc_time_features(signal):
        is_anomaly = np.max(signal) > 0.7 # Based on mock_load_wav_file
        if is_anomaly:
            return TimeDomainFeatures(rms=0.8, peak=0.9, kurtosis=5.0, skewness=0.1, crest_factor=2.2, shape_factor=1.2)
        return TimeDomainFeatures(rms=0.1, peak=0.2, kurtosis=3.0, skewness=0.0, crest_factor=2.0, shape_factor=1.0)

    with patch('src.core.benchmarking.calculate_time_domain_features', side_effect=mock_calc_time_features) as mock_time, \
         patch('src.core.benchmarking.calculate_fft_features', return_value=(None, None, {'low': 0.3, 'mid': 0.4, 'high': 0.3})):
        yield mock_time

def test_run_benchmark_test_basic(dummy_benchmark_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING, highpass_hz=10, lowpass_hz=2000, filter_order=4),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )

    # Use a more targeted mock for MTSpace in this test
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = np.random.rand(9)
        instance.inverse_covariance_matrix = np.eye(9)
        
        def mock_md(features):
            return 5.0 if features.rms > 0.7 else 1.0 # Align with mock_feature_extraction
        instance.calculate_md.side_effect = mock_md
        
        result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

        assert result is not None
        assert result.total_files == 6
        assert result.processed_files == 6
        assert instance.add_normal_sample.call_count == 5
        assert instance.calculate_md.call_count == 6
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

def test_run_benchmark_on_dcase_structure(dcase_like_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    """
    Test that the benchmark runner correctly handles a DCASE-like structure
    with multiple machine IDs, training a separate MTSpace for each.
    """
    benchmark_config = BenchmarkConfig(
        dataset_name="dcase_like_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING, highpass_hz=10, lowpass_hz=2000, filter_order=4),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )

    # We need to see if MTSpace is instantiated twice (once per machine_id)
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        # Each time MTSpace is created, it will have this behavior
        mock_instance = MagicMock()
        mock_instance.mean_vector = np.random.rand(9)
        mock_instance.inverse_covariance_matrix = np.eye(9)
        mock_instance.noise_power_spectrum_avg = np.zeros(100)
        def mock_md(features):
            return 5.0 if features.rms > 0.7 else 1.0 # Corresponds to anomaly
        mock_instance.calculate_md.side_effect = mock_md
        MockMTSpace.return_value = mock_instance

        result = run_benchmark_test(benchmark_config, dcase_like_dataset)

        # 1. Verify MTSpace was instantiated twice (once for id_00, once for id_02)
        assert MockMTSpace.call_count == 2
        
        # 2. Verify samples were added correctly for each MTSpace instance
        # The mock_instance will track calls across both instantiations
        # id_00 has 2 train files, id_02 has 3 train files.
        assert mock_instance.add_normal_sample.call_count == 5 

        # 3. Verify total files and processing
        # id_00 has 2 test files, id_02 has 4 test files
        assert result.total_files == 6
        assert result.processed_files == 6
        assert mock_instance.calculate_md.call_count == 6

        # 4. Verify accuracy
        # The logic is perfect for our mocks (anomaly has rms>0.7 -> md=5.0 > threshold)
        # All files should be classified correctly.
        assert result.accuracy == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1_score == 1.0
        
        # 5. Check file paths in results to ensure both IDs were processed
        file_paths = {str(Path(r.file_path).parent.parent.parent.name) for r in result.file_results}
        assert "id_00" in file_paths
        assert "id_02" in file_paths

def test_run_benchmark_test_with_nr_plugin(dummy_benchmark_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING, highpass_hz=10, lowpass_hz=2000, filter_order=4),
        mt_config=MTConfig(anomaly_threshold=3.0),
        nr_plugin_config={"name": "mock_nr_plugin", "params": {"param1": 1.0}}
    )

    with patch('src.core.benchmarking.plugin_manager.get_plugin') as mock_get_plugin, \
         patch('src.core.benchmarking.perform_nr_evaluation') as mock_nr_eval, \
         patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        
        mock_plugin = MagicMock()
        mock_plugin.process.return_value = np.array([0.1, 0.2]) # Dummy post-NR signal
        mock_get_plugin.return_value = mock_plugin
        
        mock_nr_eval.return_value = MagicMock(
            features_before={'rms': 0.2}, features_after={'rms': 0.1}
        )
        def asdict_side_effect(obj):
            if isinstance(obj, MagicMock):
                return {'features_before': obj.features_before, 'features_after': obj.features_after}
            return asdict(obj)

        instance = MockMTSpace.return_value
        instance.mean_vector = np.random.rand(9)
        instance.inverse_covariance_matrix = np.eye(9)
        instance.noise_power_spectrum_avg = np.zeros(100)
        def mock_md(features):
            return 5.0 if features.rms > 0.7 else 1.0
        instance.calculate_md.side_effect = mock_md

        with patch('src.core.benchmarking.asdict', side_effect=asdict_side_effect):
            result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

            assert result is not None
            assert result.total_files == 6
            assert mock_get_plugin.call_count == 6 # Called per file now
            assert mock_plugin.process.call_count == 6
            assert mock_nr_eval.call_count == 6
            
            assert result.nr_performance_metrics is not None
            assert 'avg_rms_reduction_pct' in result.nr_performance_metrics
            assert result.nr_performance_metrics['avg_rms_reduction_pct'] == pytest.approx(50.0)

def test_run_benchmark_test_no_training_files(dummy_benchmark_dataset):
    shutil.rmtree(dummy_benchmark_dataset / "train")
    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )
    # The new implementation prints a warning and continues instead of raising an error.
    # It should result in zero processed files and metrics.
    result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
    assert result.processed_files == 0
    assert result.total_files == 0
    assert result.accuracy == 0

def test_run_benchmark_test_no_test_files(dummy_benchmark_dataset):
    shutil.rmtree(dummy_benchmark_dataset / "test")
    benchmark_config = BenchmarkConfig(
        dataset_name="dummy_benchmark",
        analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
        mt_config=MTConfig(anomaly_threshold=3.0)
    )
    # Similar to the no_training_files test, this should not error out but result in 0 files.
    result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
    assert result.processed_files == 0
    assert result.total_files == 0
    assert result.accuracy == 0

def test_run_benchmark_test_mt_space_not_established(dummy_benchmark_dataset, mock_load_wav_file):
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = None # Simulate MTSpace not established
        instance.add_normal_sample.return_value = None

        benchmark_config = BenchmarkConfig(
            dataset_name="dummy_benchmark",
            analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
            mt_config=MTConfig(anomaly_threshold=3.0)
        )
        # Should not raise RuntimeError, but print a warning and continue
        result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
        assert result.processed_files == 0 # No files are processed for this machine
        assert result.total_files == 6 # Files are discovered, but not processed
        assert result.accuracy == 0
