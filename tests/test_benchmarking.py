import pytest
from pathlib import Path
import numpy as np
from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass, asdict
from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures, TimeDomainFeatures
from src.core.benchmarking import run_benchmark_test, BenchmarkConfig, MTConfig
from src.diagnostics.mt_method import MTSpace
import shutil

@pytest.fixture
def dummy_benchmark_dataset(tmp_path):
    dataset_root = tmp_path / "dummy_benchmark"
    (dataset_root / "train" / "normal").mkdir(parents=True)
    (dataset_root / "test" / "normal").mkdir(parents=True)
    (dataset_root / "test" / "anomaly").mkdir(parents=True)
    for i in range(5):
        (dataset_root / "train" / "normal" / f"normal_train_{i}.wav").touch()
    for i in range(3):
        (dataset_root / "test" / "normal" / f"normal_test_{i}.wav").touch()
        (dataset_root / "test" / "anomaly" / f"anomaly_test_{i}.wav").touch()
    return dataset_root

@pytest.fixture
def dcase_like_dataset(tmp_path):
    dataset_root = tmp_path / "dcase_like_benchmark"
    id_00_dir = dataset_root / "id_00"
    (id_00_dir / "train" / "normal").mkdir(parents=True)
    (id_00_dir / "test" / "normal").mkdir(parents=True)
    (id_00_dir / "test" / "anomaly").mkdir(parents=True)
    for i in range(2):
        (id_00_dir / "train" / "normal" / f"normal_train_00_{i}.wav").touch()
    (id_00_dir / "test" / "normal" / "normal_test_00_0.wav").touch()
    (id_00_dir / "test" / "anomaly" / "anomaly_test_00_0.wav").touch()
    id_02_dir = dataset_root / "id_02"
    (id_02_dir / "train" / "normal").mkdir(parents=True)
    (id_02_dir / "test" / "normal").mkdir(parents=True)
    (id_02_dir / "test" / "anomaly").mkdir(parents=True)
    for i in range(3):
        (id_02_dir / "train" / "normal" / f"normal_train_02_{i}.wav").touch()
    for i in range(2):
        (id_02_dir / "test" / "normal" / f"normal_test_02_{i}.wav").touch()
        (id_02_dir / "test" / "anomaly" / f"anomaly_test_02_{i}.wav").touch()
    return dataset_root

@pytest.fixture
def mock_load_wav_file():
    with patch('src.core.benchmarking.load_wav_file') as mock:
        def _mock_load(file_path_str):
            fs = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            path_obj = Path(file_path_str)
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
    def mock_calc_time_features(signal):
        is_anomaly = np.max(signal) > 0.7
        if is_anomaly:
            return TimeDomainFeatures(rms=0.8, peak=0.9, kurtosis=5.0, skewness=0.1, crest_factor=2.2, shape_factor=1.2)
        return TimeDomainFeatures(rms=0.1, peak=0.2, kurtosis=3.0, skewness=0.0, crest_factor=2.0, shape_factor=1.0)
    
    mock_freq_features = {
        'power_low': 0.3, 'power_mid': 0.4, 'power_high': 0.3, 
        'spectral_centroid': 1500.0, 'spectral_spread': 500.0, 'spectral_entropy': 8.0,
        'overall_level': 1.0, 'overall_low': 0.8, 'overall_high': 0.6
    }
    
    with patch('src.core.benchmarking.calculate_time_domain_features', side_effect=mock_calc_time_features) as mock_time, \
         patch('src.core.benchmarking.calculate_fft_features', return_value=(None, None, mock_freq_features)):
        yield mock_time

def test_run_benchmark_test_basic(dummy_benchmark_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    benchmark_config = BenchmarkConfig(dataset_name="dummy_benchmark", analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING), mt_config=MTConfig(anomaly_threshold=3.0))
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = np.random.rand(15) # Match new dimension
        def mock_md(features):
            return 5.0 if features.rms > 0.7 else 1.0
        instance.calculate_md.side_effect = mock_md
        result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
        assert result.processed_files == 6
        assert result.accuracy == 1.0

def test_run_benchmark_on_dcase_structure(dcase_like_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    benchmark_config = BenchmarkConfig(dataset_name="dcase_like_benchmark", analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING), mt_config=MTConfig(anomaly_threshold=3.0))
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        mock_instance = MagicMock()
        mock_instance.mean_vector = np.random.rand(15)
        def mock_md(features):
            return 5.0 if features.rms > 0.7 else 1.0
        mock_instance.calculate_md.side_effect = mock_md
        MockMTSpace.return_value = mock_instance
        result = run_benchmark_test(benchmark_config, dcase_like_dataset)
        assert MockMTSpace.call_count == 2
        assert result.processed_files == 6
        assert result.accuracy == 1.0

def test_run_benchmark_test_no_training_files(dummy_benchmark_dataset):
    shutil.rmtree(dummy_benchmark_dataset / "train")
    benchmark_config = BenchmarkConfig(dataset_name="dummy_benchmark", analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING), mt_config=MTConfig())
    result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
    assert result.processed_files == 0

def test_run_benchmark_test_no_test_files(dummy_benchmark_dataset):
    shutil.rmtree(dummy_benchmark_dataset / "test")
    benchmark_config = BenchmarkConfig(dataset_name="dummy_benchmark", analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING), mt_config=MTConfig())
    result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
    assert result.processed_files == 0

def test_run_benchmark_test_mt_space_not_established(dummy_benchmark_dataset, mock_load_wav_file):
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = None
        benchmark_config = BenchmarkConfig(dataset_name="dummy_benchmark", analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING), mt_config=MTConfig())
        result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
        assert result.processed_files == 0

def test_run_benchmark_test_with_optimization(dummy_benchmark_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = np.random.rand(15)
        md_side_effects = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        instance.calculate_md.side_effect = md_side_effects
        benchmark_config = BenchmarkConfig(
            dataset_name="dummy_benchmark",
            analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
            mt_config=MTConfig(anomaly_threshold=0.5),
            optimize_threshold=True
        )
        result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)
        assert result.optimized_threshold is not None
        assert 3.0 < result.optimized_threshold < 4.0
        assert result.f1_score == 1.0

def test_run_benchmark_test_new_metrics(dummy_benchmark_dataset, mock_load_wav_file, mock_signal_processing, mock_feature_extraction):
    with patch('src.core.benchmarking.MTSpace') as MockMTSpace:
        instance = MockMTSpace.return_value
        instance.mean_vector = np.random.rand(15)
        md_side_effects = [1.0, 4.0, 2.0, 2.5, 5.0, 6.0]
        instance.calculate_md.side_effect = md_side_effects

        benchmark_config = BenchmarkConfig(
            dataset_name="dummy_benchmark",
            analysis_config=AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING),
            mt_config=MTConfig(anomaly_threshold=3.0),
            optimize_threshold=False
        )

        result = run_benchmark_test(benchmark_config, dummy_benchmark_dataset)

        assert result.confusion_matrix is not None
        assert result.confusion_matrix == [[2, 1], [1, 2]]
        assert result.roc_curve is not None
        assert result.roc_auc is not None

        from sklearn.metrics import roc_auc_score as test_roc_auc_score
        binary_true_labels = [0, 0, 0, 1, 1, 1]
        assert np.isclose(result.roc_auc, test_roc_auc_score(binary_true_labels, md_side_effects))
