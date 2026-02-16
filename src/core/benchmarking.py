from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics
from src.diagnostics.mt_method import MTSpace
from src.core.plugins import plugin_manager # Import plugin manager
from src.core.evaluation import perform_nr_evaluation, NoiseReductionEvaluation # Import evaluation components
from src.utils.audit_log import AnalysisResult
from pathlib import Path


@dataclass
class MTConfig:
    anomaly_threshold: float = 3.0
    min_samples: int = 10
    recommended_samples: int = 30

@dataclass
class BenchmarkConfig:
    dataset_name: str
    analysis_config: AnalysisConfig
    mt_config: MTConfig
    nr_plugin_config: Optional[Dict[str, Any]] = None # {"name": "plugin_name", "params": {...}}

@dataclass
class FileBenchmarkResult:
    file_path: str
    actual_label: str
    predicted_label: str
    mahalanobis_distance: float
    analysis_result: Dict[str, Any] # Full analysis result for this file
    nr_evaluation: Optional[Dict[str, Any]] = None

@dataclass
class BenchmarkResult:
    benchmark_config: BenchmarkConfig
    timestamp: str
    total_files: int
    processed_files: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    classification_report: str
    avg_processing_time_ms: float
    file_results: List[FileBenchmarkResult]
    nr_performance_metrics: Optional[Dict[str, Any]] = None # e.g., avg RMS reduction, avg SNR improvement


def run_benchmark_test(benchmark_config: BenchmarkConfig, dataset_root_path: Path) -> BenchmarkResult:
    """
    Runs a benchmark test using the specified configuration and dataset.
    """
    start_time = time.time()
    
    # --- 1. Load Dataset Paths ---
    train_normal_dir = dataset_root_path / "train" / "normal"
    test_normal_dir = dataset_root_path / "test" / "normal"
    test_anomaly_dir = dataset_root_path / "test" / "anomaly"

    train_normal_files = list(train_normal_dir.rglob("*.wav"))
    test_normal_files = list(test_normal_dir.rglob("*.wav"))
    test_anomaly_files = list(test_anomaly_dir.rglob("*.wav"))

    if not train_normal_files:
        raise ValueError(f"No normal training files found in {train_normal_dir}")
    if not test_normal_files and not test_anomaly_files:
        raise ValueError(f"No test files found in {dataset_root_path / 'test'}")

    all_test_files_with_labels = [
        (f, "normal") for f in test_normal_files
    ] + [
        (f, "anomaly") for f in test_anomaly_files
    ]
    
    # --- 2. Train MT Space and Learn Noise Profile ---
    mt_space = MTSpace(
        min_samples=benchmark_config.mt_config.min_samples,
        recommended_samples=benchmark_config.mt_config.recommended_samples
    )
    
    # Use benchmark_config.analysis_config for training
    # This involves processing normal files and adding them to mt_space
    for file_path in train_normal_files:
        fs_hz, data_normalized, _ = load_wav_file(str(file_path))
        processed_dc_removed = remove_dc_offset(data_normalized)
        processed_final_pre_nr = apply_butterworth_filter(
            processed_dc_removed, fs_hz, 
            benchmark_config.analysis_config.highpass_hz, 
            benchmark_config.analysis_config.lowpass_hz, 
            benchmark_config.analysis_config.filter_order
        )
        
        time_features = calculate_time_domain_features(processed_final_pre_nr)
        _, _, power_bands_dict = calculate_fft_features(
            processed_final_pre_nr, fs_hz, benchmark_config.analysis_config.window
        )
        normal_features = VibrationFeatures(
            **asdict(time_features),
            power_low=power_bands_dict['low'],
            power_mid=power_bands_dict['mid'],
            power_high=power_bands_dict['high']
        )
        mt_space.add_normal_sample(normal_features, processed_final_pre_nr, fs_hz, benchmark_config.analysis_config)
    
    if mt_space.mean_vector is None:
        raise RuntimeError("MT Space could not be established from normal training data.")

    # --- 3. Evaluate Test Files ---
    file_results: List[FileBenchmarkResult] = []
    true_labels = []
    predicted_labels = []
    total_processing_time = 0

    for file_path, actual_label in all_test_files_with_labels:
        file_start_time = time.time()
        
        # Load and preprocess
        fs_hz, data_normalized, file_hash = load_wav_file(str(file_path))
        processed_dc_removed = remove_dc_offset(data_normalized)
        
        signal_pre_nr = apply_butterworth_filter(
            processed_dc_removed, fs_hz, 
            benchmark_config.analysis_config.highpass_hz, 
            benchmark_config.analysis_config.lowpass_hz, 
            benchmark_config.analysis_config.filter_order
        )
        
        nr_eval_results = None
        processed_final = signal_pre_nr

        if benchmark_config.nr_plugin_config and mt_space.noise_power_spectrum_avg is not None:
            plugin_name = benchmark_config.nr_plugin_config["name"]
            plugin_params = benchmark_config.nr_plugin_config["params"]
            selected_plugin = plugin_manager.get_plugin(plugin_name)
            
            if selected_plugin:
                if plugin_name == "spectral_subtraction":
                    signal_post_nr = selected_plugin.process(
                        signal_pre_nr,
                        fs_hz,
                        p_noise_avg=mt_space.noise_power_spectrum_avg, # Pass learned noise profile
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
                print(f"Warning: NR Plugin '{plugin_name}' not found. Skipping NR for {file_path.name}")
        
        # Extract features from processed_final
        time_features = calculate_time_domain_features(processed_final)
        _, _, power_bands_dict = calculate_fft_features(processed_final, fs_hz, benchmark_config.analysis_config.window)
        
        vibration_features = VibrationFeatures(
            **asdict(time_features),
            power_low=power_bands_dict['low'],
            power_mid=power_bands_dict['mid'],
            power_high=power_bands_dict['high']
        )
        
        # Calculate MD
        md = mt_space.calculate_md(vibration_features)
        
        # Predict label
        predicted_label = "anomaly" if md > benchmark_config.mt_config.anomaly_threshold else "normal"
        
        file_processing_time = (time.time() - file_start_time) * 1000 # ms
        total_processing_time += file_processing_time

        file_results.append(FileBenchmarkResult(
            file_path=str(file_path),
            actual_label=actual_label,
            predicted_label=predicted_label,
            mahalanobis_distance=md,
            analysis_result={"features": asdict(vibration_features)}, # Simplified, could include more
            nr_evaluation=asdict(nr_eval_results) if nr_eval_results else None
        ))
        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)

    # --- 4. Calculate Overall Metrics ---
    total_files = len(all_test_files_with_labels)
    processed_files = len(file_results)
    
    accuracy = accuracy_score(true_labels, predicted_labels) if processed_files > 0 else 0
    precision = precision_score(true_labels, predicted_labels, pos_label='anomaly', zero_division=0) if processed_files > 0 else 0
    recall = recall_score(true_labels, predicted_labels, pos_label='anomaly', zero_division=0) if processed_files > 0 else 0
    f1 = f1_score(true_labels, predicted_labels, pos_label='anomaly', zero_division=0) if processed_files > 0 else 0
    
    report = classification_report(true_labels, predicted_labels, target_names=['normal', 'anomaly'], output_dict=False, zero_division=0)

    avg_processing_time_ms = total_processing_time / processed_files if processed_files > 0 else 0
    
    # NR performance metrics (e.g., average RMS reduction)
    nr_performance_metrics = {}
    if benchmark_config.nr_plugin_config:
        rms_reductions = [
            (res.nr_evaluation['features_before']['rms'] - res.nr_evaluation['features_after']['rms']) / res.nr_evaluation['features_before']['rms'] * 100
            for res in file_results if res.nr_evaluation and res.nr_evaluation['features_before']['rms'] != 0
        ]
        if rms_reductions:
            nr_performance_metrics['avg_rms_reduction_pct'] = np.mean(rms_reductions)
            nr_performance_metrics['min_rms_reduction_pct'] = np.min(rms_reductions)
            nr_performance_metrics['max_rms_reduction_pct'] = np.max(rms_reductions)

    total_duration = time.time() - start_time

    return BenchmarkResult(
        benchmark_config=benchmark_config,
        timestamp=datetime.now().isoformat(),
        total_files=total_files,
        processed_files=processed_files,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        classification_report=report,
        avg_processing_time_ms=avg_processing_time_ms,
        file_results=file_results,
        nr_performance_metrics=nr_performance_metrics if nr_performance_metrics else None
    )
