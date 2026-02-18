from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from datetime import datetime
from pathlib import Path

from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics
from src.diagnostics.mt_method import MTSpace
from src.core.plugins import plugin_manager
from src.core.evaluation import perform_nr_evaluation, NoiseReductionEvaluation
from src.utils.audit_log import AnalysisResult


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
    Runs a benchmark test on a dataset, handling multiple machine IDs if present.

    If subdirectories matching 'id_*' are found in `dataset_root_path`, it runs
    a separate benchmark for each (training a unique MTSpace per ID) and aggregates
    the results. Otherwise, it runs a single benchmark on the root directory.
    """
    overall_start_time = time.time()

    # --- 1. Identify Machine ID Directories ---
    machine_id_dirs = [p for p in dataset_root_path.glob('id_*') if p.is_dir()]
    if not machine_id_dirs:
        # If no id_* subdirectories, treat the root as the single machine directory
        machine_id_dirs = [dataset_root_path]

    # --- Aggregated results across all machine IDs ---
    all_file_results: List[FileBenchmarkResult] = []
    all_true_labels = []
    all_predicted_labels = []
    total_processing_time = 0
    total_files_count = 0

    for machine_dir in machine_id_dirs:
        print(f"--- Processing Machine Directory: {machine_dir.name} ---")
        
        # --- 2. Load Dataset Paths for the current machine ---
        train_normal_dir = machine_dir / "train" / "normal"
        test_normal_dir = machine_dir / "test" / "normal"
        test_anomaly_dir = machine_dir / "test" / "anomaly"

        train_normal_files = list(train_normal_dir.rglob("*.wav"))
        test_normal_files = list(test_normal_dir.rglob("*.wav"))
        test_anomaly_files = list(test_anomaly_dir.rglob("*.wav"))

        if not train_normal_files:
            print(f"Warning: No normal training files found in {train_normal_dir}. Skipping machine {machine_dir.name}.")
            continue
        if not test_normal_files and not test_anomaly_files:
            print(f"Warning: No test files found in {machine_dir / 'test'}. Skipping machine {machine_dir.name}.")
            continue

        machine_test_files_with_labels = [
            (f, "normal") for f in test_normal_files
        ] + [
            (f, "anomaly") for f in test_anomaly_files
        ]
        total_files_count += len(machine_test_files_with_labels)

        # --- 3. Train MT Space for the current machine ---
        mt_space = MTSpace(
            min_samples=benchmark_config.mt_config.min_samples,
            recommended_samples=benchmark_config.mt_config.recommended_samples
        )
        
        for file_path in train_normal_files:
            fs_hz, data, _ = load_wav_file(str(file_path))
            processed = remove_dc_offset(data)
            processed = apply_butterworth_filter(
                processed, fs_hz,
                benchmark_config.analysis_config.highpass_hz,
                benchmark_config.analysis_config.lowpass_hz,
                benchmark_config.analysis_config.filter_order
            )
            time_features = calculate_time_domain_features(processed)
            _, _, power_bands = calculate_fft_features(
                processed, fs_hz, benchmark_config.analysis_config.window
            )
            features = VibrationFeatures(
                **asdict(time_features),
                power_low=power_bands['low'],
                power_mid=power_bands['mid'],
                power_high=power_bands['high']
            )
            mt_space.add_normal_sample(features, processed, fs_hz, benchmark_config.analysis_config)

        if mt_space.mean_vector is None:
            print(f"Warning: MT Space could not be established for machine {machine_dir.name}. Skipping.")
            continue

        # --- 4. Evaluate Test Files for the current machine ---
        for file_path, actual_label in machine_test_files_with_labels:
            file_start_time = time.time()
            
            fs_hz, data, _ = load_wav_file(str(file_path))
            processed_dc_removed = remove_dc_offset(data)
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
                plugin_params = benchmark_config.nr_plugin_config.get("params", {})
                selected_plugin = plugin_manager.get_plugin(plugin_name)
                
                if selected_plugin:
                    process_params = {"signal": signal_pre_nr, "fs_hz": fs_hz, **plugin_params}
                    if plugin_name == "spectral_subtraction":
                        process_params["p_noise_avg"] = mt_space.noise_power_spectrum_avg
                    
                    signal_post_nr = selected_plugin.process(**process_params)
                    nr_eval_results = perform_nr_evaluation(signal_pre_nr, signal_post_nr)
                    processed_final = signal_post_nr

            time_features = calculate_time_domain_features(processed_final)
            _, _, power_bands = calculate_fft_features(processed_final, fs_hz, benchmark_config.analysis_config.window)
            vibration_features = VibrationFeatures(
                **asdict(time_features),
                power_low=power_bands['low'],
                power_mid=power_bands['mid'],
                power_high=power_bands['high']
            )
            
            md = mt_space.calculate_md(vibration_features)
            predicted_label = "anomaly" if md > benchmark_config.mt_config.anomaly_threshold else "normal"
            
            file_processing_time = (time.time() - file_start_time) * 1000
            total_processing_time += file_processing_time

            all_file_results.append(FileBenchmarkResult(
                file_path=str(file_path),
                actual_label=actual_label,
                predicted_label=predicted_label,
                mahalanobis_distance=md,
                analysis_result={"features": asdict(vibration_features)},
                nr_evaluation=asdict(nr_eval_results) if nr_eval_results else None
            ))
            all_true_labels.append(actual_label)
            all_predicted_labels.append(predicted_label)

    # --- 5. Calculate Overall Metrics ---
    processed_files_count = len(all_file_results)
    
    accuracy = accuracy_score(all_true_labels, all_predicted_labels) if processed_files_count > 0 else 0
    precision = precision_score(all_true_labels, all_predicted_labels, pos_label='anomaly', zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, pos_label='anomaly', zero_division=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, pos_label='anomaly', zero_division=0)
    
    report = classification_report(all_true_labels, all_predicted_labels, target_names=['normal', 'anomaly'], output_dict=False, zero_division=0)

    avg_processing_time_ms = total_processing_time / processed_files_count if processed_files_count > 0 else 0
    
    nr_performance_metrics = {}
    if benchmark_config.nr_plugin_config:
        rms_reductions = [
            (res.nr_evaluation['features_before']['rms'] - res.nr_evaluation['features_after']['rms']) / res.nr_evaluation['features_before']['rms'] * 100
            for res in all_file_results if res.nr_evaluation and res.nr_evaluation['features_before']['rms'] != 0
        ]
        if rms_reductions:
            nr_performance_metrics['avg_rms_reduction_pct'] = np.mean(rms_reductions)

    return BenchmarkResult(
        benchmark_config=benchmark_config,
        timestamp=datetime.now().isoformat(),
        total_files=total_files_count,
        processed_files=processed_files_count,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        classification_report=report,
        avg_processing_time_ms=avg_processing_time_ms,
        file_results=all_file_results,
        nr_performance_metrics=nr_performance_metrics if nr_performance_metrics else None
    )
