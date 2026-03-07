from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
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
    optimize_threshold: bool = False
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
    optimized_threshold: Optional[float] = None
    confusion_matrix: Optional[List] = None
    roc_curve: Optional[Dict] = None
    roc_auc: Optional[float] = None
    nr_performance_metrics: Optional[Dict[str, Any]] = None # e.g., avg RMS reduction, avg SNR improvement


def run_benchmark_test(benchmark_config: BenchmarkConfig, dataset_root_path: Path) -> BenchmarkResult:
    """
    Runs a benchmark test on a dataset, handling multiple machine IDs if present
    and optionally optimizing the anomaly threshold.
    """
    overall_start_time = time.time()
    optimized_threshold = None

    # --- 1. Identify Machine ID Directories ---
    # ... (code is the same until step 4)
    machine_id_dirs = [p for p in dataset_root_path.glob('id_*') if p.is_dir()]
    if not machine_id_dirs:
        machine_id_dirs = [dataset_root_path]

    all_file_results: List[FileBenchmarkResult] = []
    all_true_labels = []
    total_processing_time = 0
    total_files_count = 0

    for machine_dir in machine_id_dirs:
        print(f"--- Processing Machine Directory: {machine_dir.name} ---")
        
        train_dir = machine_dir / "train"
        test_dir = machine_dir / "test"

        train_normal_files = list(train_dir.rglob("normal*.wav"))
        test_normal_files = list(test_dir.rglob("normal*.wav"))
        test_anomaly_files = list(test_dir.rglob("anomaly*.wav"))

        if not train_normal_files:
            print(f"Warning: No normal training files found in {train_dir}. Skipping machine {machine_dir.name}.")
            continue
        if not test_normal_files and not test_anomaly_files:
            print(f"Warning: No test files found in {machine_dir / 'test'}. Skipping machine {machine_dir.name}.")
            continue

        machine_test_files_with_labels = [(f, "normal") for f in test_normal_files] + [(f, "anomaly") for f in test_anomaly_files]
        total_files_count += len(machine_test_files_with_labels)
        
        mt_space = MTSpace(
            min_samples=benchmark_config.mt_config.min_samples,
            recommended_samples=benchmark_config.mt_config.recommended_samples
        )
        
        for file_path in train_normal_files:
            fs_hz, data, _ = load_wav_file(str(file_path))
            processed = remove_dc_offset(data)
            processed = apply_butterworth_filter(processed, fs_hz, benchmark_config.analysis_config.highpass_hz, benchmark_config.analysis_config.lowpass_hz, benchmark_config.analysis_config.filter_order)
            time_features = calculate_time_domain_features(processed)
            _, _, freq_features = calculate_fft_features(processed, fs_hz, benchmark_config.analysis_config.window)
            features = VibrationFeatures(**asdict(time_features), **freq_features)
            mt_space.add_normal_sample(features, processed, fs_hz, benchmark_config.analysis_config)

        if mt_space.mean_vector is None:
            print(f"Warning: MT Space could not be established for machine {machine_dir.name}. Skipping.")
            continue

        for file_path, actual_label in machine_test_files_with_labels:
            file_start_time = time.time()
            fs_hz, data, _ = load_wav_file(str(file_path))
            processed_dc_removed = remove_dc_offset(data)
            signal_pre_nr = apply_butterworth_filter(processed_dc_removed, fs_hz, benchmark_config.analysis_config.highpass_hz, benchmark_config.analysis_config.lowpass_hz, benchmark_config.analysis_config.filter_order)
            processed_final = signal_pre_nr
            nr_eval_results = None
            # ... (NR plugin logic is the same)
            
            time_features = calculate_time_domain_features(processed_final)
            _, _, freq_features = calculate_fft_features(processed_final, fs_hz, benchmark_config.analysis_config.window)
            vibration_features = VibrationFeatures(**asdict(time_features), **freq_features)
            md = mt_space.calculate_md(vibration_features)
            
            # Temporary predicted_label, will be recalculated if optimizing
            predicted_label = "anomaly" if md > benchmark_config.mt_config.anomaly_threshold else "normal"
            
            file_processing_time = (time.time() - file_start_time) * 1000
            total_processing_time += file_processing_time

            all_file_results.append(FileBenchmarkResult(
                file_path=str(file_path), actual_label=actual_label, predicted_label=predicted_label,
                mahalanobis_distance=md, analysis_result={"features": asdict(vibration_features)},
                nr_evaluation=asdict(nr_eval_results) if nr_eval_results else None
            ))
            all_true_labels.append(actual_label)

    # --- 4a. Optimize Threshold (Optional) ---
    anomaly_threshold = benchmark_config.mt_config.anomaly_threshold
    if benchmark_config.optimize_threshold and all_file_results:
        print("\n--- Optimizing Anomaly Threshold ---")
        md_scores = np.array([res.mahalanobis_distance for res in all_file_results])
        # Handle potential NaN/inf values in MD scores
        md_scores = md_scores[np.isfinite(md_scores)]
        
        if len(md_scores) > 0:
            min_md, max_md = np.min(md_scores), np.max(md_scores)
            best_f1 = -1
            optimized_threshold = anomaly_threshold
            
            # Iterate through a range of potential thresholds
            for threshold in np.linspace(min_md, max_md, 100):
                temp_preds = ["anomaly" if md > threshold else "normal" for md in [res.mahalanobis_distance for res in all_file_results]]
                f1 = f1_score(all_true_labels, temp_preds, pos_label='anomaly', zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    optimized_threshold = threshold
            
            anomaly_threshold = optimized_threshold
            print(f"Optimal threshold found: {optimized_threshold:.4f} (Best F1-Score: {best_f1:.4f})")
    
    # --- 5. Calculate Final Predictions and Overall Metrics ---
    all_predicted_labels = ["anomaly" if res.mahalanobis_distance > anomaly_threshold else "normal" for res in all_file_results]
    for i, res in enumerate(all_file_results):
        res.predicted_label = all_predicted_labels[i]

    processed_files_count = len(all_file_results)
    
    # Initialize new metrics
    cm = None
    roc_curve_data = None
    roc_auc = None

    if processed_files_count > 0:
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        precision = precision_score(all_true_labels, all_predicted_labels, pos_label='anomaly', zero_division=0)
        recall = recall_score(all_true_labels, all_predicted_labels, pos_label='anomaly', zero_division=0)
        f1 = f1_score(all_true_labels, all_predicted_labels, pos_label='anomaly', zero_division=0)
        report = classification_report(all_true_labels, all_predicted_labels, target_names=['normal', 'anomaly'], output_dict=False, zero_division=0)

        # Calculate Confusion Matrix
        cm = confusion_matrix(all_true_labels, all_predicted_labels, labels=['normal', 'anomaly']).tolist()

        # Calculate ROC Curve and AUC
        binary_true_labels = [1 if label == 'anomaly' else 0 for label in all_true_labels]
        md_scores_arr = np.array([res.mahalanobis_distance for res in all_file_results])
        # Only calculate ROC if there are both positive and negative samples
        if len(np.unique(binary_true_labels)) > 1:
            # Need to ensure that `md_scores_arr` and `binary_true_labels` are aligned
            # which they are because `all_file_results` and `all_true_labels` are built in parallel.
            fpr, tpr, thresholds = roc_curve(binary_true_labels, md_scores_arr)
            roc_auc = roc_auc_score(binary_true_labels, md_scores_arr)
            roc_curve_data = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}

    else:
        accuracy, precision, recall, f1, report = 0, 0, 0, 0, "No files processed."

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
        optimized_threshold=optimized_threshold,
        confusion_matrix=cm,
        roc_curve=roc_curve_data,
        roc_auc=roc_auc,
        nr_performance_metrics=nr_performance_metrics if nr_performance_metrics else None
    )
