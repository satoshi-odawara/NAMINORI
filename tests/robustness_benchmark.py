# -*- coding: utf-8 -*-
"""
Script for benchmarking the MT method's robustness against varied noise levels.

This script will:
1. Define a baseline 'normal' and 'anomaly' signal configuration.
2. Define a range of noise levels (SNRs) to test.
3. For each SNR level:
    a. Generate a synthetic benchmark dataset (train/normal, test/normal, test/anomaly).
    b. Create a BenchmarkConfig.
    c. Run the benchmark using the core benchmarking function.
    d. Print and store the results.
4. Finally, it will print a summary of how performance metrics (like F1-score)
   degrade as noise increases.
"""
import shutil
import tempfile
from pathlib import Path
import numpy as np

# This script is intended to be run from the project root directory.
# Ensure the source is in the path.
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import synthetic_data_generator as sdg
from src.core.benchmarking import run_benchmark_test, BenchmarkConfig, MTConfig
from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction
from scipy.io.wavfile import write as write_wav

def create_synthetic_dataset(root_dir: Path, signal_config: sdg.SignalConfig, num_train: int, num_test_normal: int, num_test_anomaly: int):
    """
    Generates and saves a synthetic dataset based on signal configurations.
    """
    # Create directories
    (root_dir / "train" / "normal").mkdir(parents=True)
    (root_dir / "test" / "normal").mkdir(parents=True)
    (root_dir / "test" / "anomaly").mkdir(parents=True)

    # Generate and save normal training files
    print(f"Generating {num_train} normal training files...")
    for i in range(num_train):
        signal = sdg.generate_signal(signal_config)
        wav_data = (signal * 32767).astype(np.int16)
        write_wav(root_dir / "train" / "normal" / f"train_normal_{i}.wav", signal_config.fs_hz, wav_data)

    # Generate and save normal test files
    print(f"Generating {num_test_normal} normal test files...")
    for i in range(num_test_normal):
        signal = sdg.generate_signal(signal_config)
        wav_data = (signal * 32767).astype(np.int16)
        write_wav(root_dir / "test" / "normal" / f"test_normal_{i}.wav", signal_config.fs_hz, wav_data)

    # Generate and save anomaly test files
    print(f"Generating {num_test_anomaly} anomaly test files...")
    anomaly_config = sdg.SignalConfig(**sdg.asdict(signal_config)) # Create a copy
    # Introduce a slight anomaly - e.g., add AM modulation
    if not anomaly_config.am_config:
        anomaly_config.am_config = sdg.AMConfig(mod_freq_hz=15.0, mod_index=0.3)
    else: # If AM already exists, change it
        anomaly_config.am_config.mod_index += 0.2
        
    for i in range(num_test_anomaly):
        signal = sdg.generate_signal(anomaly_config)
        wav_data = (signal * 32767).astype(np.int16)
        write_wav(root_dir / "test" / "anomaly" / f"test_anomaly_{i}.wav", signal_config.fs_hz, wav_data)


def main():
    """
    Main function to run the robustness benchmark.
    """
    # --- 1. Define Benchmark Scenarios ---
    snr_levels_db = [30, 20, 10, 5, 0, -5] # SNR levels to test
    
    base_signal_config = sdg.SignalConfig(
        fs_hz=48000,
        duration_s=2.0,
        base_freq_hz=120.0,
        base_amplitude=0.6,
    )
    
    analysis_config = AnalysisConfig(
        quantity=SignalQuantity.ACCEL,
        window=WindowFunction.HANNING,
        highpass_hz=10.0,
        lowpass_hz=20000.0,
        filter_order=4
    )
    mt_config = MTConfig(anomaly_threshold=3.0)

    results_summary = []
    
    # Create a temporary directory for all benchmark runs
    with tempfile.TemporaryDirectory() as base_tmp_dir:
        base_tmp_path = Path(base_tmp_dir)

        for snr in snr_levels_db:
            print("
" + "="*50)
            print(f"--- Running Benchmark for SNR = {snr} dB ---")
            print("="*50)

            # --- 2. Generate Dataset for current SNR ---
            scenario_dir = base_tmp_path / f"snr_{snr}db"
            
            current_signal_config = sdg.SignalConfig(**sdg.asdict(base_signal_config))
            current_signal_config.noise_config = sdg.NoiseConfig(noise_type=sdg.NoiseType.WHITE, snr_db=snr)
            
            create_synthetic_dataset(
                root_dir=scenario_dir,
                signal_config=current_signal_config,
                num_train=30,
                num_test_normal=10,
                num_test_anomaly=10
            )

            # --- 3. Run Benchmark ---
            benchmark_config = BenchmarkConfig(
                dataset_name=f"RobustnessTest_SNR_{snr}dB",
                analysis_config=analysis_config,
                mt_config=mt_config
            )
            
            try:
                result = run_benchmark_test(benchmark_config, scenario_dir)
                
                print(f"
--- Results for SNR = {snr} dB ---")
                print(f"  Accuracy:  {result.accuracy:.2%}")
                print(f"  Precision: {result.precision:.2%}")
                print(f"  Recall:    {result.recall:.2%}")
                print(f"  F1-Score:  {result.f1_score:.3f}")

                results_summary.append({
                    "SNR (dB)": snr,
                    "Accuracy": result.accuracy,
                    "Precision": result.precision,
                    "Recall": result.recall,
                    "F1-Score": result.f1_score,
                })
            except Exception as e:
                print(f"!!! ERROR during benchmark for SNR={snr}dB: {e}")
                results_summary.append({
                    "SNR (dB)": snr,
                    "Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0,
                    "Error": str(e)
                })

    # --- 4. Print Final Summary ---
    print("
" + "="*60)
    print("      MT Method Robustness Benchmark Summary")
    print("="*60)
    if results_summary:
        import pandas as pd
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
    else:
        print("No results were generated.")


if __name__ == "__main__":
    main()
