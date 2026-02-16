# Evaluation and Benchmarking Framework

This document explains the noise reduction evaluation framework and the newly implemented benchmarking feature, including their metrics and how to interpret the results within the Vibration Analysis Web Application.

## 1. Introduction

The Noise Reduction Evaluation Framework provides a mechanism to objectively assess the effectiveness of any applied noise reduction plugin. When a noise reduction algorithm is selected and applied, this framework automatically compares the signal's characteristics *before* and *after* the noise reduction process, offering quantitative and visual feedback. This helps users understand the impact of the chosen filter settings.

The **Benchmarking Feature** extends this by allowing systematic evaluation of the entire anomaly detection pipeline (including NR and MT methods) against standardized datasets with known ground truth, providing quantitative performance metrics for the overall system.

## 2. Noise Reduction Evaluation Metrics

The evaluation focuses on comparing key signal properties at two stages:
*   **Before Noise Reduction (Before NR):** The signal after initial preprocessing (DC removal and Butterworth filtering), but *before* the noise reduction plugin is applied.
*   **After Noise Reduction (After NR):** The signal *after* the selected noise reduction plugin has been applied.

The framework primarily uses **Time-Domain Features** for quantitative comparison and **Frequency-Domain Spectra** for visual assessment.

### 2.1 Time-Domain Feature Impact

The framework calculates and compares the following time-domain features:
*   **RMS (Root Mean Square):** Measures the signal's power. A reduction in RMS might indicate successful noise removal if the noise contributes significantly to overall signal power.
*   **Peak:** The maximum absolute amplitude. Noise reduction can lower peak values by removing transient noise.
*   **Kurtosis:** A measure of the "peakedness" or "tailedness" of the signal's probability distribution. It's sensitive to impulsive noise. A reduction often indicates successful removal of impulsive noise.
*   **Skewness:** A measure of the asymmetry of the signal's probability distribution. Can indicate non-linear signal components or uneven noise distribution.
*   **Crest Factor:** The ratio of the peak amplitude to the RMS value. High crest factors often indicate impulsive signals or noise.
*   **Shape Factor:** The ratio of the RMS value to the mean absolute value.

For each of these features, the framework presents the "Before NR" value, the "After NR" value, and a "Delta (%)" showing the percentage change. A significant negative delta (reduction) for RMS, Peak, or Kurtosis might indicate effective noise removal, depending on the nature of the noise.

### 2.2 Removed Signal

The framework also conceptually considers the **"Removed Signal"**, which is mathematically derived as:
`Removed Signal = Signal Before NR - Signal After NR`

Analyzing the characteristics (especially the frequency spectrum) of this removed signal can provide insight into what frequencies and types of components the noise reduction algorithm is primarily targeting.

## 3. Interpreting the Noise Reduction UI

In the Streamlit application, when a noise reduction plugin is active, a dedicated "ノイズ除去 評価" (Noise Reduction Evaluation) section appears.

### 3.1 特徴量へのインパクト (Impact on Features)

このテーブルは、ノイズ除去フィルターの適用前後における主要な時間領域特徴量の値を比較します。

*   **Before:** ノイズ除去適用前の特徴量の値。
*   **After:** ノイズ除去適用後の特徴量の値。
*   **Delta (%)**: ノイズ除去による特徴量の変化率。例えば、RMSのDeltaが負の大きい値であれば、ノイズ除去により信号のパワーが効果的に減少したことを示唆します。

### 3.2 スペクトル比較 (Spectral Comparison)

このグラフは、以下の3つの信号の周波数スペクトルを重ねて表示します。

*   **Before NR (lightblue):** ノイズ除去適用前の信号のFFTスペクトル。
*   **After NR (blue):** ノイズ除去適用後の信号のFFTスペクトル。
*   **Removed Signal (red, dashed):** フィルターによって除去された信号のFFTスペクトル。

この可視化により、ユーザーはどの周波数帯域のノイズがどの程度除去されたかを直感的に理解できます。例えば、ノッチフィルターが適用された場合、特定の周波数帯域で「After NR」のスペクトルが「Before NR」よりも顕著に低下し、「Removed Signal」のスペクトルがその周波数帯域でピークを持つことが期待されます。

## 4. Benchmarking Feature

The benchmarking feature allows for a systematic and quantitative evaluation of the entire vibration analysis pipeline against a known dataset. This is crucial for validating algorithm performance, comparing different configurations, and ensuring robustness.

### 4.1 How to Use the Benchmarking Feature

1.  **Navigate to the "ベンチマーク" (Benchmark) Page:** Select "ベンチマーク" from the sidebar menu.
2.  **Select a Benchmark Dataset:** Choose from the available benchmark datasets. Currently, an example dataset like "DCASE-2020-Task2-Dataset/pump" is integrated. These datasets typically contain `train/normal`, `test/normal`, and `test/anomaly` subfolders.
3.  **Configure Analysis Settings:**
    *   **物理量種別 (Signal Quantity):** `ACCEL`, `VELOCITY`, `DISPLACEMENT`.
    *   **窓関数 (Window Function):** `hanning`, `flattop`.
    *   **HPF (Hz) / LPF (Hz) / フィルタ次数 (Filter Order):** Standard Butterworth filter settings applied to all signals.
4.  **Configure MT Method Settings:**
    *   **異常判定閾値 (MD):** The Mahalanobis Distance threshold above which a sample is classified as "anomaly".
    *   **最小正常サンプル数 (単位空間構築):** Minimum number of normal training samples required to build the MT space.
    *   **推奨正常サンプル数:** Recommended number of normal training samples for a robust MT space.
5.  **Configure Noise Reduction Plugin (Optional):**
    *   Select a noise reduction algorithm (e.g., `notch_filter`, `band_stop_filter`, `spectral_subtraction`).
    *   Adjust its specific parameters (e.g., `freq_hz`, `q_factor` for notch filter). If `spectral_subtraction` is selected, the system will automatically attempt to use the noise profile learned from the normal training data of the benchmark dataset.
6.  **Run Benchmark:** Click the "ベンチマークを実行" button. The application will process the training data to build the MT space and learn any required noise profiles, then analyze all test files.

### 4.2 Interpreting Benchmark Results

After a benchmark run, the application displays a comprehensive summary:

*   **Overall Metrics:**
    *   **精度 (Accuracy):** The proportion of correctly classified samples (both normal and anomaly).
    *   **適合率 (Precision):** Of all samples predicted as anomaly, what proportion were actually anomaly. High precision means fewer false positives.
    *   **再現率 (Recall):** Of all actual anomaly samples, what proportion were correctly identified. High recall means fewer false negatives.
    *   **F1スコア (F1-score):** The harmonic mean of precision and recall, providing a balanced measure.
*   **分類レポート (Classification Report):** A detailed text report showing precision, recall, f1-score, and support for both "normal" and "anomaly" classes.
*   **平均処理時間 (ファイルあたり):** The average time taken to process each file in the test set, indicating computational efficiency.
*   **総ファイル数 / 処理済みファイル数:** Total number of files in the test set and how many were successfully processed.
*   **ノイズ除去性能 (ベンチマーク全体):** If a noise reduction plugin was used, this section provides aggregated metrics such as `avg_rms_reduction_pct`, indicating the average percentage reduction in RMS across all test files due to the NR plugin.
*   **ファイルごとの詳細結果:** An expandable section (table) showing results for each individual test file, including its actual label, predicted label, Mahalanobis Distance, and detailed analysis/NR evaluation results.

These metrics allow users to quantitatively compare the effectiveness of different algorithm configurations and noise reduction strategies on a standardized dataset.

### 4.3 Custom and Synthetic Benchmarking Datasets

To use your own custom or synthetic datasets for benchmarking, ensure they follow a specific directory structure. This structure allows the application to correctly identify training (normal) and test (normal/anomaly) files.

The expected structure within your custom dataset's root directory should be:

```
your_custom_dataset_name/
├── train/
│   └── normal/
│       ├── normal_train_01.wav
│       ├── normal_train_02.wav
│       └── ...
├── test/
│   ├── normal/
│   │   ├── normal_test_01.wav
│   │   ├── normal_test_02.wav
│   │   └── ...
│   └── anomaly/
│       ├── anomaly_test_01.wav
│       ├── anomaly_test_02.wav
│       └── ...
```

*   **`your_custom_dataset_name/`**: This is the root directory for your dataset. When selecting a dataset in the UI, you would point to this directory (or one of its parent directories if the UI provides a full path selector).
*   **`train/normal/`**: Contains WAV files representing *normal* operating conditions, used for training the MT space and learning noise profiles (if applicable).
*   **`test/normal/`**: Contains WAV files representing *normal* operating conditions, used as part of the test set for evaluation.
*   **`test/anomaly/`**: Contains WAV files representing *anomalous* operating conditions, used as part of the test set for evaluation.

All WAV files should be mono, 16-bit PCM, and have a consistent sampling rate within a given dataset. Adhering to this structure ensures that the benchmarking framework can correctly process and evaluate your custom data.