# User Guide

This guide provides comprehensive instructions on how to effectively use the Vibration Analysis Web Application.

## 1. Getting Started

The Vibration Analysis Web Application is designed to provide powerful tools for analyzing vibration data. It focuses on physical correctness, reproducibility, and extensibility, making it suitable for industrial and operational settings.

To start the application, ensure you have followed the [installation instructions in `README.md`](../README.md). Then, run the application using:

```bash
python -m streamlit run src/app.py
```

Open the provided local URL (e.g., `http://localhost:8501`) in your web browser.

## 2. Uploading Data

The application supports vibration data from both WAV files (LPCM 16/24/32bit) and CSV files. **Multiple files can be uploaded simultaneously for batch analysis.**

*   **Clear All Files:** Click the "🗑️ 全ファイルをクリア" (Clear All Files) button to remove all uploaded files from the list and reset the analysis results.

### For WAV Files:
*   The application will display the file name, sampling frequency (Fs), and duration of the loaded data.
*   An audio player will also appear, allowing you to listen to the uploaded file.

### For CSV Files:
When a CSV file is uploaded, a new section **"CSV解析設定" (CSV Parsing Settings)** will appear in the sidebar:
*   **加速度データ列を選択 (Select Columns):** Choose one or more columns containing vibration data.
*   **多軸合成を行う (Multi-axis Synthesis):** If multiple columns (e.g., X, Y, Z) are selected, you can toggle vector magnitude synthesis ($\sqrt{X^2 + Y^2 + Z^2}$). This removes DC bias (like gravity) from each axis before synthesis to focus on dynamic vibration.
*   **解析対象とするメインの軸を選択 (Select Main Axis):** If synthesis is OFF, you must select one specific axis as the primary target for feature extraction and MT diagnosis. Other axes will still be shown in the background of plots for comparison.
*   **タイムスタンプ列を使用する (Use Timestamp):** Automatically infer sampling frequency from a timestamp column. Supports ISO 8601 and Unix timestamps.
*   **サンプリング周波数 (Hz) を入力:** Manually specify frequency if no timestamp is available.

## 3. Analysis Configuration & Presets

### ⚙️ Analysis Presets (解析プリセット)
To improve operational efficiency, you can save and load analysis configurations for specific machines.
*   **Apply Preset:** Select a saved preset (e.g., "Pump A - Outlet") from the dropdown and click "プリセットを適用". All settings including filters, windowing, and peak settings will be updated.
*   **Save Current Settings:** Enter a name and click "現在の設定を保存" to persist your current configuration to `data/analysis_presets.json`.

### Core Settings
*   **物理量種別 (Quantity Type):** `accel` (m/s²), `velocity` (mm/s), or `disp` (μm). Affects units and physical interpretation.
*   **窓関数 (Window):** `hanning` (default) or `flattop`.
*   **フィルタ設定 (Filters):** HPF and LPF can be toggled independently. Cutoff frequencies are visually synchronized with FFT and Spectrogram charts.
*   **FFTピーク設定:** Configure the sensitivity of automated peak annotation.

## 4. Interpreting Results

### 📋 Diagnosis Summary (診断サマリー)
When multiple files are uploaded, a table at the top displays a summary of all results:
*   **MD値 (異常度):** Mahalanobis Distance. Values > 3.0 indicate potential anomalies.
*   **判定 (Status):** 🟢 Normal, 🟡 Warning, or 🔴 Anomaly.
*   **信頼度 (Confidence):** Data quality score (0-100%).

### Visual Analysis (Tabs)
The application provides three interactive tabs for deep-dive analysis:

1.  **🕒 時間領域 (Waveform Comparison):**
    *   Compares the **Processed Signal** (filtered) with the **Raw Signal** (DC removed only).
    *   If multi-axis data is available, all axes are shown in the background to help identify the direction of impact.
    *   Includes automatic downsampling for smooth interaction with large datasets.

2.  **📊 周波数領域 (FFT):**
    *   Displays the frequency spectrum with automated peak annotations (Hz and Amplitude).
    *   Supports **Linear/Logarithmic** X-axis switching.
    *   Shows filter cutoff boundaries as red dashed lines.
    *   Displays individual spectra for each axis if multi-axis data is loaded.

3.  **🌈 時間-周波数領域 (Spectrogram):**
    *   Shows how frequency components change over time.
    *   Adjustable **Resolution (Window Size)** to balance time/frequency precision.
    *   Provides an overview of each individual axis to pinpoint time-localized events.

### Data Quality & Confidence
*   **クリッピング率:** Ratio of saturated samples. High values indicate the sensor range was exceeded.
*   **S/N 比:** Signal-to-noise ratio.
*   **🤔 診断信頼度:** A composite score based on clipping, SNR, and data length. Aim for > 80% for reliable diagnosis.

## 5. MT Method (Mahalanobis-Taguchi Method)

Anomaly detection requires a "Unit Space" (baseline) built from normal data.
1.  Expand **"MT法設定"** in the sidebar.
2.  Upload 10 or more (30+ recommended) normal files.
3.  Click **"単位空間を構築/更新"**. The system will now use this baseline to calculate MD values for any evaluated data.

## 6. Audit Log (監査ログ)

For reproducibility and compliance, all analysis parameters and results are available in JSON format at the bottom of the page. You can download this log to recreate the exact analysis conditions later.
