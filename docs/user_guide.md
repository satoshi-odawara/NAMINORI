# User Guide

This guide provides comprehensive instructions on how to effectively use the Vibration Analysis Web Application.

## 1. Getting Started

The Vibration Analysis Web Application is designed to provide powerful tools for analyzing vibration data from WAV files. It focuses on physical correctness, reproducibility, and extensibility, making it suitable for industrial and operational settings.

To start the application, ensure you have followed the [installation instructions in `README.md`](../README.md). Then, run the application using:

```bash
python -m streamlit run src/app.py
```

Open the provided local URL (e.g., `http://localhost:8501`) in your web browser.

## 2. Uploading Data

The application supports vibration data from both WAV files (LPCM 16/24/32bit) and CSV files.

*   **To upload a file:** Click the "評価用WAVまたはCSVファイルをアップロード" (Upload WAV or CSV file for evaluation) button in the main content area and select your `.wav` or `.csv` file.

### For WAV Files:
*   The application will display the file name, sampling frequency (Fs), and duration of the loaded data.
*   An audio player will also appear, allowing you to listen to the uploaded file.

### For CSV Files:
When a CSV file is uploaded, a new section "CSV解析設定" (CSV Parsing Settings) will appear in the sidebar, providing options to configure how the data is interpreted:
*   **加速度データ列を選択 (Select Acceleration Data Column):** Choose the column from your CSV file that contains the vibration (acceleration) data. The application will attempt to pre-select a suitable numeric column.
*   **タイムスタンプ列を使用する (Use Timestamp Column):** Check this box if your CSV file contains a column with timestamps.
    *   **タイムスタンプ列を選択 (Select Timestamp Column):** If the above box is checked, choose the column containing your timestamp information. The application supports various datetime formats, including ISO 8601 (e.g., `YYYY-MM-DD HH:MM:SS.sss`) and Unix timestamps (seconds or milliseconds since epoch). The sampling frequency will be automatically inferred from these timestamps.
    *   **サンプリング周波数 (Hz) を入力 (Enter Sampling Frequency (Hz)):** If you choose *not* to use a timestamp column, you must manually input the sampling frequency of your data in Hertz.
*   **CSVプレビュー (CSV Preview):** The sidebar will also show a preview of the first few rows of your CSV file to help you verify column selections.

The application will display the file name, inferred or specified sampling frequency (Fs), and duration of the loaded data. If inconsistent sampling intervals are detected when using a timestamp column, a warning will be issued, and an average frequency will be used.

## 3. Analysis Settings

The sidebar on the left provides various settings to configure your analysis.

*   **物理量種別 (Physical Quantity Type):** Select the physical quantity that your WAV file represents (e.g., `accel` for acceleration, `velocity` for velocity, `disp` for displacement). This affects unit display and feature calculations.
*   **窓関数 (Window Function):** Choose the window function for FFT calculation. `hanning` is the default, suitable for general purpose. `flattop` provides better amplitude accuracy for single tones.
*   **標準フィルタ設定 (Standard Filter Settings):**
    *   **HPF (Hz):** High-Pass Filter cutoff frequency. Frequencies below this will be attenuated. Set to 0 for no HPF.
    *   **LPF (Hz):** Low-Pass Filter cutoff frequency. Frequencies above this will be attenuated. Set to Nyquist frequency (Fs/2) for no LPF.
    *   **フィルタ次数 (Filter Order):** The order of the Butterworth filter. Higher orders result in a steeper rolloff but can introduce more phase distortion. Default is 4.
*   **ノイズ除去プラグイン (Noise Reduction Plugins):**
    *   **ノイズ除去アルゴリズム (Noise Reduction Algorithm):** Select "None" to apply no noise reduction, or choose from available plugins like "Notch Filter" or "Band-Stop Filter".
    *   **Plugin Specific Settings:** If a plugin is selected, its configurable parameters (e.g., frequency, Q-factor, band limits, order) will appear dynamically below the selection. Adjust these as needed for your specific noise reduction task.
*   **FFTピーク設定 (FFT Peak Settings):**
    *   **ピーク表示数 (Number of Peaks):** FFTスペクトルに表示するピークの最大数。
    *   **最小ピーク高さ（最大値に対する%）(Min Peak Height (% of Max)):** FFTスペクトル全体での最大振幅に対する相対的な最小高さで、これ以下のピークは無視されます。
    *   **ピーク最小距離（Hz）(Min Peak Distance (Hz)):** 識別されるピーク間の最小周波数間隔。

## 4. Interpreting Results

The main content area displays the analysis results in several sections.

*   **適用中の解析設定 (Current Analysis Settings):** サイドバーで選択した主要な解析設定の概要を表示します。特に、適用されているノイズ除去プラグインとそのパラメータもここに表示されます。
*   **時間領域 特徴量 (Time-Domain Features):**
    *   **RMS:** 信号の実効値。振動のエネルギーを示す。
    *   **Peak:** 信号の最大振幅。
    *   **Kurtosis:** 信号の尖度。衝撃性の指標。
    *   **Crest Factor:** ピーク値とRMS値の比率。衝撃性の指標。
*   **データ品質 (Data Quality):**
    *   **クリッピング率 (Clipping Ratio):** データが飽和（クリッピング）している割合。値が高いほどデータの信頼性が低下します。
    *   **S/N 比 (SNR):** 信号対ノイズ比。
    *   **診断信頼度 (Confidence Score):** クリッピング率やS/N比などに基づいて算出される、解析結果の信頼度（0-100%）。信頼度が低い場合は赤色で警告されます。
*   **MT法診断 (MD) (MT Method Diagnosis):** 正常データで単位空間が構築されている場合に表示されます。
    *   **MD (Mahalanobis Distance):** 測定データが正常データの単位空間からどれだけ逸脱しているかを示すマハラノビス距離。値が大きいほど異常の可能性が高いことを示します。異常の兆候に応じて色が変わります（緑: 正常, オレンジ: 注意, 赤: 異常）。
*   **時間波形 (Time Waveform):** 処理後の信号の時間領域波形をPlotlyで表示します。
*   **FFT スペクトル (FFT Spectrum):** 処理後の信号の周波数スペクトルをPlotlyで表示します。FFTピークは自動的にアノテーションされ、最も顕著なピークの周波数（Hz）が表示されます。

## 5. MT Method (Mahalanobis-Taguchi Method)

MT法による異常診断を使用するには、まず正常な状態のデータで「単位空間」を構築する必要があります。

*   **単位空間の構築/更新 (Build/Update Unit Space):**
    1.  サイドバーの「MT法設定」セクションを展開します。
    2.  「正常時のWAVファイルをアップロード (複数可)」で、複数の正常データWAVファイルをアップロードします。
    3.  「HPF (訓練用)」「LPF (訓練用)」「フィルタ次数 (訓練用)」で、単位空間構築時に適用するフィルタ条件を設定します。
    4.  「単位空間を構築/更新」ボタンをクリックして単位空間を計算します。
    5.  「単位空間ステータス」で現在の状態を確認できます。推奨サンプル数に満たない場合は「暫定単位空間」と表示され、信頼度が低下します。

## 6. Noise Reduction Evaluation

ノイズ除去プラグインが適用されている場合、「ノイズ除去 評価」セクションが表示され、ノイズ除去の効果を詳細に確認できます。

*   **特徴量へのインパクト (Impact on Features):**
    *   ノイズ除去フィルター適用前と後で、時間領域特徴量（RMS, Peakなど）がどのように変化したかをテーブルで表示します。
    *   「Delta (%)」列で変化率を確認できます。
*   **スペクトル比較 (Spectral Comparison):**
    *   ノイズ除去前、ノイズ除去後、およびフィルターによって「除去された信号」の3つの周波数スペクトルをPlotlyグラフで重ねて表示します。
    *   これにより、どの周波数帯域のノイズがどれだけ除去されたかを視覚的に把握できます。

## 7. Audit Log (監査ログ)

すべての解析結果はJSON形式の監査ログとして保存されます。

*   **監査ログ (JSON):** 画面下部のエキスパンダーを展開すると、現在の解析で使用されたすべてのパラメータと結果がJSON形式で表示されます。
*   **Download Log:** ボタンをクリックすると、このJSONデータをファイルとしてダウンロードできます。これにより、解析の再現性が保証されます。
