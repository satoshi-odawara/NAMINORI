# Noise Reduction Evaluation Framework

This document explains the noise reduction evaluation framework, including its metrics and how to interpret the results within the Vibration Analysis Web Application.

## 1. Introduction

The Noise Reduction Evaluation Framework provides a mechanism to objectively assess the effectiveness of any applied noise reduction plugin. When a noise reduction algorithm is selected and applied, this framework automatically compares the signal's characteristics *before* and *after* the noise reduction process, offering quantitative and visual feedback. This helps users understand the impact of the chosen filter settings.

## 2. Evaluation Metrics

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

## 3. Interpreting the UI

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
