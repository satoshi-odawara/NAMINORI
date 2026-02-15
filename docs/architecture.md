# Technical Architecture Overview

This document provides a high-level overview of the Vibration Analysis Web Application's technical architecture, intended for developers and contributors.

## 1. System Overview

The Vibration Analysis Web Application is built as a single-page application using Streamlit for the frontend, backed by a robust Python signal processing and machine learning core. The system emphasizes modularity, extensibility (via a plugin framework), and reproducibility.

**Key Components:**
*   **Streamlit Application (`src/app.py`):** Serves as the user interface, handling file uploads, user inputs for analysis settings, and displaying results through interactive Plotly graphs.
*   **Core Modules (`src/core/`):** Contains the fundamental signal processing algorithms, feature extraction logic, data quality checks, and the newly implemented plugin and evaluation frameworks.
*   **Diagnostics Modules (`src/diagnostics/`):** Houses specialized algorithms like the Mahalanobis-Taguchi (MT) method for anomaly detection.
*   **Plugin System (`src/plugins/`):** Allows for dynamic loading of user-defined noise reduction algorithms, enhancing the application's flexibility.
*   **Utility Modules (`src/utils/`):** Provides common functionalities such as audit logging.

## 2. Data Flow

The typical data flow when a user uploads a WAV file for analysis is as follows:

1.  **File Upload (`src/app.py`):** User uploads a WAV file via the Streamlit UI. The file is temporarily saved.
2.  **WAV Loading (`src/core/signal_processing.py`):** The WAV file is loaded, its sampling frequency (`fs_hz`) extracted, and the raw audio data is normalized (converted to float64 between -1.0 and 1.0). A SHA256 hash of the file is generated for reproducibility.
3.  **Preprocessing (`src/core/signal_processing.py`):**
    *   **DC Offset Removal:** `remove_dc_offset` function removes any DC bias from the signal.
    *   **Butterworth Filtering:** `apply_butterworth_filter` applies standard high-pass, low-pass, or band-pass filtering based on user settings. This produces `signal_pre_nr`.
4.  **Noise Reduction Plugin Application (`src/app.py` -> `src/core/plugins.py` -> `src/plugins/noise_reduction/*.py`):**
    *   If a noise reduction plugin is selected in the UI, the `PluginManager` identifies the corresponding plugin.
    *   The plugin's `process` method is called with `signal_pre_nr` and user-defined parameters, producing `signal_post_nr`.
    *   If no plugin is selected, `signal_post_nr` is identical to `signal_pre_nr`.
5.  **Noise Reduction Evaluation (`src/core/evaluation.py`):** If a plugin was applied, `perform_nr_evaluation` compares `signal_pre_nr` and `signal_post_nr` to calculate before/after features and the "removed signal".
6.  **Feature Extraction (`src/core/feature_extraction.py`):**
    *   `calculate_time_domain_features` extracts statistical features (RMS, Peak, Kurtosis, etc.) from the `processed_final` signal (`signal_post_nr`).
    *   `calculate_fft_features` computes the FFT spectrum and power band contributions from the `processed_final` signal.
7.  **Quality Check (`src/core/quality_check.py`):** `calculate_quality_metrics` assesses the data quality (clipping, SNR) and `get_confidence_score` derives a confidence level for the analysis.
8.  **MT Method Diagnosis (`src/diagnostics/mt_method.py`):** If a unit space has been trained, `MTSpace.calculate_md` computes the Mahalanobis Distance for the extracted features against the normal unit space.
9.  **Result Display (`src/app.py`):** All extracted features, quality metrics, MT method results, and evaluation data (if applicable) are displayed in the Streamlit UI using Plotly for interactive graphs.
10. **Audit Logging (`src/utils/audit_log.py`):** All analysis parameters and results are bundled into an `AnalysisResult` object, which can be serialized to a JSON audit log for reproducibility.

## 3. Module Breakdown

The project follows a modular structure to separate concerns:

*   **`src/app.py`:** The main Streamlit application script, responsible for UI layout, user interaction, and orchestrating calls to backend logic.
*   **`src/core/`:** Contains fundamental building blocks:
    *   `models.py`: Data models (dataclasses, Enums) for configuration and results.
    *   `signal_processing.py`: Core signal loading and filtering functions.
    *   `feature_extraction.py`: Algorithms for extracting time and frequency domain features.
    *   `quality_check.py`: Logic for assessing data quality and confidence.
    *   `plugins.py`: Abstract base classes and manager for the plugin system.
    *   `evaluation.py`: Logic and data models for noise reduction evaluation.
*   **`src/diagnostics/`:** Advanced diagnostic algorithms:
    *   `mt_method.py`: Implementation of the Mahalanobis-Taguchi method.
*   **`src/plugins/noise_reduction/`:** Directory for discoverable noise reduction plugin implementations.
*   **`src/utils/`:** General utility functions:
    *   `audit_log.py`: Handles saving and loading of analysis audit logs.
*   **`tests/`:** Comprehensive unit, integration, and regression tests.
*   **`docs/`:** Dedicated documentation files.

## 4. Plugin Architecture

The application implements a pluggable architecture for noise reduction algorithms. This design allows developers to extend the application's capabilities without modifying the core codebase.

*   **Plugin Discovery:** The `PluginManager` (a singleton defined in `src/core/plugins.py`) scans the `src/plugins/noise_reduction/` directory on startup.
*   **Plugin Definition:** Custom noise reduction algorithms are implemented as Python classes inheriting from `NoiseReductionPlugin` (an Abstract Base Class in `src/core/plugins.py`).
*   **Plugin Interface:** Each plugin must define methods such as `get_name()`, `get_display_name()`, `get_parameters()` (to describe its UI configurable parameters), and `process()` (the core filtering logic).
*   **Dynamic UI:** The Streamlit frontend dynamically renders input widgets based on the `PluginParameter` objects returned by `get_parameters()`.
*   **Integration:** The `run_full_analysis_pipeline` (conceptually in `app.py`) calls the `process()` method of the selected plugin, passing the signal data and user-defined parameters.

For detailed instructions on creating custom noise reduction plugins, refer to the [Custom Noise Reduction Plugin Guide](../plugins/custom_plugins.md).
