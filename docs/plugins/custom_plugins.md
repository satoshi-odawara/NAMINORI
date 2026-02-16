# Custom Noise Reduction Plugin Guide

This guide explains how to create your own custom noise reduction algorithms and integrate them into the Vibration Analysis Web Application.

## 1. Introduction

The application supports a pluggable architecture for noise reduction. This allows you to write your own Python classes that perform signal processing and have them automatically discovered and made available in the Streamlit UI.

## 2. File Location

Your custom plugin must be a Python file (`.py`) placed in the following directory:

```
src/plugins/noise_reduction/
```

Any `.py` file in this directory (that doesn't start with `__`) will be automatically scanned by the application on startup.

## 3. The Plugin Contract

A valid plugin is a Python class that meets the following criteria:

1.  It inherits from the `NoiseReductionPlugin` abstract base class.
2.  It implements all the abstract methods defined in the base class.

You must import the necessary components from `src.core.plugins`:

```python
from src.core.plugins import NoiseReductionPlugin, PluginParameter
```

## 4. Required Methods

Here is a detailed breakdown of the methods you must implement in your plugin class.

### `get_name(self) -> str`

*   **Purpose:** Returns a unique, machine-readable name for your plugin.
*   **Rules:**
    *   Must be a simple string.
    *   Use snake\_case (e.g., `my_custom_filter`).
    *   This name is used internally and for logging. It **must be unique** across all plugins.
*   **Example:**
    ```python
    def get_name(self) -> str:
        return "gain_filter"
    ```

### `get_display_name(self) -> str`

*   **Purpose:** Returns a human-readable name that will be shown in the UI's dropdown menu.
*   **Example:**
    ```python
    def get_display_name(self) -> str:
        return "Gain (Amplify/Attenuate)"
    ```

### `get_parameters(self) -> List[PluginParameter]`

*   **Purpose:** Defines the parameters that your algorithm needs, which will be rendered automatically in the Streamlit sidebar.
*   **Returns:** A list of `PluginParameter` objects.
*   The `PluginParameter` constructor takes the following arguments:
    *   `name` (str): The internal name of the parameter. This will be the keyword argument key passed to your `process` method.
    *   `label` (str): The text label shown next to the widget in the UI.
    *   `param_type` (Literal): The type of widget to render. Can be `'number_input'` or `'slider'`.
    *   `default` (float): The default value for the parameter.
    *   `min_value` (float, optional): The minimum allowed value.
    *   `max_value` (float, optional): The maximum allowed value (required for sliders).
    *   `help_text` (str, optional): A tooltip that appears in the UI.
*   **Example:**
    ```python
    def get_parameters(self) -> List[PluginParameter]:
        return [
            PluginParameter(
                name="gain_factor",
                label="Gain Factor",
                param_type="slider",
                default=1.0,
                min_value=0.0,
                max_value=5.0,
                help_text="The factor by which to multiply the signal. >1 amplifies, <1 attenuates."
            )
        ]
    ```

### `process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray`

*   **Purpose:** This is the core of your plugin. It takes the signal data and applies your custom processing.
*   **Arguments:**
    *   `data` (np.ndarray): The input signal data.
    *   `fs` (int): The sampling frequency of the data.
    *   `**params` (Any): A dictionary containing the current values of the parameters you defined in `get_parameters`. You can get a specific parameter value with `params.get("your_param_name", default_value)`.
*   **Returns:** A NumPy array of the processed signal data. It must be the same length as the input data.
*   **Example:**
    ```python
    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        gain = params.get("gain_factor", 1.0)
        
        # It's good practice to validate parameters
        if gain < 0:
            raise ValueError("Gain factor cannot be negative.")
            
        processed_data = data * gain
        return processed_data
    ```

## 5. Complete Example: `gain_plugin.py`

Here is a complete, ready-to-use example. You can save this code as `src/plugins/noise_reduction/gain_plugin.py` and it will appear in the UI automatically.

```python
from typing import List, Any
import numpy as np

# All plugins must import the base classes
from src.core.plugins import NoiseReductionPlugin, PluginParameter

# The class name can be anything, but it must inherit from NoiseReductionPlugin
class GainPlugin(NoiseReductionPlugin):
    """
    A simple example plugin that multiplies the signal by a "gain" factor.
    This can be used to amplify or attenuate the entire signal.
    """
    
    def get_name(self) -> str:
        """A unique machine-readable name."""
        return "gain_plugin"

    def get_display_name(self) -> str:
        """A human-readable name for the UI."""
        return "Gain Adjuster"

    def get_parameters(self) -> List[PluginParameter]:
        """Define the UI widgets for this plugin's parameters."""
        return [
            PluginParameter(
                name="gain",
                label="Gain Factor",
                param_type="slider",
                default=1.0,
                min_value=0.0,
                max_value=10.0,
                help_text="The factor to multiply the signal by. >1 amplifies, <1 attenuates."
            )
        ]

    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        """The core signal processing logic."""
        gain_factor = params.get("gain", 1.0)

        # Apply the gain
        processed_data = data * gain_factor

        return processed_data

```

---

## 6. Spectral Subtraction Plugin

The Spectral Subtraction plugin provides a method for reducing broadband noise by estimating a noise profile from normal operating conditions and subtracting its power spectrum from the noisy signal's power spectrum.

### How it Works

1.  **Noise Profile Learning:** When you train the MT (Mahalanobis-Taguchi) method's unit space, the application simultaneously learns an averaged noise power spectrum from the provided normal WAV files. This learned profile represents the characteristic environmental noise.
2.  **Spectral Subtraction:** When the Spectral Subtraction plugin is selected for analysis, the application takes the input signal, performs a Fast Fourier Transform (FFT) to get its power and phase spectra. It then subtracts the *learned noise power spectrum* (scaled by an `alpha` factor) from the input signal's power spectrum. A spectral floor (`floor`) is applied to prevent negative power values. The resulting "clean" magnitude spectrum is then combined with the original signal's phase spectrum, and an Inverse FFT (IFFT) is performed to reconstruct the noise-reduced time-domain signal.
3.  **Post-filtering:** Optionally, a low-pass filter can be applied after IFFT to smooth out any residual artifacts.

### Parameters

*   **Alpha (Over-subtraction Factor):** Controls how aggressively the noise is subtracted. A higher value (e.g., 2.0 to 5.0) leads to more noise reduction but can also introduce more signal distortion. Adjust this to find a balance.
    *   `param_type`: `number_input`
    *   `default`: `2.0`
    *   `min_value`: `1.0`
    *   `max_value`: `5.0`
*   **Floor (Noise Floor, 0-1):** Sets a minimum power level to prevent negative power values during subtraction. This is often a small fraction of the estimated noise power or the noisy signal's power.
    *   `param_type`: `number_input`
    *   `default`: `0.02`
    *   `min_value`: `0.0`
    *   `max_value`: `0.5`
*   **Post-filter LPF (Hz, 0 for None):** Applies a Low-Pass Filter after the Inverse FFT to smooth out any potential "musical noise" or other high-frequency artifacts introduced by the spectral subtraction process. Set to `0` to disable this post-filtering.
    *   `param_type`: `number_input`
    *   `default`: `0.0`
    *   `min_value`: `0.0`
*   **Post-filter Order:** The order of the Butterworth low-pass filter applied as a post-filter. Higher orders provide a steeper rolloff but can introduce more phase distortion.
    *   `param_type`: `number_input`
    *   `default`: `4`
    *   `min_value`: `1`

### Usage

1.  **Train MT Method:** Before using Spectral Subtraction, ensure you have trained the MT method's unit space by uploading normal WAV files in the sidebar's "MT法設定" section and clicking "単位空間を構築/更新". This step is crucial for learning the `noise_power_spectrum_avg`.
2.  **Select Plugin:** In the "ノイズ除去プラグイン" section of the sidebar, select "Spectral Subtraction" from the "ノイズ除去アルゴリズム" dropdown.
3.  **Adjust Parameters:** Configure the Alpha, Floor, and optional Post-filter settings as needed.
4.  **Analyze:** Upload your evaluation WAV file. The Spectral Subtraction will be applied, and its effect can be reviewed in the "ノイズ除去 評価" section.
