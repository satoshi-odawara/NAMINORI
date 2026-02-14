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
