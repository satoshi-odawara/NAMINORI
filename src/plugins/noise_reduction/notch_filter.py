from typing import List, Any
import numpy as np
from scipy import signal

from src.core.plugins import NoiseReductionPlugin, PluginParameter

class NotchFilterPlugin(NoiseReductionPlugin):
    """
    A plugin to apply a notch filter, removing a specific frequency.
    """
    def get_name(self) -> str:
        return "notch_filter"

    def get_display_name(self) -> str:
        return "Notch Filter"

    def get_parameters(self) -> List[PluginParameter]:
        return [
            PluginParameter(
                name="freq_hz",
                label="Notch Frequency (Hz)",
                param_type="number_input",
                default=60.0,
                min_value=0.1,
                help_text="The center frequency to remove."
            ),
            PluginParameter(
                name="q_factor",
                label="Q Factor",
                param_type="number_input",
                default=30.0,
                min_value=0.1,
                help_text="The quality factor of the filter, which determines its bandwidth. Higher Q = narrower filter."
            )
        ]

    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        freq_hz = params.get("freq_hz", 60.0)
        q_factor = params.get("q_factor", 30.0)

        if not (0 < freq_hz < fs / 2):
            raise ValueError(f"Notch frequency ({freq_hz} Hz) must be between 0 and Nyquist frequency ({fs / 2} Hz).")
        if q_factor <= 0:
            raise ValueError("Notch Q-factor must be positive.")

        b, a = signal.iirnotch(freq_hz, q_factor, fs)
        return signal.filtfilt(b, a, data)
