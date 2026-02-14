from typing import List, Any
import numpy as np
from scipy import signal

from src.core.plugins import NoiseReductionPlugin, PluginParameter

class BandStopFilterPlugin(NoiseReductionPlugin):
    """
    A plugin to apply a band-stop filter, removing a range of frequencies.
    """
    def get_name(self) -> str:
        return "band_stop_filter"

    def get_display_name(self) -> str:
        return "Band-Stop Filter"

    def get_parameters(self) -> List[PluginParameter]:
        return [
            PluginParameter(
                name="low_hz",
                label="Stop-band Lower Freq (Hz)",
                param_type="number_input",
                default=50.0,
                min_value=0.1,
                help_text="The lower bound of the frequency band to remove."
            ),
            PluginParameter(
                name="high_hz",
                label="Stop-band Upper Freq (Hz)",
                param_type="number_input",
                default=70.0,
                min_value=0.1,
                help_text="The upper bound of the frequency band to remove."
            ),
            PluginParameter(
                name="order",
                label="Filter Order",
                param_type="number_input",
                default=4,
                min_value=1,
                help_text="The order of the Butterworth filter. Higher orders have a steeper rolloff."
            )
        ]

    def process(self, data: np.ndarray, fs: int, **params: Any) -> np.ndarray:
        low_hz = params.get("low_hz", 50.0)
        high_hz = params.get("high_hz", 70.0)
        order = int(params.get("order", 4))

        if not (0 < low_hz < high_hz < fs / 2):
            raise ValueError(f"Band-stop frequencies ({low_hz}-{high_hz} Hz) must be between 0 and Nyquist ({fs / 2} Hz) and low < high.")

        sos = signal.butter(order, [low_hz, high_hz], btype='bandstop', fs=fs, output='sos')
        return signal.sosfilt(sos, data)
