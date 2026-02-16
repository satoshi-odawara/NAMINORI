from typing import List, Any
import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
import warnings # Added for warnings.warn

from src.core.plugins import NoiseReductionPlugin, PluginParameter

class SpectralSubtractionPlugin(NoiseReductionPlugin):
    """
    A noise reduction plugin implementing a power domain spectral subtraction method.
    It learns a noise profile from provided averaged normal power spectrum and subtracts it
    from the noisy signal's power spectrum.
    """
    def get_name(self) -> str:
        return "spectral_subtraction"

    def get_display_name(self) -> str:
        return "Spectral Subtraction"

    def get_parameters(self) -> List[PluginParameter]:
        return [
            PluginParameter(
                name="alpha",
                label="Over-subtraction Factor (Alpha)",
                param_type="number_input",
                default=2.0,
                min_value=1.0,
                max_value=5.0,
                help_text="Controls how much noise is subtracted. Higher values mean more aggressive subtraction."
            ),
            PluginParameter(
                name="floor",
                label="Noise Floor (0-1)",
                param_type="number_input",
                default=0.02,
                min_value=0.0,
                max_value=0.5,
                help_text="Minimum power level to prevent negative power values (as a fraction of noise power)."
            ),
            PluginParameter(
                name="post_filter_cutoff_hz",
                label="Post-filter LPF (Hz, 0 for None)",
                param_type="number_input",
                default=0.0,
                min_value=0.0,
                help_text="Apply a low-pass filter after IFFT to smooth artifacts. Set to 0 to disable."
            ),
            PluginParameter(
                name="post_filter_order",
                label="Post-filter Order",
                param_type="number_input",
                default=4,
                min_value=1,
                help_text="Order of the low-pass post-filter."
            )
        ]

    def process(self, data: np.ndarray, fs: int, p_noise_avg: np.ndarray, **params: Any) -> np.ndarray:
        """
        Applies spectral subtraction noise reduction.

        Args:
            data: The noisy input signal (time domain).
            fs: The sampling frequency.
            p_noise_avg: The averaged noise power spectrum learned from normal data.
                         Must have the same length as the FFT of 'data'.
            **params: Plugin parameters (alpha, floor, post_filter_cutoff_hz, post_filter_order).

        Returns:
            The noise-reduced signal (time domain).
        """
        N = len(data) # Define N once at the start

        if p_noise_avg is None:
            raise ValueError("Averaged noise power spectrum (p_noise_avg) must be provided for Spectral Subtraction.")
        
        # Original length check was incorrect, N is length of data
        if len(p_noise_avg) != N:
            warnings.warn(f"p_noise_avg length ({len(p_noise_avg)}) does not match data length ({N}). "
                          f"This might lead to incorrect results. Resizing p_noise_avg.", UserWarning)
            p_noise_avg_resampled = np.interp(np.arange(N),
                                               np.arange(len(p_noise_avg)) * N / len(p_noise_avg),
                                               p_noise_avg)
            p_noise_avg = p_noise_avg_resampled

        alpha = params.get("alpha", 2.0)
        floor = params.get("floor", 0.02)
        post_filter_cutoff_hz = params.get("post_filter_cutoff_hz", 0.0)
        post_filter_order = int(params.get("post_filter_order", 4))

        # Validate post-filter cutoff
        if post_filter_cutoff_hz > 0 and not (0 < post_filter_cutoff_hz < fs / 2):
            raise ValueError(f"Post-filter cutoff frequency ({post_filter_cutoff_hz} Hz) must be between 0 and Nyquist frequency ({fs / 2} Hz).")
        
        # 1. FFT of noisy signal
        
        # Apply a window function if desired for consistent spectral estimation
        # For simplicity here, we'll use a rectangular window (no window applied)
        # In a real scenario, applying a window to both noise learning and subtraction is important
        
        noisy_fft = fft(data)
        noisy_mag = np.abs(noisy_fft)
        noisy_phase = np.angle(noisy_fft)
        noisy_power = noisy_mag**2

        # 2. Spectral Subtraction in power domain
        # Ensure p_noise_avg is non-negative and scaled if necessary
        # p_noise_avg = np.maximum(0, p_noise_avg) # Already handled in MTSpace learning
        
        # Prevent division by zero or negative values in p_noise_avg (should be handled during learning)
        p_noise_avg_clipped = np.maximum(p_noise_avg, 1e-10) # Clip to a very small positive number

        # Power subtraction
        subtracted_power = noisy_power - alpha * p_noise_avg_clipped
        
        # Apply spectral floor
        p_clean = np.maximum(subtracted_power, floor * noisy_power) # Floor as a fraction of noisy signal power
                                                                     # or floor * p_noise_avg_clipped

        # 3. Reconstruct complex spectrum
        clean_mag = np.sqrt(p_clean)
        clean_fft = clean_mag * np.exp(1j * noisy_phase)

        # 4. IFFT to time domain
        clean_signal = np.real(ifft(clean_fft)) # Take real part to discard imaginary residuals

        # 5. Optional Post-filter
        if post_filter_cutoff_hz > 0 and post_filter_cutoff_hz < fs / 2:
            sos = signal.butter(post_filter_order, post_filter_cutoff_hz, btype='low', fs=fs, output='sos')
            clean_signal = signal.sosfilt(sos, clean_signal)

        return clean_signal
