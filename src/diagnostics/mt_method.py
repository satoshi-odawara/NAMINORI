import numpy as np
from src.core.models import VibrationFeatures, AnalysisConfig, WindowFunction
from src.core.feature_extraction import calculate_fft_features # Import for power spectrum calculation
from scipy.fft import fft
from scipy import signal # Added for window functions
from typing import List, Optional

class MTSpace:
    """
    Manages the Mahalanobis-Taguchi (MT) unit space for anomaly detection
    and learns an averaged noise power spectrum from normal data for noise reduction.
    """
    def __init__(self, min_samples: int = 10, recommended_samples: int = 30):
        self.normal_samples_vectors: List[np.ndarray] = []
        self.mean_vector: Optional[np.ndarray] = None
        self.inverse_covariance_matrix: Optional[np.ndarray] = None
        self.min_samples = min_samples
        self.recommended_samples = recommended_samples
        self.is_provisional = True
        
        # For noise profile learning
        self.individual_normal_power_spectra: List[np.ndarray] = []
        self.noise_power_spectrum_avg: Optional[np.ndarray] = None

    def add_normal_sample(self, features: VibrationFeatures, processed_signal: np.ndarray, fs_hz: int, analysis_config: AnalysisConfig):
        """
        Adds a new normal sample (VibrationFeatures) to the unit space and
        contributes to learning the averaged noise power spectrum.
        Automatically updates the unit space if enough samples are available.
        """
        self.normal_samples_vectors.append(features.to_vector())
        
        # Calculate power spectrum for noise profile learning
        _, _, power_bands = calculate_fft_features(processed_signal, fs_hz, analysis_config.window)
        # We need the full power spectrum, not just bands. Recalculate or modify calculate_fft_features
        # For simplicity here, we'll re-do FFT to get full spectrum
        N = len(processed_signal)
        
        selected_window = analysis_config.window.value # Use dynamically selected window
        if selected_window == "hanning":
            win = np.hanning(N)
        elif selected_window == "flattop":
            win = signal.windows.flattop(N)
        else:
            win = np.ones(N) # Rectangular window if none specified (or for custom)

        data_windowed = processed_signal * win
        fft_result = fft(data_windowed)
        
        # Power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_result)**2
        self.individual_normal_power_spectra.append(power_spectrum)

        self._update_unit_space()

    def _update_unit_space(self):
        """
        Updates the mean vector, inverse covariance matrix, and averaged noise power spectrum
        if enough normal samples are available.
        """
        if len(self.normal_samples_vectors) < self.min_samples:
            self.mean_vector = None
            self.inverse_covariance_matrix = None
            self.is_provisional = True
            self.noise_power_spectrum_avg = None
            return

        feature_vectors = np.array(self.normal_samples_vectors)

        # Calculate mean vector
        self.mean_vector = np.mean(feature_vectors, axis=0)

        # Calculate covariance matrix
        covariance_matrix = np.cov(feature_vectors, rowvar=False)

        try:
            self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            regularization_term = np.identity(covariance_matrix.shape[0]) * 1e-9
            self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix + regularization_term)

        self.is_provisional = len(self.normal_samples_vectors) < self.recommended_samples
        
        # Calculate averaged noise power spectrum
        if self.individual_normal_power_spectra:
            self.noise_power_spectrum_avg = np.mean(self.individual_normal_power_spectra, axis=0)
            self.individual_normal_power_spectra.clear() # Clear to save memory


    def calculate_md(self, features: VibrationFeatures) -> float:
        """
        Calculates the Mahalanobis Distance (MD) for a given VibrationFeatures instance
        against the established unit space.

        Args:
            features: VibrationFeatures instance to calculate MD for.

        Returns:
            float: Mahalanobis Distance. Returns np.inf if unit space is not established.
        """
        if self.mean_vector is None or self.inverse_covariance_matrix is None:
            return np.inf # Cannot calculate MD without a unit space

        x = features.to_vector()
        diff = x - self.mean_vector
        md_squared = diff.T @ self.inverse_covariance_matrix @ diff
        return np.sqrt(md_squared)

    def get_status(self) -> str:
        """
        Returns the current status of the unit space.
        """
        if len(self.normal_samples_vectors) == 0:
            return "No samples"
        elif len(self.normal_samples_vectors) < self.min_samples:
            return f"Insufficient samples ({len(self.normal_samples_vectors)}/{self.min_samples} min)"
        elif self.is_provisional:
            return f"Provisional unit space ({len(self.normal_samples_vectors)}/{self.recommended_samples} recommended)"
        else:
            return f"Established unit space ({len(self.normal_samples_vectors)} samples)"
