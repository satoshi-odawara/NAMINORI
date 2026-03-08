import numpy as np
from src.core.models import VibrationFeatures, AnalysisConfig, WindowFunction
from src.core.feature_extraction import calculate_fft_features # Import for power spectrum calculation
from scipy.fft import fft
from scipy import signal # Added for window functions
from typing import List, Optional

class MTSpace:
    """
    Manages the Mahalanobis-Taguchi (MT) unit space for anomaly detection
    and learns an averaged magnitude spectrum from normal data for visualization.
    """
    def __init__(self, min_samples: int = 10, recommended_samples: int = 30):
        self.normal_samples_vectors: List[np.ndarray] = []
        self.mean_vector: Optional[np.ndarray] = None
        self.inverse_covariance_matrix: Optional[np.ndarray] = None
        self.min_samples = min_samples
        self.recommended_samples = recommended_samples
        self.is_provisional = True
        
        # For noise profile learning (averaged magnitude spectrum)
        self.normal_magnitude_spectra: List[np.ndarray] = []
        self.average_magnitude_spectrum: Optional[np.ndarray] = None

    def add_normal_sample(self, features: VibrationFeatures, magnitude_spectrum: np.ndarray):
        """
        Adds a new normal sample (VibrationFeatures and its magnitude spectrum).
        """
        self.normal_samples_vectors.append(features.to_vector())
        self.normal_magnitude_spectra.append(magnitude_spectrum)
        self._update_unit_space()

    def build_unit_space(self, features_list: List[VibrationFeatures], magnitude_spectra: List[np.ndarray]):
        """
        Builds the unit space from a list of features and their corresponding spectra.
        """
        self.normal_samples_vectors = [f.to_vector() for f in features_list]
        self.normal_magnitude_spectra = magnitude_spectra
        self._update_unit_space()

    def _update_unit_space(self):
        """
        Updates the mean vector, inverse covariance matrix, and average magnitude spectrum.
        """
        if len(self.normal_samples_vectors) == 0:
            self.mean_vector = None
            self.inverse_covariance_matrix = None
            self.is_provisional = True
            self.average_magnitude_spectrum = None
            return

        feature_vectors = np.array(self.normal_samples_vectors)
        num_features = feature_vectors.shape[1]

        # Calculate mean vector and average magnitude spectrum (always possible if len > 0)
        self.mean_vector = np.mean(feature_vectors, axis=0)
        
        if self.normal_magnitude_spectra:
            lengths = [len(s) for s in self.normal_magnitude_spectra]
            if len(set(lengths)) == 1:
                spectra_array = np.array(self.normal_magnitude_spectra)
                self.average_magnitude_spectrum = np.sqrt(np.mean(spectra_array**2, axis=0))
            else:
                self.average_magnitude_spectrum = self.normal_magnitude_spectra[-1]

        # Guard for inverse covariance matrix (requires more samples than features)
        if len(self.normal_samples_vectors) <= num_features:
            self.inverse_covariance_matrix = None
            self.is_provisional = True
            return

        covariance_matrix = np.cov(feature_vectors, rowvar=False)

        try:
            self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            regularization_term = np.identity(covariance_matrix.shape[0]) * 1e-9
            self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix + regularization_term)

        self.is_provisional = len(self.normal_samples_vectors) < self.recommended_samples

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
        return np.sqrt(np.maximum(0, md_squared))

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
