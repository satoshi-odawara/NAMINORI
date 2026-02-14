import numpy as np
from src.core.models import VibrationFeatures
from typing import List, Optional

class MTSpace:
    """
    Manages the Mahalanobis-Taguchi (MT) unit space for anomaly detection.
    """
    def __init__(self, min_samples: int = 10, recommended_samples: int = 30):
        self.normal_samples_vectors: List[np.ndarray] = []
        self.mean_vector: Optional[np.ndarray] = None
        self.inverse_covariance_matrix: Optional[np.ndarray] = None
        self.min_samples = min_samples
        self.recommended_samples = recommended_samples
        self.is_provisional = True

    def add_normal_sample(self, features: VibrationFeatures):
        """
        Adds a new normal sample (VibrationFeatures) to the unit space.
        Automatically updates the unit space if enough samples are available.
        """
        self.normal_samples_vectors.append(features.to_vector())
        self._update_unit_space()

    def _update_unit_space(self):
        """
        Updates the mean vector and inverse covariance matrix if enough normal samples are available.
        """
        if len(self.normal_samples_vectors) < self.min_samples:
            self.mean_vector = None
            self.inverse_covariance_matrix = None
            self.is_provisional = True
            return

        feature_vectors = np.array(self.normal_samples_vectors)

        # Calculate mean vector
        self.mean_vector = np.mean(feature_vectors, axis=0)

        # Calculate covariance matrix
        # Ensure that feature_vectors has enough variance for inverse_covariance_matrix calculation
        covariance_matrix = np.cov(feature_vectors, rowvar=False)

        # Handle singular matrix case (e.g., if all samples are identical or not enough variance)
        try:
            self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)
        except np.linalg.LinAlgError:
            # Add a small regularization term to the diagonal (Tikhonov regularization)
            # The magnitude of regularization (1e-6) can be tuned.
            regularization_term = np.identity(covariance_matrix.shape[0]) * 1e-6
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
