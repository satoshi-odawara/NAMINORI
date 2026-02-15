from dataclasses import dataclass
from src.core.models import TimeDomainFeatures
from src.core.feature_extraction import calculate_time_domain_features
import numpy as np

@dataclass
class NoiseReductionEvaluation:
    """
    Holds the results of comparing a signal before and after noise reduction.
    """
    features_before: TimeDomainFeatures
    features_after: TimeDomainFeatures
    
    # Store the signals themselves for plotting
    signal_pre_nr: np.ndarray
    signal_post_nr: np.ndarray
    
    @property
    def removed_signal(self) -> np.ndarray:
        """The signal that was removed by the filter."""
        return self.signal_pre_nr - self.signal_post_nr

def perform_nr_evaluation(signal_pre_nr: np.ndarray, signal_post_nr: np.ndarray) -> NoiseReductionEvaluation:
    """Calculates features before and after noise reduction for evaluation."""
    features_before = calculate_time_domain_features(signal_pre_nr)
    features_after = calculate_time_domain_features(signal_post_nr)
    
    return NoiseReductionEvaluation(
        features_before=features_before,
        features_after=features_after,
        signal_pre_nr=signal_pre_nr,
        signal_post_nr=signal_post_nr
    )
