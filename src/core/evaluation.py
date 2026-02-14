from dataclasses import dataclass
from src.core.models import TimeDomainFeatures
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
