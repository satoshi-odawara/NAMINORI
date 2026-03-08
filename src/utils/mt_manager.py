import pickle
import os
from pathlib import Path
from typing import Dict, Optional, List
from src.diagnostics.mt_method import MTSpace
from src.core.models import AnalysisConfig

UNIT_SPACE_DIR = Path("data/unit_spaces")

def save_unit_space(name: str, mt_space: MTSpace, config: AnalysisConfig):
    """
    Saves the established unit space to a file.
    
    Args:
        name: Name of the unit space (e.g., machine ID).
        mt_space: The MTSpace instance containing the learned parameters.
        config: The AnalysisConfig used to create this unit space.
    """
    if mt_space.mean_vector is None or mt_space.inverse_covariance_matrix is None:
        raise ValueError("Unit space is not established and cannot be saved.")
    
    UNIT_SPACE_DIR.mkdir(parents=True, exist_ok=True)
    
    file_path = UNIT_SPACE_DIR / f"{name}.pkl"
    
    data = {
        "name": name,
        "mean_vector": mt_space.mean_vector,
        "inverse_covariance_matrix": mt_space.inverse_covariance_matrix,
        "average_magnitude_spectrum": mt_space.average_magnitude_spectrum, # Add average magnitude spectrum
        "config": config,
        "sample_count": len(mt_space.normal_samples_vectors)
    }
    
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

def load_unit_space(name: str) -> Optional[dict]:
    """
    Loads a unit space from a file.
    
    Returns:
        A dictionary containing unit space parameters and config, or None if not found.
    """
    file_path = UNIT_SPACE_DIR / f"{name}.pkl"
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading unit space '{name}': {e}")
        return None

def list_saved_unit_spaces() -> List[str]:
    """Returns a list of names of saved unit spaces."""
    if not UNIT_SPACE_DIR.exists():
        return []
    return [f.stem for f in UNIT_SPACE_DIR.glob("*.pkl")]
