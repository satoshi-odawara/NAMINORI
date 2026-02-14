import json
from dataclasses import asdict
from src.core.models import AnalysisResult
from src.core.quality_check import get_confidence_score

def save_audit_log(result: AnalysisResult, path: str):
    """
    Saves the analysis result as a JSON audit log for reproducibility.

    Args:
        result: The AnalysisResult object to save.
        path: The file path to save the JSON log.
    """
    data = {
        'timestamp': result.timestamp,
        'file_hash': result.file_hash,
        'fs_hz': result.fs_hz,
        'quantity': result.config.quantity.value,
        'window': result.config.window.value,
        'filter_config': {
            'highpass_hz': result.config.highpass_hz,
            'lowpass_hz': result.config.lowpass_hz,
            'order': result.config.filter_order
        },
        'features': asdict(result.features),
        'quality': asdict(result.quality),
        'confidence_score': get_confidence_score(result.quality),
        'app_version': result.app_version
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)