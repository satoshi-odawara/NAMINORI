import numpy as np
from typing import Tuple

from src.core.models import QualityMetrics

def calculate_quality_metrics(
    data: np.ndarray,
    fs_hz: int,
    rms: float, # RMS of the signal
    magnitude: np.ndarray # FFT magnitude array for SNR
) -> QualityMetrics:
    """
    Calculates various data quality metrics.

    Args:
        data: Input signal (NumPy array, typically normalized).
        fs_hz: Sampling frequency in Hz.
        rms: RMS value of the signal.
        magnitude: FFT magnitude array.

    Returns:
        QualityMetrics: Object containing calculated quality metrics.
    """
    clipping_ratio = np.sum(np.abs(data) >= 0.99) / len(data)

    # S/N比推定（簡易版） - from GEMINI.md
    signal_power = rms**2
    noise_floor = np.percentile(magnitude, 10)**2  # 下位10%を雑音と仮定
    snr_db = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 60

    data_length_s = len(data) / fs_hz

    return QualityMetrics(
        clipping_ratio=clipping_ratio,
        snr_db=snr_db,
        data_length_s=data_length_s
    )

def get_confidence_score(quality: QualityMetrics) -> float:
    """
    Calculates the confidence score (0-100%) based on quality metrics.

    Args:
        quality: QualityMetrics object.

    Returns:
        float: Confidence score (0-100%).
    """
    # クリッピングペナルティ
    clip_score = max(0, 100 - quality.clipping_ratio * 10000)
    
    # S/N比スコア（20dB以上で満点）
    snr_score = min(100, quality.snr_db * 5)
    
    # データ長スコア（10秒以上で満点）
    length_score = min(100, quality.data_length_s * 10)
    
    return np.mean([clip_score, snr_score, length_score])
