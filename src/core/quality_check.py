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
        data: Input signal (NumPy array).
        fs_hz: Sampling frequency in Hz.
        rms: RMS value of the signal.
        magnitude: FFT magnitude array.

    Returns:
        QualityMetrics: Object containing calculated quality metrics.
    """
    # Detect if the data is likely normalized (WAV) or raw physical values
    max_abs = np.max(np.abs(data))
    
    # If data is within ~1.0, it's likely normalized (WAV)
    # If it significantly exceeds 1.0, it's likely physical units (m/s^2, etc.)
    if max_abs <= 1.05:
        threshold = 0.99
    else:
        # For physical units, we don't know the sensor's range easily, 
        # but we can check for values staying at the exact same max/min for multiple samples.
        # As a heuristic, if 90% of max is exceeded and it's very large, we might warn,
        # but to avoid false positives for physical data, we use a much higher threshold 
        # or disable it if we can't be sure of the range.
        # Here we'll use a heuristic: if values are within a typical sensor range (e.g. 20G),
        # we don't call it clipping unless it's near the very peak of the observed data.
        threshold = max_abs * 0.999 

    clipping_ratio = np.sum(np.abs(data) >= threshold) / len(data) if max_abs > 0 else 0

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

def get_confidence_score(quality: QualityMetrics) -> Tuple[float, dict[str, float]]:
    """
    Calculates the confidence score (0-100%) and its breakdown.

    Args:
        quality: QualityMetrics object.

    Returns:
        Tuple[float, dict]: (Composite score, breakdown dictionary)
    """
    # クリッピングペナルティ
    # 以前は 1% で 0点 (10000倍) と厳しすぎたため、5% で 0点 (2000倍) に緩和
    clip_score = max(0, 100 - quality.clipping_ratio * 2000)
    
    # S/N比スコア（20dB以上で満点）
    snr_score = min(100, max(0, quality.snr_db * 5))
    
    # データ長スコア（10秒以上で満点）
    length_score = min(100, quality.data_length_s * 10)
    
    breakdown = {
        "飽和回避 (Clipping)": float(clip_score),
        "ノイズ耐性 (SNR)": float(snr_score),
        "データ量 (Length)": float(length_score)
    }
    
    return float(np.mean([clip_score, snr_score, length_score])), breakdown
