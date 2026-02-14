from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np # Added for to_vector method

class SignalQuantity(Enum):
    ACCEL = "accel"
    VELOCITY = "velocity"
    DISPLACEMENT = "disp"

class WindowFunction(Enum):
    HANNING = "hanning"
    FLATTOP = "flattop"

@dataclass
class AnalysisConfig:
    """解析条件（再現性保証用）"""
    quantity: SignalQuantity
    window: WindowFunction
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None
    filter_order: int = 4

@dataclass
class QualityMetrics:
    """データ品質評価"""
    clipping_ratio: float  # クリッピング率
    snr_db: float         # S/N比
    data_length_s: float  # 有効データ長

@dataclass
class VibrationFeatures:
    """振動特徴量（MT法対応）"""
    rms: float
    peak: float
    kurtosis: float
    skewness: float
    crest_factor: float
    shape_factor: float
    power_low: float
    power_mid: float
    power_high: float

    def to_vector(self) -> np.ndarray:
        """
        Converts the vibration features into a NumPy array (vector) for MT method.
        """
        return np.array([
            self.rms, self.peak, self.kurtosis, self.skewness,
            self.crest_factor, self.shape_factor,
            self.power_low, self.power_mid, self.power_high
        ])

@dataclass
class AnalysisResult:
    """解析結果（全情報含む）"""
    features: VibrationFeatures
    quality: QualityMetrics
    config: AnalysisConfig
    timestamp: str
    file_hash: str
    fs_hz: float
    app_version: str = "1.0.0"
