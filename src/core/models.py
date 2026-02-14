from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np # Added for to_vector method

class SignalQuantity(Enum):
    ACCEL = "accel"
    VELOCITY = "velocity"
    DISPLACEMENT = "disp"

    @property
    def unit_str(self) -> str:
        """Returns the string representation of the unit for the quantity."""
        if self == SignalQuantity.ACCEL:
            return "m/s^2"
        elif self == SignalQuantity.VELOCITY:
            return "mm/s"
        elif self == SignalQuantity.DISPLACEMENT:
            return "μm"
        return ""

class WindowFunction(Enum):
    HANNING = "hanning"
    FLATTOP = "flattop"

class NoiseReductionFilterType(Enum):
    NONE = "none"
    NOTCH = "notch" # ノッチフィルタ (特定の周波数を除去)
    # 将来的に他のノイズ除去フィルタを追加可能

@dataclass
class AnalysisConfig:
    """解析条件（再現性保証用）"""
    quantity: SignalQuantity
    window: WindowFunction
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None
    filter_order: int = 4
    
    # ノイズ除去フィルタ設定
    noise_reduction_type: NoiseReductionFilterType = NoiseReductionFilterType.NONE
    notch_freq_hz: Optional[float] = None # ノッチフィルタの中心周波数
    notch_q_factor: Optional[float] = None # ノッチフィルタのQ値 (帯域幅の逆数)

@dataclass
class QualityMetrics:
    """データ品質評価"""
    clipping_ratio: float  # クリッピング率
    snr_db: float         # S/N比
    data_length_s: float  # 有効データ長

@dataclass
class TimeDomainFeatures:
    """Time-domain vibration features"""
    rms: float
    peak: float
    kurtosis: float
    skewness: float
    crest_factor: float
    shape_factor: float

@dataclass
class VibrationFeatures(TimeDomainFeatures):
    """Vibration features including frequency-domain power bands"""
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
