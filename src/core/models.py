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


@dataclass
class AnalysisConfig:
    """解析条件（再現性保証用）"""
    quantity: SignalQuantity
    window: WindowFunction
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None
    filter_order: int = 4
    hpf_enabled: bool = False
    lpf_enabled: bool = False
    
    # New pluggable noise reduction settings
    noise_reduction_plugin_name: Optional[str] = None
    noise_reduction_plugin_params: Optional[dict] = None


@dataclass
class AnalysisPreset:
    """解析条件のプリセット設定"""
    name: str
    config: AnalysisConfig
    top_n_peaks: int = 5
    min_peak_height_percent: float = 10.0
    peak_distance_hz: float = 10.0
    fft_log_x: bool = False
    show_raw_signal: bool = True
    spec_nperseg: int = 512
    unit_space_name: Optional[str] = None # Associated unit space

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
    """Vibration features including frequency-domain power bands and spectral shape features"""
    power_low: float
    power_mid: float
    power_high: float
    spectral_centroid: float
    spectral_spread: float
    spectral_entropy: float
    overall_level: float # FFT Overall value (Total)
    overall_low: float   # FFT Overall value (< 1000 Hz)
    overall_high: float  # FFT Overall value (>= 1000 Hz)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Returns the list of feature names in the same order as to_vector()."""
        return [
            "RMS", "Peak", "Kurtosis", "Skewness", "CrestFactor", "ShapeFactor",
            "PowerLow", "PowerMid", "PowerHigh",
            "SpectralCentroid", "SpectralSpread", "SpectralEntropy",
            "OverallTotal", "OverallLow", "OverallHigh"
        ]

    def to_vector(self) -> np.ndarray:
        """
        Converts the vibration features into a NumPy array (vector) for MT method.
        """
        return np.array([
            self.rms, self.peak, self.kurtosis, self.skewness,
            self.crest_factor, self.shape_factor,
            self.power_low, self.power_mid, self.power_high,
            self.spectral_centroid, self.spectral_spread, self.spectral_entropy,
            self.overall_level, self.overall_low, self.overall_high
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

@dataclass
class MTConfig:
    anomaly_threshold: float = 3.0
    min_samples: int = 10
    recommended_samples: int = 30

@dataclass
class BenchmarkConfig:
    dataset_name: str
    analysis_config: AnalysisConfig
    mt_config: MTConfig
    nr_plugin_config: Optional[dict] = None # {"name": "plugin_name", "params": {...}}
