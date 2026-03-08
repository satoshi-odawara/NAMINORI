import numpy as np
import pytest
from src.diagnostics.mt_method import MTSpace
from src.core.models import VibrationFeatures

def test_mt_space_rms_averaging_logic():
    """
    基準スペクトルの算出が「パワー平均（RMS平均）」で行われているかを検証する。
    物理的妥当性：振幅 3.0 と 4.0 のスペクトルを平均した場合、
    算術平均なら 3.5 になるが、パワー平均（RMS）なら sqrt((3^2 + 4^2)/2) = 3.535... になるべき。
    """
    mt_space = MTSpace()
    
    # ダミーの特徴量（MT法の計算には影響しないが引数として必要）
    dummy_features = VibrationFeatures(
        rms=1.0, peak=1.0, kurtosis=3.0, skewness=0.0, 
        crest_factor=1.0, shape_factor=1.0,
        power_low=0.1, power_mid=0.1, power_high=0.1,
        spectral_centroid=100.0, spectral_spread=10.0, spectral_entropy=0.5,
        overall_level=1.0, overall_low=0.5, overall_high=0.5
    )
    
    # 2つのサンプルスペクトルを用意（長さ 5）
    spec1 = np.array([3.0, 3.0, 3.0, 3.0, 3.0])
    spec2 = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    
    # 単位空間を一括構築
    mt_space.build_unit_space([dummy_features, dummy_features], [spec1, spec2])
    
    # 期待されるRMS平均値の計算
    expected_rms = np.sqrt((3.0**2 + 4.0**2) / 2.0)
    
    # 検証
    assert mt_space.average_magnitude_spectrum is not None
    np.testing.assert_allclose(mt_space.average_magnitude_spectrum, expected_rms, rtol=1e-5)
    
    # 算術平均（3.5）ではないことを念のため確認
    assert not np.isclose(mt_space.average_magnitude_spectrum[0], 3.5)

def test_mt_space_phase_insensitivity():
    """
    位相がずれた（または周波数がわずかに異なる）サンプルのエネルギーが保存されるかを検証。
    同じ振幅 1.0 のサンプルが複数ある場合、平均も 1.0 になるべき。
    """
    mt_space = MTSpace()
    dummy_features = VibrationFeatures(
        rms=1.0, peak=1.0, kurtosis=3.0, skewness=0.0, 
        crest_factor=1.0, shape_factor=1.0,
        power_low=0.1, power_mid=0.1, power_high=0.1,
        spectral_centroid=100.0, spectral_spread=10.0, spectral_entropy=0.5,
        overall_level=1.0, overall_low=0.5, overall_high=0.5
    )
    
    # すべて振幅 1.0 のスペクトル（サンプル数 10）
    specs = [np.ones(10) for _ in range(10)]
    features = [dummy_features for _ in range(10)]
    
    mt_space.build_unit_space(features, specs)
    
    # 平均しても振幅 1.0 が維持されていること
    np.testing.assert_allclose(mt_space.average_magnitude_spectrum, 1.0, rtol=1e-7)
