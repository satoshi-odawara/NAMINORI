import numpy as np
import pytest
from scipy import signal
from src.core.models import WindowFunction

def calculate_scaling_correction(nperseg, window_type):
    """
    App.py に実装したスケーリング補正ロジックの参照実装。
    STFTの振幅スケールをFFTの振幅スケールに合わせるための補正係数 alpha を計算する。
    """
    if window_type == WindowFunction.HANNING:
        win = np.hanning(nperseg)
    else:
        win = np.ones(nperseg) # simplified for test
    
    # alpha = sqrt(nperseg * sum(win**2)) / sum(win)
    return np.sqrt(nperseg * np.sum(win**2)) / np.sum(win)

def test_spectrogram_scaling_correction():
    """
    スペクトログラムの振幅が、FFTの振幅と物理的に一致するかを検証する。
    正弦波（振幅1.0）を入力し、補正後のスペクトログラムのピークが1.0になるかを確認。
    """
    fs = 1000
    t = np.arange(0, 1.0, 1/fs)
    freq = 100
    # 振幅 1.0 の正弦波
    x = 1.0 * np.sin(2 * np.pi * freq * t)
    
    nperseg = 256
    # STFT実行 (power scale)
    f, t_s, Sxx = signal.spectrogram(x, fs, window='hann', nperseg=nperseg, scaling='spectrum', mode='psd')
    
    # 補正係数の計算
    alpha = calculate_scaling_correction(nperseg, WindowFunction.HANNING)
    
    # 振幅への変換と補正
    mag_spec = np.sqrt(Sxx) * alpha
    
    # ピーク値の取得 (100Hz付近)
    peak_val = np.max(mag_spec)
    
    # 検証: 1.0 に近い値であること (窓関数の影響で微減するが、補正によりほぼ1.0になる)
    # Hanning窓の場合、ACF補正後のピークは1.0になるべき
    assert np.isclose(peak_val, 1.0, rtol=0.05)

def test_frequency_interpolation():
    """
    異なる解像度間での周波数補間が正しく行われるかを検証。
    """
    # 高解像度基準 (1001点)
    orig_freqs = np.linspace(0, 500, 1001)
    orig_mags = np.zeros(1001)
    orig_mags[200] = 5.0 # 100Hzにピーク
    
    # 低解像度ターゲット (129点)
    target_freqs = np.linspace(0, 500, 129)
    
    # 補間実行
    interp_mags = np.interp(target_freqs, orig_freqs, orig_mags)
    
    # ターゲット側で 100Hz 付近にピークが転送されていること
    target_peak_idx = np.argmin(np.abs(target_freqs - 100))
    assert interp_mags[target_peak_idx] > 0
    # 面積（エネルギー）がおよそ保存されていることを確認
    assert np.isclose(np.sum(interp_mags), np.sum(orig_mags) * (len(target_freqs)/len(orig_freqs)), rtol=0.2)
