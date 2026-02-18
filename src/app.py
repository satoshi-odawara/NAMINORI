import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import AnalysisResult, get_serializable_audit_log
from src.core.plugins import plugin_manager
from src.core.evaluation import perform_nr_evaluation
import pandas as pd
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json
from dataclasses import asdict
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.benchmarking import BenchmarkConfig, MTConfig, run_benchmark_test
from pathlib import Path
import io
from scipy.io.wavfile import write as write_wav
from src.utils import synthetic_data_generator as sdg

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

# --- MT法 単位空間の初期化 ---
if 'mt_space' not in st.session_state:
    st.session_state.mt_space = MTSpace(min_samples=10, recommended_samples=30)

# Page selection in sidebar
page_selection = st.sidebar.radio(
    "機能選択",
    ("通常解析", "ベンチマーク", "合成データ生成"),
    key="page_selector"
)

if page_selection == "通常解析":
    # --- Main Application ---
    uploaded_file = st.file_uploader("評価用WAVファイルをアップロード", type=["wav"], key="evaluation_uploader")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        st.audio(uploaded_file, format="audio/wav")

        try:
            fs_hz, data_raw, file_hash = load_wav_file(tmp_file_path)
            st.write(f"ファイル名: {uploaded_file.name}, サンプリング周波数: {fs_hz} Hz, データ長: {len(data_raw) / fs_hz:.2f} 秒")

            st.sidebar.header("評価用データ解析設定")
            quantity = st.sidebar.selectbox("物理量種別", list(SignalQuantity), format_func=lambda x: x.value, key="eval_quantity")
            window = st.sidebar.selectbox("窓関数", list(WindowFunction), format_func=lambda x: x.value, key="eval_window")
            
            nyquist = fs_hz / 2
            hpf = st.sidebar.number_input("HPF (Hz)", 0.0, nyquist, 10.0, key="eval_hpf")
            lpf = st.sidebar.number_input("LPF (Hz)", 0.0, nyquist, nyquist, key="eval_lpf")
            order = st.sidebar.number_input("フィルタ次数", 1, 10, 4, key="eval_order")
            
            st.sidebar.header("FFTピーク設定")
            top_n_peaks = st.sidebar.slider("ピーク表示数", 1, 20, 5, key="peak_count")
            min_peak_height_percent = st.sidebar.slider("最小ピーク高さ（最大値に対する%）", 0, 100, 10, key="peak_height")
            peak_distance_hz = st.sidebar.slider("ピーク最小距離（Hz）", 1, 100, 10, key="peak_distance")

            config = AnalysisConfig(quantity=quantity, window=window, highpass_hz=float(hpf), lowpass_hz=float(lpf), filter_order=order)
            
            processed_dc_removed = remove_dc_offset(data_raw)
            processed = apply_butterworth_filter(processed_dc_removed, fs_hz, config.highpass_hz, config.lowpass_hz, config.filter_order)
            
            time_features = calculate_time_domain_features(processed)
            freqs, mags, freq_features = calculate_fft_features(processed, fs_hz, config.window)
            quality = calculate_quality_metrics(data_raw, fs_hz, time_features.rms, mags)
            confidence = get_confidence_score(quality)
            unit = config.quantity.unit_str

            all_features = VibrationFeatures(**asdict(time_features), **freq_features)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("時間領域 特徴量")
                st.metric(f"RMS ({unit})", f"{all_features.rms:.3f}")
                st.metric(f"Peak ({unit})", f"{all_features.peak:.3f}")
                st.metric("Kurtosis", f"{all_features.kurtosis:.3f}")

                st.subheader("周波数領域 特徴量")
                st.metric("Spectral Centroid (Hz)", f"{all_features.spectral_centroid:.2f}")
                st.metric("Spectral Spread (Hz)", f"{all_features.spectral_spread:.2f}")
                st.metric("Spectral Entropy", f"{all_features.spectral_entropy:.3f}")
                
                st.subheader("データ品質")
                st.metric("クリッピング率", f"{quality.clipping_ratio:.2%}")
                st.metric("S/N 比", f"{quality.snr_db:.2f} dB")
                
                color = "green" if confidence >= 80 else "orange"
                st.markdown(f'#### 診断信頼度: <span style="color:{color};">{confidence:.1f}%</span>', unsafe_allow_html=True)
                
                if 'mt_space' in st.session_state and st.session_state.mt_space.mean_vector is not None:
                    md = st.session_state.mt_space.calculate_md(all_features)
                    md_color = "green" if md < 3.0 else "orange" if md < 5.0 else "red"
                    st.markdown(f'#### MT法診断 (MD): <span style="color:{md_color};">{md:.2f}</span>', unsafe_allow_html=True)
            # ... (rest of the page)

        except Exception as e:
            st.error(f"解析エラー: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

elif page_selection == "ベンチマーク":
    st.header("ベンチマーク テスト")
    st.subheader("ベンチマーク設定")
    benchmark_dataset_options = ["DCASE-2020-Task2-Dataset/pump"]
    selected_benchmark_dataset = st.selectbox("ベンチマークデータセットを選択", options=benchmark_dataset_options)
    dataset_root_path = Path("bench") / selected_benchmark_dataset
    
    with st.expander("解析設定 (ベンチマーク用)", expanded=True):
        # ...
        benchmark_analysis_config = AnalysisConfig(quantity=SignalQuantity.ACCEL, window=WindowFunction.HANNING)

    with st.expander("MT法設定 (ベンチマーク用)", expanded=True):
        benchmark_anomaly_threshold = st.number_input("異常判定閾値 (MD)", 0.1, 10.0, 3.0)
        optimize_threshold_checkbox = st.checkbox("異常判定閾値を自動で最適化する (F1スコア基準)")
        benchmark_mt_config = MTConfig(anomaly_threshold=benchmark_anomaly_threshold)

    benchmark_config = BenchmarkConfig(
        dataset_name=selected_benchmark_dataset,
        analysis_config=benchmark_analysis_config,
        mt_config=benchmark_mt_config,
        optimize_threshold=optimize_threshold_checkbox
    )

    if st.button("ベンチマークを実行"):
        with st.spinner("ベンチマーク処理中..."):
            result = run_benchmark_test(benchmark_config, dataset_root_path)
        st.success("ベンチマークが完了しました！")
        if result.optimized_threshold is not None:
            st.metric("最適化された異常判定閾値", f"{result.optimized_threshold:.4f}")
        # ... (rest of results display)

elif page_selection == "合成データ生成":
    # ... (code for this page)
    pass

