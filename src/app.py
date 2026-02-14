import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace # Not fully implemented yet, but included for structure
from src.utils.audit_log import save_audit_log
import tempfile
import os
from datetime import datetime # Added for AnalysisResult
import plotly.graph_objects as go
from scipy.signal import find_peaks # For peak annotation

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

# 1. WAVファイルアップロード
uploaded_file = st.file_uploader("WAVファイルをアップロードしてください", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.audio(uploaded_file, format="audio/wav")

    # Perform initial loading and analysis
    try:
        fs_hz, data_raw, file_hash = load_wav_file(tmp_file_path)
        st.write(f"ファイル名: {uploaded_file.name}")
        st.write(f"サンプリング周波数: {fs_hz} Hz")
        st.write(f"データ長: {len(data_raw) / fs_hz:.2f} 秒")
        
        # Display basic analysis configuration options
        st.sidebar.header("解析設定")
        selected_quantity = st.sidebar.selectbox(
            "物理量種別",
            options=list(SignalQuantity),
            format_func=lambda x: x.value,
            index=0
        )
        
        selected_window = st.sidebar.selectbox(
            "窓関数",
            options=list(WindowFunction),
            format_func=lambda x: x.value,
            index=0
        )

        # Placeholder for filter settings
        # Max value for cutoff frequencies should be strictly less than Nyquist frequency (fs_hz/2)
        nyquist_freq = fs_hz / 2
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            highpass_hz = st.number_input("HPF (Hz)", value=10, min_value=0, max_value=int(nyquist_freq * 0.99), step=1)
        with col2:
            lowpass_hz = st.number_input("LPF (Hz)", value=int(nyquist_freq * 0.99), min_value=0, max_value=int(nyquist_freq * 0.99), step=1)
        filter_order = st.sidebar.number_input("フィルタ次数", value=4, min_value=1, max_value=10, step=1)

        analysis_config = AnalysisConfig(
            quantity=selected_quantity,
            window=selected_window,
            highpass_hz=float(highpass_hz) if highpass_hz > 0 else None,
            lowpass_hz=float(lowpass_hz) if lowpass_hz < nyquist_freq * 0.99 else None, # Check this condition again
            filter_order=filter_order
        )
        
        st.sidebar.write("---")
        st.sidebar.subheader("現在のフィルタ条件:")
        if analysis_config.highpass_hz:
            st.sidebar.write(f"HPF: {analysis_config.highpass_hz} Hz")
        if analysis_config.lowpass_hz:
            st.sidebar.write(f"LPF: {analysis_config.lowpass_hz} Hz")
        if analysis_config.highpass_hz or analysis_config.lowpass_hz:
            st.sidebar.write(f"次数: {analysis_config.filter_order}")
        else:
            st.sidebar.write("フィルタなし")
        st.sidebar.write(f"窓関数: {analysis_config.window.value}")


        # Perform processing (placeholder for actual processing chain)
        processed_data = remove_dc_offset(data_raw)
        processed_data = apply_butterworth_filter(
            processed_data,
            fs_hz,
            analysis_config.highpass_hz,
            analysis_config.lowpass_hz,
            analysis_config.filter_order
        )

        # --- Display results ---
        col_summary, col_time_plot = st.columns([1, 2])

        with col_summary:
            st.subheader("解析結果 (時間領域)")
            time_features = calculate_time_domain_features(processed_data)
            st.write(f"RMS: {time_features.rms:.3f}")
            st.write(f"Peak: {time_features.peak:.3f}")
            st.write(f"Kurtosis: {time_features.kurtosis:.3f}")
            st.write(f"Skewness: {time_features.skewness:.3f}")
            st.write(f"Crest Factor: {time_features.crest_factor:.3f}")
            st.write(f"Shape Factor: {time_features.shape_factor:.3f}")
        
        with col_time_plot:
            st.subheader("時間波形")
            time_axis = np.linspace(0, len(processed_data) / fs_hz, len(processed_data))
            fig_time = go.Figure(data=[go.Scatter(x=time_axis, y=processed_data, mode='lines')])
            fig_time.update_layout(
                title="時間波形",
                xaxis_title="時間 (s)",
                yaxis_title=f"{analysis_config.quantity.value} (normalized)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_time, use_container_width=True)


        st.subheader("解析結果 (周波数領域 & 品質)")
        freq_hz, magnitude, power_low, power_mid, power_high = \
            calculate_fft_features(processed_data, fs_hz, analysis_config.window)
        
        rms_after_processing = np.sqrt(np.mean(processed_data**2))
        quality = calculate_quality_metrics(data_raw, fs_hz, rms_after_processing, magnitude)
        confidence_score = get_confidence_score(quality)

        col_freq_summary, col_freq_plot = st.columns([1, 2])

        with col_freq_summary:
            st.write(f"パワー寄与率 (低周波): {power_low:.2%}")
            st.write(f"パワー寄与率 (中周波): {power_mid:.2%}")
            st.write(f"パワー寄与率 (高周波): {power_high:.2%}")

            st.write(f"クリッピング率: {quality.clipping_ratio:.2%}")
            st.write(f"S/N比: {quality.snr_db:.2f} dB")
            st.write(f"診断信頼度: {confidence_score:.1f}%")

            if confidence_score < 50:
                st.warning("診断信頼度が低いです。データの品質を確認してください。")
            
            # Frequency axis scale toggle
            freq_scale_type = st.radio("周波数軸スケール", ("線形", "対数"), horizontal=True)

        with col_freq_plot:
            fig_fft = go.Figure(data=[go.Scatter(x=freq_hz, y=magnitude, mode='lines')])
            fig_fft.update_layout(
                title="FFT振幅スペクトル",
                xaxis_title="周波数 (Hz)",
                yaxis_title="振幅",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            if freq_scale_type == "対数":
                fig_fft.update_xaxes(type="log")
            
            # Automatic peak annotation (Top 5 peaks)
            # Find peaks only in a reasonable frequency range and above a threshold
            min_peak_height = np.max(magnitude) * 0.1 # Peaks must be at least 10% of max magnitude
            peak_indices, _ = find_peaks(magnitude, height=min_peak_height, distance=10) # distance to avoid adjacent noise peaks
            
            # Sort peaks by magnitude and take top N
            top_n = 5
            sorted_peak_indices = peak_indices[np.argsort(magnitude[peak_indices])[-top_n:]]

            for idx in sorted_peak_indices:
                fig_fft.add_annotation(
                    x=freq_hz[idx],
                    y=magnitude[idx],
                    text=f"{freq_hz[idx]:.1f}Hz",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40
                )
            st.plotly_chart(fig_fft, use_container_width=True)

    except ValueError as e:
        st.error(f"データ処理エラー: {e}")
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

else:
    st.info("WAVファイルをアップロードして解析を開始してください。")
