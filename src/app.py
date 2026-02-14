import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics, NoiseReductionFilterType
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import AnalysisResult
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json
from dataclasses import asdict
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter, apply_noise_reduction_filter

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

if 'mt_space' not in st.session_state:
    st.session_state.mt_space = MTSpace()

def get_serializable_audit_log(result: AnalysisResult) -> dict:
    log_data = asdict(result)
    log_data['config']['quantity'] = result.config.quantity.value
    log_data['config']['window'] = result.config.window.value
    log_data['config']['noise_reduction_type'] = result.config.noise_reduction_type.value # Add this
    if result.config.notch_freq_hz:
        log_data['config']['notch_freq_hz'] = result.config.notch_freq_hz
    if result.config.notch_q_factor:
        log_data['config']['notch_q_factor'] = result.config.notch_q_factor
    return log_data

# --- Sidebar ---
st.sidebar.header("MT法設定")
with st.sidebar.expander("正常データで単位空間を構築"):
    normal_files = st.file_uploader("正常時のWAVファイルをアップロード (複数可)", type=["wav"], accept_multiple_files=True, key="mt_normal_uploader")
    train_hpf = st.number_input("HPF (訓練用)", 0, 22050, 10, key="train_hpf")
    train_lpf = st.number_input("LPF (訓練用)", 0, 22050, 20000, key="train_lpf")
    train_order = st.number_input("フィルタ次数 (訓練用)", 1, 10, 4, key="train_order")
    # Noise Reduction Filter settings for training - using defaults or user input, but for simplicity, let's keep it None for training for now
    # Or, we should expose these to the UI as well for full control
    train_nr_type = NoiseReductionFilterType.NONE # Default to none for training for now
    train_nr_freq = None
    train_nr_q = None


    if st.button("単位空間を構築/更新", key="build_mt_space"):
        if normal_files:
            st.session_state.mt_space = MTSpace()
            for i, file in enumerate(normal_files):
                st.info(f"処理中: {file.name} ({i+1}/{len(normal_files)})") # Using st.info for progress as st.text was not showing
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                    tmp_f.write(file.getvalue())
                    fs, data, _ = load_wav_file(tmp_f.name)
                os.remove(tmp_f.name)
                
                processed = remove_dc_offset(data)
                processed = apply_butterworth_filter(processed, fs, float(train_hpf) if train_hpf > 0 else None, float(train_lpf) if train_lpf > 0 else None, train_order)
                # Apply noise reduction filter to training data if configured
                processed = apply_noise_reduction_filter(
                    processed,
                    fs,
                    train_nr_type,
                    train_nr_freq,
                    train_nr_q
                )
                
                time_feats = calculate_time_domain_features(processed)
                _, _, power_bands_dict = calculate_fft_features(processed, fs, WindowFunction.HANNING)
                
                st.session_state.mt_space.add_normal_sample(VibrationFeatures(
                    **asdict(time_feats),
                    power_low=power_bands_dict['low'],
                    power_mid=power_bands_dict['mid'],
                    power_high=power_bands_dict['high']
                ))
            st.success(f"{len(normal_files)}個のファイルで単位空間を構築/更新しました。")

st.sidebar.info(f"単位空間ステータス: {st.session_state.mt_space.get_status()}")

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
        
        st.sidebar.header("ノイズ除去フィルタ設定")
        nr_type = st.sidebar.selectbox("ノイズ除去タイプ", list(NoiseReductionFilterType), format_func=lambda x: x.value, key="nr_type")
        nr_freq = None
        nr_q = None

        if nr_type == NoiseReductionFilterType.NOTCH:
            nr_freq = st.sidebar.number_input("ノッチ周波数 (Hz)", 0.0, nyquist, 60.0, key="nr_freq")
            nr_q = st.sidebar.number_input("ノッチQ値", 0.1, 100.0, 30.0, key="nr_q")

        st.sidebar.header("FFTピーク設定")
        top_n_peaks = st.sidebar.slider("ピーク表示数", 1, 20, 5, key="peak_count")
        min_peak_height_percent = st.sidebar.slider("最小ピーク高さ（最大値に対する%）", 0, 100, 10, key="peak_height")
        peak_distance_hz = st.sidebar.slider("ピーク最小距離（Hz）", 1, 100, 10, key="peak_distance")

        config = AnalysisConfig(
            quantity, window,
            float(hpf) if hpf > 0 else None,
            float(lpf) if lpf < nyquist else None,
            order,
            noise_reduction_type=nr_type,
            notch_freq_hz=float(nr_freq) if nr_freq else None,
            notch_q_factor=float(nr_q) if nr_q else None
        )

        with st.container():
            st.subheader("適用中の解析設定")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("物理量", config.quantity.value)
            with col2:
                st.markdown(f"""
                **標準フィルタ設定:**
                - HPF: `{config.highpass_hz or 'N/A'}`
                - LPF: `{config.lowpass_hz or 'N/A'}`
                - Order: `{config.filter_order}`
                """)
            with col3:
                st.metric("窓関数", config.window.value)
            
            # ノイズ除去フィルタの表示
            if config.noise_reduction_type != NoiseReductionFilterType.NONE:
                st.markdown(f"""
                **ノイズ除去フィルタ:** `{config.noise_reduction_type.value}`
                - 周波数: `{config.notch_freq_hz or 'N/A'} Hz`
                - Q値: `{config.notch_q_factor or 'N/A'}`
                """)
        st.write("---")

        processed = remove_dc_offset(data_raw)
        processed = apply_butterworth_filter(
            processed,
            fs_hz,
            config.highpass_hz,
            config.lowpass_hz,
            config.filter_order
        )
        # Apply noise reduction filter
        processed = apply_noise_reduction_filter(
            processed,
            fs_hz,
            config.noise_reduction_type,
            config.notch_freq_hz,
            config.notch_q_factor
        )
        
        time_features = calculate_time_domain_features(processed)
        freqs, mags, power_bands = calculate_fft_features(processed, fs_hz, config.window)
        quality = calculate_quality_metrics(data_raw, fs_hz, time_features.rms, mags)
        confidence = get_confidence_score(quality)
        unit = config.quantity.unit_str

        # --- Display ---
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("時間領域 特徴量")
            st.metric(f"RMS ({unit})", f"{time_features.rms:.3f}")
            st.metric(f"Peak ({unit})", f"{time_features.peak:.3f}")
            st.metric("Kurtosis", f"{time_features.kurtosis:.3f}")
            st.metric("Crest Factor", f"{time_features.crest_factor:.3f}")
            
            st.subheader("データ品質")
            st.metric("クリッピング率", f"{quality.clipping_ratio:.2%}")
            st.metric("S/N 比", f"{quality.snr_db:.2f} dB")
            
            color = "green" if confidence >= 80 else "orange" if confidence >= 50 else "red"
            st.markdown(f'#### 診断信頼度: <span style="color:{color};">{confidence:.1f}%</span>', unsafe_allow_html=True)
            
            if st.session_state.mt_space.mean_vector is not None:
                features = VibrationFeatures(
                    **asdict(time_features),
                    power_low=power_bands['low'],
                    power_mid=power_bands['mid'],
                    power_high=power_bands['high']
                )
                md = st.session_state.mt_space.calculate_md(features)
                md_color = "green" if md < 3.0 else "orange" if md < 5.0 else "red"
                st.markdown(f'#### MT法診断 (MD): <span style="color:{md_color};">{md:.2f}</span>', unsafe_allow_html=True)
        with col2:
            st.subheader("時間波形")
            fig = go.Figure(go.Scatter(x=np.arange(len(processed))/fs_hz, y=processed))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title=f"Amplitude ({unit})", height=250, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("FFT スペクトル")
            fig2 = go.Figure(go.Scatter(x=freqs, y=mags, name='Spectrum'))

            distance_samples = int(peak_distance_hz / (freqs[1] - freqs[0])) if freqs[1] > freqs[0] else 1
            peaks, _ = find_peaks(mags, height=np.max(mags)*(min_peak_height_percent/100.0), distance=distance_samples)
            
            sorted_peak_indices = peaks[np.argsort(mags[peaks])[-top_n_peaks:]]
            
            fig2.add_trace(go.Scatter(x=freqs[sorted_peak_indices], y=mags[sorted_peak_indices], mode='markers', marker_symbol='x', name='Peaks', marker_color='red', marker_size=10))

            for idx in sorted_peak_indices:
                fig2.add_annotation(x=freqs[idx], y=mags[idx], text=f"{freqs[idx]:.1f}Hz", showarrow=True, arrowhead=2, ax=0, ay=-40)
            
            fig2.update_layout(xaxis_title="Frequency (Hz)", yaxis_title=f"Amplitude ({unit})", height=250, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig2, use_container_width=True)
            
        # --- Audit Log ---
        features_for_audit = VibrationFeatures(
            **asdict(time_features),
            power_low=power_bands['low'],
            power_mid=power_bands['mid'],
            power_high=power_bands['high']
        )
        result = AnalysisResult(features_for_audit, quality, config, datetime.now().isoformat(), file_hash, fs_hz)
        with st.expander("監査ログ (JSON)"):
            log_data = get_serializable_audit_log(result)
            st.json(log_data)
            st.download_button("Download Log", json.dumps(log_data, indent=2), f"audit_{file_hash[:8]}.json", "application/json")

    except Exception as e:
        st.error(f"解析エラー: {e}")
    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
else:
    st.info("解析を開始するには評価用WAVファイルをアップロードしてください。")
