import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import AnalysisResult
from src.core.plugins import plugin_manager
from src.core.evaluation import NoiseReductionEvaluation
import pandas as pd
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json
from dataclasses import asdict
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

# Load all noise reduction plugins on startup
plugin_manager.load_plugins()


if 'mt_space' not in st.session_state:
    st.session_state.mt_space = MTSpace()

def get_serializable_audit_log(result: AnalysisResult) -> dict:
    """Converts the analysis result to a serializable dictionary for the audit log."""
    log_data = asdict(result)
    log_data['config']['quantity'] = result.config.quantity.value
    log_data['config']['window'] = result.config.window.value
    
    # Add plugin info if it was used
    if result.config.noise_reduction_plugin_name:
        log_data['config']['noise_reduction_plugin_name'] = result.config.noise_reduction_plugin_name
    if result.config.noise_reduction_plugin_params:
        log_data['config']['noise_reduction_plugin_params'] = result.config.noise_reduction_plugin_params

    return log_data

def perform_nr_evaluation(signal_pre_nr: np.ndarray, signal_post_nr: np.ndarray) -> NoiseReductionEvaluation:
    """Calculates features before and after noise reduction for evaluation."""
    features_before = calculate_time_domain_features(signal_pre_nr)
    features_after = calculate_time_domain_features(signal_post_nr)
    
    return NoiseReductionEvaluation(
        features_before=features_before,
        features_after=features_after,
        signal_pre_nr=signal_pre_nr,
        signal_post_nr=signal_post_nr
    )


# --- Sidebar ---
st.sidebar.header("MT法設定")
with st.sidebar.expander("正常データで単位空間を構築"):
    normal_files = st.file_uploader("正常時のWAVファイルをアップロード (複数可)", type=["wav"], accept_multiple_files=True, key="mt_normal_uploader")
    train_hpf = st.number_input("HPF (訓練用)", 0, 22050, 10, key="train_hpf")
    train_lpf = st.number_input("LPF (訓練用)", 0, 22050, 20000, key="train_lpf")
    train_order = st.number_input("フィルタ次数 (訓練用)", 1, 10, 4, key="train_order")

    if st.button("単位空間を構築/更新", key="build_mt_space"):
        if normal_files:
            st.session_state.mt_space = MTSpace()
            for i, file in enumerate(normal_files):
                st.info(f"処理中: {file.name} ({i+1}/{len(normal_files)})")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                    tmp_f.write(file.getvalue())
                    fs, data, _ = load_wav_file(tmp_f.name)
                os.remove(tmp_f.name)
                
                processed = remove_dc_offset(data)
                processed = apply_butterworth_filter(processed, fs, float(train_hpf) if train_hpf > 0 else None, float(train_lpf) if train_lpf > 0 else None, train_order)
                # NOTE: For simplicity, noise reduction is not applied to training data in this refactoring.
                # A more advanced implementation might allow configuring this.
                
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
        
        st.sidebar.header("ノイズ除去プラグイン")
        # Get available plugins and add a "None" option
        available_plugins = plugin_manager.list_plugins()
        plugin_options = {plugin.get_display_name(): plugin for plugin in available_plugins}
        plugin_options["None"] = None
        
        selected_plugin_name = st.sidebar.selectbox(
            "ノイズ除去アルゴリズム",
            options=list(plugin_options.keys()),
            key="nr_plugin_selector"
        )
        
        selected_plugin = plugin_options[selected_plugin_name]
        plugin_params = {}
        plugin_conf_name = None

        if selected_plugin:
            plugin_conf_name = selected_plugin.get_name()
            st.sidebar.markdown(f"**{selected_plugin.get_display_name()} 設定**")
            for param in selected_plugin.get_parameters():
                if param.param_type == "number_input":
                    plugin_params[param.name] = st.sidebar.number_input(
                        label=param.label,
                        min_value=param.min_value,
                        value=param.default,
                        help=param.help_text,
                        key=f"nr_{plugin_conf_name}_{param.name}"
                    )
                elif param.param_type == "slider":
                    plugin_params[param.name] = st.sidebar.slider(
                        label=param.label,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        value=param.default,
                        help=param.help_text,
                        key=f"nr_{plugin_conf_name}_{param.name}"
                    )
        
        st.sidebar.header("FFTピーク設定")
        top_n_peaks = st.sidebar.slider("ピーク表示数", 1, 20, 5, key="peak_count")
        min_peak_height_percent = st.sidebar.slider("最小ピーク高さ（最大値に対する%）", 0, 100, 10, key="peak_height")
        peak_distance_hz = st.sidebar.slider("ピーク最小距離（Hz）", 1, 100, 10, key="peak_distance")

        config = AnalysisConfig(
            quantity=quantity, 
            window=window,
            highpass_hz=float(hpf) if hpf > 0 else None,
            lowpass_hz=float(lpf) if lpf < nyquist else None,
            filter_order=order,
            noise_reduction_plugin_name=plugin_conf_name,
            noise_reduction_plugin_params=plugin_params
        )

        with st.container():
            st.subheader("適用中の解析設定")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("物理量", config.quantity.value)
                st.metric("窓関数", config.window.value)
            with col2:
                st.markdown(f"""
                **標準フィルタ設定:**
                - HPF: `{config.highpass_hz or 'N/A'}`
                - LPF: `{config.lowpass_hz or 'N/A'}`
                - Order: `{config.filter_order}`
                """)
            with col3:
                # Display noise reduction plugin info
                if config.noise_reduction_plugin_name and selected_plugin:
                    nr_details = f"**ノイズ除去プラグイン:** `{selected_plugin.get_display_name()}`"
                    for key, value in config.noise_reduction_plugin_params.items():
                        nr_details += f"\n- {key}: `{value}`"
                    st.markdown(nr_details)
                else:
                    st.metric("ノイズ除去", "None")

        st.write("---")

        # Main processing chain
        processed_dc_removed = remove_dc_offset(data_raw)
        signal_pre_nr = apply_butterworth_filter(
            processed_dc_removed,
            fs_hz,
            config.highpass_hz,
            config.lowpass_hz,
            config.filter_order
        )
        
        nr_eval_results = None
        # Apply noise reduction plugin if selected and perform evaluation
        if selected_plugin and config.noise_reduction_plugin_params is not None:
            signal_post_nr = selected_plugin.process(
                signal_pre_nr,
                fs_hz,
                **config.noise_reduction_plugin_params
            )
            nr_eval_results = perform_nr_evaluation(signal_pre_nr, signal_post_nr)
            processed = signal_post_nr
        else:
            processed = signal_pre_nr

        
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
            
        # --- Noise Reduction Evaluation ---
        if nr_eval_results:
            with st.expander("ノイズ除去 評価", expanded=True):
                st.subheader("プラグイン効果の評価")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 特徴量へのインパクト")
                    
                    before_features = asdict(nr_eval_results.features_before)
                    after_features = asdict(nr_eval_results.features_after)
                    
                    df_data = []
                    for key in before_features:
                        before_val = before_features[key]
                        after_val = after_features[key]
                        delta = ((after_val - before_val) / before_val) * 100 if before_val != 0 else 0
                        df_data.append({
                            "Feature": key.replace('_', ' ').title(),
                            "Before": f"{before_val:.3f}",
                            "After": f"{after_val:.3f}",
                            "Delta (%)": f"{delta:+.2f}%"
                        })
                    
                    eval_df = pd.DataFrame(df_data)
                    st.table(eval_df.set_index("Feature"))

                with col2:
                    st.markdown("##### スペクトル比較")
                    
                    # Calculate FFTs for all three signals
                    freqs_pre, mags_pre, _ = calculate_fft_features(nr_eval_results.signal_pre_nr, fs_hz, config.window)
                    freqs_post, mags_post, _ = calculate_fft_features(nr_eval_results.signal_post_nr, fs_hz, config.window)
                    freqs_rem, mags_rem, _ = calculate_fft_features(nr_eval_results.removed_signal, fs_hz, config.window)

                    fig_eval = go.Figure()
                    fig_eval.add_trace(go.Scatter(x=freqs_pre, y=mags_pre, name='Before NR', line=dict(color='lightblue')))
                    fig_eval.add_trace(go.Scatter(x=freqs_post, y=mags_post, name='After NR', line=dict(color='blue')))
                    fig_eval.add_trace(go.Scatter(x=freqs_rem, y=mags_rem, name='Removed Signal', line=dict(color='red', dash='dash')))

                    fig_eval.update_layout(
                        xaxis_title="Frequency (Hz)", yaxis_title=f"Amplitude ({unit})", 
                        height=300, margin=dict(l=20,r=20,t=40,b=20),
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig_eval, use_container_width=True)

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
