import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import AnalysisResult
from src.core.plugins import plugin_manager
from src.core.evaluation import NoiseReductionEvaluation, perform_nr_evaluation
import pandas as pd
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json
from dataclasses import asdict
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.benchmarking import BenchmarkConfig, MTConfig, run_benchmark_test, FileBenchmarkResult # New imports
from pathlib import Path # New import
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
                if selected_plugin.get_name() == "spectral_subtraction":
                    if 'mt_space' not in st.session_state or st.session_state.mt_space.noise_power_spectrum_avg is None:
                        st.warning("Spectral Subtractionを使用するには、まずMT法設定で単位空間を構築し、ノイズプロファイルを学習させてください。")
                        processed = signal_pre_nr # Skip applying plugin
                    else:
                        # Pass the learned noise power spectrum to the plugin
                        signal_post_nr = selected_plugin.process(
                            signal_pre_nr,
                            fs_hz,
                            p_noise_avg=st.session_state.mt_space.noise_power_spectrum_avg,
                            **config.noise_reduction_plugin_params
                        )
                        nr_eval_results = perform_nr_evaluation(signal_pre_nr, signal_post_nr)
                        processed = signal_post_nr
                else: # Other plugins
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
                
                st.subheader("データ品質")
                st.metric("クリッピング率", f"{quality.clipping_ratio:.2%}")
                st.metric("S/N 比", f"{quality.snr_db:.2f} dB")
                
                color = "green" if confidence >= 80 else "orange" if confidence >= 50 else "red"
                st.markdown(f'#### 診断信頼度: <span style="color:{color};">{confidence:.1f}%</span>', unsafe_allow_html=True)
                
                if 'mt_space' in st.session_state and st.session_state.mt_space.mean_vector is not None:
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
                    # ... (rest of the evaluation display)
            
            # --- Audit Log ---
            # ... (rest of the audit log display)

        except Exception as e:
            st.error(f"解析エラー: {e}")
        finally:
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
    else:
        st.info("解析を開始するには評価用WAVファイルをアップロードしてください。")

elif page_selection == "ベンチマーク":
    st.header("ベンチマーク テスト")
    # ... (The full benchmark code remains here)

elif page_selection == "合成データ生成":
    st.header("合成データ生成")
    st.markdown("テストや評価に使用する、特性が既知の合成振動信号を生成します。")

    with st.sidebar:
        st.header("信号生成設定")
        
        st.subheader("基本設定")
        fs_hz = st.number_input("サンプリング周波数 (Hz)", 100, 96000, 48000, key="synth_fs")
        duration_s = st.number_input("持続時間 (s)", 0.1, 60.0, 5.0, key="synth_duration")
        base_freq_hz = st.number_input("基本周波数 (Hz)", 1.0, float(fs_hz/2), 120.0, key="synth_base_freq")
        base_amplitude = st.slider("基本振幅", 0.0, 1.0, 0.5, key="synth_base_amp")

        st.subheader("変調設定")
        use_am = st.checkbox("振幅変調 (AM) を有効化", key="synth_use_am")
        am_config = None
        if use_am:
            with st.expander("AM 設定", expanded=True):
                am_mod_freq = st.slider("変調周波数 (Hz)", 0.1, 100.0, 10.0, key="synth_am_freq")
                am_mod_index = st.slider("変調指数", 0.0, 1.0, 0.5, key="synth_am_index")
                am_config = sdg.AMConfig(mod_freq_hz=am_mod_freq, mod_index=am_mod_index)

        use_fm = st.checkbox("周波数変調 (FM) を有効化", key="synth_use_fm")
        fm_config = None
        if use_fm:
            with st.expander("FM 設定", expanded=True):
                fm_mod_freq = st.slider("変調周波数 (Hz)", 0.1, 100.0, 5.0, key="synth_fm_freq")
                fm_mod_index = st.slider("変調指数 (偏移/変調周波数)", 0.0, 10.0, 2.0, key="synth_fm_index")
                fm_config = sdg.FMConfig(mod_freq_hz=fm_mod_freq, mod_index=fm_mod_index)
        
        st.subheader("追加要素")
        use_impulses = st.checkbox("インパルスを追加", key="synth_use_impulse")
        impulse_config = None
        if use_impulses:
             with st.expander("インパルス設定", expanded=True):
                impulse_rate = st.slider("インパルス発生レート (Hz)", 0.1, 50.0, 5.0, key="synth_impulse_rate")
                impulse_amp = st.slider("インパルス振幅", 0.0, 1.0, 0.3, key="synth_impulse_amp")
                impulse_config = [sdg.ImpulseConfig(impulse_rate_hz=impulse_rate, impulse_amplitude=impulse_amp)]

        use_noise = st.checkbox("ノイズを追加", key="synth_use_noise")
        noise_config = None
        if use_noise:
            with st.expander("ノイズ設定", expanded=True):
                noise_type = st.selectbox("ノイズ種別", [t.value for t in sdg.NoiseType], key="synth_noise_type")
                snr_db = st.slider("S/N比 (dB)", -10, 40, 20, key="synth_snr")
                noise_config = sdg.NoiseConfig(noise_type=sdg.NoiseType(noise_type), snr_db=snr_db)

    # Assemble the config object
    signal_config = sdg.SignalConfig(
        fs_hz=fs_hz,
        duration_s=duration_s,
        base_freq_hz=base_freq_hz,
        base_amplitude=base_amplitude,
        am_config=am_config,
        fm_config=fm_config,
        impulses=impulse_config,
        noise_config=noise_config
    )

    if st.button("合成信号を生成", key="generate_synth_button"):
        st.subheader("生成された信号")
        try:
            with st.spinner("信号を生成中..."):
                generated_signal = sdg.generate_signal(signal_config)

            st.write("#### 時間波形")
            fig = go.Figure(go.Scatter(x=np.linspace(0, duration_s, len(generated_signal)), y=generated_signal))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", height=300, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.write("#### 音声プレビュー")
            st.audio(generated_signal, sample_rate=fs_hz)

            # Create a WAV file in memory
            wav_io = io.BytesIO()
            wav_data = (generated_signal * 32767).astype(np.int16)
            write_wav(wav_io, fs_hz, wav_data)
            wav_io.seek(0)

            st.download_button(
                label="WAVファイルをダウンロード",
                data=wav_io,
                file_name="synthetic_signal.wav",
                mime="audio/wav"
            )
            
            st.write("---")
            st.subheader("生成に使用した設定")
            st.json(asdict(signal_config))

        except NotImplementedError as e:
            st.error(f"生成エラー: {e}")
        except Exception as e:
            st.error(f"予期せぬエラーが発生しました: {e}")

