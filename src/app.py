import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics, AnalysisResult
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.core.quality_check import calculate_quality_metrics, get_confidence_score
from src.diagnostics.mt_method import MTSpace
from src.utils.audit_log import save_audit_log # This function could be used to save to file, but we display direct.
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
from scipy.signal import find_peaks
import json # Added for audit log display and download
from dataclasses import asdict # Added for audit log serialization

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

# Initialize MTSpace in session state
if 'mt_space' not in st.session_state:
    st.session_state.mt_space = MTSpace()

# MT Method Configuration in Sidebar
st.sidebar.header("MT法設定")
with st.sidebar.expander("正常データで単位空間を構築"):
    normal_files = st.file_uploader(
        "正常時のWAVファイルをアップロードしてください (複数選択可)",
        type=["wav"],
        accept_multiple_files=True,
        key="mt_normal_uploader"
    )

    # Use default analysis config for training if main config not yet available (e.g., no file uploaded yet)
    # or allow user to specify training specific filter settings
    st.write("単位空間構築時のフィルタ設定:")
    default_fs_for_training_ui = 44100 # A common default, used for UI max values
    default_nyquist_for_training_ui = default_fs_for_training_ui / 2

    col_train_hpf, col_train_lpf = st.columns(2)
    with col_train_hpf:
        train_hpf_val = st.number_input("HPF (訓練用)", value=10, min_value=0, max_value=int(default_nyquist_for_training_ui * 0.99), step=1, key="train_hpf_input")
    with col_train_lpf:
        train_lpf_val = st.number_input("LPF (訓練用)", value=int(default_nyquist_for_training_ui * 0.99), min_value=0, max_value=int(default_nyquist_for_training_ui * 0.99), step=1, key="train_lpf_input")
    train_filter_order = st.number_input("フィルタ次数 (訓練用)", value=4, min_value=1, max_value=10, step=1, key="train_order_input")

    # The actual fs_hz from the uploaded files will be used in load_wav_file
    # This config is for feature extraction *during training*
    training_analysis_config = AnalysisConfig(
        quantity=st.session_state.get('selected_quantity', SignalQuantity.ACCEL), # Use session state if available
        window=st.session_state.get('selected_window', WindowFunction.HANNING),   # Use session state if available
        highpass_hz=float(train_hpf_val) if train_hpf_val > 0 else None,
        lowpass_hz=float(train_lpf_val) if train_lpf_val < default_nyquist_for_training_ui * 0.99 else None,
        filter_order=train_filter_order
    )


    if st.button("単位空間を構築/更新", key="build_mt_space"):
        if normal_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_count = 0
            
            # Clear previous samples before re-building/updating
            st.session_state.mt_space = MTSpace() # Re-initialize for a fresh start
            
            for i, file in enumerate(normal_files):
                status_text.text(f"処理中: {file.name} ({i+1}/{len(normal_files)})")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_f:
                    tmp_f.write(file.getvalue())
                    tmp_normal_file_path = tmp_f.name
                
                try:
                    # Process file using the same pipeline as main analysis
                    fs_normal, data_normal_raw, _ = load_wav_file(tmp_normal_file_path)
                    
                    # Ensure consistent processing with training_analysis_config
                    processed_normal_data = remove_dc_offset(data_normal_raw)
                    processed_normal_data = apply_butterworth_filter(
                        processed_normal_data,
                        fs_normal,
                        training_analysis_config.highpass_hz,
                        training_analysis_config.lowpass_hz,
                        training_analysis_config.filter_order
                    )
                    
                    normal_time_features = calculate_time_domain_features(processed_normal_data)
                    
                    # FFT and Power contribution are part of VibrationFeatures
                    _, _, power_low, power_mid, power_high = \
                        calculate_fft_features(processed_normal_data, fs_normal, training_analysis_config.window)
                    
                    # Create VibrationFeatures instance
                    normal_features = VibrationFeatures(
                        rms=normal_time_features.rms,
                        peak=normal_time_features.peak,
                        kurtosis=normal_time_features.kurtosis,
                        skewness=normal_time_features.skewness,
                        crest_factor=normal_time_features.crest_factor,
                        shape_factor=normal_time_features.shape_factor,
                        power_low=power_low,
                        power_mid=power_mid,
                        power_high=power_high
                    )
                    st.session_state.mt_space.add_normal_sample(normal_features)
                    processed_count += 1
                except Exception as e:
                    st.error(f"ファイル {file.name} の処理中にエラーが発生しました: {e}")
                finally:
                    if os.path.exists(tmp_normal_file_path):
                        os.remove(tmp_normal_file_path)
                progress_bar.progress((i + 1) / len(normal_files))
            
            status_text.text(f"処理完了。{processed_count}個のファイルが単位空間に追加されました。")
            st.success("単位空間が構築/更新されました。")
        else:
            st.warning("単位空間を構築するには、正常時のWAVファイルをアップロードしてください。")

st.sidebar.info(f"単位空間ステータス: {st.session_state.mt_space.get_status()}")

# Main application content
# --------------------------------------------------------------------------------------------------

# 1. WAVファイルアップロード
uploaded_file = st.file_uploader("評価用WAVファイルをアップロードしてください", type=["wav"], key="evaluation_uploader")

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
        st.sidebar.header("評価用データ解析設定")
        selected_quantity = st.sidebar.selectbox(
            "物理量種別",
            options=list(SignalQuantity),
            format_func=lambda x: x.value,
            index=0,
            key="eval_quantity"
        )
        # Store selected_quantity in session_state for training config
        st.session_state.selected_quantity = selected_quantity
        
        selected_window = st.sidebar.selectbox(
            "窓関数",
            options=list(WindowFunction),
            format_func=lambda x: x.value,
            index=0,
            key="eval_window"
        )
        # Store selected_window in session_state for training config
        st.session_state.selected_window = selected_window


        # Placeholder for filter settings
        # Max value for cutoff frequencies should be strictly less than Nyquist frequency (fs_hz/2)
        nyquist_freq = fs_hz / 2
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            highpass_hz = st.number_input("HPF (Hz)", value=10, min_value=0, max_value=int(nyquist_freq * 0.99), step=1, key="eval_hpf")
        with col2:
            lowpass_hz = st.number_input("LPF (Hz)", value=int(nyquist_freq * 0.99), min_value=0, max_value=int(nyquist_freq * 0.99), step=1, key="eval_lpf")
        filter_order = st.sidebar.number_input("フィルタ次数", value=4, min_value=1, max_value=10, step=1, key="eval_order")

        analysis_config = AnalysisConfig(
            quantity=selected_quantity,
            window=selected_window,
            highpass_hz=float(highpass_hz) if highpass_hz > 0 else None,
            lowpass_hz=float(lowpass_hz) if lowpass_hz < nyquist_freq * 0.99 else None, # Check this condition again
            filter_order=filter_order
        )
        
        st.sidebar.write("---")
        st.sidebar.subheader("現在の評価用データフィルタ条件:")
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
            
            # Display Time Domain KPIs side-by-side
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            with kpi_col1:
                st.metric(label="RMS", value=f"{time_features.rms:.3f}")
                st.metric(label="Peak", value=f"{time_features.peak:.3f}")
            with kpi_col2:
                st.metric(label="Kurtosis", value=f"{time_features.kurtosis:.3f}")
                st.metric(label="Skewness", value=f"{time_features.skewness:.3f}")
            with kpi_col3:
                st.metric(label="Crest Factor", value=f"{time_features.crest_factor:.3f}")
                st.metric(label="Shape Factor", value=f"{time_features.shape_factor:.3f}")
        
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
            # Display Power Contributions side-by-side
            power_col1, power_col2, power_col3 = st.columns(3)
            with power_col1:
                st.metric(label="低周波パワー寄与率", value=f"{power_low:.2%}")
            with power_col2:
                st.metric(label="中周波パワー寄与率", value=f"{power_mid:.2%}")
            with power_col3:
                st.metric(label="高周波パワー寄与率", value=f"{power_high:.2%}")

            st.write("---") # Separator
            
            # Display Quality Metrics side-by-side
            quality_col1, quality_col2 = st.columns(2) # Two columns for Clipping and SNR
            with quality_col1:
                st.metric(label="クリッピング率", value=f"{quality.clipping_ratio:.2%}")
            with quality_col2:
                st.metric(label="S/N比", value=f"{quality.snr_db:.2f} dB")
            
            # Color-coded Confidence Score (Red, Yellow, Green)
            st.write("") # Add some vertical space
            st.markdown("### 診断信頼度") # Custom header for confidence score

            if confidence_score >= 80:
                color = "green"
                status_text = "良好"
            elif confidence_score >= 50:
                color = "orange" # Using orange for warning, as red is critical
                status_text = "注意"
            else:
                color = "red"
                status_text = "要確認"

            st.markdown(
                f"""
                <div style="
                    text-align: center; 
                    padding: 10px; 
                    border-radius: 5px; 
                    background-color: #333333; 
                    border: 2px solid {color};
                ">
                    <span style="font-size: 1.2em; color: white;">信頼度スコア</span><br>
                    <span style="font-size: 2.5em; font-weight: bold; color: {color};">{confidence_score:.1f}%</span><br>
                    <span style="font-size: 1.0em; color: {color};">{status_text}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # --- MT Method Anomaly Detection ---
            st.write("---")
            st.markdown("### MT法診断結果")
            if st.session_state.mt_space.mean_vector is not None:
                # Prepare features for evaluation
                evaluation_features = VibrationFeatures(
                    rms=time_features.rms,
                    peak=time_features.peak,
                    kurtosis=time_features.kurtosis,
                    skewness=time_features.skewness,
                    crest_factor=time_features.crest_factor,
                    shape_factor=time_features.shape_factor,
                    power_low=power_low,
                    power_mid=power_mid,
                    power_high=power_high
                )
                md = st.session_state.mt_space.calculate_md(evaluation_features)

                md_threshold_warning = 3.0 # Arbitrary threshold for warning
                md_threshold_anomaly = 5.0 # Arbitrary threshold for anomaly

                if md == np.inf:
                    md_color = "gray"
                    md_status_text = "単位空間未確立"
                elif md < md_threshold_warning:
                    md_color = "green"
                    md_status_text = "正常"
                elif md < md_threshold_anomaly:
                    md_color = "orange"
                    md_status_text = "要確認 (軽度異常)"
                else:
                    md_color = "red"
                    md_status_text = "異常 (重度異常)"
                
                st.markdown(
                    f"""
                    <div style="
                        text-align: center; 
                        padding: 10px; 
                        border-radius: 5px; 
                        background-color: #333333; 
                        border: 2px solid {md_color};
                    ">
                        <span style="font-size: 1.2em; color: white;">マハラノビス距離 (MD)</span><br>
                        <span style="font-size: 2.5em; font-weight: bold; color: {md_color};">{md:.2f}</span><br>
                        <span style="font-size: 1.0em; color: {md_color};">{md_status_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("MT法診断には単位空間の構築が必要です。")

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
    st.info("評価用WAVファイルをアップロードして解析を開始してください。")

# --- Audit Log Section ---
# This section needs to be outside the 'if uploaded_file is not None' block
# so that the AnalysisResult object is available to generate and download the log
# after processing.
# Moving this to outside of the uploaded_file block to ensure it's always accessible
# after analysis is complete.
if uploaded_file is not None and 'fs_hz' in locals(): # Ensure analysis has run
    # Construct AnalysisResult object
    # The datetime.now().isoformat() should be captured when analysis starts
    # For now, let's reconstruct it if analysis completes successfully.
    # This part needs to be carefully placed to ensure all variables are in scope.
    try:
        current_timestamp = datetime.now().isoformat()
        
        # Ensure all necessary variables are available from the try block above
        # This means moving the AnalysisResult construction inside the try block
        # and storing it in session_state or passing it around.
        
        # For simplicity for this issue, let's create a dummy AnalysisResult for display if not fully in scope.
        # But ideally, the AnalysisResult should be the actual result of the analysis.
        # Let's assume the analysis_config, time_features, quality, etc. are available.
        analysis_result = AnalysisResult(
            features=time_features, # This needs to be the full features object, which means updating the time_features to include power.
            quality=quality,
            config=analysis_config,
            timestamp=current_timestamp,
            file_hash=file_hash,
            fs_hz=fs_hz,
            app_version="1.0.0"
        )
        # Update features in analysis_result to include power_low, power_mid, power_high
        analysis_result.features.power_low = power_low
        analysis_result.features.power_mid = power_mid
        analysis_result.features.power_high = power_high

        audit_log_data = asdict(analysis_result)
        audit_log_json = json.dumps(audit_log_data, indent=2, ensure_ascii=False)

        st.write("---")
        with st.expander("監査ログ (JSON) を表示 / ダウンロード"):
            st.json(audit_log_data)
            
            st.download_button(
                label="監査ログをダウンロード",
                data=audit_log_json,
                file_name=f"audit_log_{file_hash[:8]}_{current_timestamp.replace(':', '-')}.json",
                mime="application/json"
            )
    except NameError:
        st.info("解析が完了すると監査ログが利用可能になります。")
    except Exception as e:
        st.error(f"監査ログの生成中にエラーが発生しました: {e}")

