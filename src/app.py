import streamlit as st
import numpy as np
from src.core.models import SignalQuantity, AnalysisConfig, WindowFunction, VibrationFeatures, QualityMetrics
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features, calculate_spectrogram
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
from src.utils import csv_parser
from src.utils import preset_manager
from src.core.models import AnalysisPreset

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

# --- Initialize Presets and Session State ---
if 'presets' not in st.session_state:
    st.session_state.presets = preset_manager.load_presets()

def apply_preset(preset_name):
    if preset_name in st.session_state.presets:
        p = st.session_state.presets[preset_name]
        st.session_state.eval_quantity = p.config.quantity
        st.session_state.eval_window = p.config.window
        st.session_state.eval_hpf_enabled = p.config.hpf_enabled
        st.session_state.eval_hpf = p.config.highpass_hz if p.config.highpass_hz else 10.0
        st.session_state.eval_lpf_enabled = p.config.lpf_enabled
        st.session_state.eval_lpf = p.config.lowpass_hz if p.config.lowpass_hz else 20000.0 # Default fallback
        st.session_state.eval_order = p.config.filter_order
        st.session_state.peak_count = p.top_n_peaks
        st.session_state.peak_height = int(p.min_peak_height_percent)
        st.session_state.peak_distance = int(p.peak_distance_hz)
        st.session_state.fft_log_x = p.fft_log_x
        st.session_state.show_raw_signal = p.show_raw_signal
        st.session_state.spec_nperseg = p.spec_nperseg

# --- MT法 単位空間の初期化 ---
if 'mt_space' not in st.session_state:
    st.session_state.mt_space = MTSpace(min_samples=10, recommended_samples=30)

# Page selection in sidebar
page_selection = st.sidebar.radio(
    "機能選択",
    ("通常解析", "ベンチマーク", "合成データ生成"),
    key="page_selector"
)

# --- Preset Management in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.header("⚙️ 解析プリセット")
preset_options = ["新規作成"] + list(st.session_state.presets.keys())
selected_preset = st.sidebar.selectbox("プリセットを選択", options=preset_options, key="selected_preset_name")

if selected_preset != "新規作成":
    if st.sidebar.button("プリセットを適用"):
        apply_preset(selected_preset)
        st.sidebar.success(f"プリセット '{selected_preset}' を適用しました。")

new_preset_name = st.sidebar.text_input("プリセット名", value="" if selected_preset == "新規作成" else selected_preset)
if st.sidebar.button("現在の設定を保存"):
    if new_preset_name:
        # Create config from current settings
        # We need to handle the case where some session state keys might not be initialized yet
        current_config = AnalysisConfig(
            quantity=st.session_state.get("eval_quantity", SignalQuantity.ACCEL),
            window=st.session_state.get("eval_window", WindowFunction.HANNING),
            highpass_hz=st.session_state.get("eval_hpf"),
            lowpass_hz=st.session_state.get("eval_lpf"),
            filter_order=st.session_state.get("eval_order", 4),
            hpf_enabled=st.session_state.get("eval_hpf_enabled", False),
            lpf_enabled=st.session_state.get("eval_lpf_enabled", False)
        )
        
        new_preset = AnalysisPreset(
            name=new_preset_name,
            config=current_config,
            top_n_peaks=st.session_state.get("peak_count", 5),
            min_peak_height_percent=st.session_state.get("peak_height", 10.0),
            peak_distance_hz=st.session_state.get("peak_distance", 10.0),
            fft_log_x=st.session_state.get("fft_log_x", False),
            show_raw_signal=st.session_state.get("show_raw_signal", True),
            spec_nperseg=st.session_state.get("spec_nperseg", 512)
        )
        
        st.session_state.presets[new_preset_name] = new_preset
        preset_manager.save_presets(st.session_state.presets)
        st.sidebar.success(f"プリセット '{new_preset_name}' を保存しました。")
        st.rerun()
    else:
        st.sidebar.error("名前を入力してください。")

if page_selection == "通常解析":
    # --- Main Application ---
    # Initialize uploader key if not present
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    col_up1, col_up2 = st.columns([4, 1])
    with col_up1:
        # Enhanced file uploader for multiple files with dynamic key for resetting
        uploaded_files = st.file_uploader(
            "評価用WAVまたはCSVファイルをアップロード (複数可)", 
            type=["wav", "csv"], 
            accept_multiple_files=True, 
            key=f"evaluation_uploader_{st.session_state.uploader_key}"
        )
    with col_up2:
        st.write("") # Spacer
        st.write("") # Spacer
        if st.button("🗑️ 全ファイルをクリア", help="アップロードされたすべてのファイルをリストから削除します。"):
            st.session_state.uploader_key += 1
            st.rerun()

    if uploaded_files:
        summary_results = []
        
        # We need a way to select which file to show details for
        file_names = [f.name for f in uploaded_files]
        selected_file_name = st.selectbox("詳細表示するファイルを選択", options=file_names)
        
        # Process all files for the summary table
        st.subheader("📋 診断サマリー")
        progress_bar = st.progress(0)
        
        # Cache the processing results to avoid re-calculating everything on detail switch
        # (For simplicity in this turn, we'll process the summary quickly)
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Basic analysis for summary
                if file_extension == "wav":
                    fs_tmp, data_tmp, _ = load_wav_file(tmp_path)
                else:
                    # For CSV summary, we use first detected column and default settings
                    # (In a real scenario, we might want to apply the current preset)
                    data_tmp, fs_tmp, _ = csv_parser.parse_csv_data(Path(tmp_path), data_columns=[pd.read_csv(tmp_path).columns[0]], sampling_frequency_hz=1000.0)
                
                # Apply current filter settings
                p_tmp = remove_dc_offset(data_tmp)
                p_tmp = apply_butterworth_filter(
                    p_tmp, fs_tmp, 
                    st.session_state.get("eval_hpf"), st.session_state.get("eval_lpf"), st.session_state.get("eval_order", 4),
                    hpf_enabled=st.session_state.get("eval_hpf_enabled", False),
                    lpf_enabled=st.session_state.get("eval_lpf_enabled", False)
                )
                
                t_feat = calculate_time_domain_features(p_tmp)
                f_hz, mags, f_feat = calculate_fft_features(p_tmp, fs_tmp, st.session_state.get("eval_window", WindowFunction.HANNING))
                all_f = VibrationFeatures(**asdict(t_feat), **f_feat)
                qual = calculate_quality_metrics(data_tmp, fs_tmp, t_feat.rms, mags)
                conf = get_confidence_score(qual)
                
                md = None
                if 'mt_space' in st.session_state and st.session_state.mt_space.mean_vector is not None:
                    md = st.session_state.mt_space.calculate_md(all_f)
                
                summary_results.append({
                    "ファイル名": uploaded_file.name,
                    "MD値 (異常度)": f"{md:.2f}" if md is not None else "-",
                    "判定": "🔴 異常" if md and md > 10 else "🟡 警告" if md and md > 3 else "🟢 正常" if md else "-",
                    "信頼度": f"{conf:.1f}%",
                    "RMS": f"{all_f.rms:.3f}",
                    "クリッピング": f"{qual.clipping_ratio:.2%}"
                })
            except:
                summary_results.append({"ファイル名": uploaded_file.name, "判定": "ERROR"})
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.dataframe(pd.DataFrame(summary_results), width='stretch')
        st.markdown("---")

        # Now show the details for the single selected file
        target_file = next(f for f in uploaded_files if f.name == selected_file_name)
        uploaded_file = target_file # Reuse existing detail logic below
        
        # --- (Existing detail logic starts here) ---
        file_extension = uploaded_file.name.split('.')[-1].lower()
        data_raw = None
        fs_hz = None
        file_hash = None
        tmp_file_path = None
        column_data_map = None # Initialize to None to prevent undefined error in detail view

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Display audio player for WAV files
        if file_extension == "wav":
            st.audio(uploaded_file, format="audio/wav")
        
        try:
            if file_extension == "wav":
                fs_hz, data_raw, file_hash = load_wav_file(tmp_file_path)
            elif file_extension == "csv":
                st.sidebar.header("CSV解析設定")
                csv_df_preview = pd.read_csv(tmp_file_path, skipinitialspace=True)
                
                # Automatically detect potential data and timestamp columns
                potential_numeric_cols = csv_df_preview.select_dtypes(include=np.number).columns.tolist()
                potential_datetime_cols = csv_df_preview.select_dtypes(include='datetime').columns.tolist()
                
                # Check for columns that look like timestamps but are objects
                for col in csv_df_preview.columns:
                    if csv_df_preview[col].dtype == 'object':
                        try:
                            pd.to_datetime(csv_df_preview[col], errors='raise')
                            potential_datetime_cols.append(col)
                        except:
                            pass
                
                # Intelligent column inference
                detected_cols = csv_parser.infer_vibration_columns(csv_df_preview.columns.tolist())
                
                if not detected_cols:
                    detected_cols = [potential_numeric_cols[0]] if potential_numeric_cols else [csv_df_preview.columns[0]]

                data_columns = st.sidebar.multiselect(
                    "加速度データ列を選択 (複数選択可)",
                    options=csv_df_preview.columns.tolist(),
                    default=detected_cols,
                    key="csv_data_columns"
                )
                
                synthesize = False
                main_axis_col = None
                if len(data_columns) > 1:
                    synthesize = st.sidebar.toggle("多軸合成を行う (合成ベクトル加速度)", value=True, key="csv_synthesize", 
                                                   help="物理的妥当性に基づき、各軸のDCオフセット（重力等）を除去した後に合成加速度 sqrt(X^2 + Y^2 + Z^2) を計算します。")
                    
                    if not synthesize:
                        main_axis_col = st.sidebar.selectbox(
                            "解析対象とするメインの軸を選択",
                            options=data_columns,
                            key="csv_main_axis_selection",
                            help="選択した軸のデータが特徴量抽出および異常診断（MT法）の対象となります。他の軸は参考としてグラフ表示されます。"
                        )

                use_timestamp = st.sidebar.checkbox("タイムスタンプ列を使用する", value=bool(potential_datetime_cols), key="csv_use_timestamp")

                timestamp_column = None
                input_sampling_frequency_hz = None

                if use_timestamp:
                    default_timestamp_col = potential_datetime_cols[0] if potential_datetime_cols else csv_df_preview.columns[0]
                    timestamp_column = st.sidebar.selectbox(
                        "タイムスタンプ列を選択",
                        options=csv_df_preview.columns.tolist(),
                        index=csv_df_preview.columns.tolist().index(default_timestamp_col),
                        key="csv_timestamp_column"
                    )
                else:
                    input_sampling_frequency_hz = st.sidebar.number_input(
                        "サンプリング周波数 (Hz) を入力",
                        min_value=1.0, value=1000.0, step=1.0,
                        key="csv_sampling_frequency"
                    )

                if data_columns:
                    st.sidebar.markdown("---")
                    st.sidebar.subheader("選択列の統計量")
                    st.sidebar.dataframe(csv_df_preview[data_columns].describe().transpose()[['mean', 'std', 'min', 'max']])

                st.sidebar.markdown("---")
                st.sidebar.subheader("CSVプレビュー (Top 5)")
                st.sidebar.dataframe(csv_df_preview.head())

                if not data_columns:
                    st.error("解析対象の列を少なくとも1つ選択してください。")
                    st.stop()

                data_raw, fs_hz, column_data_map = csv_parser.parse_csv_data(
                    Path(tmp_file_path),
                    data_columns=data_columns,
                    sampling_frequency_hz=input_sampling_frequency_hz,
                    timestamp_column=timestamp_column,
                    synthesize=synthesize
                )
                
                # If synthesize is off and user picked a main axis, override data_raw
                if not synthesize and main_axis_col and column_data_map and main_axis_col in column_data_map:
                    data_raw = column_data_map[main_axis_col]
                    col_info = main_axis_col
                else:
                    col_info = "+".join(data_columns) if synthesize else data_columns[0]
                
                file_hash = f"csv_hash_{uploaded_file.name}_{col_info}" # Simple hash for CSV

            st.write(f"ファイル名: {uploaded_file.name}, サンプリング周波数: {fs_hz} Hz, データ長: {len(data_raw) / fs_hz:.2f} 秒")
            
            # --- Analysis Configuration (same as before) ---
            st.sidebar.header("評価用データ解析設定")
            quantity = st.sidebar.selectbox("物理量種別", list(SignalQuantity), format_func=lambda x: x.value, key="eval_quantity")
            window = st.sidebar.selectbox("窓関数", list(WindowFunction), format_func=lambda x: x.value, key="eval_window")
            
            nyquist = fs_hz / 2
            hpf_enabled = st.sidebar.checkbox("HPF有効", value=False, key="eval_hpf_enabled")
            hpf = st.sidebar.number_input("HPF (Hz)", 0.0, nyquist, 10.0, key="eval_hpf", disabled=not hpf_enabled)
            
            lpf_enabled = st.sidebar.checkbox("LPF有効", value=False, key="eval_lpf_enabled")
            lpf = st.sidebar.number_input("LPF (Hz)", 0.0, nyquist, nyquist, key="eval_lpf", disabled=not lpf_enabled)
            
            order = st.sidebar.number_input("フィルタ次数", 1, 10, 4, key="eval_order")
            
            st.sidebar.header("FFTピーク設定")
            top_n_peaks = st.sidebar.slider("ピーク表示数", 1, 20, 5, key="peak_count")
            min_peak_height_percent = st.sidebar.slider("最小ピーク高さ（最大値に対する%）", 0, 100, 10, key="peak_height")
            peak_distance_hz = st.sidebar.slider("ピーク最小距離（Hz）", 1, 100, 10, key="peak_distance")

            st.sidebar.header("表示設定")
            fft_log_x = st.sidebar.checkbox("FFT周波数軸を対数にする", value=False, key="fft_log_x")
            show_raw_signal = st.sidebar.checkbox("生信号(DC除去のみ)を表示する", value=True, key="show_raw_signal")
            spec_nperseg = st.sidebar.select_slider("スペクトログラム解像度 (Window Size)", options=[128, 256, 512, 1024, 2048], value=512, key="spec_nperseg")

            config = AnalysisConfig(
                quantity=quantity, 
                window=window, 
                highpass_hz=float(hpf) if hpf_enabled else None, 
                lowpass_hz=float(lpf) if lpf_enabled else None, 
                filter_order=order,
                hpf_enabled=hpf_enabled,
                lpf_enabled=lpf_enabled
            )
            
            processed_dc_removed = remove_dc_offset(data_raw)
            processed = apply_butterworth_filter(
                processed_dc_removed, 
                fs_hz, 
                config.highpass_hz, 
                config.lowpass_hz, 
                config.filter_order,
                hpf_enabled=config.hpf_enabled,
                lpf_enabled=config.lpf_enabled
            )
            
            time_features = calculate_time_domain_features(processed)
            freqs, mags, freq_features = calculate_fft_features(processed, fs_hz, config.window)
            quality = calculate_quality_metrics(data_raw, fs_hz, time_features.rms, mags)
            confidence = get_confidence_score(quality)
            unit = config.quantity.unit_str

            all_features = VibrationFeatures(**asdict(time_features), **freq_features)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("時間領域 特徴量")
                st.metric(f"RMS ({unit})", f"{all_features.rms:.3f}", help="振動エネルギーの総量。機械の全体的な振動の強さを表します。")
                st.metric(f"Peak ({unit})", f"{all_features.peak:.3f}", help="波形の最大値。瞬間的な衝撃（ガタや突発的な異常）を捉えます。")
                st.metric("尖度 (Kurtosis)", f"{all_features.kurtosis:.3f}", help="波形の鋭さ。軸受の傷などによる衝撃波が発生すると値が大きくなります。正常値は約3.0です。")

                st.subheader("周波数領域 特徴量")
                st.metric("Spectral Centroid (Hz)", f"{all_features.spectral_centroid:.2f}", help="スペクトルの「重心」。振動の主成分がどの周波数帯にあるかを示します。")
                st.metric("Spectral Spread (Hz)", f"{all_features.spectral_spread:.2f}", help="スペクトルの「広がり」。振動が特定の周波数に集中しているか、広帯域に分散しているかを示します。")
                st.metric("Spectral Entropy", f"{all_features.spectral_entropy:.3f}", help="スペクトルの「複雑さ（乱雑さ）」。不規則なノイズが多いほど値が大きくなります。")
                
                st.subheader("データ品質", help="解析に使用したデータの健全性を評価します。各指標が悪いと『診断信頼度』が低下します。")
                st.metric("クリッピング率", f"{quality.clipping_ratio:.2%}", help="センサの測定範囲を超えて波形が飽和（潰れた）割合。1%を超えると信頼度が大幅に低下します。")
                st.metric("S/N 比", f"{quality.snr_db:.2f} dB", help="背景ノイズに対する信号の強さ。20dB以上が理想的です。値が小さい場合はノイズ除去プラグインの活用を検討してください。")
                
                color = "green" if confidence >= 80 else "orange"
                st.markdown(f'#### 診断信頼度: <span style="color:{color};" title="データ品質（クリッピング、S/N比、データ長）に基づいた、この解析結果の確からしさを示します。">{confidence:.1f}%</span>', unsafe_allow_html=True)
                
                with st.expander("🤔 診断信頼度とは？"):
                    st.write("""
                    この数値は、以下の3つの観点から「今回の解析結果をどの程度信じてよいか」を判定しています。
                    
                    1. **波形の欠落（クリッピング）**: 振動が大きすぎてセンサの限界を超えていないか。
                    2. **ノイズの影響（S/N比）**: 周囲の雑音に埋もれず、対象の振動をクリアに捉えられているか。
                    3. **録音時間（データ長）**: 物理現象を正しく捉えるのに十分な長さ（10秒以上推奨）があるか。
                    
                    **【解釈の目安】**
                    - **80%以上**: 信頼できる結果です。
                    - **50〜80%**: 参考値です。ノイズ除去の適用や、センサのレンジ見直しを推奨します。
                    - **50%未満**: データの品質に問題があります。設置状況や測定条件を再確認してください。
                    """)
                
                if 'mt_space' in st.session_state and st.session_state.mt_space.mean_vector is not None:
                    md = st.session_state.mt_space.calculate_md(all_features)
                    md_color = "green" if md < 3.0 else "orange" if md < 5.0 else "red"
                    st.markdown(f'#### MT法診断 (MD): <span style="color:{md_color};" title="単位空間（正常状態）からの距離。3.0を超えると異常の兆候、10.0を超えると明らかな異常とみなされます。">{md:.2f}</span>', unsafe_allow_html=True)
            
            with col2:
                tab1, tab2, tab3 = st.tabs(["🕒 時間領域 (波形比較)", "📊 周波数領域 (FFT)", "🌈 時間-周波数領域 (スペクトログラム)"])
                
                # Dynamic downsampling for performance
                MAX_POINTS = 10000
                
                with tab1:
                    st.subheader("時間領域波形")
                    time_axis = np.arange(len(processed)) / fs_hz
                    
                    # Performance optimization: Downsample for visualization if data is too large
                    if len(processed) > MAX_POINTS:
                        step = len(processed) // MAX_POINTS
                        plot_time = time_axis[::step]
                        plot_processed = processed[::step]
                        plot_raw_dc = processed_dc_removed[::step]
                        st.warning(f"データ長が大きいため、描画用に {len(processed)} 点から {len(plot_processed)} 点へダウンサンプリングしました。")
                    else:
                        plot_time = time_axis
                        plot_processed = processed
                        plot_raw_dc = processed_dc_removed

                    fig_time = go.Figure()
                    
                    if show_raw_signal:
                        if column_data_map:
                            for col_name, col_arr in column_data_map.items():
                                # Remove DC for visualization if it's raw data
                                p_col_arr = col_arr[::step] if len(col_arr) > MAX_POINTS else col_arr
                                fig_time.add_trace(go.Scatter(x=plot_time, y=p_col_arr - np.mean(col_arr), mode='lines', name=f'軸: {col_name} (DC除去)', opacity=0.4))
                        else:
                            fig_time.add_trace(go.Scatter(x=plot_time, y=plot_raw_dc, mode='lines', name='生信号 (DC除去のみ)', line=dict(color='rgba(150, 150, 150, 0.4)')))
                    
                    fig_time.add_trace(go.Scatter(x=plot_time, y=plot_processed, mode='lines', name='解析対象信号 (フィルタ後)', line=dict(color='blue', width=2)))
                    
                    fig_time.update_layout(
                        xaxis_title="時間 (s)",
                        yaxis_title=f"振幅 ({unit})",
                        height=500,
                        margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_time, width='stretch')

                with tab2:
                    st.subheader("FFT スペクトル")
                    
                    # FFT usually has fewer points than time domain, but we can still limit it
                    if len(mags) > MAX_POINTS:
                        step_f = len(mags) // MAX_POINTS
                        plot_freqs = freqs[::step_f]
                        plot_mags = mags[::step_f]
                    else:
                        plot_freqs = freqs
                        plot_mags = mags

                    fig_fft = go.Figure()
                    
                    # Add component FFTs if available
                    if column_data_map:
                        for col_name, col_arr in column_data_map.items():
                            # Apply the same filter as the main signal for physical consistency
                            c_processed = remove_dc_offset(col_arr)
                            c_processed = apply_butterworth_filter(
                                c_processed, fs_hz, config.highpass_hz, config.lowpass_hz, config.filter_order,
                                hpf_enabled=config.hpf_enabled, lpf_enabled=config.lpf_enabled
                            )
                            f_comp, m_comp, _ = calculate_fft_features(c_processed, fs_hz, config.window)
                            if len(m_comp) > MAX_POINTS:
                                f_comp = f_comp[::len(m_comp)//MAX_POINTS]
                                m_comp = m_comp[::len(m_comp)//MAX_POINTS]
                            fig_fft.add_trace(go.Scatter(x=f_comp, y=m_comp, mode='lines', name=f'軸: {col_name}', opacity=0.4))

                    fig_fft.add_trace(go.Scatter(x=plot_freqs, y=plot_mags, mode='lines', name='解析対象信号', line=dict(color='blue', width=2)))
                    
                    # Add vertical lines for Filter Cutoffs
                    if config.hpf_enabled and config.highpass_hz:
                        fig_fft.add_vline(x=config.highpass_hz, line_dash="dash", line_color="red", annotation_text=f"HPF {config.highpass_hz}Hz")
                    if config.lpf_enabled and config.lowpass_hz:
                        fig_fft.add_vline(x=config.lowpass_hz, line_dash="dash", line_color="red", annotation_text=f"LPF {config.lowpass_hz}Hz")

                    # Peak Annotation (only for the main processed signal)
                    peaks, _ = find_peaks(mags, height=max(mags) * min_peak_height_percent / 100, distance=peak_distance_hz * (len(freqs) / (fs_hz/2)))
                    sorted_peaks = peaks[np.argsort(mags[peaks])][::-1][:top_n_peaks]
                    
                    for p in sorted_peaks:
                        fig_fft.add_annotation(
                            x=freqs[p], y=mags[p],
                            text=f"{freqs[p]:.1f}Hz<br>{mags[p]:.3f}{unit}",
                            showarrow=True, arrowhead=1,
                            ax=0, ay=-40
                        )
                    
                    fig_fft.update_layout(
                        xaxis_title="周波数 (Hz)",
                        yaxis_title=f"振幅 ({unit})",
                        xaxis_type="log" if fft_log_x else "linear",
                        height=500,
                        margin=dict(l=20, r=20, t=20, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_fft, width='stretch')

                with tab3:
                    st.subheader("スペクトログラム (解析対象信号)")
                    spec_f, spec_t, Sxx = calculate_spectrogram(processed, fs_hz, config.window, nperseg=spec_nperseg)
                    
                    # Convert to dB for visualization
                    Sxx_db = 10 * np.log10(Sxx + 1e-12)
                    
                    filter_status = f"HPF:{config.highpass_hz}Hz " if config.hpf_enabled else ""
                    filter_status += f"LPF:{config.lowpass_hz}Hz" if config.lpf_enabled else ""
                    if not filter_status: filter_status = "None"

                    fig_spec = go.Figure(data=go.Heatmap(
                        x=spec_t, y=spec_f, z=Sxx_db,
                        colorscale='Viridis',
                        colorbar=dict(title=f"Power (dB rel. {unit}^2/Hz)")
                    ))
                    
                    fig_spec.update_layout(
                        title=f"Main Spectrogram (Window: {config.window.value}, Size: {spec_nperseg}, Filter: {filter_status})",
                        xaxis_title="時間 (s)",
                        yaxis_title="周波数 (Hz)",
                        yaxis_type="log" if fft_log_x else "linear",
                        height=500,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_spec, width='stretch')

                    # Add component spectrograms if available
                    if column_data_map:
                        st.markdown("---")
                        st.subheader("各軸の個別スペクトログラム")
                        for col_name, col_arr in column_data_map.items():
                            # Remove DC for spectrogram calculation
                            f_c, t_c, Sxx_c = calculate_spectrogram(col_arr - np.mean(col_arr), fs_hz, config.window, nperseg=spec_nperseg)
                            Sxx_c_db = 10 * np.log10(Sxx_c + 1e-12)
                            
                            fig_c = go.Figure(data=go.Heatmap(
                                x=t_c, y=f_c, z=Sxx_c_db,
                                colorscale='Viridis',
                                showscale=False # Hide colorbar to save space in small plots
                            ))
                            fig_c.update_layout(
                                title=f"軸: {col_name} (DC除去)",
                                xaxis_title="時間 (s)",
                                yaxis_title="周波数 (Hz)",
                                yaxis_type="log" if fft_log_x else "linear",
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_c, width='stretch')

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
        col_ana1, col_ana2 = st.columns(2)
        with col_ana1:
            benchmark_quantity = st.selectbox("物理量種別", list(SignalQuantity), format_func=lambda x: x.value, key="benchmark_quantity")
            benchmark_window = st.selectbox("窓関数", list(WindowFunction), format_func=lambda x: x.value, key="benchmark_window")
        with col_ana2:
            # Assuming a fixed fs for benchmark, or derived from dataset
            # For simplicity, using dummy fs to prevent errors for nyquist calculation
            dummy_fs = 48000 
            benchmark_hpf_enabled = st.checkbox("HPF有効", value=False, key="benchmark_hpf_enabled")
            benchmark_hpf = st.number_input("HPF (Hz)", 0.0, float(dummy_fs/2), 10.0, key="benchmark_hpf", disabled=not benchmark_hpf_enabled)
            
            benchmark_lpf_enabled = st.checkbox("LPF有効", value=False, key="benchmark_lpf_enabled")
            benchmark_lpf = st.number_input("LPF (Hz)", 0.0, float(dummy_fs/2), float(dummy_fs/2), key="benchmark_lpf", disabled=not benchmark_lpf_enabled)
            
            benchmark_order = st.number_input("フィルタ次数", 1, 10, 4, key="benchmark_order")
        
        benchmark_analysis_config = AnalysisConfig(
            quantity=benchmark_quantity, 
            window=benchmark_window,
            highpass_hz=float(benchmark_hpf) if benchmark_hpf_enabled else None, 
            lowpass_hz=float(benchmark_lpf) if benchmark_lpf_enabled else None, 
            filter_order=benchmark_order,
            hpf_enabled=benchmark_hpf_enabled,
            lpf_enabled=benchmark_lpf_enabled
        )

    with st.expander("MT法設定 (ベンチマーク用)", expanded=True):
        benchmark_anomaly_threshold = st.number_input("異常判定閾値 (MD)", 0.1, 10.0, 3.0, key="benchmark_anomaly_threshold")
        benchmark_min_samples = st.number_input("最小正常サンプル数 (単位空間構築)", 1, 50, 10, key="benchmark_min_samples")
        benchmark_recommended_samples = st.number_input("推奨正常サンプル数", 10, 100, 30, key="benchmark_recommended_samples")
        optimize_threshold_checkbox = st.checkbox("異常判定閾値を自動で最適化する (F1スコア基準)", key="benchmark_optimize_threshold", help="有効にすると、テストデータを用いてF1スコアが最大になるように異常判定閾値を自動で探索します。")
        
        benchmark_mt_config = MTConfig(
            anomaly_threshold=benchmark_anomaly_threshold,
            min_samples=benchmark_min_samples,
            recommended_samples=benchmark_recommended_samples
        )

    # NR Plugin Configuration (reusing existing UI elements)
    with st.expander("ノイズ除去プラグイン設定 (ベンチマーク用)", expanded=True):
        # Get available plugins and add a "None" option
        available_plugins_benchmark = plugin_manager.list_plugins()
        plugin_options_benchmark = {plugin.get_display_name(): plugin for plugin in available_plugins_benchmark}
        plugin_options_benchmark["None"] = None
        
        selected_plugin_name_benchmark = st.selectbox(
            "ノイズ除去アルゴリズム",
            options=list(plugin_options_benchmark.keys()),
            key="nr_plugin_selector_benchmark"
        )
        
        selected_plugin_benchmark = plugin_options_benchmark[selected_plugin_name_benchmark]
        benchmark_plugin_params = {}
        benchmark_plugin_conf_name = None
        benchmark_nr_plugin_config = None

        if selected_plugin_benchmark:
            benchmark_plugin_conf_name = selected_plugin_benchmark.get_name()
            st.markdown(f"**{selected_plugin_benchmark.get_display_name()} 設定**")
            for param in selected_plugin_benchmark.get_parameters():
                if param.param_type == "number_input":
                    benchmark_plugin_params[param.name] = st.number_input(
                        label=param.label,
                        min_value=param.min_value,
                        value=param.default,
                        help=param.help_text,
                        key=f"nr_benchmark_{benchmark_plugin_conf_name}_{param.name}"
                    )
                elif param.param_type == "slider":
                    benchmark_plugin_params[param.name] = st.slider(
                        label=param.label,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        value=param.default,
                        help=param.help_text,
                        key=f"nr_benchmark_{benchmark_plugin_conf_name}_{param.name}"
                    )
            benchmark_nr_plugin_config = {
                "name": benchmark_plugin_conf_name,
                "params": benchmark_plugin_params
            }
        
    benchmark_config = BenchmarkConfig(
        dataset_name=selected_benchmark_dataset,
        analysis_config=benchmark_analysis_config,
        mt_config=benchmark_mt_config,
        optimize_threshold=optimize_threshold_checkbox,
        nr_plugin_config=benchmark_nr_plugin_config
    )

    if st.button("ベンチマークを実行", key="run_benchmark_button"):
        st.info("ベンチマークを実行中...しばらくお待ちください。")
        try:
            with st.spinner("ベンチマーク処理中..."):
                benchmark_result = run_benchmark_test(benchmark_config, dataset_root_path)
            st.success("ベンチマークが完了しました！")

            st.subheader("ベンチマーク結果概要")
            
            if benchmark_result.optimized_threshold is not None:
                st.metric("最適化された異常判定閾値", f"{benchmark_result.optimized_threshold:.4f}")
            else:
                st.metric("使用された異常判定閾値", f"{benchmark_result.benchmark_config.mt_config.anomaly_threshold:.4f}")


            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            col_res1.metric("精度 (Accuracy)", f"{benchmark_result.accuracy:.2%}")
            col_res2.metric("適合率 (Precision)", f"{benchmark_result.precision:.2%}")
            col_res3.metric("再現率 (Recall)", f"{benchmark_result.recall:.2%}")
            col_res4.metric("F1スコア", f"{benchmark_result.f1_score:.2f}")

            st.markdown("##### 分類レポート")
            st.code(benchmark_result.classification_report)
            
            # Confusion Matrix
            if benchmark_result.confusion_matrix:
                st.markdown("##### 混同行列")
                cm_labels = ['normal', 'anomaly']
                cm_fig = go.Figure(data=go.Heatmap(
                       z=benchmark_result.confusion_matrix,
                       x=cm_labels,
                       y=cm_labels,
                       colorscale='Blues',
                       colorbar={"title": "Count"}))
                cm_fig.update_layout(
                    xaxis_title="予測ラベル",
                    yaxis_title="実測ラベル",
                    height=400, margin=dict(l=20,r=20,t=40,b=20),
                    yaxis=dict(autorange="reversed") # To show 'anomaly' at top
                )
                # Add text annotations
                for i in range(len(cm_labels)):
                    for j in range(len(cm_labels)):
                        cm_fig.add_annotation(
                            x=cm_labels[j],
                            y=cm_labels[i],
                            text=str(benchmark_result.confusion_matrix[i][j]),
                            showarrow=False,
                            font=dict(color="black")
                        )
                st.plotly_chart(cm_fig, width='stretch')

            # ROC Curve
            if benchmark_result.roc_curve and benchmark_result.roc_auc is not None:
                st.markdown("##### ROC曲線")
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=benchmark_result.roc_curve['fpr'], y=benchmark_result.roc_curve['tpr'], mode='lines', name=f'ROC (AUC = {benchmark_result.roc_auc:.2f})'))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
                roc_fig.update_layout(xaxis_title="偽陽性率 (FPR)", yaxis_title="真陽性率 (TPR)", height=400, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(roc_fig, width='stretch')

            # Mahalanobis Distance Scatter Plot
            if benchmark_result.file_results:
                st.markdown("##### マハラノビス距離分布")
                md_df = pd.DataFrame([
                    {'md': f.mahalanobis_distance, 'label': f.actual_label, 'predicted': f.predicted_label}
                    for f in benchmark_result.file_results
                ])
                md_fig = go.Figure()
                # Use a specific y-coordinate for each label for better visualization
                md_df['y_coord'] = md_df['label'].apply(lambda x: 0 if x == 'normal' else 1)

                md_fig.add_trace(go.Scatter(
                    x=md_df[md_df['label'] == 'normal']['md'],
                    y=md_df[md_df['label'] == 'normal']['y_coord'],
                    mode='markers',
                    name='正常データ',
                    marker=dict(color='blue', size=8, opacity=0.7)
                ))
                md_fig.add_trace(go.Scatter(
                    x=md_df[md_df['label'] == 'anomaly']['md'],
                    y=md_df[md_df['label'] == 'anomaly']['y_coord'],
                    mode='markers',
                    name='異常データ',
                    marker=dict(color='red', size=8, opacity=0.7)
                ))
                
                # Add threshold line if optimized or set
                threshold_to_plot = benchmark_result.optimized_threshold if benchmark_result.optimized_threshold is not None else benchmark_result.benchmark_config.mt_config.anomaly_threshold
                md_fig.add_shape(type="line", x0=threshold_to_plot, y0=-0.5, x1=threshold_to_plot, y1=1.5,
                                 line=dict(color="green", width=2, dash="dash"),
                                 name="判定閾値")
                md_fig.update_layout(
                    xaxis_title="マハラノビス距離 (MD)",
                    yaxis_title="ラベル (0: 正常, 1: 異常)",
                    yaxis=dict(tickvals=[0, 1], ticktext=['正常', '異常'], range=[-0.5, 1.5], showgrid=False),
                    height=400, margin=dict(l=20,r=20,t=40,b=20),
                    showlegend=True
                )
                st.plotly_chart(md_fig, width='stretch')


            st.markdown("---")
            st.subheader("ベンチマーク構成 (監査用)")
            st.json(asdict(benchmark_result.benchmark_config))

        except Exception as e:
            st.error(f"ベンチマーク実行中にエラーが発生しました: {e}")

elif page_selection == "合成データ生成":
    # ... (code for this page)
    pass

