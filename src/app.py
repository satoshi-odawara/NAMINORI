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
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
from src.utils import mt_manager
from src.core.models import AnalysisPreset

st.set_page_config(layout="wide", page_title="振動解析Webアプリ")

st.title("振動解析Webアプリケーション")

# --- Initialize Presets and Session State ---
if 'presets' not in st.session_state:
    st.session_state.presets = preset_manager.load_presets()

# --- MT法 単位空間の初期化 ---
if 'mt_space' not in st.session_state:
    st.session_state.mt_space = MTSpace(min_samples=10, recommended_samples=30)

def load_mt_space_into_session(space_name):
    """単位空間をファイルから読み込み、セッション状態を更新する。バージョン不整合時は拒否する。"""
    if space_name == "未選択":
        st.session_state.mt_space.mean_vector = None
        st.session_state.mt_space.inverse_covariance_matrix = None
        st.session_state.mt_space.is_provisional = True
        return False
    
    space_data = mt_manager.load_unit_space(space_name)
    if space_data:
        # Version Check
        v = space_data.get("version", "1.0")
        if v != mt_manager.DATA_VERSION:
            st.sidebar.error(f"❌ 基準データ '{space_name}' は非対応バージョン (v{v}) です。最新の v{mt_manager.DATA_VERSION} で再構築してください。")
            return False
            
        st.session_state.mt_space.mean_vector = space_data["mean_vector"]
        st.session_state.mt_space.inverse_covariance_matrix = space_data["inverse_covariance_matrix"]
        st.session_state.mt_space.average_magnitude_spectrum = space_data.get("average_magnitude_spectrum") 
        st.session_state.mt_space.is_provisional = False
        
        # Initialize normal samples placeholder
        current_dim = len(VibrationFeatures.get_feature_names())
        st.session_state.mt_space.normal_samples_vectors = [np.zeros(current_dim)] * space_data["sample_count"]
        return True
    return False

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
        
        # Automatically load associated unit space
        if p.unit_space_name:
            if load_mt_space_into_session(p.unit_space_name):
                st.session_state.load_mt_space_name = p.unit_space_name
        else:
            st.session_state.load_mt_space_name = "未選択"
            load_mt_space_into_session("未選択")

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
        # Physical validity: Capture all current settings including UI state and plugin config
        new_config = AnalysisConfig(
            quantity=st.session_state.get("eval_quantity", SignalQuantity.ACCEL),
            window=st.session_state.get("eval_window", WindowFunction.HANNING),
            highpass_hz=st.session_state.get("eval_hpf", 10.0),
            lowpass_hz=st.session_state.get("eval_lpf", 20000.0),
            filter_order=st.session_state.get("eval_order", 4),
            hpf_enabled=st.session_state.get("eval_hpf_enabled", False),
            lpf_enabled=st.session_state.get("eval_lpf_enabled", False),
            noise_reduction_plugin_name=st.session_state.get("nr_plugin_selection") if st.session_state.get("nr_evaluation_enabled") else None,
            noise_reduction_plugin_params=st.session_state.get("nr_plugin_params", {}) if st.session_state.get("nr_evaluation_enabled") else None
        )
        
        # Unit space association
        selected_space = st.session_state.get("load_mt_space_name")
        unit_space_name = selected_space if selected_space != "未選択" else None

        new_preset = AnalysisPreset(
            name=new_preset_name,
            config=new_config,
            top_n_peaks=st.session_state.get("peak_count", 5),
            min_peak_height_percent=float(st.session_state.get("peak_height", 10.0)),
            peak_distance_hz=float(st.session_state.get("peak_distance", 10.0)),
            fft_log_x=st.session_state.get("fft_log_x", False),
            show_raw_signal=st.session_state.get("show_raw_signal", True),
            spec_nperseg=st.session_state.get("spec_nperseg", 512),
            unit_space_name=unit_space_name
        )
        
        st.session_state.presets[new_preset_name] = new_preset
        preset_manager.save_presets(st.session_state.presets)
        st.sidebar.success(f"プリセット '{new_preset_name}' を保存しました。")
        # Removed st.rerun() to keep current analysis results visible
    else:
        st.sidebar.error("プリセット名を入力してください。")

# --- MT Method Management (Load only) moved here for visibility ---
st.sidebar.markdown("---")
with st.sidebar.expander("🛠️ MT法 (正常基準) 管理", expanded=True):
    st.markdown("##### 保存済み単位空間の呼出")
    saved_spaces = mt_manager.list_saved_unit_spaces()
    
    # Callback to handle automatic loading
    def on_mt_space_selection_change():
        selected = st.session_state.load_mt_space_name
        if load_mt_space_into_session(selected):
            # We can't use st.sidebar.success directly in callback easily without it disappearing, 
            # but the UI will update naturally.
            pass

    selected_space_name = st.selectbox(
        "単位空間を選択", 
        options=["未選択"] + saved_spaces, 
        key="load_mt_space_name",
        on_change=on_mt_space_selection_change,
        help="保存済みの基準値を読み込みます。選択すると即座に解析に反映されます。"
    )
    
    if selected_space_name != "未選択":
        st.sidebar.caption(f"✅ '{selected_space_name}' ロード済み")

if page_selection == "通常解析":
    # --- Sidebar Analysis Settings (Physicality & Traceability) ---
    st.sidebar.header("🔬 解析設定")
    
    with st.sidebar.expander("基本設定", expanded=True):
        # Physical validity: Ensure session values are Enum types and find their index robustly
        q_val = st.session_state.get("eval_quantity", SignalQuantity.ACCEL)
        # Use a list of values to find the index, which is robust against object identity mismatches
        q_options = list(SignalQuantity)
        q_values = [m.value for m in q_options]
        # Get the value to search for (handle both Enum objects and strings)
        target_q_val = q_val.value if hasattr(q_val, 'value') else q_val
        q_index = q_values.index(target_q_val) if target_q_val in q_values else 0
        
        st.session_state.eval_quantity = st.selectbox(
            "物理量種別", 
            q_options, 
            format_func=lambda x: x.value,
            key="eval_quantity_selector",
            index=q_index
        )
        st.session_state.eval_quantity = st.session_state.eval_quantity_selector

        w_val = st.session_state.get("eval_window", WindowFunction.HANNING)
        w_options = list(WindowFunction)
        w_values = [m.value for m in w_options]
        target_w_val = w_val.value if hasattr(w_val, 'value') else w_val
        w_index = w_values.index(target_w_val) if target_w_val in w_values else 0

        st.session_state.eval_window = st.selectbox(
            "窓関数", 
            w_options, 
            format_func=lambda x: x.value,
            key="eval_window_selector",
            index=w_index
        )
        st.session_state.eval_window = st.session_state.eval_window_selector

    with st.sidebar.expander("フィルタ設定 (DC除去は自動適用)", expanded=True):
        st.session_state.eval_hpf_enabled = st.checkbox("HPF (高域通過) 有効", value=st.session_state.get("eval_hpf_enabled", False))
        st.session_state.eval_hpf = st.number_input(
            "HPF カットオフ (Hz)", 
            min_value=0.1, max_value=20000.0, 
            value=st.session_state.get("eval_hpf", 10.0),
            disabled=not st.session_state.eval_hpf_enabled
        )
        
        st.session_state.eval_lpf_enabled = st.checkbox("LPF (低域通過) 有効", value=st.session_state.get("eval_lpf_enabled", False))
        st.session_state.eval_lpf = st.number_input(
            "LPF カットオフ (Hz)", 
            min_value=1.0, max_value=100000.0, 
            value=st.session_state.get("eval_lpf", 20000.0),
            disabled=not st.session_state.eval_lpf_enabled
        )
        
        st.session_state.eval_order = st.slider("フィルタ次数", 1, 10, st.session_state.get("eval_order", 4))

    # --- Noise Reduction (Plugin System) ---
    with st.sidebar.expander("🧹 ノイズ除去 (評価用)", expanded=False):
        st.session_state.nr_evaluation_enabled = st.checkbox("ノイズ除去評価を有効化", value=st.session_state.get("nr_evaluation_enabled", False))
        
        if st.session_state.nr_evaluation_enabled:
            available_plugins = plugin_manager.list_plugins()
            plugin_names = [p.get_display_name() for p in available_plugins]
            
            selected_plugin_name = st.selectbox(
                "アルゴリズム選択", 
                options=plugin_names,
                index=0 if not st.session_state.get("nr_plugin_selection") else plugin_names.index(st.session_state.nr_plugin_selection) if st.session_state.nr_plugin_selection in plugin_names else 0
            )
            st.session_state.nr_plugin_selection = selected_plugin_name
            
            selected_plugin = next(p for p in available_plugins if p.get_display_name() == selected_plugin_name)
            
            # Dynamic Parameter UI from Plugin
            plugin_params = {}
            for param in selected_plugin.get_parameters():
                if param.param_type == "number_input":
                    plugin_params[param.name] = st.number_input(
                        label=param.label,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        value=param.default if param.name not in st.session_state.get("nr_plugin_params", {}) else st.session_state.nr_plugin_params[param.name],
                        help=param.help_text
                    )
                elif param.param_type == "slider":
                    plugin_params[param.name] = st.slider(
                        label=param.label,
                        min_value=param.min_value,
                        max_value=param.max_value,
                        value=param.default if param.name not in st.session_state.get("nr_plugin_params", {}) else st.session_state.nr_plugin_params[param.name],
                        help=param.help_text
                    )
            st.session_state.nr_plugin_params = plugin_params

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
        all_mags = []
        all_freqs = []
        all_f_vectors = []
        all_filenames = []
        all_mds = []
        
        # Physical validity: Add a reference row from the established unit space if available
        if 'mt_space' in st.session_state and st.session_state.mt_space.mean_vector is not None:
            mv = st.session_state.mt_space.mean_vector
            ref_name = f"🔵 [基準] {st.session_state.get('load_mt_space_name', '単位空間')}"
            # The indices must match VibrationFeatures.to_vector() exactly
            summary_results.append({
                "ファイル名": ref_name,
                "MD値": "1.00",
                "判定": "🟢 基準",
                "信頼度": "100.0%",
                "RMS": f"{mv[0]:.3f}",
                "Overall": f"{mv[12]:.3f}",
                "OA(LF)": f"{mv[13]:.3f}",
                "OA(HF)": f"{mv[14]:.3f}",
                "Peak": f"{mv[1]:.3f}",
                "尖度": f"{mv[2]:.2f}",
                "歪度": f"{mv[3]:.2f}",
                "CrestFactor": f"{mv[4]:.2f}",
                "ShapeFactor": f"{mv[5]:.2f}",
                "重心周波数": f"{mv[9]:.1f}",
                "周波数分散": f"{mv[10]:.1f}",
                "スペクトルエントロピー": f"{mv[11]:.3f}",
                "S/N比(dB)": "-",
                "クリッピング": "-",
                "データ長(s)": "-",
                "fs(Hz)": "-"
            })
            all_f_vectors.append(mv)
            all_filenames.append(ref_name)
            all_mds.append(1.0)
            # Ensure index alignment: even if spectrum is missing, add None
            all_mags.append(st.session_state.mt_space.average_magnitude_spectrum)
            all_freqs.append(None) 

        # Process all files for the summary table
        st.subheader("📋 診断サマリー")
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Basic analysis logic... (same as before)
                if file_extension == "wav":
                    fs_tmp, data_tmp, _ = load_wav_file(tmp_path)
                else:
                    df_tmp = pd.read_csv(tmp_path, nrows=5, skipinitialspace=True)
                    df_tmp.columns = [c.strip() for c in df_tmp.columns]
                    selected_cols = st.session_state.get("csv_data_columns", [])
                    active_cols = [c for c in selected_cols if c in df_tmp.columns]
                    if not active_cols:
                        active_cols = csv_parser.infer_vibration_columns(df_tmp.columns.tolist())
                    if not active_cols:
                        potential = df_tmp.select_dtypes(include=np.number).columns.tolist()
                        active_cols = [potential[0]] if potential else [df_tmp.columns[0]]
                    
                    s_fs = st.session_state.get("csv_sampling_frequency", 1000.0)
                    s_ts = st.session_state.get("csv_timestamp_column") if st.session_state.get("csv_use_timestamp") else None
                    s_syn = st.session_state.get("csv_synthesize", False)
                    s_main_axis = st.session_state.get("csv_main_axis_selection")

                    data_tmp, fs_tmp, col_map_tmp = csv_parser.parse_csv_data(
                        Path(tmp_path), 
                        data_columns=active_cols, 
                        sampling_frequency_hz=s_fs,
                        timestamp_column=s_ts if s_ts in df_tmp.columns else None,
                        synthesize=s_syn if len(active_cols) > 1 else False
                    )
                    
                    if not s_syn and s_main_axis and col_map_tmp and s_main_axis in col_map_tmp:
                        data_tmp = col_map_tmp[s_main_axis]
                
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
                conf, _ = get_confidence_score(qual)
                
                md = None
                if 'mt_space' in st.session_state and st.session_state.mt_space.mean_vector is not None:
                    md = st.session_state.mt_space.calculate_md(all_f)
                
                summary_results.append({
                    "ファイル名": uploaded_file.name,
                    "MD値": f"{md:.2f}" if md is not None else "-",
                    "判定": "🔴 異常" if md and md > 10 else "🟡 警告" if md and md > 3 else "🟢 正常" if md else "-",
                    "信頼度": f"{conf:.1f}%",
                    "RMS": f"{all_f.rms:.3f}",
                    "Overall": f"{all_f.overall_level:.3f}",
                    "OA(LF)": f"{all_f.overall_low:.3f}",
                    "OA(HF)": f"{all_f.overall_high:.3f}",
                    "Peak": f"{all_f.peak:.3f}",
                    "尖度": f"{all_f.kurtosis:.2f}",
                    "歪度": f"{all_f.skewness:.2f}",
                    "CrestFactor": f"{all_f.crest_factor:.2f}",
                    "ShapeFactor": f"{all_f.shape_factor:.2f}",
                    "重心周波数": f"{all_f.spectral_centroid:.1f}",
                    "周波数分散": f"{all_f.spectral_spread:.1f}",
                    "スペクトルエントロピー": f"{all_f.spectral_entropy:.3f}",
                    "S/N比(dB)": f"{qual.snr_db:.1f}",
                    "クリッピング": f"{qual.clipping_ratio:.2%}",
                    "データ長(s)": f"{qual.data_length_s:.1f}",
                    "fs(Hz)": int(fs_tmp)
                })

                # Data collection: All lists must have same length
                all_mags.append(mags)
                all_freqs.append(f_hz)
                all_f_vectors.append(all_f.to_vector())
                all_filenames.append(uploaded_file.name)
                all_mds.append(md if md is not None else 1.0)

            except Exception as e:
                summary_results.append({
                    "ファイル名": uploaded_file.name, 
                    "判定": "🔴 ERROR",
                    "備考": f"詳細エラー: {str(e)}"
                })
                # Critical: Add dummy data to maintain index alignment
                all_mags.append(None)
                all_freqs.append(None)
                all_f_vectors.append(np.zeros(23)) 
                all_filenames.append(uploaded_file.name)
                all_mds.append(0.0)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)
            
        # --- Summary Table and Clustering UI ---
        df_summary = pd.DataFrame(summary_results)
        
        # Physical validity: Verify dimensions before array conversion
        current_dim = len(VibrationFeatures.get_feature_names())
        for i, v in enumerate(all_f_vectors):
            if len(v) != current_dim:
                st.error(f"FATAL: 次元不整合を検出しました (Expected {current_dim}, Found {len(v)} at Index {i})。解析設定または基準データを確認してください。")
                st.stop()
        
        X_all = np.array(all_f_vectors)
        
        col_sum1, col_sum2 = st.columns([2, 1])
        with col_sum1:
            st.subheader("📋 診断サマリー")
        with col_sum2:
            with st.expander("📊 自動グルーピング設定", expanded=False):
                use_grouping = st.checkbox("グルーピングを有効化", value=False, help="似た性質のデータを自動でグループ分けし、表示を集約します。大量データ解析時に有効です。")
                n_clusters = st.slider("グループ数", 2, 10, 3)
                run_grouping = st.button("グルーピングを実行/更新")

        # Perform clustering if requested
        X_all = np.array(all_f_vectors) 
        cluster_labels = st.session_state.get('last_cluster_labels')
        
        if use_grouping:
            # Check if we need to run or re-run clustering
            # We run if button is pressed OR if we don't have labels yet
            if run_grouping or cluster_labels is None or len(cluster_labels) != len(all_f_vectors):
                try:
                    with st.spinner("グループ分類を計算中..."):
                        # Physical validity: Use ONLY Band RMS (last 8 features) for intuitive clustering
                        X_cluster = X_all[:, 15:] # Extract only Band RMS components
                        
                        # Standardize before K-Means
                        X_scaled = StandardScaler().fit_transform(X_cluster)
                        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
                        cluster_labels = kmeans.fit_predict(X_scaled)
                        
                        # Save to session state to persist across UI refreshes
                        st.session_state.last_cluster_labels = cluster_labels
                        st.session_state.last_cluster_n = n_clusters
                except Exception as e:
                    st.error(f"グルーピング計算中にエラーが発生しました: {e}")
                    use_grouping = False
            
            # Apply labels to summary dataframe
            if cluster_labels is not None:
                df_summary['Cluster'] = cluster_labels
        else:
            # Clear stored labels if feature is disabled
            if 'last_cluster_labels' in st.session_state:
                del st.session_state.last_cluster_labels
            cluster_labels = None

        if not use_grouping or cluster_labels is None:
            st.dataframe(df_summary, use_container_width=True)
        else:
            st.info(f"💡 周波数帯域別の特性（Band RMS）に基づき、データを {st.session_state.get('last_cluster_n', n_clusters)} 個のグループに分類しました。")
            
            # Aggregated view per cluster
            for c_id in range(n_clusters):
                cluster_df = df_summary[df_summary['Cluster'] == c_id]
                count = len(cluster_df)
                
                # Calculate cluster-wide statistics
                is_ref_cluster = any(cluster_df["ファイル名"].str.contains("🔵 \[基準\]"))
                avg_md = cluster_df[cluster_df["MD値"] != "-"]["MD値"].astype(float).mean() if not cluster_df[cluster_df["MD値"] != "-"].empty else 1.0
                
                header_text = f"Group {c_id+1}: {count} 件 "
                header_text += "(正常群/基準を含む)" if is_ref_cluster else f"(平均MD: {avg_md:.2f})"
                
                with st.expander(header_text):
                    # Visual representation of the cluster profile (Band RMS)
                    cluster_indices = cluster_df.index
                    # Get average Band RMS for this cluster
                    avg_band_rms = X_all[cluster_indices, 15:].mean(axis=0)
                    
                    col_prof1, col_prof2 = st.columns([1, 2])
                    with col_prof1:
                        st.markdown("**グループ特性 (Band RMS)**")
                        fig_prof = go.Figure(data=[go.Bar(
                            x=[f"B{i+1}" for i in range(8)],
                            y=avg_band_rms,
                            marker_color='rgb(55, 83, 109)'
                        )])
                        fig_prof.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="帯域", yaxis_title="RMS")
                        st.plotly_chart(fig_prof, use_container_width=True)
                    
                    with col_prof2:
                        st.write(f"このグループに含まれるデータ（上位10件を表示）:")
                        st.dataframe(cluster_df.head(10), use_container_width=True)
                        if count > 10:
                            st.caption(f"...他 {count-10} 件のデータが含まれています。")

        # --- Advanced Trend Analysis ---
        if len(all_f_vectors) >= 2:
            st.subheader("📈 傾向分析 (データセット全体の俯瞰解析)")
            
            # Physical validity: Get unit and feature names for common use in tabs
            current_q = st.session_state.get("eval_quantity", SignalQuantity.ACCEL)
            unit = current_q.unit_str if hasattr(current_q, 'unit_str') else current_q.value
            feature_names = VibrationFeatures.get_feature_names()
            
            tab_trend1, tab_trend_diff, tab_trend2, tab_trend3, tab_trend4 = st.tabs([
                "🌈 全データ周波数ヒートマップ", 
                "⚖️ 全データ差分ヒートマップ",
                "🧩 特徴量PCA分布 (類似度分析)", 
                "📊 特徴量統計比較 (分布)",
                "📈 劣化トレンド推移"
            ])
            
            # --- Common Data for Heatmaps ---
            valid_freqs = [f for f in all_freqs if f is not None]
            common_freq_grid = None
            if valid_freqs:
                max_nyquist = max([f[-1] for f in valid_freqs])
                common_freq_grid = np.linspace(0, max_nyquist, 1024)

            with tab_trend1:
                st.markdown("##### 複数データの周波数特性比較")
                st.caption("全アップロードデータの周波数特性を並べて表示します。周波数帯ごとの変化や共通のピークを把握できます。")
                
                if common_freq_grid is not None:
                    heatmap_data = []
                    for m, f in zip(all_mags, all_freqs):
                        if f is None: f = valid_freqs[0]
                        m_interp = np.interp(common_freq_grid, f, m)
                        heatmap_data.append(m_interp)
                    
                    heatmap_array_db = 20 * np.log10(np.array(heatmap_data) + 1e-12)
                    
                    # Sort heatmap by cluster if active
                    y_labels = all_filenames
                    z_data = heatmap_array_db
                    if cluster_labels is not None:
                        sort_idx = np.argsort(cluster_labels)
                        # Physical validity: Enhance labels with Group ID for clear attribution
                        y_labels = [f"[G{cluster_labels[i]+1}] {all_filenames[i]}" for i in sort_idx]
                        z_data = heatmap_array_db[sort_idx]
                        st.caption("※ グルーピング有効時のため、ファイルをグループ順に並べ替え、ラベルに [G番号] を付与しています。")

                    fig_heat = go.Figure(data=go.Heatmap(
                        x=common_freq_grid,
                        y=y_labels,
                        z=z_data,
                        colorscale='Viridis',
                        colorbar=dict(title="振幅 (dB)")
                    ))
                    fig_heat.update_layout(
                        xaxis_title="周波数 (Hz)",
                        yaxis_title="ファイル名",
                        height=max(400, len(all_filenames) * 20),
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.warning("周波数データの収集に失敗したため、ヒートマップを表示できません。")

            with tab_trend_diff:
                st.markdown("##### 全データの基準からの逸脱度 (dB差分)")
                st.caption("基準（正常）と各データの振幅を引き算し、変化分のみを一覧表示します。赤色は増加、青色は減少を示し、異常の発生タイミングと周波数を一目で特定できます。")
                
                if 'mt_space' in st.session_state and st.session_state.mt_space.average_magnitude_spectrum is not None and common_freq_grid is not None:
                    ref_mags_raw = st.session_state.mt_space.average_magnitude_spectrum
                    ref_freqs_orig = np.linspace(0, max_nyquist, len(ref_mags_raw)) # Assume fs matches
                    ref_mags_interp = np.interp(common_freq_grid, ref_freqs_orig, ref_mags_raw)
                    ref_db = 20 * np.log10(ref_mags_interp + 1e-12)

                    diff_heatmap_data = []
                    for m, f in zip(all_mags, all_freqs):
                        if f is None: f = valid_freqs[0]
                        m_interp = np.interp(common_freq_grid, f, m)
                        m_db = 20 * np.log10(m_interp + 1e-12)
                        diff_heatmap_data.append(m_db - ref_db)
                    
                    diff_array = np.array(diff_heatmap_data)
                    y_labels_diff = all_filenames
                    if cluster_labels is not None:
                        sort_idx = np.argsort(cluster_labels)
                        y_labels_diff = [f"[G{cluster_labels[i]+1}] {all_filenames[i]}" for i in sort_idx]
                        diff_array = diff_array[sort_idx]

                    max_diff = 40.0
                    fig_heat_diff = go.Figure(data=go.Heatmap(
                        x=common_freq_grid,
                        y=y_labels_diff,
                        z=diff_array,
                        colorscale='RdBu_r',
                        zmid=0, zmin=-max_diff, zmax=max_diff,
                        colorbar=dict(title="差分 (dB)")
                    ))
                    fig_heat_diff.update_layout(
                        xaxis_title="周波数 (Hz)",
                        yaxis_title="ファイル名",
                        height=max(400, len(all_filenames) * 20),
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_heat_diff, use_container_width=True)
                else:
                    st.info("基準データ（単位空間）が読み込まれていないか、有効な指紋データがありません。")

            with tab_trend2:
                st.markdown("##### 特徴量空間におけるデータの類似度 (PCA Biplot)")
                st.caption("15種類の特徴量を2次元に圧縮して表示します。矢印（ベクトル）は各特徴量がどの方向に影響を与えているかを示します。")
                
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(np.array(all_f_vectors))
                    
                    pca = PCA(n_components=2)
                    X_pca = pca.fit_transform(X_scaled)
                    
                    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    pca_df['FileName'] = all_filenames
                    pca_df['MD'] = all_mds
                    if cluster_labels is not None:
                        pca_df['Cluster'] = [f"Group {c+1}" for c in cluster_labels]
                    
                    var_exp = pca.explained_variance_ratio_
                    
                    fig_pca = go.Figure()
                    
                    # Color points by Cluster if available, else by MD
                    color_col = 'Cluster' if cluster_labels is not None else 'MD'
                    color_title = "グループ" if cluster_labels is not None else "MD値 (異常度)"
                    
                    if cluster_labels is not None:
                        # Categorical plot for clusters
                        for group in sorted(pca_df['Cluster'].unique()):
                            grp_df = pca_df[pca_df['Cluster'] == group]
                            fig_pca.add_trace(go.Scatter(
                                x=grp_df['PC1'], y=grp_df['PC2'],
                                mode='markers',
                                name=group,
                                text=grp_df['FileName'],
                                marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')),
                                hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
                            ))
                    else:
                        # Continuous plot for MD
                        fig_pca.add_trace(go.Scatter(
                            x=pca_df['PC1'],
                            y=pca_df['PC2'],
                            mode='markers',
                            text=pca_df['FileName'],
                            marker=dict(
                                size=12,
                                color=pca_df['MD'],
                                colorscale='Reds',
                                showscale=True,
                                colorbar=dict(title=color_title),
                                line=dict(width=1, color='DarkSlateGrey')
                            ),
                            hovertemplate="<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>MD: %{marker.color:.2f}<extra></extra>"
                        ))
                    
                    # Biplot: Add arrows for features (Base 15 only to avoid clutter)
                    scale_factor = np.max(np.abs(X_pca)) * 0.8
                    loadings = pca.components_.T[:15] * scale_factor # Use only first 15 base features
                    
                    for i, (lx, ly) in enumerate(loadings):
                        fig_pca.add_trace(go.Scatter(
                            x=[0, lx], y=[0, ly],
                            mode='lines+text',
                            line=dict(color='rgba(100, 100, 100, 0.5)', width=1),
                            text=["", feature_names[i]],
                            textposition="bottom center",
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                        fig_pca.add_annotation(
                            x=lx, y=ly, ax=0, ay=0,
                            xref="x", yref="y", axref="x", ayref="y",
                            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor='rgba(100, 100, 100, 0.5)'
                        )

                    fig_pca.update_layout(
                        xaxis_title=f"主成分1 (寄与率: {var_exp[0]:.1%})",
                        yaxis_title=f"主成分2 (寄与率: {var_exp[1]:.1%})",
                        height=650,
                        margin=dict(l=20, r=20, t=40, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_pca, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"PCA解析中にエラーが発生しました: {e}")

            with tab_trend3:
                st.markdown("##### 特徴量別の分布比較")
                st.caption("基準（単位空間）と、アップロードされた全データ群の分布を比較します。どの物理指標が変化しているかを特定します。")
                
                if len(all_f_vectors) >= 2:
                    feature_to_compare = st.selectbox(
                        "比較する特徴量を選択", 
                        options=feature_names,
                        index=0
                    )
                    feat_idx = feature_names.index(feature_to_compare)
                    
                    # Prepare DataFrame for Boxplot
                    box_data = []
                    for i, vec in enumerate(all_f_vectors):
                        is_ref = all_filenames[i].startswith("🔵 [基準]")
                        if is_ref:
                            group_name = "基準値"
                        elif cluster_labels is not None:
                            group_name = f"Group {cluster_labels[i]+1}"
                        else:
                            group_name = "評価データ"
                            
                        box_data.append({
                            "グループ": group_name,
                            "値": vec[feat_idx],
                            "ファイル名": all_filenames[i]
                        })
                    
                    df_box = pd.DataFrame(box_data)
                    
                    fig_box = go.Figure()
                    # Determine groups to plot: Reference first, then Clusters or Evaluation Data
                    plot_groups = ["基準値"]
                    if cluster_labels is not None:
                        plot_groups += [f"Group {c+1}" for c in range(n_clusters)]
                    else:
                        plot_groups.append("評価データ")

                    for group in plot_groups:
                        group_df = df_box[df_box["グループ"] == group]
                        if group_df.empty: continue
                        
                        fig_box.add_trace(go.Box(
                            y=group_df["値"],
                            name=group,
                            boxmean='sd',
                            jitter=0.3,
                            pointpos=-1.8,
                            boxpoints='all',
                            text=group_df["ファイル名"]
                        ))
                    
                    fig_box.update_layout(
                        yaxis_title=f"{feature_to_compare} の値",
                        height=500,
                        margin=dict(l=20, r=20, t=20, b=20)
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.info("分布比較を行うには、基準データまたは複数のアップロードデータが必要です。")

            with tab_trend4:
                st.markdown("##### 劣化トレンド推移 (指標選択・平滑化解析)")
                st.caption("表示したい特徴量を選択してください。デフォルトはRMS（振動強さ）です。")
                
                # Dynamic indicator selection
                trend_features = ["MD値"] + feature_names
                col_tr1, col_tr2 = st.columns([2, 1])
                with col_tr1:
                    selected_trend_feats = st.multiselect(
                        "プロットする指標を選択", 
                        options=trend_features, 
                        default=["RMS"]
                    )
                with col_tr2:
                    sort_order = st.radio("X軸の並び順", ["ファイル順 (時間軸)", "グループ順 (比較軸)"], horizontal=True)

                trend_df_data_list = []
                for i, vec in enumerate(all_f_vectors):
                    if all_filenames[i].startswith("🔵 [基準]"): continue
                    
                    row = {
                        "OriginalIndex": i,
                        "ファイル名": all_filenames[i],
                        "MD値": all_mds[i],
                        "Cluster": cluster_labels[i] if cluster_labels is not None else 0
                    }
                    for f_idx, f_name in enumerate(feature_names):
                        row[f_name] = vec[f_idx]
                    trend_df_data_list.append(row)
                
                trend_df_data = pd.DataFrame(trend_df_data_list)
                
                if not trend_df_data.empty and selected_trend_feats:
                    # Handle Sorting
                    if sort_order == "グループ順 (比較軸)" and cluster_labels is not None:
                        trend_df_data = trend_df_data.sort_values(by=["Cluster", "OriginalIndex"]).reset_index(drop=True)
                        trend_df_data["PlotIndex"] = trend_df_data.index
                    else:
                        trend_df_data["PlotIndex"] = trend_df_data["OriginalIndex"]

                    # Case A: Single Feature Selected -> Show Subplots (Global + Clusters)
                    if len(selected_trend_feats) == 1:
                        target_feat = selected_trend_feats[0]
                        max_val = trend_df_data[target_feat].max() * 1.1
                        if target_feat == "MD値": max_val = max(max_val, 10.0)
                        
                        plot_n_clusters = n_clusters if cluster_labels is not None else 0
                        fig_trend = make_subplots(
                            rows=1 + plot_n_clusters, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.05,
                            subplot_titles=[f"統合トレンド: {target_feat} (全体)"] + [f"Group {i+1} 個別トレンド" for i in range(plot_n_clusters)]
                        )
                        
                        # Row 1: Global
                        fig_trend.add_trace(go.Scatter(
                            x=trend_df_data["PlotIndex"], y=trend_df_data[target_feat],
                            mode='lines', name=f'{target_feat} (全体)',
                            line=dict(color='rgba(150, 150, 150, 0.3)', width=1),
                            showlegend=False
                        ), row=1, col=1)
                        
                        # Visual Segment Partitions
                        if sort_order == "グループ順 (比較軸)" and cluster_labels is not None:
                            boundaries = trend_df_data[trend_df_data["Cluster"].diff() != 0].index.tolist()
                            for b_idx in boundaries:
                                if b_idx == 0: continue
                                for row_idx in range(1, 2 + plot_n_clusters):
                                    fig_trend.add_vline(x=b_idx - 0.5, line_dash="dash", line_color="gray", opacity=0.5, row=row_idx, col=1)

                        if cluster_labels is not None:
                            for c_id in range(n_clusters):
                                c_df = trend_df_data[trend_df_data["Cluster"] == c_id]
                                if c_df.empty: continue
                                
                                # Integrated Plot Marker
                                fig_trend.add_trace(go.Scatter(
                                    x=c_df["PlotIndex"], y=c_df[target_feat],
                                    mode='markers', name=f'Group {c_id+1}',
                                    marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                                    text=c_df["ファイル名"]
                                ), row=1, col=1)
                                
                                # Individual Rows
                                fig_trend.add_trace(go.Scatter(
                                    x=trend_df_data["PlotIndex"], y=trend_df_data[target_feat],
                                    mode='markers', marker=dict(color='rgba(200, 200, 200, 0.2)', size=4),
                                    showlegend=False, hoverinfo='skip'
                                ), row=c_id+2, col=1)
                                
                                fig_trend.add_trace(go.Scatter(
                                    x=c_df["PlotIndex"], y=c_df[target_feat],
                                    mode='markers', name=f'G{c_id+1} 生データ',
                                    marker=dict(size=8),
                                    text=c_df["ファイル名"], showlegend=False
                                ), row=c_id+2, col=1)
                                
                                # Moving Average (Only within segment)
                                if len(c_df) >= 2:
                                    ma_val = c_df[target_feat].rolling(window=5, min_periods=1, center=True).mean()
                                    fig_trend.add_trace(go.Scatter(
                                        x=c_df["PlotIndex"], y=ma_val,
                                        mode='lines', name=f'G{c_id+1} 傾向',
                                        line=dict(width=3), opacity=0.8
                                    ), row=c_id+2, col=1)
                        else:
                            fig_trend.add_trace(go.Scatter(
                                x=trend_df_data["PlotIndex"], y=trend_df_data[target_feat],
                                mode='markers', name=target_feat, marker=dict(color='red', size=8)
                            ), row=1, col=1)
                            ma_val_global = trend_df_data[target_feat].rolling(window=5, min_periods=1, center=True).mean()
                            fig_trend.add_trace(go.Scatter(
                                x=trend_df_data["PlotIndex"], y=ma_val_global,
                                mode='lines', name='全体傾向', line=dict(color='red', width=3)
                            ), row=1, col=1)

                        fig_trend.update_layout(
                            height=300 + (300 * plot_n_clusters),
                            margin=dict(l=20, r=20, t=40, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode="x unified"
                        )
                        for i in range(1 + plot_n_clusters):
                            feat_unit = unit if any(k in target_feat for k in ["RMS", "Overall", "Peak"]) else ""
                            y_title = f"{target_feat} ({feat_unit})" if feat_unit else target_feat
                            fig_trend.update_yaxes(title_text=y_title, range=[0, max_val], row=i+1, col=1)
                        
                        fig_trend.update_xaxes(title_text="並び順 (インデックス)", row=1 + plot_n_clusters, col=1)
                        st.plotly_chart(fig_trend, use_container_width=True)

                    # Case B: Multiple Features Selected -> Show Unified Multi-axis Plot
                    else:
                        st.info("💡 複数指標が選択されたため、1つの統合グラフに表示します。")
                        fig_multi = go.Figure()
                        
                        # Add segment boundaries for multi-plot
                        if sort_order == "グループ順 (比較軸)" and cluster_labels is not None:
                            boundaries = trend_df_data[trend_df_data["Cluster"].diff() != 0].index.tolist()
                            for b_idx in boundaries:
                                if b_idx == 0: continue
                                fig_multi.add_vline(x=b_idx - 0.5, line_dash="dash", line_color="gray", opacity=0.5)
                                # Label segments
                                c_id_before = trend_df_data.loc[b_idx-1, "Cluster"]
                                fig_multi.add_annotation(x=b_idx-1, y=1.05, yref="paper", text=f"Group {int(c_id_before+1)}", showarrow=False)
                            # Last segment label
                            last_idx = len(trend_df_data) - 1
                            fig_multi.add_annotation(x=last_idx, y=1.05, yref="paper", text=f"Group {int(trend_df_data.loc[last_idx, 'Cluster']+1)}", showarrow=False)

                        for f_name in selected_trend_feats:
                            # Grouped lines: break connections between segments by plotting separately
                            if sort_order == "グループ順 (比較軸)" and cluster_labels is not None:
                                for c_id in range(n_clusters):
                                    c_df = trend_df_data[trend_df_data["Cluster"] == c_id]
                                    fig_multi.add_trace(go.Scatter(
                                        x=c_df["PlotIndex"], y=c_df[f_name],
                                        mode='lines+markers', name=f"{f_name} (G{c_id+1})",
                                        legendgroup=f_name,
                                        showlegend=(c_id == 0),
                                        hovertemplate=f"<b>{f_name}</b>: %{{y:.3f}}<br>G{c_id+1}<extra></extra>"
                                    ))
                            else:
                                fig_multi.add_trace(go.Scatter(
                                    x=trend_df_data["PlotIndex"], y=trend_df_data[f_name],
                                    mode='lines+markers', name=f_name,
                                    hovertemplate=f"<b>{f_name}</b>: %{{y:.3f}}<extra></extra>"
                                ))
                        
                        fig_multi.update_layout(
                            xaxis_title="並び順 (インデックス)",
                            yaxis_title="値 (各指標の単位)",
                            height=600,
                            margin=dict(l=20, r=20, t=60, b=20),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_multi, use_container_width=True)
                elif not selected_trend_feats:
                    st.warning("プロットする指標を少なくとも1つ選択してください。")
                else:
                    st.info("トレンドを表示する評価データがありません。")
        # --- Multi-file Unit Space Construction ---
        st.markdown("##### 🛠️ MT法 単位空間の一括構築・保存")
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            save_name_main = st.text_input("単位空間の保存名 (空欄なら保存せず構築のみ)", placeholder="例: ポンプA_20260308", key="main_unit_space_save_name")
        with col_c2:
            st.write("") # Spacer
            build_btn = st.button("📥 単位空間を構築", use_container_width=True, help="現在の全ファイルから正常基準を構築します。名前を入力している場合は自動保存されます。")

        if build_btn:
            norm_features_list = []
            norm_magnitude_list = [] # Added to collect spectra for physical reference
            construction_progress = st.progress(0)

            for i, uploaded_file in enumerate(uploaded_files):
                # ... (Features extraction logic)
                file_extension = uploaded_file.name.split('.')[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    # Same logic as summary to extract features
                    if file_extension == "wav":
                        fs_tmp, data_tmp, _ = load_wav_file(tmp_path)
                    else:
                        df_tmp = pd.read_csv(tmp_path, nrows=5, skipinitialspace=True)
                        df_tmp.columns = [c.strip() for c in df_tmp.columns]
                        selected_cols = st.session_state.get("csv_data_columns", [])
                        active_cols = [c for c in selected_cols if c in df_tmp.columns]
                        if not active_cols: active_cols = csv_parser.infer_vibration_columns(df_tmp.columns.tolist())
                        if not active_cols:
                            potential = df_tmp.select_dtypes(include=np.number).columns.tolist()
                            active_cols = [potential[0]] if potential else [df_tmp.columns[0]]
                        s_fs = st.session_state.get("csv_sampling_frequency", 1000.0)
                        s_ts = st.session_state.get("csv_timestamp_column") if st.session_state.get("csv_use_timestamp") else None
                        s_syn = st.session_state.get("csv_synthesize", False)
                        s_main_axis = st.session_state.get("csv_main_axis_selection")
                        data_tmp, fs_tmp, col_map_tmp = csv_parser.parse_csv_data(Path(tmp_path), data_columns=active_cols, sampling_frequency_hz=s_fs, timestamp_column=s_ts, synthesize=s_syn)
                        if not s_syn and s_main_axis and col_map_tmp and s_main_axis in col_map_tmp:
                            data_tmp = col_map_tmp[s_main_axis]
                    
                    p_tmp = remove_dc_offset(data_tmp)
                    p_tmp = apply_butterworth_filter(p_tmp, fs_tmp, st.session_state.get("eval_hpf"), st.session_state.get("eval_lpf"), st.session_state.get("eval_order", 4), hpf_enabled=st.session_state.get("eval_hpf_enabled", False), lpf_enabled=st.session_state.get("eval_lpf_enabled", False))
                    t_feat = calculate_time_domain_features(p_tmp)
                    _, mags, f_feat = calculate_fft_features(p_tmp, fs_tmp, st.session_state.get("eval_window", WindowFunction.HANNING))
                    all_f = VibrationFeatures(**asdict(t_feat), **f_feat)
                    norm_features_list.append(all_f)
                    norm_magnitude_list.append(mags) # Store magnitude spectrum for averaging
                except Exception as e:
                    st.error(f"ファイル '{uploaded_file.name}' からの特徴量抽出に失敗しました: {e}")
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                construction_progress.progress((i + 1) / len(uploaded_files))
            
            if len(norm_features_list) >= 2:
                # Call with both features and magnitude spectra (Fix for TypeError)
                st.session_state.mt_space.build_unit_space(norm_features_list, norm_magnitude_list)
                st.sidebar.success(f"✅ {len(norm_features_list)} 個のファイルを用いて単位空間を構築しました。")
                
                # Auto-save if name provided
                if save_name_main:
                    try:
                        # Physical validity: Use session state to ensure variables are available regardless of execution order
                        current_config = AnalysisConfig(
                            quantity=st.session_state.get("eval_quantity", SignalQuantity.ACCEL),
                            window=st.session_state.get("eval_window", WindowFunction.HANNING),
                            highpass_hz=st.session_state.get("eval_hpf", 10.0),
                            lowpass_hz=st.session_state.get("eval_lpf", 20000.0),
                            filter_order=st.session_state.get("eval_order", 4),
                            hpf_enabled=st.session_state.get("eval_hpf_enabled", False),
                            lpf_enabled=st.session_state.get("eval_lpf_enabled", False)
                        )
                        mt_manager.save_unit_space(save_name_main, st.session_state.mt_space, current_config)
                        st.session_state.load_mt_space_name = save_name_main # Set as currently selected
                        st.sidebar.info(f"💾 単位空間 '{save_name_main}' をファイルに保存しました。")
                    except Exception as e:
                        st.sidebar.error(f"保存に失敗しました: {e}")
                
                st.rerun()
            else:
                st.error("単位空間の構築には少なくとも2つ以上の有効なデータが必要です。")

        # --- AI Reporting Assistant ---
        st.markdown("---")
        with st.expander("🤖 生成AI 高度レポートアシスタント (コンテキスト統合 ＆ 深層解析型)", expanded=False):
            st.markdown("##### 診断の「目的」や「対象」を整理し、生成AI（Gemini / ChatGPT 等）に渡す最適なプロンプトを作成します。")
            
            col_ai1, col_ai2 = st.columns(2)
            with col_ai1:
                st.subheader("🎯 分析のコンテキスト設定")
                ai_purpose = st.selectbox(
                    "分析の目的 (Purpose)",
                    options=[
                        "予兆検知 (いつもと違う兆候を見つけたい)",
                        "異常分類 (故障箇所や原因を特定したい)",
                        "品質評価 (測定環境やデータの信頼性を確認したい)",
                        "総合報告 (現状の健康状態を要約したい)"
                    ]
                )
                machine_type = st.text_input("設備の特性 (対象物)", placeholder="例: 定速回転ポンプ (1500rpm), モーター直結, 過去に軸受損傷歴あり")
                
                # --- ISO 10816 Classification ---
                st.markdown("**ISO 10816 評価基準の適用**")
                iso_class = st.selectbox(
                    "設備クラスを選択 (ISO 10816)",
                    options=[
                        "適用しない",
                        "Class I: 小形機械 (出力 < 15kW)",
                        "Class II: 中形機械 (15kW <= 出力 < 75kW)",
                        "Class III: 大形機械 (出力 >= 75kW, 強固な基礎)",
                        "Class IV: 大形機械 (出力 >= 75kW, 柔軟な基礎)"
                    ],
                    help="選択すると、国際規格に基づいた振動レベル評価基準をAIに教育します。"
                )
            
            with col_ai2:
                sensor_env = st.text_input("センサ・環境 (測定条件)", placeholder="例: 加速度センサ磁石取付, 軸受ハウジング垂直方向, 周囲に別機の振動あり")
                report_target = st.radio("レポートの提出先", ["現場保全担当者", "管理職・経営層", "AIとの技術的ディスカッション"], horizontal=True)
                use_anonymization = st.checkbox("自動匿名化を有効にする", value=True, help="ファイル名を 'Sample_001' 等に置換してプライバシーを保護します。")

            # --- Statistical Narrative Engine (Pre-calculation) ---
            # Calculate physical trends to help AI logic
            stats_narrative = "## Statistical Narrative (Pre-calculated Trends)\n"
            if len(df_summary) >= 2:
                # 1. Overall Trend
                valid_mds = [float(x) for x in df_summary["MD値"] if x != "-"]
                if len(valid_mds) >= 2:
                    growth_rate = (valid_mds[-1] - valid_mds[0]) / (valid_mds[0] + 1e-6)
                    stats_narrative += f"- Overall MD Change Rate: {growth_rate:+.1%}\n"
                
                # 2. Peak Anomaly Sample analysis
                if any(df_summary["判定"] == "🔴 異常"):
                    max_md_idx = df_summary[df_summary["MD値"] != "-"]["MD値"].astype(float).idxmax()
                    max_sample = df_summary.loc[max_md_idx]
                    stats_narrative += f"- Most Severe Sample: Index {max_md_idx}, MD {max_sample['MD値']}, Peak {max_sample['Peak']}\n"
                    # Add Band RMS profile if available
                    if 'X_all' in locals():
                        peak_bands = X_all[max_md_idx, 15:]
                        stats_narrative += f"  - Profile: Band RMS is dominant in {'High' if np.argmax(peak_bands) >= 5 else 'Low/Mid'} frequencies.\n"

            # --- Advanced Physical Analysis Engine (Spectral Peaks & Gaps) ---
            adv_physics_narrative = "## Advanced Physical Reasoning Material\n"
            
            # 1. Spectral Peaks Analysis (for max anomaly or cluster centroids)
            if 'all_mags' in locals() and len(all_mags) > 0:
                # Analyze peaks for the most severe anomaly among VALID data
                valid_indices = [i for i, m in enumerate(all_mags) if m is not None]
                if valid_indices:
                    # Find max MD only within valid indices
                    valid_mds_subset = [all_mds[i] for i in valid_indices]
                    max_subset_idx = np.argmax(valid_mds_subset)
                    max_md_idx_abs = valid_indices[max_subset_idx]
                    
                    mag_data = all_mags[max_md_idx_abs]
                    freq_axis = all_freqs[max_md_idx_abs] if all_freqs[max_md_idx_abs] is not None else np.linspace(0, (fs_tmp/2 if 'fs_tmp' in locals() else 1000), len(mag_data))
                    
                    # Detect peaks
                    peaks_idx, _ = find_peaks(mag_data, height=np.max(mag_data)*0.2)
                    top_peaks_idx = peaks_idx[np.argsort(mag_data[peaks_idx])][::-1][:3]
                    peak_info = [f"{freq_axis[p]:.1f}Hz ({mag_data[p]:.3f}{unit})" for p in top_peaks_idx]
                    
                    adv_physics_narrative += f"### Anomaly Peak Signatures (Sample with MD {all_mds[max_md_idx_abs]:.1f})\n"
                    adv_physics_narrative += f"- Top 3 Spectral Peaks: {', '.join(peak_info)}\n"

            # 2. Group Gap Analysis (Normalized difference between groups)
            if cluster_labels is not None and n_clusters >= 2:
                adv_physics_narrative += "### Statistical Gap Analysis (Relative to Baseline Group)\n"
                # Find which group is likely the baseline (lowest avg MD)
                cluster_avg_mds = []
                for c_id in range(n_clusters):
                    c_df = df_summary[df_summary['Cluster'] == c_id]
                    c_avg_md = c_df[c_df["MD値"] != "-"]["MD値"].astype(float).mean() if not c_df[c_df["MD値"] != "-"].empty else 1.0
                    cluster_avg_mds.append(c_avg_md)
                
                baseline_cid = np.argmin(cluster_avg_mds)
                target_cid = np.argmax(cluster_avg_mds)
                
                if baseline_cid != target_cid:
                    b_idx = df_summary[df_summary['Cluster'] == baseline_cid].index
                    t_idx = df_summary[df_summary['Cluster'] == target_cid].index
                    
                    b_rms = X_all[b_idx, 0].mean()
                    t_rms = X_all[t_idx, 0].mean()
                    b_kurt = X_all[b_idx, 2].mean()
                    t_kurt = X_all[t_idx, 2].mean()
                    
                    adv_physics_narrative += f"- Comparison: Group {target_cid+1} (Anomaly) vs Group {baseline_cid+1} (Healthy)\n"
                    adv_physics_narrative += f"  - RMS Ratio: {t_rms/(b_rms+1e-6):.2f}x increase\n"
                    adv_physics_narrative += f"  - Kurtosis Delta: {t_kurt - b_kurt:+.2f} shift\n"

            # --- System context and physical definitions for AI ---
            roles = {
                "予兆検知": "振動保全の予兆管理スペシャリスト",
                "異常分類": "設備診断・故障物理エンジニア",
                "品質評価": "計測工学・信号処理専門家",
                "総合報告": "製造現場の保全コンサルタント"
            }
            selected_role = next((v for k, v in roles.items() if k in ai_purpose), "振動解析エンジニア")

            # ISO Definition Injection
            iso_context = ""
            if iso_class != "適用しない":
                iso_thresholds = {
                    "Class I": {"good": 0.71, "satisfactory": 1.8, "unsatisfactory": 4.5, "unacceptable": 7.1},
                    "Class II": {"good": 1.12, "satisfactory": 2.8, "unsatisfactory": 7.1, "unacceptable": 11.0},
                    "Class III": {"good": 1.8, "satisfactory": 4.5, "unsatisfactory": 11.0, "unacceptable": 18.0},
                    "Class IV": {"good": 2.8, "satisfactory": 7.1, "unsatisfactory": 18.0, "unacceptable": 45.0}
                }
                cls_key = iso_class.split(":")[0]
                th = iso_thresholds.get(cls_key, iso_thresholds["Class I"])
                iso_context = f"""
# ISO 10816-3 Diagnostic Standards ({cls_key})
振動速度 (mm/s RMS) の評価基準:
- < {th['good']} mm/s: 良好 (Good)
- < {th['satisfactory']} mm/s: 良 (Satisfactory)
- < {th['unsatisfactory']} mm/s: 注意 (Unsatisfactory)
- > {th['unacceptable']} mm/s: 不可 (Unacceptable)
※ 現在の物理量が速度以外の場合は、この基準を傾向判断の参考として扱ってください。
"""

            ai_system_context = f"""
# Role
あなたは【{selected_role}】として、提供されたデータを深く分析し、{report_target}向けに最適なレポートを作成してください。

# Physics & Diagnostic Context
- 物理量: {st.session_state.get('eval_quantity', SignalQuantity.ACCEL).value}
- MD値: 1.0(正常)からの距離。3.0超(注意)、10.0超(異常)。
- 尖度: 衝撃の鋭さ。3.0が正常、大きいほど剥離等の衝撃成分あり。
- Band RMS (B1-B8): 全帯域を8等分した成分。低域(B1-2), 中域(B3-5), 高域(B6-8)。
- SPC指標: 0-1.0。基準スペクトル形状との一致度。
{iso_context}
"""

            # --- Extract and Anonymize Data ---
            total_count = len(df_summary)
            anomaly_count = len(df_summary[df_summary["判定"] == "🔴 異常"])
            
            cluster_info_text = ""
            if cluster_labels is not None:
                cluster_info_text = "## Clustering Analysis (Group Characteristics)\n"
                for c_id in range(n_clusters):
                    c_df = df_summary[df_summary['Cluster'] == c_id]
                    c_avg_md = c_df[c_df["MD値"] != "-"]["MD値"].astype(float).mean() if not c_df[c_df["MD値"] != "-"].empty else 1.0
                    c_indices = c_df.index
                    avg_bands = X_all[c_indices, 15:].mean(axis=0)
                    cluster_info_text += f"- Group {c_id+1}: {len(c_df)} files, Avg MD: {c_avg_md:.2f}, Profile: {np.round(avg_bands, 4).tolist()}\n"

            # Prepare summarized material (Anonymized)
            material_df = df_summary.copy()
            if use_anonymization:
                # Replace file names with Sample ID
                material_df["ファイル名"] = [f"Sample_{i+1:03d}" for i in range(len(material_df))]
            
            cols_to_keep = ["ファイル名", "MD値", "判定", "信頼度", "RMS", "尖度", "重心周波数"]
            if "Cluster" in material_df.columns: cols_to_keep.append("Cluster")
            
            # Pack summary results with fallback for missing tabulate dependency
            try:
                summary_markdown = material_df[cols_to_keep].head(50).to_markdown(index=False)
            except ImportError:
                summary_markdown = "## Data Material (CSV Format)\n" + material_df[cols_to_keep].head(50).to_csv(index=False)
            
            # --- Theme specific tasks ---
            task_instruction = ""
            if "予兆検知" in ai_purpose:
                task_instruction = "時系列のMD値とRMSの変化点に着目し、劣化がいつから、どの程度の速さで始まっているか特定してください。突発的な異常か、漸増的な摩耗かを判断してください。"
            elif "異常分類" in ai_purpose:
                task_instruction = "Band RMSプロファイルと物理指標の変化から、不具合の物理的原因（アンバランス、ミスアライメント、軸受傷、ガタ、潤滑不良等）を推定してください。"
            elif "品質評価" in ai_purpose:
                task_instruction = "信頼度スコアとS/N比、クリッピング率を精査し、現在の測定データが診断に耐えうるものか評価してください。測定環境やセンサ設置の改善点を提案してください。"
            else:
                task_instruction = "全データの傾向を要約し、現時点で最も深刻なリスクと、直近で実施すべきアクションプランを提案してください。"

            # --- Final Prompt Composition ---
            full_prompt = f"""{ai_system_context}

# Targeted Machine & Environment
- Target: {machine_type if machine_type else "不明 (一般的回転機として扱う)"}
- Environment: {sensor_env if sensor_env else "不明 (標準的な取付と仮定)"}

# Analysis Goal
{ai_purpose}
{task_instruction}

# Dataset Summary
- Samples: {total_count}
- Anomaly Hits: {anomaly_count}
{cluster_info_text}
{stats_narrative}
{adv_physics_narrative}

# Visual Evidence (Reference)
提供されたテキストデータに加え、解析画面に表示されている以下のグラフ（画像）も重要な証拠です。
1. 「全データ差分ヒートマップ」: どの周波数帯がいつ変化したかの2次元エビデンス。
2. 「劣化トレンド推移」: 異常度(MD)の時系列的な進行エビデンス。
3. 「PCA Biplot」: 特徴量空間でのデータの塊（クラスター）とその物理的要因エビデンス。

# Data Material (Sample Records)
{summary_markdown}

# Constraints
- {report_target}が納得できるよう、数値的な根拠（どの指標がどう変化したか）を明記すること。
- ハルシネーション（憶測による結論）を避け、提供されたデータの範囲内で推論すること。
- 日本語で出力すること。
"""
            st.subheader("📋 生成された高度プロンプト ＆ 深層解析材料")
            st.info("以下のボックス内のテキストをすべてコピーして、生成AI（Gemini 等）に貼り付けてください。")
            st.code(full_prompt, language="markdown")
            
            st.markdown("---")
            st.markdown("##### 📄 AIレポートの受け取り後のヒント")
            st.caption("AIから回答を得た後は、必要に応じて以下のように深掘りしてください。")
            st.info("「Group 2 の Band RMS 特性が〇〇である理由を物理的に詳しく説明して」\n「この劣化速度に基づき、あと何ヶ月稼働可能か故障物理モデルから予測して」")

        # --- Detail Selection ---
        st.markdown("---")
        file_names = [f.name for f in uploaded_files]
        selected_file_name = st.selectbox("詳細表示するファイルを選択", options=file_names)

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
            
            st.sidebar.header("FFTピーク設定")
            top_n_peaks = st.sidebar.slider("ピーク表示数", 1, 20, 5, key="peak_count")
            min_peak_height_percent = st.sidebar.slider("最小ピーク高さ（最大値に対する%）", 0, 100, 10, key="peak_height")
            peak_distance_hz = st.sidebar.slider("ピーク最小距離（Hz）", 1, 100, 10, key="peak_distance")

            st.sidebar.header("表示設定")
            fft_log_x = st.sidebar.checkbox("FFT周波数軸を対数にする", value=False, key="fft_log_x")
            show_raw_signal = st.sidebar.checkbox("生信号(DC除去のみ)を表示する", value=True, key="show_raw_signal")
            spec_nperseg = st.sidebar.select_slider("スペクトログラム解像度 (Window Size)", options=[128, 256, 512, 1024, 2048], value=512, key="spec_nperseg")

            # Physical validity: Nyquist frequency is half of the sampling rate
            nyquist = fs_hz / 2.0

            # Physical validity: Retrieve config values from session state (updated by sidebar)
            quantity = st.session_state.get("eval_quantity", SignalQuantity.ACCEL)
            window = st.session_state.get("eval_window", WindowFunction.HANNING)
            hpf_enabled = st.session_state.get("eval_hpf_enabled", False)
            hpf = st.session_state.get("eval_hpf", 10.0)
            lpf_enabled = st.session_state.get("eval_lpf_enabled", False)
            lpf = st.session_state.get("eval_lpf", nyquist)
            order = st.session_state.get("eval_order", 4)

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
            confidence, conf_breakdown = get_confidence_score(quality)
            unit = config.quantity.unit_str

            all_features = VibrationFeatures(**asdict(time_features), **freq_features)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.subheader("時間領域 特徴量")
                st.metric(f"RMS ({unit})", f"{all_features.rms:.3f}", help="振動エネルギーの総量。機械の全体的な振動の強さを表します。")
                st.metric(f"Peak ({unit})", f"{all_features.peak:.3f}", help="波形の最大値。瞬間的な衝撃（ガタや突発的な異常）を捉えます。")
                st.metric("尖度 (Kurtosis)", f"{all_features.kurtosis:.3f}", help="波形の鋭さ。軸受の傷などによる衝撃波が発生すると値が大きくなります。正常値は約3.0です。")

                st.subheader("周波数領域 特徴量")
                st.metric(f"Overall (Total) ({unit})", f"{all_features.overall_level:.3f}", help="全周波数帯域の振動エネルギーを統合した実効値。理論上、時間領域のRMSと一致します。")
                col_oa1, col_ana_oa2 = st.columns(2)
                with col_oa1:
                    st.metric(f"Overall (LF) ({unit})", f"{all_features.overall_low:.3f}", help="1,000Hz未満の振動成分のみを統合した実効値。アンバランスやガタの影響を捉えます。")
                with col_ana_oa2:
                    st.metric(f"Overall (HF) ({unit})", f"{all_features.overall_high:.3f}", help="1,000Hz以上の振動成分のみを統合した実効値。軸受損傷や摩擦の影響を捉えます。")
                st.metric("Spectral Centroid (Hz)", f"{all_features.spectral_centroid:.2f}", help="スペクトルの「重心」。振動の主成分がどの周波数帯にあるかを示します。")
                st.metric("Spectral Spread (Hz)", f"{all_features.spectral_spread:.2f}", help="スペクトルの「広がり」。振動が特定の周波数に集中しているか、広帯域に分散しているかを示します。")
                st.metric("Spectral Entropy", f"{all_features.spectral_entropy:.3f}", help="スペクトルの「複雑さ（乱雑さ）」。不規則なノイズが多いほど値が大きくなります。")
                
                st.subheader("データ品質", help="解析に使用したデータの健全性を評価します。各指標が悪いと『診断信頼度』が低下します。")
                st.metric("クリッピング率", f"{quality.clipping_ratio:.2%}", help="センサの測定範囲を超えて波形が飽和（潰れた）割合。1%を超えると信頼度が大幅に低下します。")
                st.metric("S/N 比", f"{quality.snr_db:.2f} dB", help="背景ノイズに対する信号の強さ。20dB以上が理想的です。値が小さい場合はノイズ除去プラグインの活用を検討してください。")
                
                color = "green" if confidence >= 80 else "orange"
                st.markdown(f'#### 診断信頼度: <span style="color:{color};" title="データ品質（クリッピング、S/N比、データ長）に基づいた、この解析結果の確からしさを示します。">{confidence:.1f}%</span>', unsafe_allow_html=True)
                
                with st.expander("🤔 診断信頼度とは？"):
                    st.write("解析結果の妥当性を以下の3指標（各100点満点）で評価しています：")
                    for label, score in conf_breakdown.items():
                        st.write(f"{label}: {score:.1f}")
                        st.progress(score / 100.0)
                    
                    st.markdown("---")
                    st.write("""
                    1. **飽和回避 (Clipping)**: 振動が大きすぎてセンサの限界を超えていないか。
                    2. **ノイズ耐性 (SNR)**: 周囲の雑音に埋もれず、対象の振動をクリアに捉えられているか。
                    3. **データ量 (Length)**: 物理現象を正しく捉えるのに十分な長さ（10秒以上推奨）があるか。
                    
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
                    
                    # Physical validity: Overlay reference spectrum and calculate difference
                    if 'mt_space' in st.session_state and st.session_state.mt_space.average_magnitude_spectrum is not None:
                        ref_mags_raw = st.session_state.mt_space.average_magnitude_spectrum
                        
                        # Guard: Only plot if frequency resolution matches (length of spectrum)
                        if len(ref_mags_raw) == len(freqs):
                            # Downsample logic to match plot_freqs
                            plot_ref_mags = ref_mags_raw[::step_f] if len(ref_mags_raw) > MAX_POINTS else ref_mags_raw
                            
                            fig_fft.add_trace(go.Scatter(
                                x=plot_freqs, y=plot_ref_mags, 
                                mode='lines', 
                                name=f'基準: {st.session_state.get("load_mt_space_name", "単位空間")}', 
                                line=dict(color='rgba(100, 100, 100, 0.6)', width=1.5, dash='dot')
                            ))

                            # --- Anomaly Signature (Difference Spectrum) ---
                            # Calculate delta: Current - Reference (only positive increases)
                            # This helps focus on added energy/vibration.
                            diff_mags_raw = np.maximum(0, mags - ref_mags_raw)
                            plot_diff_mags = diff_mags_raw[::step_f] if len(diff_mags_raw) > MAX_POINTS else diff_mags_raw
                            
                            fig_fft.add_trace(go.Scatter(
                                x=plot_freqs, y=plot_diff_mags,
                                fill='tozeroy', # Fill area from 0 to Y
                                mode='lines',
                                name='異常成分 (増大分)',
                                line=dict(color='rgba(255, 0, 0, 0.3)', width=0.5),
                                fillcolor='rgba(255, 0, 0, 0.2)'
                            ))
                        else:
                            st.caption(f"ℹ️ 基準スペクトルと現在の解析条件（窓長等）が異なるため、重ね合わせをスキップしました (Ref:{len(ref_mags_raw)}, Cur:{len(freqs)})")
                    
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
                    st.subheader("時間-周波数解析 & パターン一致度")
                    spec_f, spec_t, Sxx = calculate_spectrogram(processed, fs_hz, config.window, nperseg=spec_nperseg)
                    
                    # Convert to dB for visualization
                    Sxx_db = 10 * np.log10(Sxx + 1e-12)
                    
                    filter_status = f"HPF:{config.highpass_hz}Hz " if config.hpf_enabled else ""
                    filter_status += f"LPF:{config.lowpass_hz}Hz" if config.lpf_enabled else ""
                    if not filter_status: filter_status = "None"

                    # --- Fingerprint Analysis (SPC: Spectral Pattern Consistency) ---
                    spc_scores = None
                    if 'mt_space' in st.session_state and st.session_state.mt_space.average_magnitude_spectrum is not None:
                        ref_mags_raw = st.session_state.mt_space.average_magnitude_spectrum
                        ref_freqs_orig = np.linspace(0, fs_hz / 2, len(ref_mags_raw))
                        ref_mags_interp = np.interp(spec_f, ref_freqs_orig, ref_mags_raw)
                        
                        # Calculate Pearson Correlation for each time frame
                        # Use magnitude (sqrt of power) for pattern matching
                        Sxx_mag_frames = np.sqrt(Sxx)
                        spc_scores = []
                        for i in range(Sxx_mag_frames.shape[1]):
                            frame = Sxx_mag_frames[:, i]
                            if np.sum(frame) > 0 and np.sum(ref_mags_interp) > 0:
                                corr = np.corrcoef(frame, ref_mags_interp)[0, 1]
                                spc_scores.append(max(0, corr))
                            else:
                                spc_scores.append(1.0) # Assume match if both silent
                        
                        st.markdown("##### 🕵️ パターン一致度推移 (SPC指標)")
                        st.caption("基準（正常）のスペクトル形状と、現在の各瞬間における一致度を示します。値が急落する箇所は、異音の混入やパターンの崩れが発生しています。")
                        
                        fig_spc = go.Figure()
                        fig_spc.add_trace(go.Scatter(
                            x=spec_t, y=spc_scores,
                            mode='lines',
                            name='一致度 (SPC)',
                            line=dict(color='green', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 255, 0, 0.1)'
                        ))
                        fig_spc.update_layout(
                            xaxis_title="時間 (s)",
                            yaxis=dict(title="一致度 (0.0 - 1.0)", range=[0, 1.05]),
                            height=250,
                            margin=dict(l=20, r=20, t=10, b=20),
                            hovermode="x"
                        )
                        st.plotly_chart(fig_spc, use_container_width=True)

                    st.markdown("##### メイン・スペクトログラム")
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
                    st.plotly_chart(fig_spec, use_container_width=True)

                    # Add component spectrograms if available
                    if column_data_map:
                        st.markdown("---")
                        st.subheader("各軸の個別スペクトログラム")
                        for col_name, col_arr in column_data_map.items():
                            f_c, t_c, Sxx_c = calculate_spectrogram(col_arr - np.mean(col_arr), fs_hz, config.window, nperseg=spec_nperseg)
                            Sxx_c_db = 10 * np.log10(Sxx_c + 1e-12)
                            
                            fig_c = go.Figure(data=go.Heatmap(
                                x=t_c, y=f_c, z=Sxx_c_db,
                                colorscale='Viridis',
                                showscale=False 
                            ))
                            fig_c.update_layout(
                                title=f"軸: {col_name} (DC除去)",
                                xaxis_title="時間 (s)",
                                yaxis_title="周波数 (Hz)",
                                yaxis_type="log" if fft_log_x else "linear",
                                height=400,
                                margin=dict(l=20, r=20, t=40, b=20)
                            )
                            st.plotly_chart(fig_c, use_container_width=True)

                    # --- Simple Difference Spectrogram (dB Subtraction) ---
                    if 'mt_space' in st.session_state and st.session_state.mt_space.average_magnitude_spectrum is not None:
                        ref_mags_raw = st.session_state.mt_space.average_magnitude_spectrum
                        ref_freqs_orig = np.linspace(0, fs_hz / 2, len(ref_mags_raw))
                        
                        # Apply smoothing to reference to match spectrogram resolution
                        if len(ref_mags_raw) > len(spec_f) * 2:
                            from scipy.ndimage import gaussian_filter1d
                            sigma = len(ref_mags_raw) / len(spec_f)
                            ref_mags_processed = gaussian_filter1d(ref_mags_raw, sigma=sigma/2)
                        else:
                            ref_mags_processed = ref_mags_raw

                        ref_mags_interp = np.interp(spec_f, ref_freqs_orig, ref_mags_processed)
                        ref_db = 20 * np.log10(ref_mags_interp + 1e-12)

                        st.markdown("---")
                        st.subheader("⚖️ 差分スペクトログラム (基準との比較)")
                        st.caption("基準（正常）と現在の音の大きさを引き算した結果です。赤色は基準より増加、青色は減少を示します。白い領域は変化がない（正常と同じ）ことを意味します。")
                        
                        # Simple subtraction in dB scale
                        # Sxx_db is current spectrogram, ref_db is baseline
                        diff_db = Sxx_db - ref_db[:, np.newaxis]
                        
                        # Limit the range for better contrast (e.g., +/- 40dB)
                        max_diff = 40.0
                        
                        fig_diff = go.Figure(data=go.Heatmap(
                            x=spec_t, y=spec_f, z=diff_db,
                            colorscale='RdBu_r', # Red for positive (increase), Blue for negative (decrease)
                            zmid=0,              # Ensure 0 is white
                            zmin=-max_diff, zmax=max_diff,
                            colorbar=dict(title="差分 (dB)")
                        ))
                        
                        fig_diff.update_layout(
                            title=f"Difference Spectrogram (vs {st.session_state.get('load_mt_space_name', 'Baseline')})",
                            xaxis_title="時間 (s)",
                            yaxis_title="周波数 (Hz)",
                            yaxis_type="log" if fft_log_x else "linear",
                            height=500,
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_diff, use_container_width=True)

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

