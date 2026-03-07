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
from src.utils import csv_parser

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
    # Combined file uploader for WAV and CSV
    uploaded_file = st.file_uploader("評価用WAVまたはCSVファイルをアップロード", type=["wav", "csv"], key="evaluation_uploader")

    data_raw = None
    fs_hz = None
    file_hash = None
    tmp_file_path = None

    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()

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
                if len(data_columns) > 1:
                    synthesize = st.sidebar.toggle("多軸合成を行う (合成ベクトル加速度)", value=True, key="csv_synthesize", 
                                                   help="物理的妥当性に基づき、各軸のDCオフセット（重力等）を除去した後に合成加速度 sqrt(X^2 + Y^2 + Z^2) を計算します。")

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

                data_raw, fs_hz = csv_parser.parse_csv_data(
                    Path(tmp_file_path),
                    data_columns=data_columns,
                    sampling_frequency_hz=input_sampling_frequency_hz,
                    timestamp_column=timestamp_column,
                    synthesize=synthesize
                )
                col_info = "+".join(data_columns) if synthesize else data_columns[0]
                file_hash = f"csv_hash_{uploaded_file.name}_{col_info}" # Simple hash for CSV

            st.write(f"ファイル名: {uploaded_file.name}, サンプリング周波数: {fs_hz} Hz, データ長: {len(data_raw) / fs_hz:.2f} 秒")
            
            # --- Analysis Configuration (same as before) ---
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
        col_ana1, col_ana2 = st.columns(2)
        with col_ana1:
            benchmark_quantity = st.selectbox("物理量種別", list(SignalQuantity), format_func=lambda x: x.value, key="benchmark_quantity")
            benchmark_window = st.selectbox("窓関数", list(WindowFunction), format_func=lambda x: x.value, key="benchmark_window")
        with col_ana2:
            # Assuming a fixed fs for benchmark, or derived from dataset
            # For simplicity, using dummy fs to prevent errors for nyquist calculation
            dummy_fs = 48000 
            benchmark_hpf = st.number_input("HPF (Hz)", 0.0, float(dummy_fs/2), 10.0, key="benchmark_hpf")
            benchmark_lpf = st.number_input("LPF (Hz)", 0.0, float(dummy_fs/2), float(dummy_fs/2), key="benchmark_lpf")
            benchmark_order = st.number_input("フィルタ次数", 1, 10, 4, key="benchmark_order")
        
        benchmark_analysis_config = AnalysisConfig(
            quantity=benchmark_quantity, 
            window=benchmark_window,
            highpass_hz=float(benchmark_hpf), 
            lowpass_hz=float(benchmark_lpf), 
            filter_order=benchmark_order
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
                st.plotly_chart(cm_fig, use_container_width=True)

            # ROC Curve
            if benchmark_result.roc_curve and benchmark_result.roc_auc is not None:
                st.markdown("##### ROC曲線")
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(x=benchmark_result.roc_curve['fpr'], y=benchmark_result.roc_curve['tpr'], mode='lines', name=f'ROC (AUC = {benchmark_result.roc_auc:.2f})'))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Classifier', line=dict(dash='dash')))
                roc_fig.update_layout(xaxis_title="偽陽性率 (FPR)", yaxis_title="真陽性率 (TPR)", height=400, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(roc_fig, use_container_width=True)

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
                st.plotly_chart(md_fig, use_container_width=True)


            st.markdown("---")
            st.subheader("ベンチマーク構成 (監査用)")
            st.json(asdict(benchmark_result.benchmark_config))

        except Exception as e:
            st.error(f"ベンチマーク実行中にエラーが発生しました: {e}")

elif page_selection == "合成データ生成":
    # ... (code for this page)
    pass

