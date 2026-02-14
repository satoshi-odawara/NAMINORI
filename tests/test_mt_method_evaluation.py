import os
import glob
from pathlib import Path
import numpy as np
from typing import List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report
from dataclasses import asdict

# Assuming src is in the Python path, for standalone script execution
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.models import AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures, NoiseReductionFilterType
from src.core.signal_processing import load_wav_file, remove_dc_offset, apply_butterworth_filter, apply_noise_reduction_filter
from src.core.feature_extraction import calculate_time_domain_features, calculate_fft_features
from src.diagnostics.mt_method import MTSpace

# --- Configuration ---
BASE_PATH = Path("bench/DCASE-2020-Task2-Dataset/pump")
TRAIN_NORMAL_DIR = BASE_PATH / "train"
TEST_DIR = BASE_PATH / "test"

ANALYSIS_CONFIG = AnalysisConfig(
    quantity=SignalQuantity.ACCEL,
    window=WindowFunction.HANNING,
    highpass_hz=10,
    lowpass_hz=None,
    filter_order=4,
    noise_reduction_type=NoiseReductionFilterType.NONE # Default to no noise reduction
)

# --- Utility Functions ---
def get_wav_files(directory: Path, pattern: str = "*.wav") -> List[Path]:
    """Recursively finds WAV files in a given directory."""
    if not directory.exists():
        print(f"Warning: Directory not found: {directory}")
        return []
    return list(directory.rglob(pattern))

def extract_features_from_file(file_path: str, config: AnalysisConfig) -> VibrationFeatures:
    """Helper function to run the modular pipeline and extract features."""
    fs_hz, data_normalized, _ = load_wav_file(file_path)
    
    # Processing pipeline
    data_processed = remove_dc_offset(data_normalized)
    data_processed = apply_butterworth_filter(
        data_processed, fs_hz, config.highpass_hz, config.lowpass_hz, config.filter_order
    )
    data_processed = apply_noise_reduction_filter(
        data_processed, fs_hz, config.noise_reduction_type,
        config.notch_freq_hz, config.notch_q_factor,
        config.band_stop_low_hz, config.band_stop_high_hz, config.band_stop_order
    )
    
    # Feature extraction
    time_features = calculate_time_domain_features(data_processed)
    _, _, power_bands = calculate_fft_features(data_processed, fs_hz, config.window)
    
    # Combine features
    vibration_features = VibrationFeatures(
        **asdict(time_features),
        power_low=power_bands['low'],
        power_mid=power_bands['mid'],
        power_high=power_bands['high']
    )
    return vibration_features

def train_unit_space(normal_data_paths: List[Path], config: AnalysisConfig) -> MTSpace:
    """Trains the MTSpace using normal data."""
    print("--- 単位空間の学習開始 ---")
    mt_space = MTSpace()
    
    for i, file_path in enumerate(normal_data_paths):
        try:
            features = extract_features_from_file(str(file_path), config)
            mt_space.add_normal_sample(features)
            print(f"  {i+1}/{len(normal_data_paths)}: '{file_path.name}' の特徴量を抽出しました。")
        except Exception as e:
            print(f"  Warning: ファイル '{file_path.name}' の特徴量抽出中にエラー: {e}")
            continue
            
    if len(mt_space.feature_vectors) == 0:
        raise ValueError("正常データのWAVファイルから特徴量を抽出できませんでした。")

    print(f"--- 単位空間の学習完了 (データ数: {len(mt_space.feature_vectors)}) ---")
    return mt_space

def evaluate_mt_method(test_data_paths: List[Path], unit_space: MTSpace, config: AnalysisConfig, anomaly_threshold: float = 3.0) -> pd.DataFrame:
    """Evaluates the MT method using test data."""
    print(f"\n--- MT法による評価開始 (閾値: {anomaly_threshold:.2f}) ---")
    results = []
    
    for i, file_path in enumerate(test_data_paths):
        label = "anomaly" if "anomaly" in file_path.name.lower() else "normal"
        try:
            features = extract_features_from_file(str(file_path), config)
            md = unit_space.calculate_md(features)
            
            prediction = "異常" if md is not None and md > anomaly_threshold else "正常"
            is_correct = (label == "normal" and prediction == "正常") or \
                         (label == "anomaly" and prediction == "異常")
            
            results.append({
                "file": file_path.name,
                "label": label,
                "mahalanobis_distance": md,
                "prediction": prediction,
                "is_correct": is_correct,
                "features": features.to_vector().tolist()
            })
            print(f"  {i+1}/{len(test_data_paths)}: '{file_path.name}' - ラベル: {label}, MD: {md:.2f} (予測: {prediction})")
            
        except Exception as e:
            print(f"  Warning: ファイル '{file_path.name}' の解析エラー: {e}")
            results.append({
                "file": file_path.name,
                "label": label,
                "mahalanobis_distance": np.nan,
                "prediction": "Error",
                "is_correct": False,
                "features": None
            })
            
    df_results = pd.DataFrame(results)

    print("\n--- 評価結果サマリー ---")
    valid_results = df_results.dropna(subset=['mahalanobis_distance'])
    correct_predictions = sum(1 for _, row in valid_results.iterrows() if row["is_correct"])
    accuracy = correct_predictions / len(valid_results) if not valid_results.empty else 0

    print(f"総テスト数: {len(df_results)}")
    print(f"有効なテスト数 (解析エラーを除く): {len(valid_results)}")
    print(f"正解数: {correct_predictions}")
    print(f"精度: {accuracy*100:.2f}%")
    
    print("\n詳細結果:")
    print(df_results.to_string())
    
    return df_results

def visualize_md_results(df_results: pd.DataFrame, anomaly_threshold: float, plot_filename: str, unit_space: MTSpace):
    """Generates and saves Plotly visualizations and a detailed HTML table of Mahalanobis Distance results."""
    
    df_valid_results = df_results.dropna(subset=['mahalanobis_distance'])
    html_content = "<h1>MT法 評価結果レポート</h1>"
    html_content += f"<p>使用した異常判定閾値: <b>{anomaly_threshold:.2f}</b></p>"
    html_content += f"<p>総テストサンプル数: <b>{len(df_results)}</b></p>"
    
    if not df_valid_results.empty:
        correct_predictions = sum(1 for _, row in df_valid_results.iterrows() if row["is_correct"])
        accuracy = correct_predictions / len(df_valid_results)
        html_content += f"<p>有効な解析サンプル数 (エラーを除く): <b>{len(df_valid_results)}</b></p>"
        html_content += f"<p>正しく判定されたサンプル数: <b>{correct_predictions}</b></p>"
        html_content += f"<p><b>総合精度: {accuracy*100:.2f}%</b></p>"
    else:
        html_content += "<p>有効な結果がないため、概要統計は計算できません。</p>"

    if not df_valid_results.empty:
        y_true = df_valid_results['label'].apply(lambda x: 0 if x == 'normal' else 1).values
        y_pred = df_valid_results['prediction'].apply(lambda x: 0 if x == '正常' else 1).values
        
        report = classification_report(y_true, y_pred, target_names=['normal', 'anomaly'], output_dict=True, zero_division=0)
        html_content += "<h2>分類性能指標</h2><ul>"
        html_content += f"<li><b>Normal Class:</b> Precision={report['normal']['precision']:.2f}, Recall={report['normal']['recall']:.2f}, F1-score={report['normal']['f1-score']:.2f}, Support={report['normal']['support']}</li>"
        html_content += f"<li><b>Anomaly Class:</b> Precision={report['anomaly']['precision']:.2f}, Recall={report['anomaly']['recall']:.2f}, F1-score={report['anomaly']['f1-score']:.2f}, Support={report['anomaly']['support']}</li>"
        html_content += f"<li><b>Accuracy:</b> {report['accuracy']:.2f}</li></ul>"
    else:
        html_content += "<h2>分類性能指標</h2><p>有効な結果がないため、分類性能指標は計算できません。</p>"

    if not df_valid_results.empty:
        fig = make_subplots(rows=2, cols=1, subplot_titles=("マハラノビス距離分布", "マハラノビス距離 (各ファイル)"))
        for label_val in df_valid_results['label'].unique():
            fig.add_trace(go.Histogram(x=df_valid_results[df_valid_results['label'] == label_val]['mahalanobis_distance'], 
                                   name=f'MD ({label_val})', marker_color='blue' if label_val == 'normal' else 'red',
                                   opacity=0.7, histnorm='density'), row=1, col=1)
        fig.update_layout(barmode='overlay')
        fig.add_trace(go.Scatter(x=df_valid_results.index, y=df_valid_results['mahalanobis_distance'], 
                                 mode='markers', marker=dict(color=['blue' if lbl == 'normal' else 'red' for lbl in df_valid_results['label']]),
                                 name='MD', text=[f"ファイル: {f}<br>ラベル: {l}<br>MD: {md:.2f}" 
                                       for f, l, md in zip(df_valid_results['file'], df_valid_results['label'], df_valid_results['mahalanobis_distance'])],
                                 hoverinfo='text'), row=2, col=1)
        fig.add_hline(y=anomaly_threshold, line_dash="dot", annotation_text="異常閾値", annotation_position="top right", line_color="orange", row=2, col=1)
        fig.update_layout(height=800, title_text="MT法 評価結果", showlegend=True)
        html_content += "<h2>可視化</h2>" + fig.to_html(full_html=False, include_plotlyjs='cdn')
    else:
        html_content += "<h2>可視化</h2><p>有効な結果がないため、可視化はスキップされました。</p>"
    
    html_content += "<h2>詳細結果テーブル</h2>"
    df_display = df_results.copy()
    if 'features' in df_display.columns:
        df_display['features'] = df_display['features'].apply(lambda x: str(np.round(x, 4).tolist()) if x is not None else "N/A")
    
    def style_row(row):
        base_style = 'text-align: left; padding: 8px; border: 1px solid #dddddd;'
        if not row['is_correct']:
            return ['background-color: #ffe6e6;' + base_style] * len(row)
        return [base_style] * len(row)
    
    styled_df = df_display.style.apply(style_row, axis=1)
    html_content += styled_df.to_html(index=False)
    
    with open(plot_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n評価結果のレポートを '{plot_filename}' に保存しました。")

def run_mt_evaluation_test():
    print("MT法 性能評価スクリプト")
    train_normal_files = get_wav_files(TRAIN_NORMAL_DIR, "normal_id_00_*.wav")
    test_normal_files = get_wav_files(TEST_DIR, "normal_id_00_*.wav")
    test_anomaly_files = get_wav_files(TEST_DIR, "anomaly_id_00_*.wav")

    if not train_normal_files:
        print(f"エラー: 学習用正常データが見つかりません。")
        return
    
    all_test_files = test_normal_files + test_anomaly_files
    if not all_test_files:
        print(f"エラー: テストデータが見つかりません。")
        return

    try:
        trained_unit_space = train_unit_space(train_normal_files, ANALYSIS_CONFIG)
    except Exception as e:
        print(f"単位空間学習中にエラー: {e}")
        return

    anomaly_threshold = 3.0
    df_results = evaluate_mt_method(all_test_files, trained_unit_space, ANALYSIS_CONFIG, anomaly_threshold)

    if not df_results.empty:
        visualize_md_results(df_results, anomaly_threshold, "mt_evaluation_results.html", trained_unit_space)

if __name__ == "__main__":
    run_mt_evaluation_test()