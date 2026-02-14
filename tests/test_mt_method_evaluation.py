import os
import glob
from pathlib import Path
import numpy as np
from typing import List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import classification_report

# Assuming src is in the Python path, for standalone script execution
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from core.signal_processing import analyze_vibration_full, AnalysisConfig, SignalQuantity, WindowFunction, VibrationFeatures
from diagnostics.mt_method import UnitSpace

# --- Configuration ---
BASE_PATH = Path("bench/DCASE-2020-Task2-Dataset/pump")
TRAIN_NORMAL_DIR = BASE_PATH / "train"
TEST_DIR = BASE_PATH / "test"

# Analysis configuration (adjust as needed based on sensor type/domain knowledge)
# Assuming 'accel' for pump data, but this might need to be clarified with user or data spec.
ANALYSIS_CONFIG = AnalysisConfig(
    quantity=SignalQuantity.ACCEL,
    window=WindowFunction.HANNING,
    highpass_hz=10, # Example: High-pass filter at 10 Hz to remove low-frequency noise/drift
    lowpass_hz=None, # No low-pass filter by default, or set based on Nyquist/application
    filter_order=4
)

# --- Utility Functions ---
def get_wav_files(directory: Path, pattern: str = "*.wav") -> List[Path]:
    """Recursively finds WAV files in a given directory."""
    # Add a check for directory existence
    if not directory.exists():
        print(f"Warning: Directory not found: {directory}")
        return []
    return list(directory.rglob(pattern))

def train_unit_space(normal_data_paths: List[Path], config: AnalysisConfig) -> UnitSpace:
    """Trains the UnitSpace using normal data."""
    print(f"--- 単位空間の学習開始 ---") # Changed to print for standalone script
    normal_feature_vectors: List[np.ndarray] = []
    
    for i, file_path in enumerate(normal_data_paths):
        try:
            # Ensure a fresh config is used for each analysis, avoid modifying shared object
            current_config = AnalysisConfig(
                quantity=config.quantity,
                window=config.window,
                highpass_hz=config.highpass_hz,
                lowpass_hz=config.lowpass_hz,
                filter_order=config.filter_order
            )
            result = analyze_vibration_full(str(file_path), current_config)
            normal_feature_vectors.append(result.features.to_vector())
            print(f"  {i+1}/{len(normal_data_paths)}: '{file_path.name}' の特徴量を抽出しました。") # Changed to print
        except Exception as e:
            print(f"  Warning: ファイル '{file_path.name}' の特徴量抽出中にエラー: {e}") # Changed to print
            continue
            
    if not normal_feature_vectors:
        raise ValueError("正常データのWAVファイルから特徴量を抽出できませんでした。")

    unit_space = UnitSpace()
    unit_space.train(normal_feature_vectors)
    print(f"--- 単位空間の学習完了 (データ数: {len(normal_feature_vectors)}) ---") # Changed to print
    return unit_space

def evaluate_mt_method(test_data_paths: List[Path], unit_space: UnitSpace, config: AnalysisConfig, anomaly_threshold: float = 3.0) -> pd.DataFrame:
    """Evaluates the MT method using test data."""
    print(f"\n--- MT法による評価開始 (閾値: {anomaly_threshold:.2f}) ---")
    results = []
    
    for i, file_path in enumerate(test_data_paths):
        # Determine label based on filename pattern for DCASE 2020 dataset
        label = "anomaly" if "anomaly" in file_path.name.lower() else "normal"
        try:
            # Ensure a fresh config for each analysis
            current_config = AnalysisConfig(
                quantity=config.quantity,
                window=config.window,
                highpass_hz=config.highpass_hz,
                lowpass_hz=config.lowpass_hz,
                filter_order=config.filter_order,
                mt_unit_space=unit_space # Pass the trained unit space
            )
            result = analyze_vibration_full(str(file_path), current_config)
            
            md = result.mahalanobis_distance
            prediction = "異常" if md is not None and md > anomaly_threshold else "正常"
            is_correct = (label == "normal" and prediction == "正常") or \
                         (label == "anomaly" and prediction == "異常")
            
            results.append({
                "file": file_path.name,
                "label": label,
                "mahalanobis_distance": md, # Store as float for plotting
                "prediction": prediction,
                "is_correct": is_correct,
                "features": result.features.to_vector().tolist() # Store feature vector as a list
            })
            print(f"  {i+1}/{len(test_data_paths)}: '{file_path.name}' - ラベル: {label}, MD: {md:.2f} (予測: {prediction})")
            
        except ValueError as e:
            print(f"  Warning: ファイル '{file_path.name}' の解析エラー: {e}")
            results.append({
                "file": file_path.name,
                "label": label,
                "mahalanobis_distance": np.nan, # Store NaN for errors
                "prediction": "Error",
                "is_correct": False,
                "features": None
            })
        except Exception as e:
            print(f"  Warning: ファイル '{file_path.name}' の予期せぬ解析エラー: {e}")
            results.append({
                "file": file_path.name,
                "label": label,
                "mahalanobis_distance": np.nan, # Store NaN for errors
                "prediction": "Error",
                "is_correct": False,
                "features": None
            })
            
    df_results = pd.DataFrame(results)

    print("\n--- 評価結果サマリー ---")
    # Filter out rows with NaN Mahalanobis Distance for accuracy calculation
    valid_results = df_results.dropna(subset=['mahalanobis_distance'])
    correct_predictions = sum(1 for _, row in valid_results.iterrows() if row["is_correct"])
    accuracy = correct_predictions / len(valid_results) if not valid_results.empty else 0

    print(f"総テスト数: {len(df_results)}")
    print(f"有効なテスト数 (解析エラーを除く): {len(valid_results)}")
    print(f"正解数: {correct_predictions}")
    print(f"精度: {accuracy*100:.2f}%")
    
    print("\n詳細結果:")
    print(df_results.to_string())
    
    return df_results # Return DataFrame

def visualize_md_results(df_results: pd.DataFrame, anomaly_threshold: float, plot_filename: str, unit_space: UnitSpace):
    """Generates and saves Plotly visualizations and a detailed HTML table of Mahalanobis Distance results."""
    
    # Filter out rows with NaN Mahalanobis Distance for plotting and metrics
    df_valid_results = df_results.dropna(subset=['mahalanobis_distance'])

    html_content = ""

    # Add descriptive text and summary
    html_content += "<h1>MT法 評価結果レポート</h1>"
    html_content += "<p>このレポートは、MT法による異常診断の性能評価結果を示します。</p>"
    html_content += f"<p>使用した異常判定閾値: <b>{anomaly_threshold:.2f}</b></p>"
    html_content += f"<p>総テストサンプル数: <b>{len(df_results)}</b></p>"
    
    # Calculate and display general metrics (accuracy from previous summary)
    if not df_valid_results.empty:
        correct_predictions = sum(1 for _, row in df_valid_results.iterrows() if row["is_correct"])
        accuracy = correct_predictions / len(df_valid_results) if not df_valid_results.empty else 0
        html_content += f"<p>有効な解析サンプル数 (エラーを除く): <b>{len(df_valid_results)}</b></p>"
        html_content += f"<p>正しく判定されたサンプル数: <b>{correct_predictions}</b></p>"
        html_content += f"<p><b>総合精度: {accuracy*100:.2f}%</b></p>"
    else:
        html_content += "<p>有効な結果がないため、概要統計は計算できません。</p>"


    # Calculate classification metrics
    if not df_valid_results.empty:
        y_true = df_valid_results['label'].apply(lambda x: 0 if x == 'normal' else 1).values
        y_pred = df_valid_results['prediction'].apply(lambda x: 0 if x == '正常' else 1).values # Assuming '正常' is normal, '異常' is anomaly
        
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=['normal', 'anomaly'], output_dict=True, zero_division=0)

        html_content += "<h2>分類性能指標</h2>"
        html_content += "<p>異常検知の性能を評価するための指標です。</p>"
        
        html_content += "<ul>"
        html_content += f"<li><b>Normal Class:</b> Precision={report['normal']['precision']:.2f}, Recall={report['normal']['recall']:.2f}, F1-score={report['normal']['f1-score']:.2f}, Support={report['normal']['support']}</li>"
        html_content += f"<li><b>Anomaly Class:</b> Precision={report['anomaly']['precision']:.2f}, Recall={report['anomaly']['recall']:.2f}, F1-score={report['anomaly']['f1-score']:.2f}, Support={report['anomaly']['support']}</li>"
        html_content += f"<li><b>Accuracy:</b> {report['accuracy']:.2f}</li>"
        html_content += f"<li><b>Macro Avg F1-score:</b> {report['macro avg']['f1-score']:.2f}</li>"
        html_content += f"<li><b>Weighted Avg F1-score:</b> {report['weighted avg']['f1-score']:.2f}</li>"
        html_content += "</ul>"
    else:
        html_content += "<h2>分類性能指標</h2><p>有効な結果がないため、分類性能指標は計算できません。</p>"


    # --- Plotly Visualizations ---
    if not df_valid_results.empty: # Use df_valid_results for plotting
        # Create subplots
        fig = make_subplots(rows=2, cols=1, 
                            subplot_titles=("マハラノビス距離分布 (ヒストグラム)", "マハラノビス距離 (各ファイル)"))

        # Plot 1: Histogram/Density Plot
        for label_val in df_valid_results['label'].unique():
            fig.add_trace(go.Histogram(x=df_valid_results[df_valid_results['label'] == label_val]['mahalanobis_distance'], 
                                   name=f'MD ({label_val})', 
                                   marker_color='blue' if label_val == 'normal' else 'red',
                                   opacity=0.7, histnorm='density'), row=1, col=1)
        
        fig.update_xaxes(title_text="マハラノビス距離", row=1, col=1)
        fig.update_yaxes(title_text="密度", row=1, col=1)
        fig.update_layout(barmode='overlay')

        # Plot 2: Scatter Plot with Threshold
        fig.add_trace(go.Scatter(x=df_valid_results.index, y=df_valid_results['mahalanobis_distance'], 
                                 mode='markers', 
                                 marker=dict(color=['blue' if lbl == 'normal' else 'red' for lbl in df_valid_results['label']]),
                                 name='MD (各ファイル)',
                                 text=[f"ファイル: {f}<br>ラベル: {l}<br>MD: {md:.2f}<br>予測: {p}<br>正解: {c}" 
                                       for f, l, md, p, c in zip(df_valid_results['file'], df_valid_results['label'], df_valid_results['mahalanobis_distance'], df_valid_results['prediction'], df_valid_results['is_correct'])],
                                 hoverinfo='text'), row=2, col=1)
        
        # Add anomaly threshold line
        fig.add_hline(y=anomaly_threshold, line_dash="dot", 
                      annotation_text=f"異常閾値 ({anomaly_threshold:.2f})", 
                      annotation_position="top right", 
                      line_color="orange", row=2, col=1)

        fig.update_xaxes(title_text="ファイルインデックス", row=2, col=1)
        fig.update_yaxes(title_text="マハラノビス距離", row=2, col=1)
        
        fig.update_layout(height=800, title_text="MT法 評価結果", showlegend=True)
        
        # Convert Plotly figure to HTML div string
        plotly_div = fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_content += "<h2>可視化</h2>"
        html_content += plotly_div
    else:
        html_content += "<h2>可視化</h2><p>有効な結果がないため、可視化はスキップされました。</p>"

    # --- Detailed HTML Table ---
    html_content += "<h2>詳細結果テーブル</h2>"
    html_content += "<p>各テストサンプルの詳細な評価結果です。誤判定された行は背景が色付けされます。</p>"

    # Convert features (list of floats) to string for display in table
    df_results_display = df_results.copy()
    # Check if 'features' column exists before applying lambda
    if 'features' in df_results_display.columns:
        df_results_display['features'] = df_results_display['features'].apply(lambda x: str(np.round(x, 4).tolist()) if x is not None else "N/A")

    def style_row(row):
        style = []
        # Base style for all cells
        base_style = 'text-align: left; padding: 8px; border: 1px solid #dddddd;'
        for _ in range(len(row)):
            style.append(base_style)
        
        if not row['is_correct']:
            # Apply specific background to the entire row if misclassified
            style = ['background-color: #ffe6e6;' + s for s in style] # Light red for incorrect
        elif row['label'] == 'anomaly' and row['prediction'] == '異常':
            # Correctly identified anomaly - potentially green
            style = ['background-color: #e6ffe6;' + s for s in style] # Light green
        elif row['label'] == 'normal' and row['prediction'] == '正常':
            # Correctly identified normal - potentially light blue
            style = ['background-color: #e6f7ff;' + s for s in style] # Light blue

        return style

    # Apply styling to the DataFrame
    # Ensure all columns are handled by the styling
    styled_df = df_results_display.style.apply(style_row, axis=1)
    
    # Custom CSS for the table (can be embedded or linked)
    table_css = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 14px;
        }
        th, td {
            text-align: left;
            padding: 8px;
            border: 1px solid #dddddd;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
    </style>
    """
    
    html_content += table_css
    html_content += styled_df.to_html(index=False) # Convert DataFrame to HTML table

    # --- Save to file ---
    with open(plot_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\n評価結果のレポートを '{plot_filename}' に保存しました。")


def run_mt_evaluation_test():
    print("MT法 性能評価スクリプト")
    print(f"データパス: {BASE_PATH}")

    # 1. Get file paths
    train_normal_files = get_wav_files(TRAIN_NORMAL_DIR, "normal_id_00_*.wav")
    test_normal_files = get_wav_files(TEST_DIR, "normal_id_00_*.wav")
    test_anomaly_files = get_wav_files(TEST_DIR, "anomaly_id_00_*.wav")

    if not train_normal_files:
        print(f"エラー: 学習用正常データが見つかりません。パス: {TRAIN_NORMAL_DIR}")
        return
    
    print(f"学習用正常データファイル数: {len(train_normal_files)}")
    print(f"テスト用正常データファイル数: {len(test_normal_files)}")
    print(f"テスト用異常データファイル数: {len(test_anomaly_files)}")

    # Combine test files
    all_test_files = test_normal_files + test_anomaly_files
    if not all_test_files:
        print(f"エラー: テストデータが見つかりません。パス: {TEST_DIR}")
        return

    # 2. Train Unit Space
    try:
        trained_unit_space = train_unit_space(train_normal_files, ANALYSIS_CONFIG)
    except ValueError as e:
        print(f"単位空間学習エラー: {e}")
        return
    except Exception as e:
        print(f"単位空間学習中に予期せぬエラー: {e}")
        return

    # 3. Evaluate MT Method
    anomaly_threshold = 3.0 # Example threshold, can be configured
    df_results = evaluate_mt_method(all_test_files, trained_unit_space, ANALYSIS_CONFIG, anomaly_threshold)

    # 4. Visualize Results
    if not df_results.empty:
        # Pass the trained_unit_space to visualize_md_results for feature vector comparison
        visualize_md_results(df_results, anomaly_threshold, "mt_evaluation_results.html", trained_unit_space)
    else:
        print("Warning: No results to visualize.")

if __name__ == "__main__":
    run_mt_evaluation_test()