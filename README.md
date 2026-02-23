# NAMINORI: 振動解析Webアプリケーション

## プロジェクト概要

NAMINORIは、製造現場や事業運用に耐える品質を目指して開発された振動解析Webアプリケーションです。振動工学と信号処理の専門知識を組み込み、物理的に正しく、保守性が高く、説明可能な解析コードを提供します。将来的なAI・自動診断拡張を考慮した設計を特徴としています。

## 特徴

* **WAVファイル解析:** LPCM 16/24/32bit WAV形式の振動データをアップロードし解析。
* **信号処理:** DC成分除去、Butterworthフィルタ（LPF/HPF/BPF）、窓関数（Hanning/Flat Top）適用。
* **プラグイン式ノイズ除去フレームワーク:** ユーザー定義のカスタムノイズ除去アルゴリズムを動的にロード・適用可能。
* **ノイズ除去効果評価フレームワーク:** 適用されたノイズ除去フィルターの効果を客観的に評価し可視化。
* **特徴量抽出:** 時間領域（RMS, Peak, Kurtosisなど）、周波数領域（FFT, パワー寄与率）の特徴量を抽出。
* **データ品質評価:** クリッピング検出、S/N比推定、診断信頼度スコア算出。
* **MT法（マハラノビス・タグチ法）による異常診断:** 正常データからの逸脱度を評価。
* **インタラクティブな可視化:** Plotlyによる時間波形、FFTスペクトルのグラフ表示（ズーム、パン、周波数軸スケール切替、ピークアノテーション）。
* **解析条件の監査ログ:** 再現性を保証するため、すべての解析条件をJSON形式で保存。

## 技術スタック

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **Data Processing:** NumPy, SciPy (signal, fft)
* **Visualization:** Plotly
* **File Format:** WAV（LPCM 16/24/32bit）

## インストール方法

### 1. リポジトリのクローン

まず、このリポジトリをローカルにクローンします。

```bash
git clone https://github.com/your-username/naminori.git # 必要に応じてURLを修正
cd naminori
```

### 2. Python環境のセットアップ

Python 3.10以上がインストールされていることを確認してください。推奨として、仮想環境の利用を強くお勧めします。

```bash
# 仮想環境の作成 (初回のみ)
python -m venv venv

# 仮想環境のアクティベート
# Windowsの場合
.\venv\Scripts\activate
# macOS/Linuxの場合
source venv/bin/activate
```

### 3. 依存ライブラリのインストール

仮想環境をアクティベートした後、必要なPythonライブラリをインストールします。

```bash
pip install -r requirements.txt
```

## 使い方

### Webアプリケーションの実行

Streamlitアプリケーションを起動し、Webブラウザでアクセスします。

```bash
python -m streamlit run src/app.py
```

上記のコマンドを実行すると、ターミナルに以下のようなURLが表示されます。

```bash
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

表示された `Local URL` をWebブラウザで開いてください。

### アプリケーションの操作

1. **WAVファイルのアップロード:** 画面左上のファイルアップローダーからWAVファイルをアップロードします。
2. **解析設定:** サイドバー（左側）で、物理量種別、窓関数、フィルタのカットオフ周波数、フィルタ次数などの解析条件を設定できます。
3. **解析結果の確認:** 時間領域・周波数領域の特徴量、データ品質、信頼度スコア、およびグラフが表示されます。
4. **グラフの操作:** Plotlyグラフはインタラクティブであり、拡大・縮小、パン、データポイントの情報の確認が可能です。FFTスペクトルでは周波数軸の線形/対数スケール切り替えやピークのアノテーションが表示されます。

## ドキュメント

より詳細な情報や、カスタムプラグインの開発方法については、[ドキュメント](docs/index.md)を参照してください。

## テスト

開発中にユニットテストを実行するには、以下のコマンドを使用します。

```bash
# 仮想環境がアクティブであることを確認
python -m pytest tests/
```

## プロジェクト構造

```text
vibration-ai/
├── src/
│   ├── app.py                    # Streamlit メインアプリ
│   ├── core/
│   │   ├── models.py             # データモデル (dataclass, Enum)
│   │   ├── signal_processing.py  # 信号処理コア
│   │   ├── feature_extraction.py # 特徴量抽出
│   │   ├── quality_check.py      # データ品質評価
│   │   ├── plugins.py            # プラグインアーキテクチャ定義
│   │   └── evaluation.py         # ノイズ除去評価フレームワーク
│   ├── diagnostics/
│   │   └── mt_method.py          # MT法実装
│   ├── plugins/                  # カスタムプラグイン
│   │   └── noise_reduction/      # ノイズ除去プラグイン
│   └── utils/
│       └── audit_log.py          # 監査ログ管理
├── data/
│   ├── samples/                  # テスト用WAVファイル (Git管理外)
│   └── unit_space/               # MT法単位空間データ (Git管理外)
├── docs/                         # ドキュメントルート
│   ├── index.md                  # ドキュメントインデックス
│   ├── user_guide.md             # ユーザーガイド
│   ├── plugins/                  # プラグイン関連ドキュメント
│   │   └── index.md
│   │   └── custom_plugins.md
│   ├── evaluation.md             # 評価フレームワークドキュメント
│   └── architecture.md           # アーキテクチャ概要
├── tests/
│   ├── test_signal_processing.py
│   ├── test_feature_extraction.py
│   ├── test_quality_check.py
│   ├── test_mt_method.py
│   ├── test_plugins.py           # プラグインマネージャーテスト
│   └── test_evaluation.py        # 評価フレームワークテスト
├── requirements.txt              # Pythonの依存ライブラリ
├── README.md                     # このファイル
└── GEMINI.md                     # プロジェクト開発規約
```

## 貢献

貢献は大歓迎です。バグ報告、機能リクエスト、プルリクエストなど、お気軽にお寄せください。
