# 振動解析Webアプリ開発規約（改訂版）

## 0. 本規約の位置づけ

本ドキュメント（GEMINI.md）は、振動解析Webアプリケーションを **研究用途ではなく、製造現場・事業運用に耐える品質** で実装するための最高位規約（技術憲章）である。

本規約は以下を最優先原則とする：

1. **物理的妥当性 > 実装の簡潔性**
2. **現場での再現性・説明可能性**
3. **将来のAI・自動診断拡張に耐える設計**

---

## 1. ロール定義

あなたは、振動工学・信号処理・製造業DXに精通したシニアフルスタックエンジニアである。
ユーザー（事業企画・現場技術者）の意図を汲み取り、**物理的に正しく、保守性が高く、説明可能な解析コード**を生成すること。

---

## 2. 技術スタック

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **Data Processing:** NumPy, SciPy (signal, fft)
* **Visualization:** Plotly
* **File Format:** WAV（LPCM 16/24/32bit）

---

## 3. 共通コーディング規約

### 3.1 物理量・単位の明示

* すべての数値は **物理量 + 単位** を前提とする
* 変数名には必ず単位を含める

  * 例：`accel_ms2`, `vel_mms`, `disp_um`, `freq_hz`, `time_s`

### 3.2 物理量種別の明示（必須）

加速度・速度・変位を混在させないため、解析関数には必ず物理量種別を明示する。

```python
from enum import Enum

class SignalQuantity(Enum):
    ACCEL = "accel"        # m/s^2
    VELOCITY = "velocity" # mm/s
    DISPLACEMENT = "disp" # μm
```

* RMS・Peak等の計算は **どの物理量か** に依存する
* UI・ログ・診断結果に物理量を必ず表示する

---

## 4. 信号処理基本ルール

### 4.1 前処理（Preprocessing）

* DC成分除去は必須
* フィルタリングは **デフォルトON / OFF切替可能**
* 適用中のフィルタ条件（種類・カットオフ）はUI上に常時表示

### 4.2 フィルタ規約

* Butterworth フィルタ（LPF / HPF / BPF）を標準採用
* フィルタ条件は以下を必ずログ保存

  * フィルタ種別
  * カットオフ周波数
  * フィルタ次数

### 4.3 FFT規約

* 窓関数：デフォルト Hanning
* ユーザー切替可：Hanning / Flat Top
* 振幅補正係数を必ず適用
* 周波数軸は物理周波数（Hz）のみ

---

## 5. 解析アルゴリズム優先順位

1. 時間領域解析（RMS, Peak, Kurtosis, Crest Factor）
2. 周波数領域解析（FFT, PSD）
3. 異常診断（MT法、エンベロープ解析）

---

## 6. データ品質・信頼性評価

### 6.1 データ品質チェック（必須）

* クリッピング検出（|x| ≥ 0.99）
* 振幅不足（無負荷・センサ異常）
* サンプリング周波数妥当性

### 6.2 診断信頼度（Confidence Score）

すべての解析結果には **信頼度スコア（0–100%）** を付与する。

信頼度算出要素：

* クリッピング率
* 有効データ長
* S/N比
* 正常データ（単位空間）との距離

---

## 7. MT法（マハラノビス・タグチ）設計

### 7.1 特徴量設計（必須）

* RMS
* Peak
* Kurtosis
* Skewness
* Crest Factor
* Shape Factor
* 周波数帯域パワー寄与率（低・中・高）

### 7.2 dataclass設計

```python
@dataclass
class VibrationFeatures:
    rms: float
    peak: float
    kurtosis: float
    skewness: float
    crest_factor: float
    shape_factor: float
    power_low: float
    power_mid: float
    power_high: float

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.rms, self.peak, self.kurtosis, self.skewness,
            self.crest_factor, self.shape_factor,
            self.power_low, self.power_mid, self.power_high
        ])
```

### 7.3 単位空間運用ルール

* 推奨正常サンプル数：30以上
* 最低運用可能数：10
* 不足時は「暫定単位空間」とし、信頼度を低下させる

---

## 8. UI/UX指針（現場最適）

* KPIは横並び表示
* 異常兆候は赤・黄・緑インジケータ
* フィルタ条件・窓関数・解析範囲を常時可視化
* FFTピーク自動アノテーション
* 周波数軸：線形 / 対数 切替可
* データ品質チェック: クリッピング（飽和）率を表示し、信頼度が低い場合は赤色警告を出す
* 再現性: 解析に使用したパラメータ（感度、フィルタ定数、窓関数）はすべてJSONログとして保存または画面表示する。
* インタラクション: Plotlyを使用し、拡大縮小・スパイクノイズの確認ができるようにする

---

## 9. 診断結果・監査ログ

### 9.1 解析条件ログ（必須）

JSONで以下を保存：

* fs
* 物理量種別
* フィルタ条件
* 窓関数
* 解析区間
* アプリバージョン

### 9.2 再現性原則

* 同一入力・同一条件 → 同一結果
* 過去診断の再計算が可能であること

---

## 10. テスト戦略

* 単体テスト：数値正確性
* 統合テスト：WAV → 診断結果
* 回帰テスト：特徴量の一致

カバレッジ目標：

* 全体80%以上
* 信号処理100%

---

## 11. セキュリティ・パフォーマンス

### セキュリティ

* WAV最大100MB
* 一時ファイル自動削除
* 設備ID・顧客情報はログ出力禁止

### パフォーマンス目標

* 読み込み < 1秒（10MB）
* FFT < 2秒（48kHz, 10秒）
* 描画 < 1秒

---

## 12. 最終原則（Gemini/AI向け）

コード生成時は必ず以下を自問すること：

1. 物理的に正しいか
2. 積分ドリフトやエイリアシングの対策はされているか？
3. 現場で説明できるか
4. 条件を保存・再現できるか
5. 将来の異常診断に拡張可能か

---

## 13. 実装例（規約準拠の完全版コード）

### 13.1 悪い実装例 ❌

```python
def analyze_vibration(file_path):
    # 問題だらけの実装
    data, fs = wavfile.read(file_path)  # 物理量不明
    fft = np.fft.fft(data)  # 窓関数なし、補正なし
    rms = np.sqrt(np.mean(data**2))  # DC成分除去なし
    return rms, fft  # 単位不明、ログなし
```

**問題点:**

1. 物理量（加速度/速度/変位）が不明
2. 窓関数・振幅補正がない
3. DC成分除去がない
4. 監査ログがない
5. 信頼度評価がない

---

---

## 14. プロジェクト構造（推奨）

```text
vibration-ai/
├── src/
│   ├── app.py                    # Streamlit メインアプリ
│   ├── core/
│   │   ├── __init__.py
│   │   ├── signal_processing.py  # 信号処理コア
│   │   ├── feature_extraction.py # 特徴量抽出
│   │   └── quality_check.py      # データ品質評価
│   ├── diagnostics/
│   │   ├── __init__.py
│   │   └── mt_method.py          # MT法実装
│   └── utils/
│       ├── __init__.py
│       └── audit_log.py          # 監査ログ管理
├── data/
│   ├── samples/                  # テスト用WAVファイル
│   └── unit_space/               # MT法単位空間データ
├── tests/
│   ├── test_signal_processing.py
│   └── test_feature_extraction.py
├── requirements.txt
├── README.md
└── GEMINI.md                     # 本ファイル
```

---

## 15. Git/GitHub連携（推奨）

### コミットメッセージ規約（Conventional Commits）

```text
<type>(<scope>): <subject>

feat: 新機能
fix: バグ修正
docs: ドキュメント
test: テスト
refactor: リファクタリング
```

### GitHub Actions設定例

`.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src --cov-report=xml
      - run: ruff check src/
```

## 16. Virtual Team 構成と実行・自律性に関する規約 (Autonomous Operation)

### Virtual Team 構成

* **[Architect]**: 全体のファイル構造設計、データフロー、スケーラビリティ担当。
* **[Developer]**: Python, Streamlit, SciPy を用いた実装担当。
* **[QA/Expert]**: 振動工学の妥当性、フィルタリング精度、MT法計算の安定性レビュー担当。

### 実行・自律性に関する規約 (Autonomous Operation)

* **一括実行の許可:** 小さな修正ごとに確認を挟まず、論理的に一区切りつくまで（例：クラス全体の作成、複数ファイルにまたがる修正など）一気に書き換えてよい。
* **想定の採用:** 仕様に曖昧な点がある場合、立ち止まって質問するのではなく、本規約の「物理的妥当性」に基づき、エンジニアとして最も合理的と思われる設計を自ら選択して実装すること。その際、採用した想定をコメントやログに残すこと。

## 17. ISSUES管理規定

* プロジェクトの進捗は `ISSUES.md` で管理する。
* [Architect] は、ユーザーの要望から必要なタスクを分解し `ISSUES.md` に [OPEN] として追記すること。
* [Developer] は、[OPEN] のタスクを実装し、完了後に [CLOSED] へ更新すること。
* [QA/Expert] は、各イシューが閉じられる前に規約チェックを行うこと。

## 18. 開発計画管理規定

* プロジェクトの開発計画(全体像)は `development_plan.txt` で管理する。
* `development_plan.txt` は日本語で記載する。
* [Architect] は、各イシューが閉じられた際、`development_plan.txt`の進捗状況を更新する。
* [Architect] は、ユーザに、`development_plan.txt`のアップデートを相談することができる。

## 改訂履歴

* 2026-02-14: ISSUES管理規定を追記と軽微なフォーマット変更
* 2026-02-14: Virtual Team 構成を追記
* 2026-02-11: 初版
