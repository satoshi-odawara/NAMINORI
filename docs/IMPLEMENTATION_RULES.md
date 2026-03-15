# 振動解析Webアプリ実装規約（実装・運用ガイドライン）

本ドキュメントは、`GEMINI.md` 技術憲章に基づき、具体的なコーディング、信号処理アルゴリズム、およびツールの運用ルールを定義する。

---

## 1. 技術スタック

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **Data Processing:** NumPy, SciPy (signal, fft)
* **Visualization:** Plotly
* **File Format:** WAV（LPCM 16/24/32bit）

---

## 2. 共通コーディング規約

### 2.1 物理量・単位の明示

* すべての数値は **物理量 + 単位** を前提とする。
* 変数名には必ず単位を含める。
  * 例：`accel_ms2`, `vel_mms`, `disp_um`, `freq_hz`, `time_s`

### 2.2 物理量種別の明示（必須）

加速度・速度・変位を混在させないため、解析関数には必ず物理量種別を明示する。

```python
from enum import Enum

class SignalQuantity(Enum):
    ACCEL = "accel"        # m/s^2
    VELOCITY = "velocity" # mm/s
    DISPLACEMENT = "disp" # μm
```

---

## 3. 信号処理基本ルール

### 3.1 前処理（Preprocessing）

* **DC成分除去:** 必須（物理的妥当性の確保のため自動適用）。
* **フィルタリング:** HPF / LPF は **デフォルトOFF**。ユーザーによる明示的なON切替が必要。
* **窓関数:** デフォルト Hanning。振幅補正係数（ACF）を必ず適用すること。

### 3.2 フィルタ規約

* **Butterworth フィルタ:** LPF / HPF / BPF を標準採用。
* **ログ保存:** フィルタ種別、カットオフ、次数を必ず記録。

---

## 4. データ品質・信頼性評価

### 4.1 診断信頼度（Confidence Score）

以下の要素に基づき 0–100% で算出する：
1. **飽和回避 (Clipping):** |x| ≥ 0.99 の割合。
2. **ノイズ耐性 (SNR):** 背景ノイズに対する信号比。
3. **データ量 (Length):** 解析に使用した信号の長さ（10秒以上推奨）。

---

## 5. MT法（マハラノビス・タグチ）設計

### 5.1 特徴量設計とベクトル順序（厳守）

`VibrationFeatures.to_vector()` は以下の **15次元** の順序を厳守すること。

1. `rms` (有効値)
2. `peak` (最大振幅)
3. `kurtosis` (尖度)
4. `skewness` (歪度)
5. `crest_factor`
6. `shape_factor`
7. `power_low` (低域パワー寄与率)
8. `power_mid` (中域パワー寄与率)
9. `power_high` (高域パワー寄与率)
10. `spectral_centroid` (重心周波数)
11. `spectral_spread` (周波数分散)
12. `spectral_entropy` (スペクトルエントロピー)
13. `overall_level` (全帯域Overall)
14. `overall_low` (低域Overall)
15. `overall_high` (高域Overall)

---

## 6. 運用・保守ガイドライン

### 6.1 `replace` ツール使用規約

* **厳密な一致:** `old_string` は改行・空白含めターゲットと完全に一致させること。
* **大規模変更:** リファクタリング時は `read_file` -> `write_file` 方式を推奨。

### 6.2 Streamlit変数のスコープ管理（NameError防止）

* **前方定義:** 物理単位 (`unit`) 等の共通変数は、それを使用する最初のブロックよりも前で定義すること。
* **独立性:** 俯瞰解析（サマリー）は、詳細表示セクションの変数に依存しないように実装する。

### 6.3 検証自動化 (CI/CD)

GitHub Actions (`.github/workflows/ci.yml`) にて以下を自動実行する。
```yaml
- run: pip install -r requirements.txt
- run: pytest --cov=src --cov-report=xml
- run: ruff check src/
- run: streamlit run src/app.py --server.headless true
```

---

## 7. 実装例（規約準拠）

### 7.1 悪い実装例 ❌
```python
def analyze(file):
    data, fs = wavfile.read(file) # 物理量・単位が不明
    fft = np.fft.fft(data)        # 窓関数・補正がない
    rms = np.sqrt(np.mean(data**2)) # DC成分除去がない
    return rms, fft
```

### 7.2 良い実装例 ✅
```python
def analyze_vibration_data(data_raw: np.ndarray, fs_hz: float, quantity: SignalQuantity):
    # 1. 前処理 (DC除去)
    data_dc_removed = remove_dc_offset(data_raw)
    
    # 2. 特徴量抽出 (物理量を考慮)
    t_feat = calculate_time_domain_features(data_dc_removed)
    f_hz, mags, f_feat = calculate_fft_features(data_dc_removed, fs_hz, WindowFunction.HANNING)
    
    # 3. 物理量単位の取得
    unit = quantity.unit_str
    
    return VibrationFeatures(**asdict(t_feat), **f_feat), unit
```
