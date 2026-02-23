# image-diff-poc

加工前後画像の差分抽出 PoC（Proof of Concept）

## セットアップ

> **注意**: 以下のコマンドはすべて **プロジェクトルート（`image-diff-poc/`）から** 実行してください。

```bash
# 1. 仮想環境を作成・有効化
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. 依存ライブラリをインストール
pip install -r requirements.txt
```

## PoC デモ（最短手順）

```bash
source .venv/bin/activate

# ① ダミー画像生成（初回のみ）
python generate_dummies.py

# ② CLI で差分抽出 → ハイライト画像 + CSV 生成
python -m image_diff \
    --before samples/before.png \
    --after  samples/after.png  \
    --output output/
```

出力例:
```
[WARNING] SSIM=0.8522 < ssim_min=0.9800. 大きな構造変化が検出されました。
❌  verdict    : FAIL
   diff_score : 0.532123  (53.21%)
   ssim_score : 0.852193

Output files:
  highlight  : output/diff_highlight.png
  report     : output/report.csv
```

| ファイル | 内容 |
|---|---|
| `output/diff_highlight.png` | 差分領域に赤矩形を描画した after 画像 |
| `output/report.csv` | verdict / diff_score / ssim_score / 理由 |

### CLI オプション一覧

```bash
python -m image_diff --help
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--before PATH` | 必須 | 加工前画像 |
| `--after  PATH` | 必須 | 加工後画像 |
| `--output DIR`  | `output` | 出力先ディレクトリ |
| `--thresh N`    | `10` | 差分 2 値化の閾値（0–255） |
| `--blur N`      | `5` | GaussianBlur カーネルサイズ（奇数） |
| `--ssim-min F`  | `0.98` | SSIM 警告閾値 |
| `--full`        | なし | アルファ破壊 + 見切れチェックも実行 |

### `--full` モード（拡張チェック）

```bash
python -m image_diff \
    --before samples/before.png \
    --after  samples/after.png  \
    --output output/ --full
```

出力例:
```
--- Full Mode: 拡張チェック ---
⚠️  alpha_check:
   destroyed_count : 35,731 px
   blob_count      : 2
   ratio           : 0.5321 (53.21%)
   blob label=1 area=20480 bbox=(472, 0, 40, 512)   ← 右端見切れ
   blob label=2 area=15251 bbox=(30, 30, 151, 101)   ← 追加要素
⚠️  clipping_check:
   edge_touch_ratio=0.0814, has_linear_edge=True, → 端付近に差分あり（要確認）
```

### 現場調整: `--thresh` の変更

| 状況 | 推奨値 |
|---|---|
| 圧縮ノイズが多い JPEG 素材 | `--thresh 20`〜`30` |
| ピクセル完全一致を期待する PNG | `--thresh 1`〜`5` |
| デフォルト（バランス） | `--thresh 10`（デフォルト） |

### 終了コード

| コード | 意味 |
|---|---|
| `0` | PASS または WARN（明確な差分なし） |
| `1` | FAIL（diff_score ≥ 1.0%） |
| `2` | 入力エラー（ファイル不存在など） |

> **パス注意**: `ModuleNotFoundError` が出た場合はカレントディレクトリがズレています。
> `cd /path/to/image-diff-poc` でルートに戻ってから再実行してください。

## 実行例（個別スクリプト）

### 環境確認

```bash
python test_env.py
```

出力例:
```
cv2 version: 4.x.x
Saved: output/test.png
```

`output/test.png` に 256×256 の黒画像（BGRA）が保存されます。

### ダミー画像生成

```bash
python generate_dummies.py
```

出力例:
```
Saved: samples/before.png
Saved: samples/after.png

Done. Dummy images generated in samples/
```

| ファイル | 内容 |
|---|---|
| `samples/before.png` | 透過背景（α=0）+ 中央に赤円（α=255） |
| `samples/after.png` | before に青矩形を追加 + 右端 40px 見切れ（白塗り） |

## 基礎知識

### NumPy 配列と画像の対応

OpenCV が扱う画像は NumPy の `ndarray` です。

```python
import cv2
import numpy as np

img = cv2.imread("samples/before.png", cv2.IMREAD_UNCHANGED)
print(img.shape)   # (height, width, channels)  例: (512, 512, 4)
print(img.dtype)   # uint8 ← 必ず確認すること
```

### BGR チャンネル順

OpenCV は **BGR** 順（R と B が逆）です。

```python
# 赤色 (Red) の指定
red_bgr  = (0, 0, 255)   # OpenCV: B=0, G=0, R=255
red_rgb  = (255, 0, 0)   # Pillow / Matplotlib の場合
```

### `IMREAD_UNCHANGED` の重要性

PNG の **アルファチャンネル（透明度）を保持** するには `cv2.IMREAD_UNCHANGED` を指定します。  
省略すると alpha が落ちて 3ch BGR になります。

```python
# NG: alpha が切り捨てられる
img = cv2.imread("samples/before.png")          # shape: (512, 512, 3)

# OK: BGRA 4ch で読み込む
img = cv2.imread("samples/before.png", cv2.IMREAD_UNCHANGED)  # shape: (512, 512, 4)
```

### `dtype=uint8` の注意点

- 画素値の範囲は **0〜255**（8bit 符号なし整数）。
- 演算時に **オーバーフロー・アンダーフロー**が起きやすい。

```python
# 危険: uint8 のままだと 255+1=0 になる
arr = np.array([254, 255], dtype=np.uint8)
print(arr + 1)   # array([255,   0], dtype=uint8)  ← 溢れる！

# 安全: 差分計算は int16 / float32 に変換してから行う
a = img_before.astype(np.int16)
b = img_after.astype(np.int16)
diff = np.abs(a - b)
```

### I/O モジュール (`image_diff/io.py`)

```bash
python test_io.py
```

```python
from image_diff.io import load_bgra, validate_pair

# 単体読み込み: 常に BGRA (H, W, 4), dtype=uint8 で返る
img = load_bgra("samples/before.png")

# ペア検証: サイズ不一致なら ValueError を送出
before, after = validate_pair("samples/before.png", "samples/after.png")
```

## ハマりポイント

### ① alpha 落ち（`IMREAD_UNCHANGED` 忘れ）

```python
# NG: alpha が切り捨てられて shape=(H,W,3) になる
img = cv2.imread("samples/before.png")

# OK: shape=(H,W,4) で読み込まれる
img = cv2.imread("samples/before.png", cv2.IMREAD_UNCHANGED)
```

> `load_bgra()` は必ず `IMREAD_UNCHANGED` を使うため、呼び出し側が意識する必要はない。
> ただし **直接 `cv2.imread` を書く場合は必ず指定すること**。

### ② dtype ミス（演算時の uint8 オーバーフロー）

```python
# 危険: uint8 のまま引き算すると 0-1=255 になる（アンダーフロー）
diff = after - before          # NG

# 安全: int16 に拡張してから計算し、最後に uint8 に戻す
diff = after.astype(np.int16) - before.astype(np.int16)   # OK
diff_abs = np.clip(np.abs(diff), 0, 255).astype(np.uint8)
```

### ③ 仮想環境でカレントディレクトリを見失ったとき

`import image_diff` が `ModuleNotFoundError` になるのはほぼカレントディレクトリがズレている。

```bash
# 症状: ModuleNotFoundError: No module named 'image_diff'
# 対処: プロジェクトルートに戻る
cd /path/to/image-diff-poc
python test_io.py
```

`image-diff-poc/` を起点にしてスクリプトを実行することを**常に意識**すること。

## ディレクトリ構成

```
image-diff-poc/
├── .venv/                  # 仮想環境（git 管理外）
├── image_diff/             # ライブラリ本体
│   ├── __init__.py
│   ├── io.py               # load_bgra, validate_pair
│   └── diff.py             # DiffExtractor, DiffResult
├── output/                 # 出力先
│   ├── test.png
│   ├── diff_mask.png       # 差分 2値マスク
│   └── diff_overlay.png    # 差分オーバーレイ画像
├── samples/                # ダミー画像
│   ├── before.png
│   └── after.png
├── generate_dummies.py
├── test_env.py
├── test_io.py
├── test_diff.py            # DiffExtractor 検証
├── requirements.txt
└── README.md
```

### 差分抽出モジュール (`image_diff/diff.py`)

```bash
python test_diff.py
```

```python
from image_diff.diff import DiffExtractor

extractor = DiffExtractor(thresh=10, blur_ksize=5, ssim_min=0.98)
result = extractor.extract_diff("samples/before.png", "samples/after.png")

print(f"diff_score : {result.diff_score:.4f}")   # 有効ピクセル内の差分率
print(f"ssim_score : {result.ssim_score:.6f}")   # 構造類似度 (1.0=完全一致)
# result.diff_mask : shape=(H,W), dtype=uint8, 差分あり=255
```

## 差分抽出の狙いと限界

### 狙い

| ステップ | 目的 |
|---|---|
| GaussianBlur | 圧縮ノイズ・サブピクセル揺れを除去して False Positive を減らす |
| SSIM チェック | 大局的な構造変化を事前に検出。閾値割れで WARNING |
| absdiff + threshold | RGB レベルの画素差分を 2 値化 |
| alpha OR マスク | どちらか一方でも不透明な領域を有効範囲とし、透明→不透明変化も検出 |
| **alpha_check** | alpha=0→255 変化ピクセルを connectedComponents で塊として検出（`--full`） |
| **clipping_check** | 端接触率 + approxPolyDP 直線判定で見切れを検出（`--full`） |

### 拡張チェックの設計意図（`--full` で有効）

#### `alpha_check` — アルファ破壊検出

透明（alpha=0）→ 不透明（alpha>0）に変化したピクセルを検出する。
`connectedComponentsWithStats` で連続する塊に分割し、
面積・bbox・塊数を返すことで「どこが・どのサイズで変化したか」を特定できる。

**狙い**: 合成ミス・レイヤー設定ミス・見切れを RGB とは独立して検出する。
**限界**: 半透明（alpha=1-254）の変化は検出しない。完全な透明−不透明の二極のみ。

#### `clipping_check` — 見切れ検出

差分ピクセルが画像端（上下左右 edge_width px 以内）に接触しているかを計測し、
`approxPolyDP` で輪郭の直線性を判定する（4頂点以下を「直線的」とみなす）。

```
clipping_score = edge_touch_ratio  （has_linear_edge=True のとき）
              = 0.0                （直線輪郭が検出されないとき）
```

| clipping_score | 判定 |
|---|---|
| > 0.3 | 高確率で見切れあり |
| > 0.05 | 端付近に差分あり（要確認） |
| ≤ 0.05 | 見切れの兆候なし |

**狙い**: 右端・上端に直線的に差分が並ぶ「見切れ」パターンを識別する。
**限界**: 非直線的な見切れ（曲線オブジェクトが端で途切れる）は `has_linear_edge=False` になりスコアがゼロになる。

### 限界（False Positive / False Negative が出るケース）

#### ① アンチエイリアス (AA) の揺れ
フォント・曲線のエッジ付近では jpeg/png の圧縮アーティファクトにより
サブピクセル単位でRGBが変化する。`blur_ksize` や `thresh` を上げれば軽減できるが、
微細な差分の見落とし（False Negative）とのトレードオフになる。

#### ② 透明領域の RGB 揺れ
PNG の alpha=0 ピクセルは「視覚的に透明」だが、RGB 値はソフトウェアによって
任意の値が入る場合がある（例: Photoshop は 0,0,0、Figma は元色を保存）。
alpha=0 の領域は有効マスクで除外しているため影響は限定的だが、
alpha が 1-254 の半透明ピクセルでは RGB 揺れが差分として現れる。

#### ③ alpha AND 問題（設計上の注意点）
`_get_valid_mask` が **OR** である理由:
before=透明/after=不透明 という「見切れ・追加要素」を正しく差分として拾うため。
**AND** にすると「両方で不透明な領域のみ」に限定され、
透明→不透明の変化（右端見切れ等）が全消滅する。

> **対処**: 背景が完全に透明な PNG で before/after を比較するときは必ず OR マスクを使うこと。

### 可視化モジュール (`image_diff/visualize.py`)

```bash
python test_viz.py
```

```python
from image_diff.visualize import visualize_diff, generate_report, _verdict

# 差分領域を赤矩形でハイライト
visualize_diff("samples/after.png", result.diff_mask, out_path="output/diff_highlight.png")

# CSV レポート生成
v = _verdict(result.diff_score)   # "PASS" / "WARN" / "FAIL"
generate_report(
    diff_score=result.diff_score,
    ssim_score=result.ssim_score,
    verdict=v,
    reason="...",
    artifacts={"file_pair": "before vs after", "highlight_path": "output/diff_highlight.png",
                "mask_path": "output/diff_mask.png", "overlay_path": ""},
    out_path="output/report.csv",
)
```

**判定基準**:

| verdict | 条件 |
|---|---|
| PASS | diff_score < 0.1%（実質同一） |
| WARN | diff_score < 1.0%（要目視確認） |
| FAIL | diff_score ≥ 1.0%（明確な差分） |

## OpenCV 初心者向け解説

### `cv2.threshold` — 2 値化

グレースケール画像の各ピクセルを「閾値以上=白(255) / 未満=黒(0)」に変換する。

```python
gray = cv2.cvtColor(diff_bgr, cv2.COLOR_BGR2GRAY)

# thresh=10: 10 より大きい差分ピクセルを白にする
ret, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
# binary: shape=(H,W), dtype=uint8, 0 or 255 のみ
```

> `ret` は使われた実際の閾値（THRESH_BINARY では thresh と同じ）。
> `_` で捨てていることが多い。

### `cv2.findContours` — 輪郭抽出

白ピクセルの集まりの「外周ライン」を座標リストとして取得する。

```python
contours, hierarchy = cv2.findContours(
    binary,               # 入力: 2値マスク (uint8, 0/255)
    cv2.RETR_EXTERNAL,    # 最外輪郭のみ取得（ネスト輪郭を無視）
    cv2.CHAIN_APPROX_SIMPLE  # 直線の中間点を省略（点数を削減）
)
# contours: list of ndarray  各輪郭の座標列
```

### `cv2.boundingRect` — 外接矩形

輪郭を囲む最小の軸平行矩形（AABB）の左上座標・幅・高さを返す。

```python
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255, 255), 2)
    # (x,y): 左上, (x+w, y+h): 右下, 色: 赤 BGRA, 線幅: 2
```

> **ハマりポイント**: 元画像に描画したい色と同じ色（例: 赤い画像に赤矩形）の場合、
> 矩形の有無をピクセル色で判定できない。
> 代わりに「描画前後の画像差分」で矩形の存在を確認すること。

## ディレクトリ構成

```
image-diff-poc/
├── .venv/
├── image_diff/
│   ├── __init__.py
│   ├── io.py           # load_bgra, validate_pair
│   ├── diff.py         # DiffExtractor, DiffResult
│   └── visualize.py    # visualize_diff, generate_report
├── output/
│   ├── test.png
│   ├── diff_mask.png
│   ├── diff_overlay.png
│   └── diff_highlight.png
│   └── report.csv
├── samples/
│   ├── before.png
│   └── after.png
├── generate_dummies.py
├── test_env.py
├── test_io.py
├── test_diff.py
├── test_viz.py
├── requirements.txt
└── README.md
```
