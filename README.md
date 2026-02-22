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

## 実行例

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
├── image_diff/             # I/O・解析ライブラリ
│   ├── __init__.py
│   └── io.py               # load_bgra, validate_pair
├── output/                 # test_env.py の出力先
│   └── test.png
├── samples/                # ダミー画像
│   ├── before.png
│   └── after.png
├── generate_dummies.py     # ダミー画像生成スクリプト
├── test_env.py             # 環境確認スクリプト
├── test_io.py              # I/O モジュール検証
├── requirements.txt        # 依存ライブラリ一覧
└── README.md
```
