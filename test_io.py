"""
test_io.py
image_diff/io.py の動作検証スクリプト。
実行: python test_io.py  (プロジェクトルート image-diff-poc/ から)
"""

import sys
import numpy as np
from image_diff.io import load_bgra, validate_pair

BEFORE = "samples/before.png"
AFTER  = "samples/after.png"

# ---------------------------------------------------------------
# ヘルパー
# ---------------------------------------------------------------
def ok(msg: str) -> None:
    print(f"  [OK] {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------
# 1. load_bgra: before.png
# ---------------------------------------------------------------
print("=== 1. load_bgra(before.png) ===")
img = load_bgra(BEFORE)

assert img.ndim == 3,          "ndim must be 3"; ok(f"ndim={img.ndim}")
assert img.shape[2] == 4,      "channels must be 4"; ok(f"channels={img.shape[2]}")
assert img.dtype == np.uint8,  "dtype must be uint8"; ok(f"dtype={img.dtype}")
ok(f"shape={img.shape}")

# alpha チャンネルが存在し、円の内側で255 であることを確認
# before.png の中央 (256, 256) 付近は赤円の中心
cx, cy = 256, 256
alpha_center = img[cy, cx, 3]
assert alpha_center == 255, f"円中心のalpha={alpha_center} expected 255"
ok(f"alpha at circle center = {alpha_center}")

# 透過背景（左上隅 0,0 は alpha=0 のはず）
alpha_corner = img[0, 0, 3]
assert alpha_corner == 0, f"左上隅のalpha={alpha_corner} expected 0"
ok(f"alpha at corner (transparent bg) = {alpha_corner}")


# ---------------------------------------------------------------
# 2. load_bgra: after.png
# ---------------------------------------------------------------
print("\n=== 2. load_bgra(after.png) ===")
img_after = load_bgra(AFTER)
assert img_after.shape == img.shape, "after.shape must match before.shape"
ok(f"shape={img_after.shape}")

# 右端40px は白・不透明（alpha=255）
alpha_right = img_after[256, -1, 3]
assert alpha_right == 255, f"右端alpha={alpha_right} expected 255"
ok(f"alpha at right edge = {alpha_right}")


# ---------------------------------------------------------------
# 3. validate_pair: 正常系（サイズ一致）
# ---------------------------------------------------------------
print("\n=== 3. validate_pair (normal) ===")
b, a = validate_pair(BEFORE, AFTER)
assert b.shape == a.shape
ok(f"pair shapes match: {b.shape}")


# ---------------------------------------------------------------
# 4. validate_pair: 異常系（存在しないファイル）
# ---------------------------------------------------------------
print("\n=== 4. load_bgra: FileNotFoundError ===")
try:
    load_bgra("samples/nonexistent.png")
    fail("例外が発生しなかった")
except FileNotFoundError as e:
    ok(f"FileNotFoundError correctly raised: {e}")


# ---------------------------------------------------------------
# 5. validate_pair: 異常系（サイズ不一致）
# ---------------------------------------------------------------
print("\n=== 5. validate_pair: size mismatch ===")
import cv2, os, tempfile

# サイズ違いの仮ファイルを一時作成
tmp_path = os.path.join(tempfile.gettempdir(), "_test_small.png")
small = np.zeros((100, 100, 4), dtype=np.uint8)
cv2.imwrite(tmp_path, small)

try:
    validate_pair(BEFORE, tmp_path)
    fail("例外が発生しなかった")
except ValueError as e:
    ok(f"ValueError correctly raised: {e}")
finally:
    os.remove(tmp_path)


# ---------------------------------------------------------------
print("\n=== All tests passed ✓ ===")
