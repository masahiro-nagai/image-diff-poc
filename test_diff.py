"""
test_diff.py
DiffExtractor の動作検証スクリプト。
実行: python test_diff.py  (プロジェクトルート image-diff-poc/ から)
"""

import os
import sys
import numpy as np
import cv2
from image_diff.diff import DiffExtractor, DiffResult

BEFORE = "samples/before.png"
AFTER  = "samples/after.png"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


extractor = DiffExtractor(thresh=10, blur_ksize=5, ssim_min=0.98)

# ---------------------------------------------------------------
# 1. 基本的な差分抽出（before vs after）
# ---------------------------------------------------------------
print("=== 1. extract_diff(before, after) ===")
result = extractor.extract_diff(BEFORE, AFTER)

assert isinstance(result, DiffResult), "戻り値は DiffResult であること"
ok(f"type(result) = {type(result).__name__}")

mask = result.diff_mask
assert mask.ndim == 2,             "diff_mask は 2次元"; ok(f"mask.ndim={mask.ndim}")
assert mask.dtype == np.uint8,     "diff_mask は uint8"; ok(f"mask.dtype={mask.dtype}")
assert set(np.unique(mask)) <= {0, 255}, "diff_mask の値は 0 か 255 のみ"
ok(f"diff_mask unique values = {sorted(np.unique(mask).tolist())}")

diffed = np.count_nonzero(mask)
assert diffed > 0, "差分ピクセルが 1 個以上あること"
ok(f"diff pixels = {diffed:,}")

assert 0.0 <= result.diff_score <= 1.0, "diff_score は 0〜1"
ok(f"diff_score = {result.diff_score:.4f}  ({result.diff_score*100:.2f}%)")

assert -1.0 <= result.ssim_score <= 1.0, "ssim_score は -1〜1"
ok(f"ssim_score = {result.ssim_score:.6f}")

# diff_mask を可視化して保存
out_path = os.path.join(OUT_DIR, "diff_mask.png")
cv2.imwrite(out_path, mask)
ok(f"Saved diff_mask → {out_path}")

# 差分を before に重ねたオーバーレイ画像を保存（目視確認用）
before_bgra = cv2.imread(BEFORE, cv2.IMREAD_UNCHANGED)
overlay = before_bgra.copy()
# 差分領域を赤で塗りつぶし（BGR+Alpha）
overlay[mask == 255] = [0, 0, 255, 255]
overlay_path = os.path.join(OUT_DIR, "diff_overlay.png")
cv2.imwrite(overlay_path, overlay)
ok(f"Saved diff_overlay → {overlay_path}")


# ---------------------------------------------------------------
# 2. 同一画像 vs 同一画像 → 差分ゼロになるはず
# ---------------------------------------------------------------
print("\n=== 2. extract_diff(before, before) → no diff ===")
result_same = extractor.extract_diff(BEFORE, BEFORE)
diffed_same = np.count_nonzero(result_same.diff_mask)
assert diffed_same == 0, f"同一画像の差分は 0 であるべきが {diffed_same}"
ok(f"diff pixels = {diffed_same}  (同一画像で差分なし ✓)")
ok(f"ssim_score  = {result_same.ssim_score:.6f}  (≒ 1.0 期待)")
ok(f"diff_score  = {result_same.diff_score:.4f}  (≒ 0.0 期待)")


# ---------------------------------------------------------------
# 3. blur_ksize が偶数のとき ValueError
# ---------------------------------------------------------------
print("\n=== 3. DiffExtractor(blur_ksize=4) → ValueError ===")
try:
    DiffExtractor(blur_ksize=4)
    fail("例外が発生しなかった")
except ValueError as e:
    ok(f"ValueError correctly raised: {e}")


# ---------------------------------------------------------------
# 4. thresh=0 → 微細な差分も全検出（ノイズ含む可能性アリ）
# ---------------------------------------------------------------
print("\n=== 4. thresh=0: all diff pixels detected ===")
ext_strict = DiffExtractor(thresh=0, blur_ksize=5)
result_strict = ext_strict.extract_diff(BEFORE, AFTER)
ok(f"diff_score (thresh=0)  = {result_strict.diff_score:.4f}")
assert result_strict.diff_score >= result.diff_score, \
    "thresh=0 の diff_score は thresh=10 以上のはず"
ok("thresh=0 の diff_score ≥ thresh=10 ✓")


# ---------------------------------------------------------------
print("\n=== All tests passed ✓ ===")
print(f"\nOutput files:")
print(f"  output/diff_mask.png    — 差分ピクセルの 2値マスク")
print(f"  output/diff_overlay.png — 差分を赤でオーバーレイした before 画像")
