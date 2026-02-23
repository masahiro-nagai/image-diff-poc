"""
test_extensions.py
alpha_check / clipping_check の動作検証スクリプト。
実行: python test_extensions.py  (プロジェクトルート image-diff-poc/ から)
"""

import os
import sys
import numpy as np
import cv2

from image_diff.io import load_bgra
from image_diff.diff import (
    DiffExtractor,
    alpha_check,  AlphaCheckResult,
    clipping_check, ClippingCheckResult,
)

BEFORE = "samples/before.png"
AFTER  = "samples/after.png"
OUT    = "output"
os.makedirs(OUT, exist_ok=True)


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    sys.exit(1)


before = load_bgra(BEFORE)
after  = load_bgra(AFTER)
extractor = DiffExtractor(thresh=10, blur_ksize=5, ssim_min=0.98)
diff_result = extractor.extract_diff(BEFORE, AFTER)


# ---------------------------------------------------------------
# 1. alpha_check: before→after でアルファ破壊を検出
# ---------------------------------------------------------------
print("=== 1. alpha_check(before, after) ===")
ac = alpha_check(before, after)

assert isinstance(ac, AlphaCheckResult)
ok(f"type = AlphaCheckResult")

assert ac.destroyed_mask.ndim  == 2
assert ac.destroyed_mask.dtype == np.uint8
ok(f"destroyed_mask shape={ac.destroyed_mask.shape}, dtype={ac.destroyed_mask.dtype}")

# after.png の右端 40px が白 (alpha=255), before では透明 → アルファ破壊あり
assert ac.destroyed_count > 0, f"destroyed_count={ac.destroyed_count}, 右端40px分が検出されるはず"
ok(f"destroyed_count = {ac.destroyed_count:,} px")

assert ac.blob_count >= 1, "少なくとも1塊検出されるはず"
ok(f"blob_count = {ac.blob_count}")

assert 0.0 < ac.ratio <= 1.0, f"ratio={ac.ratio} は (0, 1] の範囲であるべき"
ok(f"ratio = {ac.ratio:.4f} ({ac.ratio*100:.2f}%)")
for b in ac.blobs[:2]:
    ok(f"  blob: area={b['area']:,} bbox={b['bbox']}")

# 出力保存
cv2.imwrite(os.path.join(OUT, "alpha_destroyed.png"), ac.destroyed_mask)
ok(f"Saved: output/alpha_destroyed.png")


# ---------------------------------------------------------------
# 2. alpha_check: 同一画像 → 破壊ゼロ
# ---------------------------------------------------------------
print("\n=== 2. alpha_check(before, before) → no destruction ===")
ac_same = alpha_check(before, before)
assert ac_same.destroyed_count == 0, f"同一画像でアルファ変化はゼロのはず: {ac_same.destroyed_count}"
assert ac_same.blob_count       == 0
ok(f"destroyed_count={ac_same.destroyed_count}, blob_count={ac_same.blob_count} (ゼロ ✓)")


# ---------------------------------------------------------------
# 3. alpha_check: min_blob_area で微小成分フィルタ
# ---------------------------------------------------------------
print("\n=== 3. alpha_check: min_blob_area filter ===")
ac_large = alpha_check(before, after, min_blob_area=999999)
assert ac_large.blob_count == 0, "面積フィルタで全塊が除外されるはず"
ok(f"min_blob_area=999999 → blob_count={ac_large.blob_count} (全除外 ✓)")


# ---------------------------------------------------------------
# 4. clipping_check: before→after の diff_mask で見切れ検出
# ---------------------------------------------------------------
print("\n=== 4. clipping_check(diff_mask) ===")
cc = clipping_check(diff_result.diff_mask)

assert isinstance(cc, ClippingCheckResult)
ok(f"type = ClippingCheckResult")

assert cc.edge_mask.ndim  == 2
assert cc.edge_mask.dtype == np.uint8
ok(f"edge_mask shape={cc.edge_mask.shape}")

# after.png の右端 40px 見切れがあるので edge_touch_ratio > 0 のはず
assert cc.edge_touch_ratio > 0, f"edge_touch_ratio={cc.edge_touch_ratio}, 0より大きいはず"
ok(f"edge_touch_ratio = {cc.edge_touch_ratio:.4f}")
ok(f"has_linear_edge  = {cc.has_linear_edge}")
ok(f"clipping_score   = {cc.clipping_score:.4f}")
ok(f"detail: {cc.detail}")

cv2.imwrite(os.path.join(OUT, "clipping_edge.png"), cc.edge_mask)
ok(f"Saved: output/clipping_edge.png")


# ---------------------------------------------------------------
# 5. clipping_check: 差分ゼロマスク → edge_touch_ratio=0
# ---------------------------------------------------------------
print("\n=== 5. clipping_check: zero diff_mask ===")
zero_mask = np.zeros((512, 512), dtype=np.uint8)
cc_zero = clipping_check(zero_mask)
assert cc_zero.edge_touch_ratio == 0.0
assert cc_zero.clipping_score   == 0.0
ok(f"zero mask → edge_touch_ratio=0.0, clipping_score=0.0 ✓")


# ---------------------------------------------------------------
# 6. CLI --full モード動作確認
# ---------------------------------------------------------------
print("\n=== 6. CLI --full mode ===")
import subprocess
proc = subprocess.run(
    [".venv/bin/python", "-m", "image_diff",
     "--before", BEFORE, "--after", AFTER,
     "--output", OUT, "--full"],
    capture_output=True, text=True,
)
output = proc.stdout + proc.stderr
print(output.rstrip())
assert "alpha_check" in output,    "alpha_check の出力が含まれるはず"
assert "clipping_check" in output, "clipping_check の出力が含まれるはず"
assert proc.returncode == 1, f"FAIL なので exit=1 のはず: returncode={proc.returncode}"
ok("--full モード出力に alpha_check / clipping_check が含まれる ✓")
ok(f"exit code = {proc.returncode} (FAIL=1 ✓)")


# ---------------------------------------------------------------
print("\n=== All tests passed ✓ ===")
print("\nOutput files:")
for f in ["alpha_destroyed.png", "clipping_edge.png"]:
    p = os.path.join(OUT, f)
    print(f"  {p}  ({os.path.getsize(p):,} bytes)")
