"""
test_viz.py
image_diff/visualize.py の動作検証スクリプト。
実行: python test_viz.py  (プロジェクトルート image-diff-poc/ から)
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2

from image_diff.diff import DiffExtractor
from image_diff.visualize import (
    visualize_diff,
    generate_report,
    run_visualize_and_report,
    _verdict,
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


# ---------------------------------------------------------------
# 準備: DiffExtractor で差分マスク取得
# ---------------------------------------------------------------
extractor = DiffExtractor(thresh=10, blur_ksize=5, ssim_min=0.98)
result = extractor.extract_diff(BEFORE, AFTER)


# ---------------------------------------------------------------
# 1. _verdict の判定ロジック
# ---------------------------------------------------------------
print("=== 1. _verdict thresholds ===")
assert _verdict(0.0)     == "PASS"; ok("0.000 → PASS")
assert _verdict(0.0009)  == "PASS"; ok("0.0009 → PASS")
assert _verdict(0.001)   == "WARN"; ok("0.001 → WARN")
assert _verdict(0.0099)  == "WARN"; ok("0.0099 → WARN")
assert _verdict(0.01)    == "FAIL"; ok("0.01 → FAIL")
assert _verdict(1.0)     == "FAIL"; ok("1.00 → FAIL")


# ---------------------------------------------------------------
# 2. visualize_diff: ハイライト画像生成
# ---------------------------------------------------------------
print("\n=== 2. visualize_diff ===")
highlight_path = os.path.join(OUT, "diff_highlight.png")
saved = visualize_diff(AFTER, result.diff_mask, out_path=highlight_path)
assert saved == highlight_path
assert os.path.exists(highlight_path), "ファイルが存在しない"
ok(f"Saved: {saved}")

# 保存画像の形式確認
img = cv2.imread(highlight_path, cv2.IMREAD_UNCHANGED)
assert img is not None
assert img.shape[2] == 4, f"BGRA(4ch)で保存されていること: ch={img.shape[2]}"
ok(f"shape={img.shape}, dtype={img.dtype} (BGRA保持 ✓)")

# 赤矩形が描画されているか: B=0, G=0, R=255 のピクセルが存在するか確認
red_pixels = np.all(img[:, :, :3] == [0, 0, 255], axis=2)
assert red_pixels.any(), "赤矩形ピクセルが存在しない"
ok(f"赤矩形ピクセル数: {red_pixels.sum():,}px ✓")


# ---------------------------------------------------------------
# 3. generate_report: CSV 出力
# ---------------------------------------------------------------
print("\n=== 3. generate_report ===")
v = _verdict(result.diff_score)
report_path = os.path.join(OUT, "report.csv")
saved_csv = generate_report(
    diff_score=result.diff_score,
    ssim_score=result.ssim_score,
    verdict=v,
    reason=f"diff_score={result.diff_score:.4f}",
    artifacts={
        "file_pair":      f"{BEFORE} vs {AFTER}",
        "highlight_path": highlight_path,
        "mask_path":      os.path.join(OUT, "diff_mask.png"),
        "overlay_path":   os.path.join(OUT, "diff_overlay.png"),
    },
    out_path=report_path,
)
assert os.path.exists(report_path), "CSV が存在しない"
ok(f"Saved: {saved_csv}")

df = pd.read_csv(report_path)
ok(f"rows={len(df)}, cols={list(df.columns)}")
assert len(df) == 1
assert df["verdict"].iloc[0] == v,              f"verdict mismatch: {df['verdict'].iloc[0]}"
assert abs(df["diff_score"].iloc[0] - result.diff_score) < 1e-5
assert abs(df["ssim_score"].iloc[0] - result.ssim_score) < 1e-5
ok(f"verdict={df['verdict'].iloc[0]}, diff_score={df['diff_score'].iloc[0]:.6f}, ssim={df['ssim_score'].iloc[0]:.6f}")


# ---------------------------------------------------------------
# 4. run_visualize_and_report: パイプライン一括実行
# ---------------------------------------------------------------
print("\n=== 4. run_visualize_and_report (pipeline) ===")
outputs = run_visualize_and_report(
    before_path=BEFORE,
    after_path=AFTER,
    diff_mask=result.diff_mask,
    diff_score=result.diff_score,
    ssim_score=result.ssim_score,
    out_dir=OUT,
)
ok(f"verdict       = {outputs['verdict']}")
ok(f"highlight_path= {outputs['highlight_path']}")
ok(f"report_path   = {outputs['report_path']}")
assert os.path.exists(outputs["highlight_path"])
assert os.path.exists(outputs["report_path"])


# ---------------------------------------------------------------
# 5. 異常系: FileNotFoundError
# ---------------------------------------------------------------
print("\n=== 5. visualize_diff: FileNotFoundError ===")
try:
    visualize_diff("samples/nonexistent.png", result.diff_mask)
    fail("例外が発生しなかった")
except FileNotFoundError as e:
    ok(f"FileNotFoundError correctly raised: {e}")


# ---------------------------------------------------------------
# 6. 同一画像（差分ゼロ）: 矩形なし → 描画による変化がないことを確認
# ---------------------------------------------------------------
print("\n=== 6. no-diff mask: no rectangles drawn ===")
same_result = extractor.extract_diff(BEFORE, BEFORE)
no_diff_path = os.path.join(OUT, "diff_highlight_same.png")

# diff_mask がゼロであることをまず確認
assert np.count_nonzero(same_result.diff_mask) == 0, "diff_mask は全ゼロのはず"
ok("diff_mask はゼロ ✓")

visualize_diff(BEFORE, same_result.diff_mask, out_path=no_diff_path)

# 出力画像が「オリジナルの before.png と同一」であることを確認。
# 差分ゼロ → contours が空 → 矩形描画なし → 画像に変化なし
original = cv2.imread(BEFORE, cv2.IMREAD_UNCHANGED)
output   = cv2.imread(no_diff_path, cv2.IMREAD_UNCHANGED)
assert np.array_equal(original, output), \
    "差分ゼロのとき、出力画像はオリジナルと同一のはず（矩形描画なし）"
ok("差分ゼロ → 出力画像 = オリジナル（矩形描画なし） ✓")


# ---------------------------------------------------------------
print("\n=== All tests passed ✓ ===")
print("\nOutput files:")
for f in ["diff_highlight.png", "report.csv", "diff_highlight_same.png"]:
    p = os.path.join(OUT, f)
    size = os.path.getsize(p) if os.path.exists(p) else 0
    print(f"  {p}  ({size:,} bytes)")

# CSV 内容をプレビュー
print("\n--- report.csv preview ---")
print(pd.read_csv(report_path).to_string(index=False))
