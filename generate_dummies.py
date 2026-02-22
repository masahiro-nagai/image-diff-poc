"""
generate_dummies.py
ダミー画像生成スクリプト:
  - samples/before.png : 透過背景（RGBA）に中央赤円
  - samples/after.png  : before に一部色変更（青矩形）+ 右端を白にして見切れを演出
実行: python generate_dummies.py  (プロジェクトルート image-diff-poc/ から)
"""

import os
import cv2
import numpy as np

os.makedirs("samples", exist_ok=True)

W, H = 512, 512  # 画像サイズ

# ------------------------------------------------------------------
# before.png : 透過背景 (alpha=0) + 中央に赤円 (alpha=255)
# ------------------------------------------------------------------
before = np.zeros((H, W, 4), dtype=np.uint8)  # BGRA, 全透明

# 中央に赤円を描画
# OpenCV の色順は BGR なので赤 = (0, 0, 255)
center = (W // 2, H // 2)
radius = 100
# 円の内側だけ alpha=255 にするため mask を使う
cv2.circle(before, center, radius, (0, 0, 255, 255), thickness=-1)

cv2.imwrite("samples/before.png", before)
print("Saved: samples/before.png")

# ------------------------------------------------------------------
# after.png : before をコピーして差分を加える
#   1. 左上に青矩形を追加（色変更差分）
#   2. 右端 40px を白色で塗りつぶして「見切れ」を演出
# ------------------------------------------------------------------
after = before.copy()

# 1. 左上に青矩形 (BGR: 255,0,0)
cv2.rectangle(after, (30, 30), (180, 130), (255, 0, 0, 255), thickness=-1)

# 2. 右端 40px 見切れ（白・不透明）
after[:, W - 40:, :] = [255, 255, 255, 255]

cv2.imwrite("samples/after.png", after)
print("Saved: samples/after.png")

print("\nDone. Dummy images generated in samples/")
