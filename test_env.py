"""
test_env.py
環境確認スクリプト:
  - cv2 バージョン表示
  - ダミー黒画像(256x256, BGRA)を output/test.png に保存
実行: python test_env.py  (プロジェクトルート image-diff-poc/ から)
"""

import os
import cv2
import numpy as np

print(f"cv2 version: {cv2.__version__}")

# output ディレクトリ作成
os.makedirs("output", exist_ok=True)

# 256x256 の黒画像 (BGRA: 4チャンネル, dtype=uint8)
# shape: (height, width, channels) = (256, 256, 4)
dummy = np.zeros((256, 256, 4), dtype=np.uint8)

out_path = os.path.join("output", "test.png")
cv2.imwrite(out_path, dummy)
print(f"Saved: {out_path}")
