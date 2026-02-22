"""
image_diff/io.py
画像の読み込み・ペア検証ユーティリティ。

Note:
    - OpenCV は BGR 順で画像を扱う。
    - アルファチャンネルを保持するには cv2.IMREAD_UNCHANGED が必須。
    - 演算前に dtype=uint8 の溢れに注意すること（差分計算は int16/float32 推奨）。

実行: プロジェクトルート (image-diff-poc/) からスクリプト・モジュールを呼び出すこと。
      仮想環境内でカレントディレクトリを見失ったら "cd /path/to/image-diff-poc" で戻れ。
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np


def load_bgra(path: str) -> np.ndarray:
    """画像を BGRA (4チャンネル, dtype=uint8) で読み込む。

    アルファチャンネルを持たない画像（BGR 3ch 等）の場合は
    alpha=255（完全不透明）のチャンネルを自動付与して返す。

    Args:
        path: 読み込む画像ファイルのパス。

    Returns:
        shape=(H, W, 4), dtype=uint8 の NumPy 配列（チャンネル順: B, G, R, A）。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        ValueError: ファイルは存在するが OpenCV が画像として読み込めない場合、
                    または dtype が uint8 でない予期せぬ形式の場合。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"画像ファイルが見つかりません: {path!r}")

    # IMREAD_UNCHANGED: alpha チャンネルをそのまま保持して読み込む。
    # これを省略すると PNG の透明度が失われ 3ch BGR になる（ハマりポイント）。
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(
            f"OpenCV が画像を読み込めませんでした（破損 or 非対応フォーマット）: {path!r}"
        )

    # dtype チェック: uint8 以外（例: 16bit PNG）は明示的に拒否
    if img.dtype != np.uint8:
        raise ValueError(
            f"対応外の dtype={img.dtype} です（uint8 のみサポート）: {path!r}"
        )

    # チャンネル数を正規化して BGRA にそろえる
    if img.ndim == 2:
        # グレースケール → BGR に変換してから alpha 付与
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ch = img.shape[2]

    if ch == 3:
        # BGR → BGRA: alpha=255（完全不透明）を付与
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    elif ch == 4:
        # 既に BGRA: そのまま返す
        pass
    else:
        raise ValueError(
            f"非対応のチャンネル数 ch={ch} です（1/3/4ch のみ対応）: {path!r}"
        )

    return img


def validate_pair(before_path: str, after_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """before / after 画像ペアを読み込み、サイズ一致を検証する。

    両画像を :func:`load_bgra` で読み込んだ後、
    (height, width) が完全一致することを確認する。

    Args:
        before_path: 加工前画像のパス。
        after_path:  加工後画像のパス。

    Returns:
        (before_bgra, after_bgra): いずれも shape=(H, W, 4), dtype=uint8。

    Raises:
        FileNotFoundError: どちらかのファイルが存在しない場合。
        ValueError: 読み込み失敗、または両画像のサイズが一致しない場合。
    """
    before = load_bgra(before_path)
    after = load_bgra(after_path)

    b_h, b_w = before.shape[:2]
    a_h, a_w = after.shape[:2]

    if (b_h, b_w) != (a_h, a_w):
        raise ValueError(
            f"画像サイズが一致しません。"
            f"before=({b_h}x{b_w}) vs after=({a_h}x{a_w})\n"
            f"  before: {before_path!r}\n"
            f"  after:  {after_path!r}"
        )

    return before, after
