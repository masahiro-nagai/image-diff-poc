"""
image_diff/diff.py
GaussianBlur + absdiff + SSIM ベースの差分抽出モジュール。

Note:
    - headless 専用（imshow 禁止）。出力は numpy 配列 + スコアのみ。
    - 透明領域（alpha=0）はマスクで差分から除外する。
    - SSIM が ssim_min 未満の場合でも差分計算は継続するが、
      「大きな構造変化あり」として WARNING を出す。

実行: プロジェクトルート (image-diff-poc/) から呼び出すこと。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from image_diff.io import load_bgra, validate_pair


@dataclass
class DiffResult:
    """extract_diff の返り値をまとめたデータクラス。

    Attributes:
        diff_mask:   差分ピクセルを示す 2値マスク (H, W, dtype=uint8)。
                     差分あり=255、差分なし / 透明領域=0。
        diff_score:  diff_mask 内の「差分ピクセル率」（0.0〜1.0）。
                     有効ピクセル（alpha>0）に占める白ピクセルの割合。
        ssim_score:  SSIM スコア（-1.0〜1.0）。1.0 が完全一致。
    """
    diff_mask: np.ndarray
    diff_score: float
    ssim_score: float


class DiffExtractor:
    """加工前後画像の差分を抽出するクラス。

    前処理として GaussianBlur を適用し、ノイズを除去してから
    absdiff + threshold で差分マスクを生成する。
    SSIM スコアが ssim_min を下回った場合は WARNING を出力する。

    Args:
        thresh:    二値化しきい値（0〜255、デフォルト=10）。
                   値が小さいほど微細な差分も拾う。
        blur_ksize: GaussianBlur のカーネルサイズ（奇数、デフォルト=5）。
                    大きいほど細かいノイズが消える。
        ssim_min:  SSIM がこの値未満のとき「大きな構造変化」と判定し
                   WARNING を出力する（デフォルト=0.98）。
    """

    def __init__(
        self,
        thresh: int = 10,
        blur_ksize: int = 5,
        ssim_min: float = 0.98,
    ) -> None:
        if blur_ksize % 2 == 0:
            raise ValueError(f"blur_ksize は奇数でなければなりません: {blur_ksize}")
        self.thresh = thresh
        self.blur_ksize = blur_ksize
        self.ssim_min = ssim_min

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------

    def extract_diff(
        self,
        before_path: str,
        after_path: str,
    ) -> DiffResult:
        """差分マスク・差分スコア・SSIM スコアを返す。

        処理フロー:
            1. load_bgra + validate_pair でペアを読み込む。
            2. RGB チャンネルに GaussianBlur を適用（ノイズ除去）。
            3. SSIM スコアを計算し、ssim_min 未満なら WARNING。
            4. absdiff → GRAY → threshold で diff_mask 生成。
            5. 透明領域（alpha=0）を diff_mask から除外。
            6. 有効ピクセルに占める差分ピクセル率を diff_score として算出。

        Args:
            before_path: 加工前画像のパス。
            after_path:  加工後画像のパス。

        Returns:
            DiffResult(diff_mask, diff_score, ssim_score)。

        Raises:
            FileNotFoundError: いずれかの画像が存在しない場合。
            ValueError: 読み込み失敗またはサイズ不一致の場合。
        """
        # --- Step 1: 読み込み & ペア検証 ---
        before, after = validate_pair(before_path, after_path)

        # --- Step 2: GaussianBlur（RGB のみ、alpha は維持）---
        # alpha チャンネルをノイズ除去の対象から外す
        before_blurred = self._apply_blur(before)
        after_blurred = self._apply_blur(after)

        # --- Step 3: SSIM 計算（グレースケール変換後）---
        ssim_score = self._calc_ssim(before_blurred, after_blurred)
        if ssim_score < self.ssim_min:
            print(
                f"[WARNING] SSIM={ssim_score:.4f} < ssim_min={self.ssim_min:.4f}. "
                "大きな構造変化が検出されました。diff_score は参考値として扱ってください。"
            )

        # --- Step 4: absdiff → GRAY → 二値化で diff_mask 生成 ---
        # uint8 のまま absdiff する（OpenCV が飽和演算で安全に処理する）
        abs_diff = cv2.absdiff(before_blurred[:, :, :3], after_blurred[:, :, :3])
        gray_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2GRAY)
        _, diff_mask = cv2.threshold(
            gray_diff, self.thresh, 255, cv2.THRESH_BINARY
        )

        # --- Step 5: 透明領域を除外（両画像のどちらかが alpha=0 ならマスク）---
        # before と after の alpha の論理積: 両方で不透明 (alpha>0) な領域のみ有効
        alpha_valid = self._get_valid_mask(before, after)
        diff_mask = cv2.bitwise_and(diff_mask, alpha_valid)

        # --- Step 6: diff_score 計算（有効ピクセル数に対する差分ピクセル率）---
        diff_score = self._calc_diff_score(diff_mask, alpha_valid)

        return DiffResult(
            diff_mask=diff_mask,
            diff_score=diff_score,
            ssim_score=ssim_score,
        )

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """RGB チャンネルだけに GaussianBlur を適用して BGRA を返す。"""
        blurred = img.copy()
        blurred[:, :, :3] = cv2.GaussianBlur(
            img[:, :, :3],
            (self.blur_ksize, self.blur_ksize),
            0,
        )
        return blurred

    @staticmethod
    def _calc_ssim(before: np.ndarray, after: np.ndarray) -> float:
        """BGRA 画像をグレースケールに変換して SSIM スコアを返す。

        Note:
            SSIM は RGB 情報のみで評価する（alpha は除外）。
            skimage.metrics.structural_similarity を使用。
        """
        b_gray = cv2.cvtColor(before[:, :, :3], cv2.COLOR_BGR2GRAY)
        a_gray = cv2.cvtColor(after[:, :, :3], cv2.COLOR_BGR2GRAY)
        score, _ = ssim(b_gray, a_gray, full=True)
        return float(score)

    @staticmethod
    def _get_valid_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """どちらか一方でも alpha > 0 な領域を示す 8bit マスクを返す。

        論理 OR を使う理由:
            before で透明（alpha=0）だが after で不透明（白塗りなど）に変化した
            領域も「差分あり」として検出したいため。AND では両方が不透明な交差領域
            だけを対象にしてしまい、透明→不透明の変化を見逃す。

        例: after.png の右端見切れ（白塗り, alpha=255）は before では透明（alpha=0）。
            AND だとこの領域は有効範囲外になり、差分が全消滅する。
        """
        alpha_before = (before[:, :, 3] > 0).astype(np.uint8) * 255
        alpha_after  = (after[:, :, 3]  > 0).astype(np.uint8) * 255
        return cv2.bitwise_or(alpha_before, alpha_after)

    @staticmethod
    def _calc_diff_score(
        diff_mask: np.ndarray,
        alpha_valid: np.ndarray,
    ) -> float:
        """有効ピクセルに占める差分ピクセルの割合を返す。

        Args:
            diff_mask:   二値マスク（差分あり=255）。
            alpha_valid: 有効ピクセルマスク（不透明=255）。

        Returns:
            0.0〜1.0 の差分率。有効ピクセルが 0 の場合は 0.0 を返す。
        """
        valid_count = int(np.count_nonzero(alpha_valid))
        if valid_count == 0:
            return 0.0
        diff_count = int(np.count_nonzero(diff_mask))
        return diff_count / valid_count
