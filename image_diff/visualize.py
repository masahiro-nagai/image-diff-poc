"""
image_diff/visualize.py
差分マスクの可視化 + CSV レポート生成モジュール。

Note:
    - headless 専用（imshow 禁止）。
    - 出力は PNG ファイル（BGRA 保持）と CSV ファイル。

実行: プロジェクトルート (image-diff-poc/) から呼び出すこと。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# 判定ロジック
# ------------------------------------------------------------------

def _verdict(diff_score: float) -> str:
    """diff_score から PASS / WARN / FAIL を判定する。

    Args:
        diff_score: 有効ピクセルに占める差分ピクセル率（0.0〜1.0）。

    Returns:
        - "PASS" : diff_score <  0.001（0.1%未満: ほぼ同一）
        - "WARN" : diff_score <  0.01 （1.0%未満: 要確認）
        - "FAIL" : diff_score >= 0.01 （1.0%以上: 明確な差分）
    """
    if diff_score < 0.001:
        return "PASS"
    if diff_score < 0.01:
        return "WARN"
    return "FAIL"


# ------------------------------------------------------------------
# 可視化
# ------------------------------------------------------------------

def visualize_diff(
    after_path: str,
    diff_mask: np.ndarray,
    out_path: str = "output/diff_highlight.png",
) -> str:
    """差分領域を矩形でハイライトした画像を保存する。

    処理フロー:
        1. after 画像を BGRA で読み込む。
        2. diff_mask から輪郭（contours）を抽出する。
        3. 各輪郭の外接矩形（boundingRect）を赤い矩形で描画する。
        4. BGRA のまま out_path に保存する。

    Args:
        after_path: 加工後画像のパス（BGRA で読み込む）。
        diff_mask:  shape=(H, W), dtype=uint8 の 2値マスク（差分=255）。
        out_path:   出力先 PNG パス（デフォルト: output/diff_highlight.png）。

    Returns:
        保存した out_path 文字列。

    Raises:
        FileNotFoundError: after_path が存在しない場合。
        ValueError: diff_mask の shape/dtype が不正な場合。
    """
    # --- 入力バリデーション ---
    if not os.path.exists(after_path):
        raise FileNotFoundError(f"after 画像が見つかりません: {after_path!r}")
    if diff_mask.ndim != 2:
        raise ValueError(f"diff_mask は 2次元でなければなりません: ndim={diff_mask.ndim}")
    if diff_mask.dtype != np.uint8:
        raise ValueError(f"diff_mask の dtype は uint8 でなければなりません: {diff_mask.dtype}")

    # --- after 画像を BGRA で読み込む ---
    canvas = cv2.imread(after_path, cv2.IMREAD_UNCHANGED)
    if canvas is None:
        raise ValueError(f"OpenCV が after 画像を読み込めませんでした: {after_path!r}")
    # 3ch BGR なら alpha=255 の BGRA に変換して保持
    if canvas.ndim == 3 and canvas.shape[2] == 3:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)

    # --- findContours で差分輪郭を抽出 ---
    # RETR_EXTERNAL: 最外輪郭のみ取得（ネスト輪郭は無視）
    # CHAIN_APPROX_SIMPLE: 直線部分の中間点を省略して軽量化
    contours, _ = cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # --- 各輪郭の外接矩形を赤で描画 ---
    # 赤色: BGR=(0,0,255), alpha=255（完全不透明）
    RED_BGRA = (0, 0, 255, 255)
    THICKNESS = 2

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 面積が極小のノイズ矩形は除外（1px × 1px 相当）
        if w * h < 4:
            continue
        cv2.rectangle(canvas, (x, y), (x + w, y + h), RED_BGRA, THICKNESS)

    # --- 出力先ディレクトリ作成 & 保存 ---
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, canvas)

    return out_path


# ------------------------------------------------------------------
# レポート生成
# ------------------------------------------------------------------

def generate_report(
    diff_score: float,
    ssim_score: float,
    verdict: str,
    reason: str,
    artifacts: Dict[str, str],
    out_path: str = "output/report.csv",
) -> str:
    """差分結果を CSV レポートとして保存する。

    CSV カラム:
        - file_pair    : "before_path vs after_path" 形式の文字列
        - verdict      : PASS / WARN / FAIL
        - diff_score   : 差分ピクセル率（小数 6 桁表示）
        - ssim_score   : SSIM スコア（小数 6 桁表示）
        - reason       : 判定理由・コメント
        - highlight_path : 矩形ハイライト画像のパス
        - mask_path    : 差分 2 値マスクのパス
        - overlay_path : 差分オーバーレイ画像のパス（省略可）

    Args:
        diff_score: 差分ピクセル率（0.0〜1.0）。
        ssim_score: SSIM スコア（-1.0〜1.0）。
        verdict:    判定結果文字列（"PASS", "WARN", "FAIL"）。
        reason:     判定理由・コメント。
        artifacts:  出力ファイルパスの辞書。
                    期待キー: "file_pair", "highlight_path",
                              "mask_path", "overlay_path"（省略可）。
        out_path:   CSV 出力先パス（デフォルト: output/report.csv）。

    Returns:
        保存した out_path 文字列。
    """
    row: Dict[str, Any] = {
        "file_pair":      artifacts.get("file_pair", ""),
        "verdict":        verdict,
        "diff_score":     round(diff_score, 6),
        "ssim_score":     round(ssim_score, 6),
        "reason":         reason,
        "highlight_path": artifacts.get("highlight_path", ""),
        "mask_path":      artifacts.get("mask_path", ""),
        "overlay_path":   artifacts.get("overlay_path", ""),
    }
    df = pd.DataFrame([row])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")

    return out_path


# ------------------------------------------------------------------
# 一括実行ヘルパー（pipeline 用）
# ------------------------------------------------------------------

def run_visualize_and_report(
    before_path: str,
    after_path: str,
    diff_mask: np.ndarray,
    diff_score: float,
    ssim_score: float,
    out_dir: str = "output",
) -> Dict[str, str]:
    """visualize_diff + generate_report を一括実行する便利関数。

    Args:
        before_path: 加工前画像パス。
        after_path:  加工後画像パス。
        diff_mask:   DiffExtractor.extract_diff の diff_mask。
        diff_score:  同上 diff_score。
        ssim_score:  同上 ssim_score。
        out_dir:     出力先ディレクトリ（デフォルト: output/）。

    Returns:
        各出力ファイルパスを格納した辞書。
    """
    v = _verdict(diff_score)
    reason_map = {
        "PASS": "diff_score < 0.1%: 実質的な差分なし",
        "WARN": "diff_score < 1.0%: 微細な差分あり、要目視確認",
        "FAIL": f"diff_score >= 1.0%: 明確な差分を検出 (score={diff_score:.4f})",
    }

    highlight_path = os.path.join(out_dir, "diff_highlight.png")
    mask_path      = os.path.join(out_dir, "diff_mask.png")
    overlay_path   = os.path.join(out_dir, "diff_overlay.png")
    report_path    = os.path.join(out_dir, "report.csv")

    # 可視化
    visualize_diff(after_path, diff_mask, out_path=highlight_path)

    # レポート生成
    generate_report(
        diff_score=diff_score,
        ssim_score=ssim_score,
        verdict=v,
        reason=reason_map[v],
        artifacts={
            "file_pair":      f"{before_path} vs {after_path}",
            "highlight_path": highlight_path,
            "mask_path":      mask_path,
            "overlay_path":   overlay_path,
        },
        out_path=report_path,
    )

    return {
        "verdict":        v,
        "highlight_path": highlight_path,
        "report_path":    report_path,
    }
