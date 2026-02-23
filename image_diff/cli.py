"""
image_diff/cli.py
argparse ベースの CLI ロジック。
`python -m image_diff` から呼び出される。

Usage:
    python -m image_diff \\
        --before samples/before.png \\
        --after  samples/after.png  \\
        --output output/            \\
        [--thresh 10] [--blur 5] [--ssim-min 0.98]

Note:
    プロジェクトルート (image-diff-poc/) から実行すること。
    カレントディレクトリを見失ったら: cd /path/to/image-diff-poc
"""

from __future__ import annotations

import argparse
import sys

from image_diff.diff import DiffExtractor, alpha_check, clipping_check
from image_diff.io import load_bgra
from image_diff.visualize import run_visualize_and_report


def build_parser() -> argparse.ArgumentParser:
    """ArgumentParser を構築して返す。"""
    parser = argparse.ArgumentParser(
        prog="python -m image_diff",
        description="加工前後 PNG の差分を抽出し、ハイライト画像と CSV レポートを生成する。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--before", required=True, metavar="PATH",
        help="加工前画像のパス（PNG, BGRA 推奨）",
    )
    parser.add_argument(
        "--after", required=True, metavar="PATH",
        help="加工後画像のパス（PNG, BGRA 推奨）",
    )
    parser.add_argument(
        "--output", default="output", metavar="DIR",
        help="出力先ディレクトリ",
    )
    parser.add_argument(
        "--thresh", type=int, default=10, metavar="N",
        help="差分 2 値化の閾値（0–255）。小さいほど微細差分を拾う",
    )
    parser.add_argument(
        "--blur", type=int, default=5, metavar="N",
        help="GaussianBlur カーネルサイズ（奇数）。大きいほどノイズ除去が強い",
    )
    parser.add_argument(
        "--ssim-min", type=float, default=0.98, dest="ssim_min", metavar="F",
        help="SSIM がこの値未満のとき WARNING を出力する",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="フルモード: アルファ破壊チェック + 見切れチェックも実行する",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    """diff → visualize → report のパイプラインを実行する。

    Args:
        args: parse_args() の結果。

    Returns:
        終了コード（0: PASS/WARN, 1: FAIL, 2: エラー）。
    """
    # --- 差分抽出 ---
    try:
        extractor = DiffExtractor(
            thresh=args.thresh,
            blur_ksize=args.blur,
            ssim_min=args.ssim_min,
        )
        result = extractor.extract_diff(args.before, args.after)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # --- 可視化 + レポート生成 ---
    try:
        outputs = run_visualize_and_report(
            before_path=args.before,
            after_path=args.after,
            diff_mask=result.diff_mask,
            diff_score=result.diff_score,
            ssim_score=result.ssim_score,
            out_dir=args.output,
        )
    except Exception as e:
        print(f"[ERROR] 可視化/レポート生成に失敗しました: {e}", file=sys.stderr)
        return 2

    # --- 結果サマリ表示 ---
    v = outputs["verdict"]
    icon = {"PASS": "✅", "WARN": "⚠️ ", "FAIL": "❌"}[v]
    print(f"\n{icon}  verdict    : {v}")
    print(f"   diff_score : {result.diff_score:.6f}  ({result.diff_score * 100:.2f}%)")
    print(f"   ssim_score : {result.ssim_score:.6f}")
    print(f"\nOutput files:")
    print(f"  highlight  : {outputs['highlight_path']}")
    print(f"  report     : {outputs['report_path']}")

    # --- --full モード: 拡張チェック ---
    if args.full:
        print("\n--- Full Mode: 拡張チェック ---")
        try:
            before_img = load_bgra(args.before)
            after_img  = load_bgra(args.after)

            # アルファ破壊チェック
            ac = alpha_check(before_img, after_img)
            alpha_icon = "⚠️ " if ac.blob_count > 0 else "✅"
            print(f"{alpha_icon} alpha_check:")
            print(f"   destroyed_count : {ac.destroyed_count:,} px")
            print(f"   blob_count      : {ac.blob_count}")
            print(f"   ratio           : {ac.ratio:.4f} ({ac.ratio*100:.2f}%)")
            if ac.blobs:
                for b in ac.blobs[:3]:  # 最大3件表示
                    print(f"   blob label={b['label']} area={b['area']} bbox={b['bbox']}")
                if len(ac.blobs) > 3:
                    print(f"   ... and {len(ac.blobs)-3} more blob(s)")

            # 見切れチェック
            cc = clipping_check(result.diff_mask)
            clip_icon = "⚠️ " if cc.clipping_score > 0.05 else "✅"
            print(f"{clip_icon} clipping_check:")
            print(f"   {cc.detail}")

        except Exception as e:
            print(f"[WARNING] 拡張チェック中にエラー: {e}", file=sys.stderr)

    return 1 if v == "FAIL" else 0


def main() -> None:
    """CLI エントリポイント。"""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))
