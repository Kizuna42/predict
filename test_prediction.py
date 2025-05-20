#!/usr/bin/env python
# coding: utf-8

"""
空調システム室内温度予測モデルのテスト実行スクリプト
特定のゾーンとホライゾンだけを対象とした予測モデルの構築・評価を実行
"""

import argparse
import time
import sys

def main():
    """
    テスト実行のエントリーポイント
    """
    parser = argparse.ArgumentParser(description='空調システム室内温度予測モデルのテスト実行')
    parser.add_argument('--zones', type=int, nargs='+', default=[4],
                        help='処理対象のゾーン番号のリスト（例: 4 5）。デフォルトはゾーン4のみ')
    parser.add_argument('--horizons', type=int, nargs='+', default=[5],
                        help='処理対象の予測ホライゾン（分）のリスト（例: 5 10）。デフォルトは5分後予測のみ')
    parser.add_argument('--full', action='store_true',
                        help='テストモードを無効にして全ゾーン・全ホライゾンで実行')

    args = parser.parse_args()

    # メイン処理の開始時間
    start_time = time.time()

    # 実行するコマンドを構築
    if args.full:
        # 完全実行モード
        cmd = "python main.py"
        print("全ゾーン・全ホライゾンでモデルを構築します（フルモード）...")
    else:
        # テストモード
        cmd = f"python main.py --test --zones {' '.join(map(str, args.zones))} --horizons {' '.join(map(str, args.horizons))}"
        print(f"テストモード: ゾーン {args.zones}, ホライゾン {args.horizons} のモデルを構築します...")

    # コマンドを実行
    print(f"実行コマンド: {cmd}")
    print("-" * 80)
    sys.exit(os.system(cmd))

if __name__ == "__main__":
    import os
    main()
