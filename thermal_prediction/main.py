#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
温度予測メインモジュール
"""

import os
import argparse
import pandas as pd

from .utils import determine_thermo_status
from .analyze_horizon import analyze_prediction_horizons

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='温度予測モデルの学習と評価')
    parser.add_argument('--file_path', type=str, default='./AllDayData.csv', help='入力データファイルのパス')
    parser.add_argument('--output_dir', type=str, default='./output/sens_temp_predictions', help='出力ディレクトリ')
    parser.add_argument('--start_date', type=str, default='2024-06-26', help='分析開始日')
    parser.add_argument('--end_date', type=str, default='2024-09-20', help='分析終了日')
    parser.add_argument('--analyze_horizons', action='store_true', help='予測ホライゾン分析を実行する')
    parser.add_argument('--horizons', type=str, default='5,10,15,20,30', help='評価する予測ホライゾン（分）をカンマ区切りで指定')
    parser.add_argument('--zones', type=str, default='all', help='分析するゾーン番号（カンマ区切り。例: 0,2,4 または all で全ゾーン）')

    args = parser.parse_args()

    if args.analyze_horizons:
        # 予測ホライゾン分析の実行
        horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
        output_dir = os.path.join(args.output_dir, 'horizon_analysis')
        analyze_prediction_horizons(
            file_path=args.file_path,
            output_dir=output_dir,
            zones=args.zones,
            horizons=horizons,
            start_date=args.start_date,
            end_date=args.end_date
        )
    else:
        print("メインの温度予測処理はまだ実装されていません。--analyze_horizons を指定して予測ホライゾン分析を実行してください。")

if __name__ == "__main__":
    main()
