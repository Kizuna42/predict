#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
温度予測メインモジュール
"""

import os
import argparse
import pandas as pd
import sys
import pathlib
current_dir = pathlib.Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from thermal_prediction.utils import determine_thermo_status
from thermal_prediction.analyze_horizon import analyze_prediction_horizons
from thermal_prediction.config import DEFAULT_CONFIG

def main():
    """メイン関数"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='温度予測モデルの学習と評価')
    parser.add_argument('--file_path', type=str, default=DEFAULT_CONFIG['DEFAULT_FILE_PATH'], help='入力データファイルのパス')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_CONFIG['DEFAULT_OUTPUT_DIR'], help='出力ディレクトリ')
    parser.add_argument('--start_date', type=str, default=DEFAULT_CONFIG['DEFAULT_START_DATE'], help='分析開始日')
    parser.add_argument('--end_date', type=str, default=DEFAULT_CONFIG['DEFAULT_END_DATE'], help='分析終了日')
    parser.add_argument('--analyze_horizons', action='store_true', help='予測ホライゾン分析を実行する')
    parser.add_argument('--horizons', type=str, help='評価する予測ホライゾン（分）をカンマ区切りで指定')
    parser.add_argument('--zones', type=str, help='分析するゾーン番号（カンマ区切り。例: 0,2,4 または all で全ゾーン）')

    args = parser.parse_args()

    # configファイルからデフォルト値を取得し、コマンドライン引数が指定されている場合はそれを優先
    file_path = args.file_path
    output_dir = args.output_dir
    start_date = args.start_date
    end_date = args.end_date

    # analyze_horizonsの判定: コマンドライン引数で指定されていなければconfigから取得
    run_analyze_horizons = args.analyze_horizons or DEFAULT_CONFIG['RUN_ANALYZE_HORIZONS']

    # zonesの設定: コマンドライン引数なければconfigから取得
    zones = args.zones if args.zones is not None else DEFAULT_CONFIG['ZONES_TO_ANALYZE']

    # horizonsの設定: コマンドライン引数なければconfigから取得
    if args.horizons is not None:
        horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
    else:
        horizons = DEFAULT_CONFIG['HORIZONS_TO_ANALYZE']

    if run_analyze_horizons:
        # 予測ホライゾン分析の実行
        output_dir_horizons = os.path.join(output_dir, 'horizon_analysis')
        analyze_prediction_horizons(
            file_path=file_path,
            output_dir=output_dir_horizons,
            zones=zones,
            horizons=horizons,
            start_date=start_date,
            end_date=end_date
        )
    else:
        print("メインの温度予測処理はまだ実装されていません。config.pyでRUN_ANALYZE_HORIZONSをTrueに設定するか、--analyze_horizons を指定して予測ホライゾン分析を実行してください。")

if __name__ == "__main__":
    main()
