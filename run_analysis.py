#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
温度予測モジュールを実行するスクリプト
"""

import sys
import argparse
from thermal_prediction.main import main

if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description='温度予測分析ツール')
    parser.add_argument('--file_path', type=str, help='入力データファイルのパス（デフォルト: ./AllDayData.csv）')
    parser.add_argument('--output_dir', type=str, help='出力ディレクトリのパス（デフォルト: ./output/sens_temp_predictions）')
    parser.add_argument('--start_date', type=str, help='分析開始日（デフォルト: 2024-06-26）')
    parser.add_argument('--end_date', type=str, help='分析終了日（デフォルト: 2024-09-20）')
    parser.add_argument('--analyze_horizons', action='store_true', help='予測ホライゾン分析を実行する')
    parser.add_argument('--horizons', type=str, help='評価する予測ホライゾン（分）をカンマ区切りで指定（デフォルト: 5,10,15,20,30）')
    parser.add_argument('--zones', type=str, help='分析するゾーン番号（カンマ区切り。例: 0,2,4 または all で全ゾーン）')

    # コマンドライン引数をメイン関数に渡す
    main()
