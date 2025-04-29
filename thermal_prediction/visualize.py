#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
温度予測の結果をインタラクティブに可視化するスクリプト
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import glob
import pathlib

current_dir = pathlib.Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from thermal_prediction.visualization.interactive import (
    interactive_all_zones_timeseries,
    interactive_horizon_metrics,
    interactive_timeseries,
    interactive_all_horizons_timeseries,
    interactive_feature_ranks,
    interactive_horizon_scatter
)

def parse_args():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description="温度予測結果のインタラクティブな可視化")
    parser.add_argument("--output_dir", default="./output", help="出力ディレクトリ")
    parser.add_argument("--zones", default="all", help="表示するゾーン (カンマ区切り数値またはall)")
    parser.add_argument("--horizon", type=int, default=15, help="表示する予測ホライゾン (分)")
    parser.add_argument("--all_horizons", action="store_true", help="全予測ホライゾンを表示")
    parser.add_argument("--results_dir", help="予測結果があるディレクトリ（指定しない場合は自動検出）")
    parser.add_argument("--open", action="store_true", help="結果のHTMLファイルを自動で開く")
    return parser.parse_args()

def load_results(output_dir, zones, horizon):
    """保存された予測結果を辞書形式で読み込む"""
    if not os.path.exists(output_dir):
        print(f"エラー: 出力ディレクトリが見つかりません: {output_dir}")
        return None

    results_dict = {}

    # 全ゾーンの予測結果を読み込み
    for zone in zones:
        # 予測結果のCSVパスを推測
        csv_pattern = os.path.join(output_dir, f"*zone_{zone}*predictions*.csv")
        csv_files = glob.glob(csv_pattern)

        if not csv_files:
            print(f"ゾーン {zone} の予測結果が見つかりません")
            continue

        # 最新のCSVファイルを使用
        latest_csv = max(csv_files, key=os.path.getmtime)
        try:
            df = pd.read_csv(latest_csv)
            # タイムスタンプをインデックスに設定
            if 'time_stamp' in df.columns:
                df['time_stamp'] = pd.to_datetime(df['time_stamp'])
                df.set_index('time_stamp', inplace=True)

            # 予測値と実測値が含まれるか確認
            if 'actual' not in df.columns:
                if 'sens_temp' in df.columns:
                    df.rename(columns={'sens_temp': 'actual'}, inplace=True)
                else:
                    print(f"ゾーン {zone} の実測値が見つかりません")
                    continue

            # ホライゾンに対応する予測列が存在するか確認
            pred_col = f'pred_{horizon}'
            if pred_col not in df.columns:
                # 他のホライゾンを探す
                pred_cols = [col for col in df.columns if col.startswith('pred_')]
                if pred_cols:
                    print(f"ホライゾン {horizon}分の予測値が見つかりません。利用可能なホライゾン: {[int(col.split('_')[1]) for col in pred_cols]}")
                    # 一番近いホライゾンを使用
                    available_horizons = [int(col.split('_')[1]) for col in pred_cols]
                    closest_horizon = min(available_horizons, key=lambda x: abs(x - horizon))
                    print(f"代わりにホライゾン {closest_horizon}分を使用します")
                    pred_col = f'pred_{closest_horizon}'
                else:
                    print(f"ゾーン {zone} の予測値が見つかりません")
                    continue

            # 結果辞書に追加
            results_dict[str(zone)] = df
            print(f"ゾーン {zone} の予測結果を読み込みました: {latest_csv}")
        except Exception as e:
            print(f"ゾーン {zone} の予測結果の読み込みに失敗しました: {e}")

    return results_dict

def main():
    """メイン処理"""
    args = parse_args()

    # ゾーンリストの解析
    if args.zones == 'all':
        zones = list(range(12))  # 0から11までのゾーン
    else:
        zones = [int(z.strip()) for z in args.zones.split(',') if z.strip()]

    print(f"表示するゾーン: {zones}")
    print(f"予測ホライゾン: {args.horizon}分")

    # 結果ディレクトリの設定
    if args.results_dir:
        results_dir = args.results_dir
    else:
        # 一般的な結果ディレクトリを推測
        possible_dirs = [
            os.path.join(args.output_dir, 'horizon_analysis'),
            os.path.join(args.output_dir, 'sens_temp_predictions'),
            args.output_dir
        ]

        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                results_dir = dir_path
                break
        else:
            results_dir = args.output_dir

    print(f"結果ディレクトリ: {results_dir}")

    # 保存先ディレクトリの設定
    interactive_dir = os.path.join(results_dir, 'interactive')
    os.makedirs(interactive_dir, exist_ok=True)

    # 結果の読み込み
    results = load_results(results_dir, zones, args.horizon)

    if not results:
        print("表示できる予測結果がありません。まず分析を実行してください。")
        sys.exit(1)

    if len(results) == 0:
        print("表示できるゾーンがありません。")
        sys.exit(1)

    # インタラクティブな可視化の実行
    try:
        # 全ゾーンの可視化
        all_zones_paths = interactive_all_zones_timeseries(
            results=results,
            horizon=args.horizon,
            zones=zones,
            output_dir=interactive_dir
        )

        if all_zones_paths:
            print(f"全ゾーンの可視化を保存しました: {all_zones_paths[0]}")
            if args.open:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(all_zones_paths[0])}")
        else:
            print("全ゾーンの可視化に失敗しました。")

    except ImportError as e:
        print(f"インタラクティブな可視化に必要なパッケージがインストールされていません: {e}")
        print("インストール方法: pip install plotly nbformat kaleido")
    except Exception as e:
        print(f"可視化中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
