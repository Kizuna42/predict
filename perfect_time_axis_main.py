#!/usr/bin/env python
# coding: utf-8

"""
完璧な時間軸修正システム実行スクリプト
予測値と同じ時刻（予測対象時刻）の実測値を正しく比較
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import HORIZONS, OUTPUT_DIR

# 完璧な可視化システムのインポート
from src.utils.perfect_time_axis_visualization import (
    create_perfect_visualization_for_zone,
    create_perfect_visualization_for_all_zones,
    create_comprehensive_perfect_visualization,
    get_future_actual_values,
    plot_perfect_time_axis_comparison
)

# データ前処理のインポート
from src.data.preprocessing import prepare_time_features, create_future_targets


def load_original_data():
    """
    元のデータフレームを読み込む
    """
    print("元のデータフレームを読み込み中...")

    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")

        # 時間特徴量の準備
        df = prepare_time_features(df)

        return df

    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None


def create_mock_results_with_perfect_data(original_df, zones, horizons):
    """
    完璧な時間軸検証用のモックデータを作成
    """
    print("完璧な時間軸検証用のモックデータを作成中...")

    results_dict = {}

    # 時間間隔の推定
    time_diff = original_df.index.to_series().diff().dropna().value_counts().index[0]

    for zone in zones:
        results_dict[zone] = {}
        temp_col = f'sens_temp_{zone}'

        if temp_col not in original_df.columns:
            print(f"警告: {temp_col} が見つかりません")
            continue

        for horizon in horizons:
            print(f"  ゾーン {zone}, ホライゾン {horizon}分のデータ作成中...")

            # 目的変数の作成
            df_with_targets = create_future_targets(original_df, [zone], [horizon], time_diff)
            target_col = f'sens_temp_{zone}_future_{horizon}'

            # 最新2000ポイントをテストデータとして使用
            test_data = df_with_targets.iloc[-2000:].copy()

            # test_yの作成
            test_y = test_data[target_col].dropna()

            # 予測値の作成（実際の未来値に小さなノイズを加える）
            # これにより、完璧な時間軸比較が可能
            np.random.seed(42)  # 再現性のため
            noise_std = test_y.std() * 0.05  # 5%のノイズ
            test_predictions = test_y.values + np.random.normal(0, noise_std, len(test_y))

            # 結果の格納
            results_dict[zone][horizon] = {
                'test_data': test_data,
                'test_y': test_y,
                'test_predictions': test_predictions,
                'feature_importance': pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(10)],
                    'importance': np.random.random(10)
                })
            }

            print(f"    データ長: test_data={len(test_data)}, test_y={len(test_y)}, predictions={len(test_predictions)}")

    return results_dict


def run_perfect_time_axis_demonstration(target_zones=None, target_horizons=None):
    """
    完璧な時間軸修正のデモンストレーション実行

    Parameters:
    -----------
    target_zones : list, optional
        対象ゾーンのリスト
    target_horizons : list, optional
        対象ホライゾンのリスト
    """
    print("=" * 100)
    print("🚀 完璧な時間軸修正システム - デモンストレーション")
    print("=" * 100)

    # 対象の設定
    if target_horizons is None:
        target_horizons = HORIZONS[:2]  # 最初の2つのホライゾンのみ

    if target_zones is None:
        target_zones = [1, 2, 3, 4]  # 最初の4ゾーン

    print(f"対象ホライゾン: {target_horizons}")
    print(f"対象ゾーン: {target_zones}")

    # 元データの読み込み
    original_df = load_original_data()
    if original_df is None:
        print("エラー: 元データの読み込みに失敗しました")
        return

    # モックデータの作成
    results_dict = create_mock_results_with_perfect_data(original_df, target_zones, target_horizons)

    # 出力ディレクトリの作成
    perfect_output_dir = os.path.join(OUTPUT_DIR, "perfect_time_axis")
    os.makedirs(perfect_output_dir, exist_ok=True)

    # 包括的完璧可視化の実行
    comprehensive_result = create_comprehensive_perfect_visualization(
        results_dict=results_dict,
        original_df=original_df,
        horizons=target_horizons,
        save_dir=perfect_output_dir
    )

    # 結果の保存
    import json
    result_path = os.path.join(perfect_output_dir, 'perfect_time_axis_results.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_result, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n📁 完璧な時間軸修正結果保存: {result_path}")

    return comprehensive_result


def run_single_zone_perfect_demo(zone=1, horizon=15):
    """
    単一ゾーンの完璧な時間軸修正デモ

    Parameters:
    -----------
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    """
    print(f"\n{'='*80}")
    print(f"🎯 ゾーン {zone} - {horizon}分予測の完璧な時間軸修正デモ")
    print(f"{'='*80}")

    # 元データの読み込み
    original_df = load_original_data()
    if original_df is None:
        return

    # モックデータの作成
    results_dict = create_mock_results_with_perfect_data(original_df, [zone], [horizon])

    # 出力ディレクトリ
    demo_output_dir = os.path.join(OUTPUT_DIR, "perfect_demo")
    os.makedirs(demo_output_dir, exist_ok=True)

    # 単一ゾーンの完璧可視化
    result = create_perfect_visualization_for_zone(
        results_dict=results_dict,
        original_df=original_df,
        zone=zone,
        horizon=horizon,
        save_dir=demo_output_dir,
        sample_size=300
    )

    if result['success']:
        print(f"\n✅ 完璧な可視化作成成功!")
        if result['metrics']:
            metrics = result['metrics']
            print(f"📊 性能指標:")
            print(f"  MAE: {metrics['mae']:.3f}°C")
            print(f"  RMSE: {metrics['rmse']:.3f}°C")
            print(f"  相関: {metrics['correlation']:.3f}")
            print(f"  有効データ点数: {metrics['valid_points']}/{metrics['total_points']}")

        if result['file_paths']:
            print(f"📁 生成ファイル:")
            for path in result['file_paths']:
                print(f"  - {path}")
    else:
        print(f"❌ 可視化作成失敗: {result['error_message']}")

    return result


def demonstrate_time_axis_concepts():
    """
    時間軸の概念を詳しく説明するデモンストレーション
    """
    print(f"\n{'='*100}")
    print(f"📚 時間軸修正の概念説明")
    print(f"{'='*100}")

    print(f"\n🔍 問題の所在:")
    print(f"  従来の方法:")
    print(f"    - 入力時刻: 13:00")
    print(f"    - 実測値: 13:00の温度（例：20.5°C）")
    print(f"    - 予測値: 13:15の温度（例：21.2°C）")
    print(f"    - プロット: 両方とも13:00にプロット ← 問題！")

    print(f"\n  ❌ 問題点:")
    print(f"    - 異なる時刻の値を同じ時刻で比較している")
    print(f"    - 予測が実測の「後追い」に見える")
    print(f"    - モデルの真の性能が評価できない")

    print(f"\n✅ 完璧な解決策:")
    print(f"  方法1: 予測値の時間軸修正")
    print(f"    - 入力時刻: 13:00")
    print(f"    - 実測値: 13:00の温度（例：20.5°C） → 13:00にプロット")
    print(f"    - 予測値: 13:15の温度（例：21.2°C） → 13:15にプロット")

    print(f"\n  方法2: 同じ時刻での比較（推奨）")
    print(f"    - 予測対象時刻: 13:15")
    print(f"    - 実測値: 13:15の実際の温度（例：21.0°C） → 13:15にプロット")
    print(f"    - 予測値: 13:15の予測温度（例：21.2°C） → 13:15にプロット")
    print(f"    - 比較: 同じ時刻の値同士で正確な比較が可能")

    print(f"\n🎯 本システムの特徴:")
    print(f"  1. 3つの表示方法を比較")
    print(f"     - 従来の間違った方法")
    print(f"     - 部分修正された方法")
    print(f"     - 完璧な方法")

    print(f"\n  2. 正確な性能指標")
    print(f"     - MAE（平均絶対誤差）")
    print(f"     - RMSE（二乗平均平方根誤差）")
    print(f"     - 相関係数")

    print(f"\n  3. 自動的な未来値取得")
    print(f"     - 予測対象時刻の実測値を自動取得")
    print(f"     - データ不足の場合の適切な処理")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='完璧な時間軸修正システム')
    parser.add_argument('--demo', choices=['single', 'comprehensive', 'concepts'],
                       default='single',
                       help='デモンストレーションの種類')
    parser.add_argument('--zone', type=int, default=1,
                       help='単一ゾーンデモのゾーン番号')
    parser.add_argument('--horizon', type=int, default=15,
                       help='単一ゾーンデモの予測ホライゾン')
    parser.add_argument('--zones', type=int, nargs='+',
                       help='包括デモの対象ゾーン')
    parser.add_argument('--horizons', type=int, nargs='+',
                       help='包括デモの対象ホライゾン')

    args = parser.parse_args()

    if args.demo == 'concepts':
        demonstrate_time_axis_concepts()
    elif args.demo == 'single':
        run_single_zone_perfect_demo(args.zone, args.horizon)
    elif args.demo == 'comprehensive':
        run_perfect_time_axis_demonstration(args.zones, args.horizons)

    print(f"\n{'='*100}")
    print(f"🎉 完璧な時間軸修正システム実行完了")
    print(f"{'='*100}")
