#!/usr/bin/env python
# coding: utf-8

"""
統合温度予測システム - メインスクリプト
直接温度予測と差分予測の両方をサポートする統合システム
リファクタリング版：モジュール化により大幅に簡素化
"""

import pandas as pd
import warnings
import argparse
from typing import List, Optional

warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import HORIZONS, L_ZONES, M_ZONES, R_ZONES

# データ前処理関数のインポート
from src.data.preprocessing import (
    filter_temperature_outliers,
    prepare_time_features
)

# 予測実行ランナー
from src.prediction_runners import PredictionRunner


def load_and_preprocess_data() -> pd.DataFrame:
    """
    データ読み込みと基本前処理
    
    Returns:
    --------
    df : DataFrame
        前処理済みデータ
    """
    print("\n## データ読み込み...")
    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]:,}行 x {df.shape[1]}列")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        raise

    print("\n## データ前処理...")
    df = prepare_time_features(df)
    df = filter_temperature_outliers(df)
    
    return df


def get_target_zones_and_horizons(df: pd.DataFrame, zones: Optional[List[int]], 
                                 horizons: Optional[List[int]], test_mode: bool) -> tuple:
    """
    対象ゾーンとホライゾンを決定
    
    Parameters:
    -----------
    df : DataFrame
        データフレーム
    zones : list or None
        指定ゾーン
    horizons : list or None  
        指定ホライゾン
    test_mode : bool
        テストモードかどうか
        
    Returns:
    --------
    target_zones, target_horizons : tuple
        対象ゾーンとホライゾンのタプル
    """
    # ゾーン設定
    temp_cols = [col for col in df.columns if 'sens_temp' in col and 'future' not in col]
    available_zones = sorted([int(col.split('_')[2]) for col in temp_cols])

    if zones is None:
        target_zones = available_zones[:2] if test_mode else available_zones
    else:
        target_zones = [z for z in zones if z in available_zones]

    # ホライゾン設定
    if horizons is None:
        target_horizons = [15] if test_mode else HORIZONS
    else:
        target_horizons = horizons

    print(f"対象ゾーン: {target_zones}")
    print(f"対象ホライゾン: {target_horizons}")
    
    return target_zones, target_horizons


def run_temperature_prediction(mode='both', test_mode=False, zones=None, horizons=None,
                             save_models=True, create_visualizations=True, comparison_only=False):
    """
    統合温度予測システムのメイン実行関数（リファクタリング版）

    Parameters:
    -----------
    mode : str
        実行モード ('direct', 'difference', 'both', 'comparison')
    test_mode : bool
        テストモードで実行するか
    zones : list
        対象ゾーン（Noneの場合は全ゾーン）
    horizons : list
        対象ホライゾン（Noneの場合は設定ファイルから）
    save_models : bool
        モデルを保存するか
    create_visualizations : bool
        可視化を作成するか
    comparison_only : bool
        比較分析のみ実行するか
    """
    print("# 🌡️ 統合温度予測システム (リファクタリング版)")
    print(f"## 実行モード: {mode}")

    if test_mode:
        print("[テストモード] 限定実行")

    # データ読み込みと前処理
    df = load_and_preprocess_data()
    
    # 対象設定
    target_zones, target_horizons = get_target_zones_and_horizons(
        df, zones, horizons, test_mode
    )

    # 予測実行ランナーの初期化
    runner = PredictionRunner(
        save_models=save_models,
        create_visualizations=create_visualizations
    )

    # 実行
    print(f"\n## 実行開始: {mode}モード")
    
    if comparison_only or mode == 'comparison':
        print("比較分析機能は今後実装予定")
        # TODO: 比較分析機能を実装
    elif mode == 'direct':
        runner.run_direct_prediction(df, target_zones, target_horizons)
    elif mode == 'difference':
        runner.run_difference_prediction(df, target_zones, target_horizons)
    elif mode == 'both':
        print("🎯 直接予測と差分予測の両方を実行します")
        runner.run_direct_prediction(df, target_zones, target_horizons)
        runner.run_difference_prediction(df, target_zones, target_horizons)
    else:
        print(f"不明なモード: {mode}")
        return

    print("\n" + "="*80)
    print("🎉 処理完了！")
    print("="*80)
    print("📁 結果は Output/ ディレクトリに保存されました")
    print("   - models/: 学習済みモデル")
    print("   - visualizations/: 可視化結果")


def main():
    """コマンドライン実行用のメイン関数"""
    parser = argparse.ArgumentParser(description='統合温度予測システム (リファクタリング版)')

    # 基本オプション
    parser.add_argument('--mode', choices=['direct', 'difference', 'both', 'comparison'],
                       default='difference', help='実行モード (デフォルト: difference)')
    parser.add_argument('--test', action='store_true', help='テストモードで実行')
    parser.add_argument('--zones', nargs='+', type=int, help='対象ゾーン番号')
    parser.add_argument('--horizons', nargs='+', type=int, help='対象ホライゾン（分）')

    # 詳細オプション
    parser.add_argument('--no-models', action='store_true', help='モデル保存をスキップ')
    parser.add_argument('--no-visualization', action='store_true', help='可視化をスキップ')
    parser.add_argument('--comparison-only', action='store_true', help='比較分析のみ実行')

    args = parser.parse_args()

    run_temperature_prediction(
        mode=args.mode,
        test_mode=args.test,
        zones=args.zones,
        horizons=args.horizons,
        save_models=not args.no_models,
        create_visualizations=not args.no_visualization,
        comparison_only=args.comparison_only
    )


if __name__ == "__main__":
    main()
