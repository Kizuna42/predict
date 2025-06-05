#!/usr/bin/env python
# coding: utf-8

"""
細かいフィルタリング閾値最適化スクリプト
段階的最適化により最適なフィルタリング閾値を見つけます
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
import time
import sys
import os

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import SMOOTHING_WINDOWS, FEATURE_SELECTION_THRESHOLD, TEST_SIZE

# データ前処理関数のインポート
from src.data.preprocessing import (
    filter_temperature_outliers,
    create_temperature_difference_targets,
    prepare_time_features,
    get_time_based_train_test_split,
    filter_high_value_targets
)

# 特徴量エンジニアリング関数のインポート
from src.data.feature_engineering import create_difference_prediction_pipeline

# モデル訓練関数のインポート
from src.models.training import train_temperature_difference_model

# 評価関数のインポート
from src.models.evaluation import evaluate_temperature_difference_model


def test_filtering_threshold(df, zone, horizon, percentile, time_diff_seconds):
    """
    指定されたパーセンタイル閾値でフィルタリングを実行し、性能を評価

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン
    percentile : float
        フィルタリングパーセンタイル
    time_diff_seconds : float
        データのサンプリング間隔

    Returns:
    --------
    dict
        評価結果
    """
    try:
        print(f"\\n🔍 {percentile:.1f}%ileフィルタリングテスト中...")

        # 差分予測用目的変数の作成
        df_with_diff_targets = create_temperature_difference_targets(
            df, [zone], [horizon], pd.Timedelta(seconds=time_diff_seconds)
        )

        # 差分予測専用特徴量エンジニアリング
        df_processed, selected_features, feature_info = create_difference_prediction_pipeline(
            df=df_with_diff_targets,
            zone_nums=[zone],
            horizons_minutes=[horizon],
            time_diff_seconds=time_diff_seconds,
            smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
            feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
        )

        diff_target_col = f'temp_diff_{zone}_future_{horizon}'
        if diff_target_col not in df_processed.columns:
            return {'error': f'目的変数 {diff_target_col} が見つかりません'}

        # データ準備
        feature_cols = [col for col in selected_features if col in df_processed.columns]
        valid_data = df_processed.dropna(subset=[diff_target_col] + feature_cols)

        if len(valid_data) < 100:
            return {'error': f'有効データが不足 ({len(valid_data)}行)'}

        # 高値フィルタリング
        abs_diff_col = f'abs_temp_diff_{zone}_future_{horizon}'
        valid_data[abs_diff_col] = valid_data[diff_target_col].abs()

        filtered_data, filter_info = filter_high_value_targets(
            valid_data, [abs_diff_col], percentile=percentile
        )

        if len(filtered_data) < 50:
            return {'error': f'フィルタ後のデータが不足 ({len(filtered_data)}行)'}

        # 時系列分割
        train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

        X_train = train_df[feature_cols]
        y_train_diff = train_df[diff_target_col]
        X_test = test_df[feature_cols]
        y_test_diff = test_df[diff_target_col]

        if len(X_train) < 30 or len(X_test) < 15:
            return {'error': f'分割後のデータが不足 (train: {len(X_train)}, test: {len(X_test)})'}

        # モデル訓練
        start_time = time.time()
        diff_model = train_temperature_difference_model(X_train, y_train_diff)
        training_time = time.time() - start_time

        # 予測
        y_pred_diff = diff_model.predict(X_test)

        # 評価
        current_temps_test = test_df[f'sens_temp_{zone}']
        diff_metrics = evaluate_temperature_difference_model(
            y_test_diff, y_pred_diff, current_temps_test
        )

        # 結果の整理
        result = {
            'percentile': percentile,
            'original_data_count': len(valid_data),
            'filtered_data_count': len(filtered_data),
            'data_reduction_rate': (len(valid_data) - len(filtered_data)) / len(valid_data) * 100,
            'train_count': len(X_train),
            'test_count': len(X_test),
            'training_time': training_time,
            'threshold_value': filter_info['thresholds'][abs_diff_col],
            'diff_rmse': diff_metrics['diff_rmse'],
            'diff_mae': diff_metrics['diff_mae'],
            'direction_accuracy': diff_metrics['direction_accuracy'],
            'restoration_rmse': diff_metrics.get('restoration_rmse', np.nan),
            'restoration_mae': diff_metrics.get('restoration_mae', np.nan),
            'restoration_r2': diff_metrics.get('restoration_r2', np.nan),
            'error': None
        }

        print(f"✅ {percentile:.1f}%ile完了: RMSE={result['restoration_rmse']:.4f}, R²={result['restoration_r2']:.4f}")
        return result

    except Exception as e:
        print(f"❌ {percentile:.1f}%ileエラー: {e}")
        return {
            'percentile': percentile,
            'error': str(e),
            'diff_rmse': np.nan,
            'restoration_rmse': np.nan,
            'restoration_r2': np.nan
        }


def optimize_filtering_threshold(zone=1, horizon=15):
    """
    段階的フィルタリング閾値最適化を実行

    Parameters:
    -----------
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン

    Returns:
    --------
    dict
        最適化結果
    """
    print("🔥 段階的フィルタリング閾値最適化開始")
    print(f"対象: ゾーン{zone}, 予測ホライゾン{horizon}分")

    # データ読み込み
    data_path = project_root / "AllDayData.csv"
    print(f"\\nデータ読み込み中: {data_path}")

    df = pd.read_csv(data_path)
    time_diff_seconds = pd.to_datetime(df['datetime']).diff().dt.total_seconds().median()

    # 基本前処理
    df = filter_temperature_outliers(df)
    df = prepare_time_features(df)

    all_results = []

    # ステップ1: 粗い探索（20-60%、5%刻み）
    print("\\n🔍 ステップ1: 粗い探索（20-60%、5%刻み）")
    coarse_percentiles = np.arange(20, 65, 5)
    coarse_results = []

    for percentile in coarse_percentiles:
        result = test_filtering_threshold(df, zone, horizon, percentile, time_diff_seconds)
        coarse_results.append(result)
        all_results.append(result)

    # 粗い探索での最良結果を特定
    valid_coarse = [r for r in coarse_results if r.get('error') is None]
    if not valid_coarse:
        print("❌ 粗い探索で有効な結果が得られませんでした")
        return {'error': '粗い探索失敗', 'all_results': all_results}

    best_coarse = min(valid_coarse, key=lambda x: x['restoration_rmse'])
    best_percentile = best_coarse['percentile']
    print(f"\\n🎯 粗い探索での最良: {best_percentile}%ile (RMSE: {best_coarse['restoration_rmse']:.4f})")

    # ステップ2: 中程度探索（最良結果周辺、2%刻み）
    print(f"\\n🔍 ステップ2: 中程度探索（{best_percentile}%ile周辺、2%刻み）")
    medium_range = np.arange(max(20, best_percentile - 6), min(65, best_percentile + 8), 2)
    medium_results = []

    for percentile in medium_range:
        if percentile not in [r['percentile'] for r in coarse_results]:  # 既に実行済みをスキップ
            result = test_filtering_threshold(df, zone, horizon, percentile, time_diff_seconds)
            medium_results.append(result)
            all_results.append(result)

    # 中程度探索での最良結果を特定
    all_valid = [r for r in all_results if r.get('error') is None]
    best_medium = min(all_valid, key=lambda x: x['restoration_rmse'])
    best_percentile = best_medium['percentile']
    print(f"\\n🎯 中程度探索での最良: {best_percentile}%ile (RMSE: {best_medium['restoration_rmse']:.4f})")

    # ステップ3: 細かい探索（最良結果周辺、0.5%刻み）
    print(f"\\n🔍 ステップ3: 細かい探索（{best_percentile}%ile周辺、0.5%刻み）")
    fine_range = np.arange(max(20, best_percentile - 2), min(65, best_percentile + 2.5), 0.5)
    fine_results = []

    for percentile in fine_range:
        if percentile not in [r['percentile'] for r in all_results]:  # 既に実行済みをスキップ
            result = test_filtering_threshold(df, zone, horizon, percentile, time_diff_seconds)
            fine_results.append(result)
            all_results.append(result)

    # 最終的な最良結果を特定
    final_valid = [r for r in all_results if r.get('error') is None]
    if not final_valid:
        print("❌ 有効な結果が得られませんでした")
        return {'error': '最適化失敗', 'all_results': all_results}

    best_result = min(final_valid, key=lambda x: x['restoration_rmse'])

    # 結果サマリー
    print("\\n" + "="*80)
    print("🎉 最適化完了！")
    print("="*80)
    print(f"最適パーセンタイル: {best_result['percentile']:.1f}%ile")
    print(f"復元温度RMSE: {best_result['restoration_rmse']:.4f}℃")
    print(f"復元温度R²: {best_result['restoration_r2']:.4f}")
    print(f"方向精度: {best_result['direction_accuracy']:.1f}%")
    print(f"データ削減率: {best_result['data_reduction_rate']:.1f}%")
    print(f"フィルタ後データ数: {best_result['filtered_data_count']:,}行")

    # 結果を保存
    output_dir = project_root / "Output"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"filtering_optimization_zone{zone}_horizon{horizon}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_result': best_result,
            'all_results': all_results,
            'optimization_summary': {
                'zone': zone,
                'horizon': horizon,
                'total_tests': len(all_results),
                'best_percentile': best_result['percentile'],
                'best_rmse': best_result['restoration_rmse'],
                'best_r2': best_result['restoration_r2']
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\\n結果を保存: {results_file}")

    return {
        'best_result': best_result,
        'all_results': all_results,
        'optimization_summary': {
            'zone': zone,
            'horizon': horizon,
            'total_tests': len(all_results),
            'best_percentile': best_result['percentile'],
            'best_rmse': best_result['restoration_rmse'],
            'best_r2': best_result['restoration_r2']
        }
    }


if __name__ == "__main__":
    # コマンドライン引数のパース
    zone = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # 最適化実行
    results = optimize_filtering_threshold(zone, horizon)

    if 'error' not in results:
        print("\\n✅ 最適化が正常に完了しました")
    else:
        print(f"\\n❌ 最適化でエラーが発生: {results['error']}")
