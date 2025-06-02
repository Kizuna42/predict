#!/usr/bin/env python
# coding: utf-8

"""
細かいフィルタリング閾値最適化スクリプト
60%以下の範囲でより細かい刻みで最適な閾値を見つける
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
import time

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


def test_filtering_threshold_fine(df, zone, horizon, percentile, time_diff_seconds):
    """
    指定されたパーセンタイル閾値でフィルタリングを実行し、性能を評価（細かい最適化用）

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン
    percentile : float
        フィルタリングパーセンタイル（20-60）
    time_diff_seconds : float
        データのサンプリング間隔

    Returns:
    --------
    dict
        評価結果
    """
    try:
        print(f"\n🔍 {percentile:.1f}%ileフィルタリングテスト中...")

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

        if len(filtered_data) < 50:  # より厳しい最小データ数要件
            return {'error': f'フィルタ後のデータが不足 ({len(filtered_data)}行)'}

        # 時系列分割
        train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

        X_train = train_df[feature_cols]
        y_train_diff = train_df[diff_target_col]
        X_test = test_df[feature_cols]
        y_test_diff = test_df[diff_target_col]

        if len(X_train) < 30 or len(X_test) < 15:  # より厳しい最小データ数要件
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

        # 追加の統計情報
        diff_std = y_test_diff.std()
        diff_mean_abs = y_test_diff.abs().mean()
        pred_std = pd.Series(y_pred_diff).std()

        # データ品質指標
        large_changes = (y_test_diff.abs() > diff_mean_abs * 2).sum()
        small_changes = (y_test_diff.abs() < diff_mean_abs * 0.5).sum()

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
            'large_change_accuracy': diff_metrics.get('large_change_accuracy', np.nan),
            'restoration_rmse': diff_metrics.get('restoration_rmse', np.nan),
            'restoration_mae': diff_metrics.get('restoration_mae', np.nan),
            'restoration_r2': diff_metrics.get('restoration_r2', np.nan),
            # 追加統計
            'diff_std': diff_std,
            'diff_mean_abs': diff_mean_abs,
            'pred_std': pred_std,
            'large_changes_count': large_changes,
            'small_changes_count': small_changes,
            'data_diversity_score': large_changes + small_changes,  # データの多様性指標
            'error': None
        }

        print(f"✅ {percentile:.1f}%ile完了: RMSE={result['restoration_rmse']:.4f}, R²={result['restoration_r2']:.4f}, データ数={len(filtered_data)}")
        return result

    except Exception as e:
        print(f"❌ {percentile:.1f}%ileエラー: {e}")
        return {
            'percentile': percentile,
            'error': str(e),
            'diff_rmse': np.nan,
            'diff_mae': np.nan,
            'restoration_rmse': np.nan,
            'restoration_mae': np.nan,
            'restoration_r2': np.nan
        }


def optimize_filtering_fine_tuned(zone=1, horizon=15):
    """
    細かいフィルタリング閾値最適化を実行する関数

    複数の範囲で段階的に最適化：
    1. 20-60%を5%刻み（粗い探索）
    2. 最良結果周辺を2%刻み（中程度探索）
    3. さらに最良結果周辺を0.5%刻み（細かい探索）

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
    print("🔥 細かいフィルタリング閾値最適化開始")
    print(f"対象: ゾーン{zone}, {horizon}分後予測")
    print("段階的最適化: 粗い探索 → 中程度探索 → 細かい探索")

    # データ読み込み
    print("\n## データ読み込み...")
    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None

    # 基本前処理
    print("\n## データ前処理...")
    df = prepare_time_features(df)
    df = filter_temperature_outliers(df)

    # 時間差の計算
    time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

    all_results = []

    # ステップ1: 粗い探索（20-60%を5%刻み）
    print("\n" + "="*80)
    print("📊 ステップ1: 粗い探索（20-60%を5%刻み）")
    print("="*80)

    coarse_percentiles = np.arange(20, 65, 5)  # [20, 25, 30, 35, 40, 45, 50, 55, 60]
    coarse_results = []

    for i, percentile in enumerate(coarse_percentiles, 1):
        print(f"\n[{i}/{len(coarse_percentiles)}] {percentile}%ileテスト実行中...")
        result = test_filtering_threshold_fine(df, zone, horizon, percentile, time_diff_seconds)
        coarse_results.append(result)
        all_results.append(result)

    # 粗い探索の最良結果を特定
    valid_coarse = [r for r in coarse_results if r['error'] is None and not np.isnan(r['restoration_rmse'])]
    if not valid_coarse:
        print("❌ 粗い探索で有効な結果がありませんでした")
        return None

    best_coarse = min(valid_coarse, key=lambda x: x['restoration_rmse'])
    print(f"\n🏆 粗い探索最良結果: {best_coarse['percentile']}%ile (RMSE: {best_coarse['restoration_rmse']:.4f})")

    # ステップ2: 中程度探索（最良結果±10%を2%刻み）
    print("\n" + "="*80)
    print("📊 ステップ2: 中程度探索（最良結果周辺を2%刻み）")
    print("="*80)

    center = best_coarse['percentile']
    medium_start = max(20, center - 10)
    medium_end = min(60, center + 10)
    medium_percentiles = np.arange(medium_start, medium_end + 1, 2)

    # 既にテスト済みの値を除外
    tested_percentiles = {r['percentile'] for r in coarse_results}
    medium_percentiles = [p for p in medium_percentiles if p not in tested_percentiles]

    print(f"中程度探索範囲: {medium_start}%-{medium_end}% (中心: {center}%)")

    medium_results = []
    for i, percentile in enumerate(medium_percentiles, 1):
        print(f"\n[{i}/{len(medium_percentiles)}] {percentile}%ileテスト実行中...")
        result = test_filtering_threshold_fine(df, zone, horizon, percentile, time_diff_seconds)
        medium_results.append(result)
        all_results.append(result)

    # 中程度探索を含めた最良結果を特定
    all_valid = [r for r in all_results if r['error'] is None and not np.isnan(r['restoration_rmse'])]
    best_medium = min(all_valid, key=lambda x: x['restoration_rmse'])
    print(f"\n🏆 中程度探索最良結果: {best_medium['percentile']}%ile (RMSE: {best_medium['restoration_rmse']:.4f})")

    # ステップ3: 細かい探索（最良結果±5%を0.5%刻み）
    print("\n" + "="*80)
    print("📊 ステップ3: 細かい探索（最良結果周辺を0.5%刻み）")
    print("="*80)

    center = best_medium['percentile']
    fine_start = max(20, center - 5)
    fine_end = min(60, center + 5)
    fine_percentiles = np.arange(fine_start, fine_end + 0.5, 0.5)

    # 既にテスト済みの値を除外
    tested_percentiles = {r['percentile'] for r in all_results}
    fine_percentiles = [p for p in fine_percentiles if p not in tested_percentiles]

    print(f"細かい探索範囲: {fine_start}%-{fine_end}% (中心: {center}%)")

    fine_results = []
    for i, percentile in enumerate(fine_percentiles, 1):
        print(f"\n[{i}/{len(fine_percentiles)}] {percentile:.1f}%ileテスト実行中...")
        result = test_filtering_threshold_fine(df, zone, horizon, percentile, time_diff_seconds)
        fine_results.append(result)
        all_results.append(result)

    # 最終結果分析
    print("\n" + "="*80)
    print("📊 最終結果分析")
    print("="*80)

    # 有効な結果のみを抽出
    valid_results = [r for r in all_results if r['error'] is None and not np.isnan(r['restoration_rmse'])]

    if not valid_results:
        print("❌ 有効な結果がありませんでした")
        return None

    # 各指標での最良結果
    best_rmse = min(valid_results, key=lambda x: x['restoration_rmse'])
    best_mae = min(valid_results, key=lambda x: x['restoration_mae'])
    best_r2 = max(valid_results, key=lambda x: x['restoration_r2'])
    best_direction = max(valid_results, key=lambda x: x['direction_accuracy'])

    print(f"\n🏆 最良結果:")
    print(f"最低RMSE: {best_rmse['restoration_rmse']:.4f} ({best_rmse['percentile']:.1f}%ile)")
    print(f"最低MAE: {best_mae['restoration_mae']:.4f} ({best_mae['percentile']:.1f}%ile)")
    print(f"最高R²: {best_r2['restoration_r2']:.4f} ({best_r2['percentile']:.1f}%ile)")
    print(f"最高方向精度: {best_direction['direction_accuracy']:.1f}% ({best_direction['percentile']:.1f}%ile)")

    # 改良された総合スコアの計算
    for result in valid_results:
        # 正規化スコア（0-1）
        rmse_range = max(r['restoration_rmse'] for r in valid_results) - min(r['restoration_rmse'] for r in valid_results)
        mae_range = max(r['restoration_mae'] for r in valid_results) - min(r['restoration_mae'] for r in valid_results)
        r2_range = max(r['restoration_r2'] for r in valid_results) - min(r['restoration_r2'] for r in valid_results)

        if rmse_range > 0:
            rmse_score = 1 - (result['restoration_rmse'] - min(r['restoration_rmse'] for r in valid_results)) / rmse_range
        else:
            rmse_score = 1.0

        if mae_range > 0:
            mae_score = 1 - (result['restoration_mae'] - min(r['restoration_mae'] for r in valid_results)) / mae_range
        else:
            mae_score = 1.0

        if r2_range > 0:
            r2_score = (result['restoration_r2'] - min(r['restoration_r2'] for r in valid_results)) / r2_range
        else:
            r2_score = 1.0

        # データ量ボーナス（十分なデータがある場合）
        data_bonus = min(1.0, result['filtered_data_count'] / 50000)  # 50,000行を基準

        # 重み付け総合スコア（RMSE重視、データ量も考慮）
        result['composite_score'] = (0.4 * rmse_score + 0.25 * mae_score + 0.25 * r2_score + 0.1 * data_bonus)

    # 総合最良結果
    best_overall = max(valid_results, key=lambda x: x['composite_score'])

    print(f"\n🎯 総合最適閾値: {best_overall['percentile']:.1f}%ile")
    print(f"  復元RMSE: {best_overall['restoration_rmse']:.4f}")
    print(f"  復元MAE: {best_overall['restoration_mae']:.4f}")
    print(f"  復元R²: {best_overall['restoration_r2']:.4f}")
    print(f"  方向精度: {best_overall['direction_accuracy']:.1f}%")
    print(f"  データ削減: {best_overall['data_reduction_rate']:.1f}%")
    print(f"  フィルタ後データ数: {best_overall['filtered_data_count']:,}行")
    print(f"  総合スコア: {best_overall['composite_score']:.4f}")

    # 詳細結果テーブル（上位10位）
    print(f"\n📋 詳細結果（上位10位）:")
    print("Percentile | RMSE   | MAE    | R²     | Dir%  | Data数  | Score")
    print("-" * 70)
    top_results = sorted(valid_results, key=lambda x: x['composite_score'], reverse=True)[:10]
    for result in top_results:
        print(f"{result['percentile']:9.1f}% | {result['restoration_rmse']:6.4f} | "
              f"{result['restoration_mae']:6.4f} | {result['restoration_r2']:6.4f} | "
              f"{result['direction_accuracy']:5.1f} | {result['filtered_data_count']:7,} | "
              f"{result['composite_score']:5.3f}")

    # 結果保存
    output_dir = Path("Output")
    output_dir.mkdir(exist_ok=True)

    optimization_result = {
        'zone': zone,
        'horizon': horizon,
        'optimization_type': 'fine_tuned',
        'total_tests': len(all_results),
        'valid_tests': len(valid_results),
        'best_overall': best_overall,
        'best_rmse': best_rmse,
        'best_mae': best_mae,
        'best_r2': best_r2,
        'best_direction': best_direction,
        'top_10_results': top_results,
        'all_results': all_results,
        'valid_results': valid_results,
        'coarse_results': coarse_results,
        'medium_results': medium_results,
        'fine_results': fine_results
    }

    result_file = output_dir / f"filtering_optimization_fine_tuned_zone_{zone}_horizon_{horizon}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        # NaN値をNullに変換してJSON保存
        json.dump(optimization_result, f, indent=2, default=lambda x: None if pd.isna(x) else x, ensure_ascii=False)

    print(f"\n💾 結果保存: {result_file}")

    return optimization_result


if __name__ == "__main__":
    # 細かい最適化実行
    result = optimize_filtering_fine_tuned(
        zone=1,
        horizon=15
    )

    if result:
        print(f"\n🎉 細かい最適化完了！")
        print(f"推奨閾値: {result['best_overall']['percentile']:.1f}%ile")
        print(f"RMSE改善: {result['best_overall']['restoration_rmse']:.4f}")
        print(f"総テスト数: {result['total_tests']}回")
    else:
        print("\n❌ 細かい最適化に失敗しました")
