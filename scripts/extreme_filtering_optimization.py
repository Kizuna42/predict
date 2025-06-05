#!/usr/bin/env python
# coding: utf-8

"""
極端フィルタリング最適化スクリプト
上司要求: 「目的変数の値が大きいデータのみに絞って学習」
- より激しい温度変化データのみでの学習
- 5%ile～20%ileの極端フィルタリング検証
- 学習データ効率性の最大化
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import lightgbm as lgb

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

# 改善版関数のインポート
from scripts.advanced_model_improvements import (
    create_advanced_temporal_features,
    create_robust_training_weights,
    train_advanced_temperature_difference_model
)

# 評価関数のインポート
from src.models.evaluation import evaluate_temperature_difference_model


def test_extreme_filtering_threshold(df, zone, horizon, percentile, time_diff_seconds, use_advanced_features=True):
    """
    極端フィルタリング閾値でのテスト

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン
    percentile : float
        フィルタリングパーセンタイル（極端に小さい値）
    time_diff_seconds : float
        データのサンプリング間隔
    use_advanced_features : bool
        高度な特徴量を使用するか

    Returns:
    --------
    dict
        評価結果
    """
    try:
        print(f"\\n🔥 極端{percentile:.1f}%ileフィルタリングテスト中...")

        # 差分予測用目的変数の作成
        df_with_diff_targets = create_temperature_difference_targets(
            df, [zone], [horizon], pd.Timedelta(seconds=time_diff_seconds)
        )

        # 基本特徴量エンジニアリング
        df_processed, selected_features, feature_info = create_difference_prediction_pipeline(
            df=df_with_diff_targets,
            zone_nums=[zone],
            horizons_minutes=[horizon],
            time_diff_seconds=time_diff_seconds,
            smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
            feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
        )

        # 高度な時間特徴量の追加（オプション）
        if use_advanced_features:
            df_enhanced, temporal_features = create_advanced_temporal_features(
                df_processed, [zone], [horizon], time_diff_seconds
            )
            all_features = selected_features + temporal_features
            all_features = [f for f in all_features if f in df_enhanced.columns]
        else:
            df_enhanced = df_processed
            all_features = selected_features

        diff_target_col = f'temp_diff_{zone}_future_{horizon}'
        if diff_target_col not in df_enhanced.columns:
            return {'error': f'目的変数 {diff_target_col} が見つかりません'}

        # データ準備
        valid_data = df_enhanced.dropna(subset=[diff_target_col] + all_features)

        if len(valid_data) < 100:
            return {'error': f'有効データが不足 ({len(valid_data)}行)'}

        # 極端高値フィルタリング
        abs_diff_col = f'abs_temp_diff_{zone}_future_{horizon}'
        valid_data[abs_diff_col] = valid_data[diff_target_col].abs()

        filtered_data, filter_info = filter_high_value_targets(
            valid_data, [abs_diff_col], percentile=percentile
        )

        if len(filtered_data) < 50:
            return {'error': f'極端フィルタ後のデータが不足 ({len(filtered_data)}行)'}

        print(f"   📊 データ削減: {len(valid_data):,} → {len(filtered_data):,}行 ({(1-len(filtered_data)/len(valid_data))*100:.1f}%削減)")
        print(f"   🎯 閾値: {filter_info['thresholds'][abs_diff_col]:.4f}℃")

        # 時系列分割
        train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

        X_train = train_df[all_features]
        y_train_diff = train_df[diff_target_col]
        X_test = test_df[all_features]
        y_test_diff = test_df[diff_target_col]
        current_temps_test = test_df[f'sens_temp_{zone}']

        if len(X_train) < 30 or len(X_test) < 15:
            return {'error': f'分割後のデータが不足 (train: {len(X_train)}, test: {len(X_test)})'}

        # 極端重み付けの作成
        extreme_weights = create_robust_training_weights(y_train_diff, residual_threshold_factor=1.5)

        # 高性能モデル訓練
        model = train_advanced_temperature_difference_model(
            X_train, y_train_diff, sample_weights=extreme_weights
        )

        # 予測と評価
        y_pred_diff = model.predict(X_test)
        metrics = evaluate_temperature_difference_model(
            y_test_diff, y_pred_diff, current_temps_test
        )

        # 結果の整理
        result = {
            'percentile': percentile,
            'use_advanced_features': use_advanced_features,
            'original_data_count': len(valid_data),
            'filtered_data_count': len(filtered_data),
            'data_reduction_rate': (len(valid_data) - len(filtered_data)) / len(valid_data) * 100,
            'train_count': len(X_train),
            'test_count': len(X_test),
            'feature_count': len(all_features),
            'threshold_value': filter_info['thresholds'][abs_diff_col],

            # 性能指標
            'diff_rmse': metrics['diff_rmse'],
            'diff_mae': metrics['diff_mae'],
            'direction_accuracy': metrics['direction_accuracy'],
            'restoration_rmse': metrics.get('restoration_rmse', np.nan),
            'restoration_mae': metrics.get('restoration_mae', np.nan),
            'restoration_r2': metrics.get('restoration_r2', np.nan),
            'small_change_sensitivity': metrics.get('small_change_sensitivity', np.nan),
            'large_change_accuracy': metrics.get('large_change_accuracy', np.nan),

            'error': None
        }

        print(f"   ✅ 完了: RMSE={result['restoration_rmse']:.4f}℃, R²={result['restoration_r2']:.4f}, 方向={result['direction_accuracy']:.1f}%")
        return result

    except Exception as e:
        print(f"   ❌ エラー: {e}")
        return {
            'percentile': percentile,
            'error': str(e),
            'restoration_rmse': np.nan,
            'restoration_r2': np.nan,
            'direction_accuracy': np.nan
        }


def compare_with_current_best(new_results, zone, horizon):
    """
    現在のベストモデルとの比較

    Parameters:
    -----------
    new_results : dict
        新しい結果
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン

    Returns:
    --------
    dict
        比較結果
    """
    current_best_file = project_root / "Output" / f"advanced_model_analysis_zone{zone}_horizon{horizon}.json"

    if not current_best_file.exists():
        return {'error': '現在のベストモデル結果が見つかりません'}

    with open(current_best_file, 'r', encoding='utf-8') as f:
        current_data = json.load(f)

    current_metrics = current_data['advanced_metrics']

    comparison = {
        'current_best': {
            'rmse': current_metrics['restoration_rmse'],
            'r2': current_metrics['restoration_r2'],
            'direction_accuracy': current_metrics['direction_accuracy'],
            'mae': current_metrics['restoration_mae']
        },
        'new_extreme': {
            'rmse': new_results['restoration_rmse'],
            'r2': new_results['restoration_r2'],
            'direction_accuracy': new_results['direction_accuracy'],
            'mae': new_results['restoration_mae']
        }
    }

    # 改善率計算
    rmse_improvement = (comparison['current_best']['rmse'] - comparison['new_extreme']['rmse']) / comparison['current_best']['rmse'] * 100
    r2_improvement = (comparison['new_extreme']['r2'] - comparison['current_best']['r2']) / comparison['current_best']['r2'] * 100
    direction_improvement = (comparison['new_extreme']['direction_accuracy'] - comparison['current_best']['direction_accuracy']) / comparison['current_best']['direction_accuracy'] * 100
    mae_improvement = (comparison['current_best']['mae'] - comparison['new_extreme']['mae']) / comparison['current_best']['mae'] * 100

    comparison['improvements'] = {
        'rmse': rmse_improvement,
        'r2': r2_improvement,
        'direction_accuracy': direction_improvement,
        'mae': mae_improvement
    }

    return comparison


def extreme_filtering_optimization(zone=1, horizon=15):
    """
    極端フィルタリング最適化の実行

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
    print("🔥 極端フィルタリング最適化開始")
    print("=" * 60)
    print("上司要求: 「目的変数の値が大きいデータのみに絞って学習」")
    print(f"対象: ゾーン{zone}, 予測ホライゾン{horizon}分")

    # データ読み込み
    data_path = project_root / "AllDayData.csv"
    print(f"\\nデータ読み込み中: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)

    # 時間列の設定
    if 'time_stamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['time_stamp'])
        df = df.set_index('datetime')

    time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

    # 基本前処理
    df = filter_temperature_outliers(df)
    df = prepare_time_features(df)

    all_results = []

    # 極端フィルタリング閾値の検証（5%～20%、1%刻み）
    print("\\n🎯 極端フィルタリング閾値最適化")
    print("範囲: 5%ile～20%ile（より激しい変化のみに集中）")

    extreme_percentiles = np.arange(5, 21, 1)  # 5%～20%、1%刻み

    for percentile in extreme_percentiles:
        # 基本特徴量での試行
        result_basic = test_extreme_filtering_threshold(
            df, zone, horizon, percentile, time_diff_seconds, use_advanced_features=False
        )
        result_basic['feature_type'] = 'basic'
        all_results.append(result_basic)

        # 高度特徴量での試行
        result_advanced = test_extreme_filtering_threshold(
            df, zone, horizon, percentile, time_diff_seconds, use_advanced_features=True
        )
        result_advanced['feature_type'] = 'advanced'
        all_results.append(result_advanced)

    # 有効な結果のみを抽出
    valid_results = [r for r in all_results if r.get('error') is None]

    if not valid_results:
        print("❌ 有効な結果が得られませんでした")
        return {'error': '極端最適化失敗', 'all_results': all_results}

    # 最良結果の特定（複数指標で総合評価）
    print("\\n📊 総合評価による最良モデル選定...")

    for result in valid_results:
        # 総合スコア計算（RMSE重視、方向精度も考慮）
        rmse_score = 1 / (1 + result['restoration_rmse'])  # 小さいほど良い→大きいほど良いに変換
        r2_score = result['restoration_r2']  # 大きいほど良い
        direction_score = result['direction_accuracy'] / 100  # %→比率

        # 重み付き総合スコア
        result['composite_score'] = (
            rmse_score * 0.4 +  # RMSE重視
            r2_score * 0.3 +    # R²
            direction_score * 0.3  # 方向精度
        )

    # 総合スコアで最良結果を選定
    best_result = max(valid_results, key=lambda x: x['composite_score'])

    # 現在のベストモデルとの比較
    comparison = compare_with_current_best(best_result, zone, horizon)

    print("\\n" + "="*80)
    print("🎉 極端フィルタリング最適化完了！")
    print("="*80)

    print(f"🏆 最良設定:")
    print(f"   フィルタリング: {best_result['percentile']:.1f}%ile")
    print(f"   特徴量タイプ: {best_result['feature_type']}")
    print(f"   特徴量数: {best_result['feature_count']}")
    print(f"   データ削減: {best_result['data_reduction_rate']:.1f}%")

    print(f"\\n📈 性能指標:")
    print(f"   復元温度RMSE: {best_result['restoration_rmse']:.4f}℃")
    print(f"   復元温度R²: {best_result['restoration_r2']:.4f}")
    print(f"   方向精度: {best_result['direction_accuracy']:.1f}%")
    print(f"   MAE: {best_result['restoration_mae']:.4f}℃")
    print(f"   総合スコア: {best_result['composite_score']:.4f}")

    if 'improvements' in comparison:
        print(f"\\n🚀 現在ベストからの改善:")
        improvements = comparison['improvements']
        print(f"   RMSE: {improvements['rmse']:+.2f}%")
        print(f"   R²: {improvements['r2']:+.2f}%")
        print(f"   方向精度: {improvements['direction_accuracy']:+.2f}%")
        print(f"   MAE: {improvements['mae']:+.2f}%")

        # 改善判定
        significant_improvements = sum([
            improvements['rmse'] > 1,  # RMSE 1%以上改善
            improvements['r2'] > 0.5,  # R² 0.5%以上改善
            improvements['direction_accuracy'] > 2  # 方向精度 2%以上改善
        ])

        if significant_improvements >= 2:
            print("\\n✅ 有意な改善が確認されました！")
        else:
            print("\\n🤔 改善は限定的です。現在のモデルが既に高性能です。")

    # 結果保存（JSON serialization可能にするためint64をintに変換）
    output_dir = project_root / "Output"
    output_dir.mkdir(exist_ok=True)

    # numpy型をPython基本型に変換する関数
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # 結果をJSON化可能にする
    best_result = convert_numpy_types(best_result)
    comparison = convert_numpy_types(comparison)
    all_results = convert_numpy_types(all_results)

    results_file = output_dir / f"extreme_filtering_optimization_zone{zone}_horizon{horizon}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'best_result': best_result,
            'comparison_with_current': comparison,
            'all_results': all_results,
            'optimization_summary': {
                'zone': zone,
                'horizon': horizon,
                'total_tests': len(all_results),
                'valid_tests': len(valid_results),
                'best_percentile': float(best_result['percentile']),
                'best_feature_type': best_result['feature_type'],
                'best_rmse': float(best_result['restoration_rmse']),
                'best_r2': float(best_result['restoration_r2']),
                'composite_score': float(best_result['composite_score'])
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\\n結果を保存: {results_file}")

    return {
        'best_result': best_result,
        'comparison': comparison,
        'all_results': all_results
    }


if __name__ == "__main__":
    # コマンドライン引数のパース
    zone = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # 極端最適化実行
    results = extreme_filtering_optimization(zone, horizon)

    if 'error' not in results:
        print("\\n✅ 極端フィルタリング最適化が正常に完了しました")
    else:
        print(f"\\n❌ 最適化でエラーが発生: {results['error']}")
