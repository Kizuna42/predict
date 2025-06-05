#!/usr/bin/env python
# coding: utf-8

"""
高度なモデル改善スクリプト
分析結果に基づいた改善版モデルの実装:
1. 時間相関の改善（シーケンス特徴量）
2. 外れ値に対する堅牢な学習
3. 時間パターンの明示的モデリング
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

# 評価関数のインポート
from src.models.evaluation import evaluate_temperature_difference_model


def create_advanced_temporal_features(df, zone_nums, horizons_minutes, time_diff_seconds):
    """
    高度な時間特徴量の作成
    - ラグ特徴量
    - 移動統計
    - 周期性特徴量
    - 時間的相互作用

    Parameters:
    -----------
    df : DataFrame
        時系列データ
    zone_nums : list
        ゾーン番号のリスト
    horizons_minutes : list
        予測ホライゾン
    time_diff_seconds : float
        サンプリング間隔

    Returns:
    --------
    DataFrame, list
        拡張されたデータフレームと新特徴量リスト
    """
    print("\\n🔥 高度な時間特徴量を作成中...")
    df_copy = df.copy()
    temporal_features = []

    # 1. ラグ特徴量（短期から長期まで）
    lag_minutes = [5, 10, 15, 30, 60, 120]  # 様々な時間スケール

    for zone in zone_nums:
        base_temp_col = f'sens_temp_{zone}'
        if base_temp_col in df_copy.columns:

            for lag_min in lag_minutes:
                lag_steps = int(lag_min * 60 / time_diff_seconds)

                # 温度ラグ特徴量
                lag_col = f'temp_lag_{zone}_{lag_min}min'
                df_copy[lag_col] = df_copy[base_temp_col].shift(lag_steps)
                temporal_features.append(lag_col)

                # 温度変化ラグ特徴量
                temp_change_col = f'temp_change_{zone}_{lag_min}min'
                if temp_change_col in df_copy.columns:
                    lag_change_col = f'temp_change_lag_{zone}_{lag_min}min'
                    df_copy[lag_change_col] = df_copy[temp_change_col].shift(lag_steps)
                    temporal_features.append(lag_change_col)

    # 2. 移動統計特徴量（ローリング統計）
    window_minutes = [15, 30, 60, 120]

    for zone in zone_nums:
        base_temp_col = f'sens_temp_{zone}'
        if base_temp_col in df_copy.columns:

            for window_min in window_minutes:
                window_steps = int(window_min * 60 / time_diff_seconds)

                # 移動平均（既存の平滑化より長期）
                if window_min > 15:  # 既存の短期平滑化と重複を避ける
                    rolling_mean_col = f'temp_rolling_mean_{zone}_{window_min}min'
                    df_copy[rolling_mean_col] = df_copy[base_temp_col].rolling(
                        window=window_steps, min_periods=1, center=False
                    ).mean()
                    temporal_features.append(rolling_mean_col)

                # 移動標準偏差（変動性の指標）
                rolling_std_col = f'temp_rolling_std_{zone}_{window_min}min'
                df_copy[rolling_std_col] = df_copy[base_temp_col].rolling(
                    window=window_steps, min_periods=1, center=False
                ).std()
                temporal_features.append(rolling_std_col)

                # 移動最大・最小（範囲の指標）
                rolling_max_col = f'temp_rolling_max_{zone}_{window_min}min'
                rolling_min_col = f'temp_rolling_min_{zone}_{window_min}min'
                df_copy[rolling_max_col] = df_copy[base_temp_col].rolling(
                    window=window_steps, min_periods=1, center=False
                ).max()
                df_copy[rolling_min_col] = df_copy[base_temp_col].rolling(
                    window=window_steps, min_periods=1, center=False
                ).min()
                temporal_features.extend([rolling_max_col, rolling_min_col])

    # 3. 周期性特徴量の強化
    if 'hour' in df_copy.columns:
        # 時間帯の温度パターン
        for zone in zone_nums:
            base_temp_col = f'sens_temp_{zone}'
            if base_temp_col in df_copy.columns:
                # 時間別平均温度からの偏差
                hourly_mean = df_copy.groupby('hour')[base_temp_col].transform('mean')
                temp_hour_deviation_col = f'temp_hour_deviation_{zone}'
                df_copy[temp_hour_deviation_col] = df_copy[base_temp_col] - hourly_mean
                temporal_features.append(temp_hour_deviation_col)

    # 4. 時間的相互作用特徴量
    for zone in zone_nums:
        base_temp_col = f'sens_temp_{zone}'
        if base_temp_col in df_copy.columns and 'hour' in df_copy.columns:
            # 時間と温度の相互作用
            temp_hour_interaction_col = f'temp_hour_interaction_{zone}'
            df_copy[temp_hour_interaction_col] = df_copy[base_temp_col] * df_copy['hour']
            temporal_features.append(temp_hour_interaction_col)

    # 5. トレンド特徴量（短期・中期トレンド）
    trend_windows = [30, 60, 120]  # 分

    for zone in zone_nums:
        base_temp_col = f'sens_temp_{zone}'
        if base_temp_col in df_copy.columns:

            for trend_min in trend_windows:
                trend_steps = int(trend_min * 60 / time_diff_seconds)

                # 線形トレンド（最小二乗法の傾き）
                trend_col = f'temp_trend_{zone}_{trend_min}min'

                def calculate_trend(series):
                    if len(series) < 2 or series.isna().all():
                        return np.nan
                    x = np.arange(len(series))
                    y = series.values
                    valid_mask = ~np.isnan(y)
                    if valid_mask.sum() < 2:
                        return np.nan
                    return np.polyfit(x[valid_mask], y[valid_mask], 1)[0]

                df_copy[trend_col] = df_copy[base_temp_col].rolling(
                    window=trend_steps, min_periods=2, center=False
                ).apply(calculate_trend, raw=False)
                temporal_features.append(trend_col)

    print(f"✅ 高度な時間特徴量を{len(temporal_features)}個作成しました")
    return df_copy, temporal_features


def create_robust_training_weights(y_train, residual_threshold_factor=2.0):
    """
    外れ値に対する堅牢な重み付けの作成

    Parameters:
    -----------
    y_train : Series
        訓練用目的変数
    residual_threshold_factor : float
        外れ値判定の閾値因子

    Returns:
    --------
    np.array
        トレーニング重み
    """
    print("\\n🎯 堅牢な重み付け戦略を作成中...")

    # 基本統計
    y_std = y_train.std()
    y_median = y_train.median()

    # 1. 基本重み（既存の重み付け戦略）
    abs_values = y_train.abs()

    # 高い変化に重み
    high_change_weight = np.where(abs_values > abs_values.quantile(0.8), 2.0, 1.0)

    # 極小変化の検出能力向上
    tiny_change_weight = np.where(abs_values < abs_values.quantile(0.1), 1.5, 1.0)

    # 方向転換点の重要性
    direction_change_weight = np.ones(len(y_train))
    if len(y_train) > 2:
        direction_changes = ((y_train.shift(1) > 0) & (y_train < 0)) | ((y_train.shift(1) < 0) & (y_train > 0))
        direction_change_weight = np.where(direction_changes, 2.0, 1.0)

    # 2. 外れ値抑制重み
    outlier_threshold = residual_threshold_factor * y_std
    outlier_weight = np.where(abs_values > outlier_threshold, 0.5, 1.0)  # 外れ値の重みを下げる

    # 3. 時間的近接性重み（最新データに高い重み）
    temporal_weight = np.linspace(0.8, 1.2, len(y_train))  # 最新20%重み増

    # 4. 統合重み計算
    combined_weights = (high_change_weight * tiny_change_weight *
                       direction_change_weight * outlier_weight * temporal_weight)

    # 重みの正規化（平均1.0になるように）
    combined_weights = combined_weights / combined_weights.mean()

    # 重みの範囲制限（0.2～5.0）
    combined_weights = np.clip(combined_weights, 0.2, 5.0)

    print(f"✅ 堅牢な重み付け完了:")
    print(f"   平均重み: {combined_weights.mean():.3f}")
    print(f"   重み範囲: {combined_weights.min():.3f} - {combined_weights.max():.3f}")
    print(f"   外れ値抑制対象: {(abs_values > outlier_threshold).sum()}行 ({(abs_values > outlier_threshold).mean():.1%})")

    return combined_weights


def train_advanced_temperature_difference_model(X_train, y_train, sample_weights=None):
    """
    高度な温度差分予測モデルの訓練

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数
    sample_weights : array, optional
        サンプル重み

    Returns:
    --------
    lgb.LGBMRegressor
        訓練済みモデル
    """
    print("\\n🔥 高度な温度差分予測モデルを訓練中...")

    # LightGBMパラメータ（改善版）
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,  # やや増加（複雑なパターン対応）
        'learning_rate': 0.05,  # やや低下（安定性向上）
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'reg_alpha': 0.1,  # L1正則化（特徴選択）
        'reg_lambda': 0.1,  # L2正則化（過学習防止）
        'max_depth': 8,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,

        # 外れ値に対する堅牢性向上
        'extra_trees': True,  # ランダム性追加
        'path_smooth': 1.0,   # パス平滑化
    }

    # モデル作成と訓練
    model = lgb.LGBMRegressor(**lgb_params, n_estimators=2000)

    # 早期停止用の検証セット作成
    X_train_split = X_train.iloc[:-1000]
    X_val_split = X_train.iloc[-1000:]
    y_train_split = y_train.iloc[:-1000]
    y_val_split = y_train.iloc[-1000:]

    if sample_weights is not None:
        weights_train_split = sample_weights[:-1000]
        weights_val_split = sample_weights[-1000:]
    else:
        weights_train_split = None
        weights_val_split = None

    # モデル訓練
    model.fit(
        X_train_split, y_train_split,
        sample_weight=weights_train_split,
        eval_set=[(X_val_split, y_val_split)],
        eval_sample_weight=[weights_val_split] if weights_val_split is not None else None,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(0)  # ログ出力無効化
        ]
    )

    print(f"✅ 高度な差分予測モデル訓練完了 (最終イテレーション: {model.best_iteration_})")

    return model


def compare_models(original_results, advanced_results, zone, horizon):
    """
    オリジナルモデルと改善モデルの比較

    Parameters:
    -----------
    original_results : dict
        オリジナルモデルの結果
    advanced_results : dict
        改善モデルの結果
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン

    Returns:
    --------
    dict
        比較結果
    """
    print(f"\\n📊 モデル比較分析 (ゾーン{zone}, {horizon}分予測)")
    print("="*60)

    comparison = {}

    # 主要指標の比較
    metrics_to_compare = [
        ('restoration_rmse', 'RMSE (℃)'),
        ('restoration_r2', 'R²'),
        ('direction_accuracy', '方向精度 (%)'),
        ('restoration_mae', 'MAE (℃)')
    ]

    for metric_key, metric_name in metrics_to_compare:
        original_val = original_results.get(metric_key, np.nan)
        advanced_val = advanced_results.get(metric_key, np.nan)

        if not np.isnan(original_val) and not np.isnan(advanced_val):
            if metric_key in ['restoration_rmse', 'restoration_mae']:
                # 小さい方が良い指標
                improvement = (original_val - advanced_val) / original_val * 100
                comparison[metric_key] = {
                    'original': original_val,
                    'advanced': advanced_val,
                    'improvement_pct': improvement,
                    'better': advanced_val < original_val
                }
            else:
                # 大きい方が良い指標
                improvement = (advanced_val - original_val) / original_val * 100
                comparison[metric_key] = {
                    'original': original_val,
                    'advanced': advanced_val,
                    'improvement_pct': improvement,
                    'better': advanced_val > original_val
                }

            print(f"{metric_name}:")
            print(f"  オリジナル: {original_val:.4f}")
            print(f"  改善版:     {advanced_val:.4f}")
            print(f"  改善率:     {improvement:+.2f}%")
            print(f"  結果:       {'✅ 改善' if comparison[metric_key]['better'] else '❌ 悪化'}")
            print()

    return comparison


def create_advanced_model_analysis(zone=1, horizon=15):
    """
    改善版モデルの包括的分析

    Parameters:
    -----------
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン

    Returns:
    --------
    dict
        分析結果
    """
    print("🚀 高度な改善版モデル分析開始")
    print(f"対象: ゾーン{zone}, 予測ホライゾン{horizon}分")

    # データ読み込み（オリジナルと同じ前処理）
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

    # 高度な時間特徴量の追加
    df_enhanced, temporal_features = create_advanced_temporal_features(
        df_processed, [zone], [horizon], time_diff_seconds
    )

    # 全特徴量の統合
    all_features = selected_features + temporal_features
    all_features = [f for f in all_features if f in df_enhanced.columns]

    print(f"\\n📊 特徴量統計:")
    print(f"   基本特徴量: {len(selected_features)}")
    print(f"   時間特徴量: {len(temporal_features)}")
    print(f"   総特徴量数: {len(all_features)}")

    # データ準備
    diff_target_col = f'temp_diff_{zone}_future_{horizon}'
    valid_data = df_enhanced.dropna(subset=[diff_target_col] + all_features)

    # 高値フィルタリング（極端フィルタリング最適化結果：5%ileが最適）
    # 上司要求：「目的変数の値が大きいデータのみに絞って学習」
    abs_diff_col = f'abs_temp_diff_{zone}_future_{horizon}'
    valid_data[abs_diff_col] = valid_data[diff_target_col].abs()
    filtered_data, filter_info = filter_high_value_targets(
        valid_data, [abs_diff_col], percentile=5
    )

    # 時系列分割
    train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

    X_train = train_df[all_features]
    y_train_diff = train_df[diff_target_col]
    X_test = test_df[all_features]
    y_test_diff = test_df[diff_target_col]
    current_temps_test = test_df[f'sens_temp_{zone}']

    print(f"\\n📊 データ分割:")
    print(f"   訓練データ: {X_train.shape[0]}行 x {X_train.shape[1]}特徴量")
    print(f"   テストデータ: {X_test.shape[0]}行")

    # 堅牢な重み付けの作成
    robust_weights = create_robust_training_weights(y_train_diff)

    # 改善版モデル訓練
    advanced_model = train_advanced_temperature_difference_model(
        X_train, y_train_diff, sample_weights=robust_weights
    )

    # 予測と評価
    y_pred_advanced = advanced_model.predict(X_test)
    advanced_metrics = evaluate_temperature_difference_model(
        y_test_diff, y_pred_advanced, current_temps_test
    )

    # オリジナル結果の読み込み（比較用）
    original_analysis_file = project_root / "Output" / f"comprehensive_analysis_zone{zone}_horizon{horizon}.json"
    original_metrics = {}
    if original_analysis_file.exists():
        with open(original_analysis_file, 'r', encoding='utf-8') as f:
            original_analysis = json.load(f)
            original_metrics = original_analysis['performance_analysis']['basic_metrics']

    # モデル比較
    comparison_results = compare_models(original_metrics, advanced_metrics, zone, horizon)

    # 結果統合
    advanced_results = {
        'metadata': {
            'zone': zone,
            'horizon': horizon,
            'analysis_timestamp': datetime.now().isoformat(),
            'model_type': 'advanced_temporal_robust',
            'feature_count': {
                'basic': len(selected_features),
                'temporal': len(temporal_features),
                'total': len(all_features)
            }
        },
        'advanced_metrics': advanced_metrics,
        'model_comparison': comparison_results,
        'feature_importance': {
            'top_20': [
                {
                    'feature': all_features[i],
                    'importance': float(importance)
                }
                for i, importance in enumerate(advanced_model.feature_importances_)
            ][:20]
        },
        'training_info': {
            'robust_weights_used': True,
            'temporal_features_added': len(temporal_features),
            'best_iteration': int(advanced_model.best_iteration_)
        }
    }

    # 結果保存
    output_dir = project_root / "Output"
    results_file = output_dir / f"advanced_model_analysis_zone{zone}_horizon{horizon}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(advanced_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\\n結果を保存: {results_file}")

    return advanced_results


if __name__ == "__main__":
    # コマンドライン引数のパース
    zone = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # 改善版モデル分析実行
    results = create_advanced_model_analysis(zone, horizon)

    print("\\n🎉 高度な改善版モデル分析が完了しました！")
