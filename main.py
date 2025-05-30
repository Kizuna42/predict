#!/usr/bin/env python
# coding: utf-8

"""
統合温度予測システム - メインスクリプト
直接温度予測と差分予測の両方をサポートする統合システム
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import (
    HORIZONS, SMOOTHING_WINDOWS, L_ZONES, M_ZONES, R_ZONES,
    MODELS_DIR, OUTPUT_DIR, TEST_SIZE, FEATURE_SELECTION_THRESHOLD
)

# データ前処理関数のインポート
from src.data.preprocessing import (
    filter_temperature_outliers,
    apply_smoothing_to_sensors,
    create_future_targets,
    create_temperature_difference_targets,
    prepare_time_features,
    get_time_based_train_test_split
)

# 特徴量エンジニアリング関数のインポート
from src.data.feature_engineering import (
    create_optimized_features_pipeline,
)

# モデル訓練関数のインポート
from src.models.training import (
    train_physics_guided_model,
    save_model_and_features,
    train_temperature_difference_model,
    save_difference_model_and_features
)

# 評価関数のインポート
from src.models.evaluation import (
    calculate_metrics,
    print_metrics,
    evaluate_temperature_difference_model,
    print_difference_metrics,
    restore_temperature_from_difference,
    compare_difference_vs_direct_prediction,
    print_prediction_comparison
)

# 可視化関数のインポート
from src.utils.basic_plots import (
    plot_feature_importance,
    plot_time_series_comparison,
    plot_scatter_analysis,
    plot_performance_summary,
    plot_comparison_analysis,
    create_comprehensive_visualization_report
)


def setup_output_directories():
    """出力ディレクトリの構造を設定"""
    base_output = Path("Output")
    models_dir = base_output / "models"
    viz_dir = base_output / "visualizations"

    models_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    return models_dir, viz_dir


def run_direct_prediction(df, zones, horizons, save_models=True, create_visualizations=True):
    """
    直接温度予測を実行

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zones : list
        対象ゾーン
    horizons : list
        予測ホライゾン
    save_models : bool
        モデルを保存するか
    create_visualizations : bool
        可視化を作成するか

    Returns:
    --------
    dict
        実行結果
    """
    print("\n🎯 直接温度予測システム実行中...")

    models_dir, viz_dir = setup_output_directories()
    results = {'models': [], 'metrics': {}}

    # 時間差の計算
    time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

    # 直接予測用目的変数の作成
    df_with_targets = create_future_targets(df, zones, horizons,
                                          pd.Timedelta(seconds=time_diff_seconds))

    # 特徴量エンジニアリング
    df_processed, selected_features, feature_info = create_optimized_features_pipeline(
        df=df_with_targets,
        zone_nums=zones,
        horizons_minutes=horizons,
        time_diff_seconds=time_diff_seconds,
        smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
    )

    for zone in zones:
        print(f"\n🎯 ゾーン {zone} 処理中...")

        for horizon in horizons:
            print(f"\n--- {horizon}分後予測 ---")

            target_col = f'sens_temp_{zone}_future_{horizon}'
            if target_col not in df_processed.columns:
                print(f"警告: 目的変数 {target_col} が見つかりません")
                continue

            # データ準備
            feature_cols = [col for col in selected_features if col in df_processed.columns]
            valid_data = df_processed.dropna(subset=[target_col] + feature_cols)

            if len(valid_data) < 100:
                print(f"警告: 有効データが不足 ({len(valid_data)}行)")
                continue

            # 時系列分割
            train_df, test_df = get_time_based_train_test_split(valid_data, test_size=TEST_SIZE)

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            print(f"訓練データ: {X_train.shape[0]}行, テストデータ: {X_test.shape[0]}行")

            # モデル訓練
            model = train_physics_guided_model(X_train, y_train)

            # 予測
            y_pred = model.predict(X_test)

            # 評価
            metrics = calculate_metrics(y_test, y_pred)
            print_metrics(metrics, zone, horizon)

            results['metrics'][f'direct_zone_{zone}_horizon_{horizon}'] = metrics

            # モデル保存
            if save_models:
                model_path, features_path = save_model_and_features(
                    model, feature_cols, zone, horizon
                )
                if model_path:
                    results['models'].append({
                        'type': 'direct',
                        'zone': zone,
                        'horizon': horizon,
                        'model_path': model_path,
                        'features_path': features_path
                    })

            # 可視化作成
            if create_visualizations:
                try:
                    # 包括的可視化レポートの作成
                    test_timestamps = test_df.index
                    created_visualizations = create_comprehensive_visualization_report(
                        model=model,
                        feature_names=feature_cols,
                        y_true=y_test,
                        y_pred=y_pred,
                        timestamps=test_timestamps,
                        metrics=metrics,
                        zone=zone,
                        horizon=horizon,
                        model_type="Direct",
                        save_dir=str(viz_dir)
                    )

                    print(f"直接予測の包括的可視化を作成: ゾーン{zone}, {horizon}分後")

                except Exception as e:
                    print(f"可視化エラー: {e}")

    return results


def run_difference_prediction(df, zones, horizons, save_models=True, create_visualizations=True):
    """
    差分予測を実行

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zones : list
        対象ゾーン
    horizons : list
        予測ホライゾン
    save_models : bool
        モデルを保存するか
    create_visualizations : bool
        可視化を作成するか

    Returns:
    --------
    dict
        実行結果
    """
    print("\n🔥 差分予測システム実行中...")

    models_dir, viz_dir = setup_output_directories()
    results = {'models': [], 'metrics': {}}

    # 時間差の計算
    time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

    # 差分予測用目的変数の作成
    df_with_diff_targets = create_temperature_difference_targets(df, zones, horizons,
                                                               pd.Timedelta(seconds=time_diff_seconds))

    # 特徴量エンジニアリング
    df_processed, selected_features, feature_info = create_optimized_features_pipeline(
        df=df_with_diff_targets,
        zone_nums=zones,
        horizons_minutes=horizons,
        time_diff_seconds=time_diff_seconds,
        smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
    )

    for zone in zones:
        print(f"\n🎯 ゾーン {zone} 処理中...")

        for horizon in horizons:
            print(f"\n--- {horizon}分後差分予測 ---")

            diff_target_col = f'temp_diff_{zone}_future_{horizon}'
            if diff_target_col not in df_processed.columns:
                print(f"警告: 差分目的変数 {diff_target_col} が見つかりません")
                continue

            # データ準備
            feature_cols = [col for col in selected_features if col in df_processed.columns]
            valid_data = df_processed.dropna(subset=[diff_target_col] + feature_cols)

            if len(valid_data) < 100:
                print(f"警告: 有効データが不足 ({len(valid_data)}行)")
                continue

            # 時系列分割
            train_df, test_df = get_time_based_train_test_split(valid_data, test_size=TEST_SIZE)

            X_train = train_df[feature_cols]
            y_train_diff = train_df[diff_target_col]
            X_test = test_df[feature_cols]
            y_test_diff = test_df[diff_target_col]

            print(f"訓練データ: {X_train.shape[0]}行, テストデータ: {X_test.shape[0]}行")

            # 差分予測モデル訓練
            diff_model = train_temperature_difference_model(X_train, y_train_diff)

            # 予測
            y_pred_diff = diff_model.predict(X_test)

            # 評価
            current_temps_test = test_df[f'sens_temp_{zone}']
            diff_metrics = evaluate_temperature_difference_model(
                y_test_diff, y_pred_diff, current_temps_test
            )
            print_difference_metrics(diff_metrics, zone, horizon)

            results['metrics'][f'difference_zone_{zone}_horizon_{horizon}'] = diff_metrics

            # モデル保存
            if save_models:
                model_path, features_path = save_difference_model_and_features(
                    diff_model, feature_cols, zone, horizon
                )
                if model_path:
                    results['models'].append({
                        'type': 'difference',
                        'zone': zone,
                        'horizon': horizon,
                        'model_path': model_path,
                        'features_path': features_path
                    })

            # 可視化作成
            if create_visualizations:
                try:
                    # 温度復元
                    y_restored = y_pred_diff + current_temps_test
                    test_timestamps = test_df.index

                    # 包括的可視化レポートの作成
                    created_visualizations = create_comprehensive_visualization_report(
                        model=diff_model,
                        feature_names=feature_cols,
                        y_true=y_test_diff,
                        y_pred=y_pred_diff,
                        timestamps=test_timestamps,
                        metrics=diff_metrics,
                        zone=zone,
                        horizon=horizon,
                        model_type="Difference",
                        save_dir=str(viz_dir)
                    )

                    # 復元温度の時系列比較も作成
                    restored_timeseries_path = viz_dir / f"difference_restored_timeseries_zone_{zone}_horizon_{horizon}.png"
                    future_target_col = f'sens_temp_{zone}_future_{horizon}'
                    if future_target_col in test_df.columns:
                        plot_time_series_comparison(
                            test_df[future_target_col], y_restored, test_timestamps,
                            zone, horizon, str(restored_timeseries_path),
                            model_type="Difference (Restored)", save=True,
                            show_period_hours=24, detailed_mode=True
                        )

                    print(f"差分予測の包括的可視化を作成: ゾーン{zone}, {horizon}分後")

                except Exception as e:
                    print(f"可視化エラー: {e}")

    return results


def run_comparison_analysis(df, zones, horizons):
    """
    直接予測と差分予測の比較分析

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zones : list
        対象ゾーン
    horizons : list
        予測ホライゾン
    """
    print("\n📊 直接予測 vs 差分予測 比較分析...")

    models_dir, viz_dir = setup_output_directories()

    # 時間差の計算
    time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

    # 両方の目的変数を作成
    df_with_targets = create_future_targets(df, zones, horizons,
                                          pd.Timedelta(seconds=time_diff_seconds))
    df_with_both = create_temperature_difference_targets(df_with_targets, zones, horizons,
                                                       pd.Timedelta(seconds=time_diff_seconds))

    # 特徴量エンジニアリング
    df_processed, selected_features, feature_info = create_optimized_features_pipeline(
        df=df_with_both,
        zone_nums=zones,
        horizons_minutes=horizons,
        time_diff_seconds=time_diff_seconds,
        smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
    )

    for zone in zones:
        print(f"\n🎯 ゾーン {zone} 比較分析...")

        for horizon in horizons:
            print(f"\n--- {horizon}分後予測比較 ---")

            # 目的変数の確認
            direct_target_col = f'sens_temp_{zone}_future_{horizon}'
            diff_target_col = f'temp_diff_{zone}_future_{horizon}'

            if direct_target_col not in df_processed.columns or diff_target_col not in df_processed.columns:
                print(f"警告: 必要な目的変数が見つかりません")
                continue

            # データ準備
            feature_cols = [col for col in selected_features if col in df_processed.columns]
            required_cols = [direct_target_col, diff_target_col] + feature_cols
            valid_data = df_processed.dropna(subset=required_cols)

            if len(valid_data) < 100:
                print(f"警告: 有効データが不足 ({len(valid_data)}行)")
                continue

            # 時系列分割
            train_df, test_df = get_time_based_train_test_split(valid_data, test_size=TEST_SIZE)

            X_train = train_df[feature_cols]
            y_train_direct = train_df[direct_target_col]
            y_train_diff = train_df[diff_target_col]
            X_test = test_df[feature_cols]
            y_test_direct = test_df[direct_target_col]
            y_test_diff = test_df[diff_target_col]

            # 直接予測モデル
            direct_model = train_physics_guided_model(X_train, y_train_direct)
            y_pred_direct = direct_model.predict(X_test)
            direct_metrics = calculate_metrics(y_test_direct, y_pred_direct)

            # 差分予測モデル
            diff_model = train_temperature_difference_model(X_train, y_train_diff)
            y_pred_diff = diff_model.predict(X_test)
            current_temps_test = test_df[f'sens_temp_{zone}']
            diff_metrics = evaluate_temperature_difference_model(
                y_test_diff, y_pred_diff, current_temps_test
            )

            # 比較分析
            print("\n📈 性能比較:")
            print_metrics(direct_metrics, zone, horizon)
            print_difference_metrics(diff_metrics, zone, horizon)

            # 詳細比較
            comparison = compare_difference_vs_direct_prediction(
                direct_metrics, diff_metrics, current_temps_test, y_test_direct
            )
            print_prediction_comparison(comparison, zone, horizon)

            # 比較可視化の作成
            try:
                comparison_path = viz_dir / f"method_comparison_zone_{zone}_horizon_{horizon}.png"
                plot_comparison_analysis(
                    direct_metrics, diff_metrics, zone, horizon,
                    save_path=str(comparison_path), save=True
                )
                print(f"比較分析グラフ保存: {comparison_path}")
            except Exception as e:
                print(f"比較可視化エラー: {e}")


def run_temperature_prediction(mode='both', test_mode=False, zones=None, horizons=None,
                             save_models=True, create_visualizations=True, comparison_only=False):
    """
    統合温度予測システムのメイン実行関数

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
    print("# 🌡️ 統合温度予測システム")
    print(f"## 実行モード: {mode}")

    if test_mode:
        print("[テストモード] 限定実行")

    # データ読み込み
    print("\n## データ読み込み...")
    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    # 基本前処理
    print("\n## データ前処理...")
    df = prepare_time_features(df)
    df = filter_temperature_outliers(df)

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

    # 実行
    if comparison_only or mode == 'comparison':
        run_comparison_analysis(df, target_zones, target_horizons)
    elif mode == 'direct':
        run_direct_prediction(df, target_zones, target_horizons, save_models, create_visualizations)
    elif mode == 'difference':
        run_difference_prediction(df, target_zones, target_horizons, save_models, create_visualizations)
    elif mode == 'both':
        run_direct_prediction(df, target_zones, target_horizons, save_models, create_visualizations)
        run_difference_prediction(df, target_zones, target_horizons, save_models, create_visualizations)
        run_comparison_analysis(df, target_zones, target_horizons)
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
    parser = argparse.ArgumentParser(description='統合温度予測システム')

    # 基本オプション
    parser.add_argument('--mode', choices=['direct', 'difference', 'both', 'comparison'],
                       default='both', help='実行モード')
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
