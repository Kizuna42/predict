#!/usr/bin/env python
# coding: utf-8

"""
空調システム室内温度予測モデル開発
モジュール化されたコードによる実装
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import argparse
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
    prepare_time_features,
    get_time_based_train_test_split
)

# 特徴量エンジニアリング関数のインポート
from src.data.feature_engineering import (
    create_physics_based_features,
    create_future_explanatory_features,
    create_thermo_state_features,
    select_important_features,
    create_polynomial_features,
    create_optimized_features_pipeline,
    select_important_features_enhanced
)

# モデルトレーニング関数のインポート
from src.models.training import (
    train_physics_guided_model,
    save_model_and_features
)

# モデル評価関数のインポート
from src.models.evaluation import (
    calculate_metrics,
    print_metrics,
    analyze_feature_importance,
    print_lag_dependency_warning
)

# 可視化関数のインポート
from src.utils.visualization import (
    create_comprehensive_analysis_report,
    print_analysis_summary
)

# 簡素化された可視化機能のインポート
from src.utils.basic_plots import (
    plot_feature_importance
)

from src.utils.advanced_visualization import (
    plot_corrected_time_series_by_horizon,
    plot_ultra_detailed_minute_analysis
)

# 診断機能のインポート
from src.diagnostics import (
    analyze_lag_dependency,
    detect_lag_following_pattern,
    validate_prediction_timing,
    analyze_feature_patterns,
    calculate_comprehensive_metrics
)


def main(test_mode=False, target_zones=None, target_horizons=None):
    """
    メイン実行関数

    Parameters:
    -----------
    test_mode : bool
        テストモードで実行するかどうか
    target_zones : list of int, optional
        処理対象のゾーン番号のリスト（None の場合は全てのゾーン）
    target_horizons : list of int, optional
        処理対象の予測ホライゾン（分）のリスト（None の場合は全てのホライゾン）
    """
    print("# 空調システム室内温度予測モデル開発")
    if test_mode:
        print(f"[テストモード] 対象ゾーン: {target_zones}, 対象ホライゾン: {target_horizons}")

    print("## データ読み込みと前処理")
    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100

    missing_df = pd.DataFrame({
        '欠損値数': missing_values,
        '欠損率(%)': missing_percent
    })

    print("\n欠損値の状況:")
    missing_cols = missing_df[missing_df['欠損値数'] > 0].sort_values('欠損値数', ascending=False)
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("欠損値はありません")

    print("\n## 時系列データの処理")

    # 時間特徴量の追加
    df = prepare_time_features(df)

    # 温度センサーデータの概要を確認
    temp_cols = [col for col in df.columns if 'sens_temp' in col]
    print(f"\n温度センサーデータの列: {len(temp_cols)}個")

    # サンプリング間隔の確認
    time_diff = df.index.to_series().diff().dropna().value_counts().index[0]
    print(f"データの時間間隔: {time_diff}")

    # 実際のゾーン番号を抽出
    existing_zones = sorted([int(col.split('_')[2]) for col in temp_cols if 'future' not in col])
    print(f"検出されたゾーン: {existing_zones}")

    # テストモードの場合は対象ゾーンを絞り込む
    if test_mode and target_zones:
        target_zones = [z for z in target_zones if z in existing_zones]
        if not target_zones:
            print("警告: 指定されたゾーンが存在しません。全てのゾーンを使用します。")
            target_zones = existing_zones
        else:
            existing_zones = target_zones
            print(f"指定されたゾーンを使用します: {existing_zones}")

    # テストモードの場合は対象ホライゾンを絞り込む
    actual_horizons = HORIZONS
    if test_mode and target_horizons:
        target_horizons = [h for h in target_horizons if h in HORIZONS]
        if not target_horizons:
            print("警告: 指定されたホライゾンが設定に存在しません。全てのホライゾンを使用します。")
        else:
            actual_horizons = target_horizons
            print(f"指定されたホライゾンを使用します: {actual_horizons}")

    # AC_temp_* 列を削除
    ac_temp_cols = [col for col in df.columns if 'AC_temp_' in col]
    if ac_temp_cols:
        df = df.drop(columns=ac_temp_cols)
        print(f"\n{len(ac_temp_cols)}個のAC_temp_列を削除しました")

    # 電力系統データ（L, M, R）を削除
    power_cols = [col for col in df.columns if col in ['L', 'M', 'R']]
    if power_cols:
        df = df.drop(columns=power_cols)
        print(f"{len(power_cols)}個の電力系統データ列を削除しました")

    # 不要な時間特徴量を削除（day_of_week, is_weekend等）
    # 必要なのはhourのみ
    time_cols_to_remove = [col for col in df.columns if col in ['day_of_week', 'is_weekend', 'day_of_year']]
    if time_cols_to_remove:
        df = df.drop(columns=time_cols_to_remove)
        print(f"{len(time_cols_to_remove)}個の不要な時間特徴量を削除しました")

    print("\n## 目的変数の作成（将来温度の予測）")

    # 目的変数の作成
    df_with_targets = create_future_targets(df, existing_zones, actual_horizons, time_diff)
    print(f"目的変数を追加したデータシェイプ: {df_with_targets.shape}")

    # 外れ値処理の実行
    df_with_targets = filter_temperature_outliers(df_with_targets)

    print("\n## 最適化された特徴量エンジニアリング")
    # 新しい統合パイプライン関数を使用
    time_diff_seconds_val = time_diff.total_seconds()
    df_processed, selected_features, feature_info = create_optimized_features_pipeline(
        df=df_with_targets,
        zone_nums=existing_zones,
        horizons_minutes=actual_horizons,
        time_diff_seconds=time_diff_seconds_val,
        is_prediction_mode=False,  # 学習時
        smoothing_window=5,
        feature_selection_threshold='20%'  # より多くの特徴量を保持
    )

    # 作成された特徴量の詳細情報を表示
    print(f"\n作成された特徴量の詳細:")
    for category, features in feature_info.items():
        if category != 'total_features':
            print(f"  - {category}: {len(features)}個")
            # 重要カテゴリの場合は特徴量名も表示
            if category in ['thermo_features', 'future_features'] and len(features) <= 10:
                for feature in features[:5]:  # 最初の5個のみ表示
                    print(f"    · {feature}")
                if len(features) > 5:
                    print(f"    · ... 他{len(features)-5}個")

    print(f"\n統合パイプラインで作成された特徴量総数: {len(selected_features)}個")

    # 時間特徴量を追加
    if 'hour' not in selected_features and 'hour' in df_processed.columns:
        selected_features.append('hour')
        print("時間特徴量 'hour' を追加しました")

    # 多項式特徴量のベース特徴量設定（重要特徴量から選択）
    poly_base_features = []
    for feature in selected_features:
        # 重要な基本特徴量のみを多項式特徴量のベースとして使用
        if any(pattern in feature for pattern in [
            'thermo_state', 'atmospheric', 'solar', 'AC_valid',
            'smoothed', 'temp_diff'
        ]) and 'future' not in feature:  # 未来特徴量は除外
            poly_base_features.append(feature)

    # 多項式特徴量のベース数を制限（計算時間とメモリ使用量を考慮）
    poly_base_features = poly_base_features[:15]  # 最大15個に制限
    print(f"多項式特徴量のベース特徴: {len(poly_base_features)}個")

    # 最終的な特徴量リスト
    feature_cols = selected_features
    print(f"最終特徴量リスト: {len(feature_cols)}個")

    # モデルトレーニングと評価のループ
    results = {}

    for zone in existing_zones:
        zone_results = {}

        for horizon in actual_horizons:
            print(f"\n### ゾーン{zone}の{horizon}分後予測モデル")
            target_col = f'sens_temp_{zone}_future_{horizon}'

            if target_col not in df_processed.columns:
                print(f"警告: 目的変数 {target_col} が見つかりません。このホライゾンをスキップします。")
                continue

            # トレーニング/テスト分割
            train_df, test_df = get_time_based_train_test_split(df_processed, test_size=TEST_SIZE)

            # 特徴量と目的変数の準備
            available_feature_cols = [col for col in feature_cols if col in train_df.columns]
            train_X = train_df[available_feature_cols].copy()
            train_y = train_df[target_col]
            test_X = test_df[available_feature_cols].copy()
            test_y = test_df[target_col]

            # 欠損値処理
            train_X = train_X.fillna(method='ffill').fillna(method='bfill')
            test_X = test_X.fillna(method='ffill').fillna(method='bfill')

            # 多項式特徴量の生成
            train_X_poly, test_X_poly, poly_features = create_polynomial_features(
                train_X, test_X, poly_base_features, degree=2
            )
            print(f"多項式特徴量を{len(poly_features)}個追加しました")

            # 改良された特徴量選択を使用
            train_X_selected, test_X_selected, final_selected_features = select_important_features_enhanced(
                train_X_poly, train_y, test_X_poly,
                available_feature_cols + poly_features,
                threshold=FEATURE_SELECTION_THRESHOLD
            )

            # モデルトレーニング
            model = train_physics_guided_model(train_X_selected, train_y)

            # モデル評価
            train_predictions = model.predict(train_X_selected)
            test_predictions = model.predict(test_X_selected)

            # 評価指標の計算
            train_metrics = calculate_metrics(train_y, train_predictions)
            test_metrics = calculate_metrics(test_y, test_predictions)

            # 結果を表示
            print(f"\nトレーニングデータの評価:")
            print_metrics(train_metrics)
            print(f"\nテストデータの評価:")
            print_metrics(test_metrics)

            # 特徴量重要度の分析
            importance_df = analyze_feature_importance(model, final_selected_features)
            print("\n特徴量重要度 (上位10):")
            print(importance_df.head(10))

            # ラグ依存性分析（新しい診断機能を使用）
            # 上司のアドバイスに従い、ゾーンの系統（L/M/R）を特定
            zone_system = 'L' if zone in L_ZONES else ('M' if zone in M_ZONES else 'R')

            # 新しい診断機能を使用
            from src.diagnostics.lag_analysis import analyze_lag_dependency as new_analyze_lag_dependency
            lag_dependency = new_analyze_lag_dependency(importance_df, zone, horizon, zone_system)
            print("\nLAG特徴量への依存度:")
            print(lag_dependency)

            # LAG依存度警告の表示（依存度が高い場合は警告を表示）
            print_lag_dependency_warning(lag_dependency, threshold=30.0, zone=zone, horizon=horizon)

            # モデルと特徴量情報を保存
            model_path, features_path = save_model_and_features(model, final_selected_features, zone, horizon)

            zone_results[horizon] = {
                'model': model,
                'selected_features': final_selected_features,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': importance_df,
                'lag_dependency': lag_dependency,
                'model_path': model_path,
                'features_path': features_path,
                'test_data': test_df,
                'test_y': test_y,
                'test_predictions': test_predictions
            }

        results[zone] = zone_results
    print("\n## 📊 包括的分析レポートの生成")

    # 新しい統合分析インターフェースを使用
    comprehensive_report = create_comprehensive_analysis_report(
        results_dict=results,
        horizons=actual_horizons,
        save_dir=OUTPUT_DIR,
        save=True
    )

    # 分析サマリーの表示
    print_analysis_summary(comprehensive_report)

    print("\n## すべてのモデルのトレーニングと分析が完了しました")
    print(f"結果は {OUTPUT_DIR} ディレクトリに保存されています")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='室内温度予測モデルトレーニング')
    parser.add_argument('--test', action='store_true', help='テストモードで実行（サブセットのデータとゾーンで処理）')
    parser.add_argument('--zones', type=int, nargs='+', help='処理対象のゾーン番号')
    parser.add_argument('--horizons', type=int, nargs='+', help='処理対象の予測ホライゾン（分）')
    args = parser.parse_args()
    main(
        test_mode=args.test,
        target_zones=args.zones,
        target_horizons=args.horizons
    )
