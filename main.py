#!/usr/bin/env python
# coding: utf-8

"""
空調システム室内温度予測モデル開発
モジュール化されたコードによる実装
上司のアドバイスに従って修正済み
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
import argparse  # 追加: コマンドライン引数のパーサー
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
    create_lag_features,
    create_rolling_features,
    create_physics_based_features,
    create_future_explanatory_features,
    create_thermo_state_features,
    select_important_features
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
    analyze_lag_dependency
)

# 可視化関数のインポート
from src.utils.visualization import (
    plot_feature_importance,
    plot_scatter_actual_vs_predicted,
    plot_time_series,
    plot_lag_dependency_analysis,
    plot_scatter_actual_vs_predicted_by_horizon,
    plot_time_series_by_horizon
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

    # テストモードの表示
    if test_mode:
        print(f"[テストモード] 対象ゾーン: {target_zones}, 対象ホライゾン: {target_horizons}")

    print("## データ読み込みと前処理")

    # データ読み込み
    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return

    # メモリ使用状況確認
    print("メモリ使用状況:")
    mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"総メモリ使用量: {mem_usage:.2f} MB")

    print("\n## データの基本情報確認")

    # データの基本統計量
    print("\nデータの基本統計量:")
    desc_stats = df.describe()
    print(desc_stats)

    # 欠損値の確認
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100

    missing_df = pd.DataFrame({
        '欠損値数': missing_values,
        '欠損率(%)': missing_percent
    })

    # 欠損のある列のみを表示
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

    # LMRのゾーン区分を表示
    print("ゾーン区分:")
    print(f"L系統ゾーン: {[z for z in L_ZONES if z in existing_zones]}")
    print(f"M系統ゾーン: {[z for z in M_ZONES if z in existing_zones]}")
    print(f"R系統ゾーン: {[z for z in R_ZONES if z in existing_zones]}")

    # 上司のアドバイスに従い、不要な列を削除
    # AC_temp_* 列を削除
    ac_temp_cols = [col for col in df.columns if 'AC_temp_' in col]
    if ac_temp_cols:
        df = df.drop(columns=ac_temp_cols)
        print(f"\n上司のアドバイスに従い、{len(ac_temp_cols)}個のAC_temp_列を削除しました")

    # 電力系統データ（L, M, R）を削除
    power_cols = [col for col in df.columns if col in ['L', 'M', 'R']]
    if power_cols:
        df = df.drop(columns=power_cols)
        print(f"上司のアドバイスに従い、{len(power_cols)}個の電力系統データ列を削除しました")

    print("\n## 目的変数の作成（将来温度の予測）")

    # 目的変数の作成
    df_with_targets = create_future_targets(df, existing_zones, actual_horizons, time_diff)
    print(f"目的変数を追加したデータシェイプ: {df_with_targets.shape}")

    # 外れ値処理の実行
    df_with_targets = filter_temperature_outliers(df_with_targets)

    # センサーデータの平滑化処理（ノイズ対策）- 上司のアドバイスに従い強化
    print("\n上司のアドバイスに従い、センサーデータの平滑化処理を強化します")
    # スムージングウィンドウを拡張してより強力なノイズ除去
    enhanced_smoothing_windows = SMOOTHING_WINDOWS + [18, 24]  # より長いウィンドウも追加
    df_with_targets, smoothed_features = apply_smoothing_to_sensors(df_with_targets, enhanced_smoothing_windows)

    # 時系列特徴量の作成
    print("\n## 物理ベースの時系列特徴量を作成")
    # LAG特徴量の作成（長期LAGを中心に）
    df_with_targets, lag_cols = create_lag_features(df_with_targets, existing_zones)
    # 移動平均特徴量の作成（物理的意味を持つものに変更）
    df_with_targets, rolling_cols = create_rolling_features(df_with_targets, existing_zones)
    # 物理モデルベースの特徴量を追加
    df_with_targets, physics_cols = create_physics_based_features(df_with_targets, existing_zones)

    # 新しい特徴量を特定
    print(f"物理ベースのLAG特徴量を{len(lag_cols)}個追加しました")
    print(f"物理的意味を持つ移動平均特徴量を{len(rolling_cols)}個追加しました")
    print(f"熱力学ベースの特徴量を{len(physics_cols)}個追加しました")

    # 目的変数の例を表示
    target_cols = [col for col in df_with_targets.columns if 'future' in col]
    first_zone = existing_zones[0]
    print(f"\nゾーン{first_zone}の目的変数サンプル:")
    print(df_with_targets[[f'sens_temp_{first_zone}'] + [col for col in target_cols if f'_{first_zone}_future' in col]].head(10))

    print("\n## 特徴量エンジニアリング")
    # サーモ状態特徴量を作成 - 上司のアドバイスに従い強化
    df_with_targets, thermo_features = create_thermo_state_features(df_with_targets, existing_zones)
    print(f"サーモ状態特徴量を{len(thermo_features)}個作成しました")

    # 未来の説明変数のためのベース特徴量設定
    future_explanatory_base_config = []

    # 共通環境特徴量
    actual_atmo_temp_col_name = None
    for col_name in df_with_targets.columns:
        if 'atmospheric' in col_name.lower() and 'temperature' in col_name.lower():
            actual_atmo_temp_col_name = col_name
            future_explanatory_base_config.append({'name': actual_atmo_temp_col_name, 'type': 'common'})
            print(f"環境特徴量 (未来予測対象): {actual_atmo_temp_col_name}")
            break

    actual_solar_rad_col_name = None
    for col_name in df_with_targets.columns:
        if 'total' in col_name.lower() and 'solar' in col_name.lower() and 'radiation' in col_name.lower():
            actual_solar_rad_col_name = col_name
            future_explanatory_base_config.append({'name': actual_solar_rad_col_name, 'type': 'common'})
            print(f"環境特徴量 (未来予測対象): {actual_solar_rad_col_name}")
            break

    # ゾーン別特徴量 (サーモ状態, AC有効状態, ACモード) - 上司のアドバイスに従い修正
    for zone in existing_zones:
        # サーモ状態を優先
        if f'thermo_state_{zone}' in df_with_targets.columns:
            future_explanatory_base_config.append({'name': f'thermo_state_{zone}', 'type': 'zone_specific', 'zone': zone})
            print(f"サーモ状態特徴量 (未来予測対象): thermo_state_{zone}")

        # サーモ状態調整値も追加（冷暖房モードで調整したもの）
        if f'thermo_state_adjusted_{zone}' in df_with_targets.columns:
            future_explanatory_base_config.append({'name': f'thermo_state_adjusted_{zone}', 'type': 'zone_specific', 'zone': zone})
            print(f"調整済みサーモ状態特徴量 (未来予測対象): thermo_state_adjusted_{zone}")

        # 空調有効状態
        if f'AC_valid_{zone}' in df_with_targets.columns:
            future_explanatory_base_config.append({'name': f'AC_valid_{zone}', 'type': 'zone_specific', 'zone': zone})
            print(f"空調有効状態特徴量 (未来予測対象): AC_valid_{zone}")

        # 空調モード
        ac_mode_col_candidate = f'AC_mode_{zone}'
        if ac_mode_col_candidate in df_with_targets.columns:
            future_explanatory_base_config.append({'name': ac_mode_col_candidate, 'type': 'zone_specific', 'zone': zone})
            print(f"ACモード特徴量 (未来予測対象): {ac_mode_col_candidate}")

    # 未来の説明変数を生成 - 上司のアドバイスに従い修正
    time_diff_seconds_val = time_diff.total_seconds()
    df_with_targets, all_future_explanatory_features = create_future_explanatory_features(
        df_with_targets,
        future_explanatory_base_config,
        actual_horizons,
        time_diff_seconds_val
    )
    print(f"{len(all_future_explanatory_features)}個の未来の説明変数を生成しました。")

    # 基本的な特徴量のリスト
    feature_cols = []

    # センサー温度・湿度（平滑化版を優先）
    feature_cols.extend(smoothed_features)

    # サーモ状態特徴量を追加（上司のアドバイスに従い優先）
    feature_cols.extend(thermo_features)

    # 物理ベースの特徴量
    feature_cols.extend(lag_cols)
    feature_cols.extend(rolling_cols)
    feature_cols.extend(physics_cols)

    # 未来の説明変数（制御可能パラメータと環境データ）
    feature_cols.extend(all_future_explanatory_features)

    # 時間特徴量（上司のアドバイスに従い、day_of_weekとis_weekendは削除検討）
    time_features = ['hour']

    # 高次の相互作用を生成するためのベース特徴量（上司のアドバイスを反映）
    poly_base_features = []

    # 上司のアドバイスに従った重要な特徴量
    for zone in existing_zones:
        # 現在の温度・湿度
        temp_feature = f'sens_temp_{zone}_smoothed' if f'sens_temp_{zone}_smoothed' in df_with_targets.columns else f'sens_temp_{zone}'
        poly_base_features.append(temp_feature)

        # サーモ状態
        if f'thermo_state_{zone}' in df_with_targets.columns:
            poly_base_features.append(f'thermo_state_{zone}')

        # 空調発停
        if f'AC_valid_{zone}' in df_with_targets.columns:
            poly_base_features.append(f'AC_valid_{zone}')

    # 環境特徴量
    if actual_atmo_temp_col_name:
        poly_base_features.append(actual_atmo_temp_col_name)
    if actual_solar_rad_col_name:
        poly_base_features.append(actual_solar_rad_col_name)

    # 未来特徴量のベース版も追加（ホライゾンが最小のもの）
    min_horizon = min(actual_horizons)
    for base_feature in poly_base_features.copy():
        future_feature = f"{base_feature}_future_{min_horizon}"
        if future_feature in df_with_targets.columns:
            poly_base_features.append(future_feature)

    print(f"多項式特徴量のベース特徴: {poly_base_features}")

    # リスト内の重複を排除
    feature_cols = list(dict.fromkeys(feature_cols + time_features))
    print(f"特徴量リスト（多項式特徴量を除く）: {len(feature_cols)}個")

    # モデルトレーニングと評価のループ
    results = {}

    for zone in existing_zones:
        zone_results = {}

        for horizon in actual_horizons:
            print(f"\n### ゾーン{zone}の{horizon}分後予測モデル")
            target_col = f'sens_temp_{zone}_future_{horizon}'

            if target_col not in df_with_targets.columns:
                print(f"警告: 目的変数 {target_col} が見つかりません。このホライゾンをスキップします。")
                continue

            # トレーニング/テスト分割
            train_df, test_df = get_time_based_train_test_split(df_with_targets, test_size=TEST_SIZE)

            # 不要な列を削除
            train_X = train_df[feature_cols].copy()
            train_y = train_df[target_col]
            test_X = test_df[feature_cols].copy()
            test_y = test_df[target_col]

            # 欠損値処理
            train_X = train_X.fillna(method='ffill').fillna(method='bfill')
            test_X = test_X.fillna(method='ffill').fillna(method='bfill')

            # 特徴量選択
            train_X_selected, test_X_selected, selected_features = select_important_features(
                train_X, train_y, test_X, feature_cols, FEATURE_SELECTION_THRESHOLD
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
            importance_df = analyze_feature_importance(model, selected_features)
            print("\n特徴量重要度 (上位10):")
            print(importance_df.head(10))

            # ラグ依存性分析
            # 上司のアドバイスに従い、ゾーンの系統（L/M/R）を特定
            zone_system = 'L' if zone in L_ZONES else ('M' if zone in M_ZONES else 'R')
            lag_dependency = analyze_lag_dependency(importance_df, zone, horizon, zone_system)
            print("\nLAG特徴量への依存度:")
            print(lag_dependency)

            # モデルと特徴量情報を保存
            model_path, features_path = save_model_and_features(model, selected_features, zone, horizon)

            # 結果を格納
            zone_results[horizon] = {
                'model': model,
                'selected_features': selected_features,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'importance': importance_df,
                'lag_dependency': lag_dependency,
                'model_path': model_path,
                'features_path': features_path,
                'test_y': test_y,
                'test_predictions': test_predictions,
                'test_df': test_df
            }

        # ゾーンごとの結果を格納
        results[zone] = zone_results

    # ホライゾンごとにすべてのゾーンをまとめた可視化を生成
    print("\n## ホライゾンごとの集約可視化を生成中...")
    for horizon in actual_horizons:
        # 散布図（全ゾーン）
        scatter_fig = plot_scatter_actual_vs_predicted_by_horizon(results, horizon, save_dir=OUTPUT_DIR)
        if scatter_fig:
            print(f"ホライゾン {horizon}分 の散布図を生成しました")
        else:
            print(f"ホライゾン {horizon}分 の散布図生成に失敗しました")

        # 時系列プロット（全ゾーン）
        ts_fig = plot_time_series_by_horizon(results, horizon, save_dir=OUTPUT_DIR)
        if ts_fig:
            print(f"ホライゾン {horizon}分 の時系列プロットを生成しました")
        else:
            print(f"ホライゾン {horizon}分 の時系列プロット生成に失敗しました")

    print("\n## すべてのモデルのトレーニングが完了しました")
    print(f"結果は {OUTPUT_DIR} ディレクトリに保存されています")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='室内温度予測モデルトレーニング')
    parser.add_argument('--test', action='store_true', help='テストモードで実行（サブセットのデータとゾーンで処理）')
    parser.add_argument('--zones', type=int, nargs='+', help='処理対象のゾーン番号')
    parser.add_argument('--horizons', type=int, nargs='+', help='処理対象の予測ホライゾン（分）')

    args = parser.parse_args()

    # コマンドライン引数に基づいて実行
    main(
        test_mode=args.test,
        target_zones=args.zones,
        target_horizons=args.horizons
    )
