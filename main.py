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
    plot_lag_dependency_analysis
)


def main():
    """メイン実行関数"""
    print("# 空調システム室内温度予測モデル開発")
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

    # LMRのゾーン区分を表示
    print("ゾーン区分:")
    print(f"L系統ゾーン: {[z for z in L_ZONES if z in existing_zones]}")
    print(f"M系統ゾーン: {[z for z in M_ZONES if z in existing_zones]}")
    print(f"R系統ゾーン: {[z for z in R_ZONES if z in existing_zones]}")

    print("\n## 目的変数の作成（将来温度の予測）")

    # 目的変数の作成
    df_with_targets = create_future_targets(df, existing_zones, HORIZONS, time_diff)
    print(f"目的変数を追加したデータシェイプ: {df_with_targets.shape}")

    # 外れ値処理の実行
    df_with_targets = filter_temperature_outliers(df_with_targets)

    # センサーデータの平滑化処理（ノイズ対策）
    df_with_targets, smoothed_features = apply_smoothing_to_sensors(df_with_targets, SMOOTHING_WINDOWS)

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
    # サーモ状態特徴量を作成
    df_with_targets, thermo_features = create_thermo_state_features(df_with_targets, existing_zones)

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

    # ゾーン別特徴量 (サーモ状態, AC有効状態, ACモード)
    for zone in existing_zones:
        if f'thermo_state_{zone}' in df_with_targets.columns:
            future_explanatory_base_config.append({'name': f'thermo_state_{zone}', 'type': 'zone_specific', 'zone': zone})
        if f'AC_valid_{zone}' in df_with_targets.columns:
            future_explanatory_base_config.append({'name': f'AC_valid_{zone}', 'type': 'zone_specific', 'zone': zone})

        ac_mode_col_candidate = f'AC_mode_{zone}'
        if ac_mode_col_candidate in df_with_targets.columns:
            future_explanatory_base_config.append({'name': ac_mode_col_candidate, 'type': 'zone_specific', 'zone': zone})
            print(f"ACモード特徴量 (未来予測対象): {ac_mode_col_candidate}")

    # 未来の説明変数を生成
    time_diff_seconds_val = time_diff.total_seconds()
    df_with_targets, all_future_explanatory_features = create_future_explanatory_features(
        df_with_targets,
        future_explanatory_base_config,
        HORIZONS,
        time_diff_seconds_val
    )
    print(f"{len(all_future_explanatory_features)}個の未来の説明変数を生成しました。")

    # 基本的な特徴量のリスト
    feature_cols = []

    # センサー温度・湿度（平滑化版を優先）
    feature_cols.extend(smoothed_features)

    # サーモ状態特徴量を追加
    feature_cols.extend(thermo_features)

    # AC設定温度も特徴量として追加
    ac_set_features = []
    for zone in existing_zones:
        if f'AC_set_{zone}' in df_with_targets.columns:
            ac_set_features.append(f'AC_set_{zone}')
            print(f"AC設定温度特徴量を追加: {f'AC_set_{zone}'}")

    feature_cols.extend(ac_set_features)

    # 空調システム関連（AC_valid）
    ac_control_features = []
    for zone in existing_zones:
        if f'AC_valid_{zone}' in df_with_targets.columns:
            ac_control_features.append(f'AC_valid_{zone}')
            print(f"AC有効状態特徴量を追加: {f'AC_valid_{zone}'}")

        ac_mode_col = f'AC_mode_{zone}'
        if ac_mode_col in df_with_targets.columns:
            ac_control_features.append(ac_mode_col)
            print(f"ACモード特徴量を追加: {ac_mode_col}")

    feature_cols.extend(ac_control_features)

    # 環境データ（現在の外気温・日射量）
    env_features_current = []
    if actual_atmo_temp_col_name:
        feature_cols.append(actual_atmo_temp_col_name)
        env_features_current.append(actual_atmo_temp_col_name)
        print(f"現在の外気温特徴量を追加: {actual_atmo_temp_col_name}")

    if actual_solar_rad_col_name:
        feature_cols.append(actual_solar_rad_col_name)
        env_features_current.append(actual_solar_rad_col_name)
        print(f"現在の日射量特徴量を追加: {actual_solar_rad_col_name}")

    # 時間特徴量（周期的特徴量を含む）
    if 'hour_sin' in df_with_targets.columns and 'hour_cos' in df_with_targets.columns:
        feature_cols.extend(['hour_sin', 'hour_cos'])
        print("周期的時間特徴量を追加: hour_sin, hour_cos")

    # LAG特徴量と移動平均特徴量を追加
    feature_cols.extend(lag_cols)
    feature_cols.extend(rolling_cols)
    feature_cols.extend(physics_cols)

    # 重複する特徴量を削除
    feature_cols = list(dict.fromkeys(feature_cols))
    print(f"基本特徴量 (現在時刻ベース): {len(feature_cols)}個")

    # ゾーンごとの結果を格納する辞書
    all_results = {}
    lag_dependency = {}

    print("\n## モデルトレーニングと評価")
    # 各ゾーンのモデルを構築
    for zone_to_predict in existing_zones:
        zone_results = {}

        # このゾーンに対応するLMR系統を特定
        if zone_to_predict in L_ZONES:
            zone_system = 'L'
        elif zone_to_predict in M_ZONES:
            zone_system = 'M'
        elif zone_to_predict in R_ZONES:
            zone_system = 'R'
        else:
            zone_system = 'Unknown'

        print(f"\nゾーン{zone_to_predict}({zone_system}系統)のモデル構築を開始します")

        for horizon in HORIZONS:
            target_col = f'sens_temp_{zone_to_predict}_future_{horizon}'

            # 目的変数が存在するか確認
            if target_col not in df_with_targets.columns:
                print(f"警告: 列 {target_col} が見つかりません。ゾーン{zone_to_predict}の{horizon}分後予測をスキップします。")
                continue

            print(f"\nゾーン{zone_to_predict}({zone_system}系統)の{horizon}分後の温度を予測するモデルを構築します")

            # このホライゾン用の特徴量を準備
            # 1. 基本特徴量 (現在の値)
            current_features_for_model = feature_cols.copy()

            # 2. このホライゾンに対応する「未来の説明変数」を追加
            horizon_specific_future_explanatory = []
            for f_col_config in future_explanatory_base_config:
                base_name = f_col_config['name']
                # ゾーン別特徴量は予測対象ゾーンにのみ限定
                if f_col_config['type'] == 'zone_specific' and f_col_config.get('zone') != zone_to_predict:
                    continue

                future_variant_name = f"{base_name}_future_{horizon}"
                if future_variant_name in df_with_targets.columns:
                    horizon_specific_future_explanatory.append(future_variant_name)

            current_features_for_model.extend(horizon_specific_future_explanatory)
            current_features_for_model = list(dict.fromkeys(current_features_for_model)) # 重複除去

            # 目的変数と特徴量を準備
            temp_df_for_dropna = df_with_targets[current_features_for_model + [target_col]].copy()
            temp_df_for_dropna.dropna(subset=[target_col], inplace=True) # まず目的変数のNaNを除去

            # X_baseとyを再定義
            y_intermediate = temp_df_for_dropna[target_col]
            X_base = temp_df_for_dropna[current_features_for_model]

            # NaNを含む行を除外
            X_base.dropna(inplace=True)
            y_intermediate = y_intermediate.loc[X_base.index] # X_baseに合わせてyも更新

            if len(X_base) == 0:
                print(f"警告: ゾーン{zone_to_predict}の{horizon}分後予測に使用可能なデータがありません。スキップします。")
                continue

            # 時系列に基づいてデータを分割
            cutoff_date = get_time_based_train_test_split(X_base, test_size=TEST_SIZE)

            X_train_base = X_base[X_base.index <= cutoff_date]
            X_test_base = X_base[X_base.index > cutoff_date]
            y_train = y_intermediate[y_intermediate.index <= cutoff_date]
            y_test = y_intermediate[y_intermediate.index > cutoff_date]

            # 特徴量選択を行い、重要な特徴量のみを使用
            X_train_selected, X_test_selected, selected_features = select_important_features(
                X_train_base, y_train, X_test_base, X_train_base.columns.tolist(), threshold=FEATURE_SELECTION_THRESHOLD
            )

            print(f"最終的な学習特徴量数: {len(selected_features)}")

            # 物理ベースのモデルトレーニング
            print("物理法則ガイド付きLightGBMモデルをトレーニング中...")
            lgb_model = train_physics_guided_model(X_train_selected, y_train)

            # モデルと特徴量情報の保存
            save_model_and_features(
                lgb_model, selected_features, zone_to_predict, horizon
            )

            # 予測と評価
            y_pred = lgb_model.predict(X_test_selected)
            metrics = calculate_metrics(y_test, y_pred)
            print_metrics(metrics, zone_to_predict, horizon)

            # 特徴量重要度の分析
            feature_importance = analyze_feature_importance(lgb_model, selected_features)

            # 結果の保存
            zone_results[horizon] = {
                'model': lgb_model,
                'X_test': X_test_selected,
                'y_test': y_test,
                'y_pred': y_pred,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'feature_importance': feature_importance,
                'system': zone_system
            }

            # LAG依存度分析(15分後または最初のホライゾン)
            if horizon == 15 or (horizon == HORIZONS[0] and 15 not in HORIZONS):
                lag_dependency[zone_to_predict] = analyze_lag_dependency(
                    feature_importance, zone_to_predict, horizon, zone_system
                )

        all_results[zone_to_predict] = zone_results

    # 視覚化
    print("\n## 結果の可視化")

    # 特徴量重要度を可視化
    for zone, results in all_results.items():
        if not results:  # 結果がない場合はスキップ
            continue

        # ゾーンの系統を取得
        zone_system = results[list(results.keys())[0]]['system']

        # 15分後予測(またはある最初のホライゾン)の重要度を使用
        horizon = 15 if 15 in results else list(results.keys())[0]
        feature_importance = results[horizon]['feature_importance']

        plot_feature_importance(feature_importance, zone, zone_system, horizon)

    # 予測ホライゾンごとに全ゾーンの散布図をプロット
    for horizon in HORIZONS:
        plot_scatter_actual_vs_predicted(all_results, horizon)

    # 予測ホライゾンごとに全ゾーンの時系列プロット
    for horizon in HORIZONS:
        plot_time_series(all_results, horizon)

    # LAG依存度分析結果をCSVにまとめる
    lag_dependency_df = pd.DataFrame([
        {
            'ゾーン': zone,
            'ホライゾン(分)': data['horizon'],
            '系統': data.get('system', 'Unknown'),
            '現在温度依存度(%)': data['current_sensor_temp_percent'],
            '現在湿度依存度(%)': data['current_sensor_humid_percent'],
            'LAG温度依存度(%)': data['lag_temp_percent'],
            '移動平均温度依存度(%)': data['rolling_temp_percent'],
            '現在サーモ状態依存度(%)': data['thermo_state_current_percent'],
            '現在AC制御依存度(%)': data['ac_control_current_percent'],
            '現在環境データ依存度(%)': data['env_current_percent'],
            '未来サーモ状態依存度(%)': data['future_thermo_state_percent'],
            '未来AC制御依存度(%)': data['future_ac_control_percent'],
            '未来環境データ依存度(%)': data['future_env_percent'],
            '多項式特徴量依存度(%)': data['poly_interaction_percent'],
            '時間特徴量依存度(%)': data['time_percent'],
            'その他特徴量依存度(%)': data['other_percent'],
            '過去時系列合計(%)': data['total_past_time_series_percent'],
            '現在非センサー合計(%)': data['total_current_non_sensor_percent'],
            '未来説明変数合計(%)': data['total_future_explanatory_percent'],
        }
        for zone, data in lag_dependency.items()
    ])

    # LAG依存度分析の可視化
    plot_lag_dependency_analysis(lag_dependency_df)

    # CSV保存
    lag_dependency_df.to_csv(os.path.join(OUTPUT_DIR, 'lag_dependency.csv'), index=False)
    print("LAG依存度分析結果をCSVファイルに保存しました")

    # 系統別の平均性能を計算
    system_performance = pd.DataFrame([
        {
            'System': 'L',
            'Zones': [z for z in L_ZONES if z in existing_zones],
            'Avg_R2': np.mean([all_results[z][15]['r2'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['r2']
                             for z in L_ZONES if z in existing_zones and all_results.get(z)]),
            'Avg_RMSE': np.mean([all_results[z][15]['rmse'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['rmse']
                               for z in L_ZONES if z in existing_zones and all_results.get(z)])
        },
        {
            'System': 'M',
            'Zones': [z for z in M_ZONES if z in existing_zones],
            'Avg_R2': np.mean([all_results[z][15]['r2'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['r2']
                             for z in M_ZONES if z in existing_zones and all_results.get(z)]),
            'Avg_RMSE': np.mean([all_results[z][15]['rmse'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['rmse']
                               for z in M_ZONES if z in existing_zones and all_results.get(z)])
        },
        {
            'System': 'R',
            'Zones': [z for z in R_ZONES if z in existing_zones],
            'Avg_R2': np.mean([all_results[z][15]['r2'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['r2']
                             for z in R_ZONES if z in existing_zones and all_results.get(z)]),
            'Avg_RMSE': np.mean([all_results[z][15]['rmse'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['rmse']
                               for z in R_ZONES if z in existing_zones and all_results.get(z)])
        }
    ])
    system_performance.to_csv(os.path.join(OUTPUT_DIR, 'system_performance.csv'), index=False)
    print("系統別の予測性能をCSVファイルに保存しました")

    # 各ゾーンの結果をテーブルにまとめる
    summary_data = []
    for zone, results in all_results.items():
        if 15 in results:  # 15分後予測の結果があれば使用
            h = 15
        elif results:  # なければ最初のホライゾンを使用
            h = list(results.keys())[0]
        else:
            continue

        # ゾーンの系統を取得
        zone_system = results[h]['system']

        summary_data.append({
            'ゾーン': zone,
            '系統': zone_system,
            'ホライゾン(分)': h,
            'RMSE': results[h]['rmse'],
            'MAE': results[h]['mae'],
            'R²': results[h]['r2'],
            '過去時系列依存度(%)': lag_dependency[zone]['total_past_time_series_percent'],
            '現在非センサー依存度(%)': lag_dependency[zone]['total_current_non_sensor_percent'],
            '未来説明変数依存度(%)': lag_dependency[zone]['total_future_explanatory_percent'],
            '多項式特徴量依存度(%)': lag_dependency[zone]['poly_interaction_percent'],
            '重要特徴量': ', '.join(results[h]['feature_importance'].sort_values('importance', ascending=False).head(5)['feature'].tolist())
        })

    summary_df = pd.DataFrame(summary_data)
    print("各ゾーンの予測性能まとめ:")
    print(summary_df)

    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'prediction_summary.csv'), index=False)
    print("予測性能まとめをCSVファイルに保存しました")

    print("\n分析が完了しました。すべての結果はOutputディレクトリに保存されています。")

    return all_results, lag_dependency_df


if __name__ == "__main__":
    main()
