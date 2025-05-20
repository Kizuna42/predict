#!/usr/bin/env python
# coding: utf-8

"""
特徴量エンジニアリングモジュール
LAG特徴量、移動平均特徴量、物理モデルベースの特徴量など、各種特徴量生成関数を提供
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb


def create_lag_features(df, zone_nums, lag_periods=[6, 12]):
    """
    各ゾーンの過去の温度と湿度をLAG特徴量として作成
    修正: LAGに過度に依存しないように、直近の短期LAGを削除し長期LAGを使用

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    lag_periods : list
        ラグ期間（データサンプリング単位）のリスト（長期LAGのみに変更）

    Returns:
    --------
    DataFrame
        LAG特徴量を追加したデータフレーム
    list
        作成した特徴量のリスト
    """
    print("物理モデルベースのLAG特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    for zone in zone_nums:
        # 温度のLAG特徴量ではなく、温度変化率に焦点を当てる
        if f'sens_temp_{zone}' in df.columns:
            # 温度変化量（一定期間での温度差）- 物理的意味を持つ特徴量
            for lag in lag_periods:
                # 短期的な変化率（直前からの変化ではなく、より長期の変化を捉える）
                df_copy[f'temp_change_rate_{zone}_{lag}'] = (df_copy[f'sens_temp_{zone}'] - df_copy[f'sens_temp_{zone}'].shift(lag)) / lag
                created_features.append(f'temp_change_rate_{zone}_{lag}')

            # 温度の加速度（変化率の変化）- 物理的なパラメータ
            df_copy[f'temp_acceleration_{zone}'] = df_copy[f'sens_temp_{zone}'].diff().diff()
            created_features.append(f'temp_acceleration_{zone}')

            # 温度変化のトレンド（上昇/下降の持続性）- 長期的なトレンドを捉える
            df_copy[f'temp_trend_{zone}_long'] = np.sign(df_copy[f'sens_temp_{zone}'].diff(6))
            created_features.append(f'temp_trend_{zone}_long')

    print(f"作成した物理ベースの特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_rolling_features(df, zone_nums, windows=[6, 12, 24]):
    """
    各ゾーンの温度と湿度の移動平均を特徴量として作成
    重要: 未来の値を使わないようにmin_periods=1を設定し、過去のデータのみを使用
    修正: 変化率を中心とした特徴量に変更し、より長い窓幅を追加

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    windows : list
        移動平均の窓サイズ（データサンプリング単位）のリスト

    Returns:
    --------
    DataFrame
        移動平均特徴量を追加したデータフレーム
    list
        作成した特徴量のリスト
    """
    print("物理的意味を持つ移動平均特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    for zone in zone_nums:
        # 温度の移動平均ではなく、物理的な意味を持つ特徴量を作成
        if f'sens_temp_{zone}' in df.columns:
            for window in windows:
                # 温度変化率の移動平均（温度の一次微分の平滑化）
                df_copy[f'temp_rate_mean_{zone}_{window}'] = df_copy[f'sens_temp_{zone}'].diff().rolling(
                    window=window, min_periods=1).mean()
                created_features.append(f'temp_rate_mean_{zone}_{window}')

                # 温度加速度の移動平均（温度の二次微分の平滑化）
                df_copy[f'temp_accel_mean_{zone}_{window}'] = df_copy[f'sens_temp_{zone}'].diff().diff().rolling(
                    window=window, min_periods=1).mean()
                created_features.append(f'temp_accel_mean_{zone}_{window}')

                # 温度変動の振幅（温度の安定性/変動性を物理的に表現）
                df_copy[f'temp_amplitude_{zone}_{window}'] = df_copy[f'sens_temp_{zone}'].rolling(
                    window=window, min_periods=1).max() - df_copy[f'sens_temp_{zone}'].rolling(
                    window=window, min_periods=1).min()
                created_features.append(f'temp_amplitude_{zone}_{window}')

    print(f"作成した物理的特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_physics_based_features(df, zone_nums):
    """
    物理法則に基づいた特徴量を作成する関数
    改善: より詳細な熱力学モデルに基づいた特徴量を追加

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト

    Returns:
    --------
    DataFrame
        物理モデルベースの特徴量を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    print("拡張された物理モデルベースの特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    # 1. 熱力学の法則に基づく特徴量
    for zone in zone_nums:
        if f'sens_temp_{zone}' in df.columns:
            # 温度変化率（一次微分）
            df_copy[f'temp_rate_{zone}'] = df_copy[f'sens_temp_{zone}'].diff()
            created_features.append(f'temp_rate_{zone}')

            # 温度加速度（二次微分 - 温度変化の変化率）
            df_copy[f'temp_accel_{zone}'] = df_copy[f'temp_rate_{zone}'].diff()
            created_features.append(f'temp_accel_{zone}')

            # 正規化した温度変化率（過去に対する相対変化）
            df_copy[f'temp_rel_change_{zone}'] = df_copy[f'temp_rate_{zone}'] / df_copy[f'sens_temp_{zone}'].shift(1).replace(0, np.nan)
            df_copy[f'temp_rel_change_{zone}'].fillna(0, inplace=True)
            created_features.append(f'temp_rel_change_{zone}')

            # 温度の慣性（過去の変化の持続性 - より長期的な視点）
            df_copy[f'temp_momentum_{zone}'] = df_copy[f'sens_temp_{zone}'].diff(3) * df_copy[f'sens_temp_{zone}'].diff()
            created_features.append(f'temp_momentum_{zone}')

            # 1.1 設定温度との関係
            if f'AC_set_{zone}' in df.columns:
                # 設定温度との差（熱力学における駆動力）
                df_copy[f'temp_diff_to_setpoint_{zone}'] = df_copy[f'sens_temp_{zone}'] - df_copy[f'AC_set_{zone}']
                created_features.append(f'temp_diff_to_setpoint_{zone}')

                # 熱伝達の効率性指標（温度差に対する変化率の比）
                df_copy[f'heat_transfer_efficiency_{zone}'] = df_copy[f'sens_temp_{zone}'].diff() / df_copy[f'temp_diff_to_setpoint_{zone}'].abs()
                # 分母が0になる可能性があるため、無限大や無効な値を処理
                df_copy[f'heat_transfer_efficiency_{zone}'].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_copy[f'heat_transfer_efficiency_{zone}'].fillna(0, inplace=True)  # NaNを0に置換
                created_features.append(f'heat_transfer_efficiency_{zone}')

                # 設定との差の変化率（制御応答の指標）
                df_copy[f'setpoint_response_{zone}'] = df_copy[f'temp_diff_to_setpoint_{zone}'].diff()
                created_features.append(f'setpoint_response_{zone}')

                # 設定温度への収束速度の推定
                df_copy[f'convergence_rate_{zone}'] = df_copy[f'temp_diff_to_setpoint_{zone}'] / df_copy[f'temp_diff_to_setpoint_{zone}'].shift(1)
                df_copy[f'convergence_rate_{zone}'].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_copy[f'convergence_rate_{zone}'].fillna(1, inplace=True)  # NaNを1に置換（変化なし）
                created_features.append(f'convergence_rate_{zone}')

            # 1.2 空調制御状態に基づく特徴量
            if f'AC_valid_{zone}' in df.columns:
                # AC有効状態と温度変化の交互作用
                df_copy[f'ac_temp_interaction_{zone}'] = df_copy[f'AC_valid_{zone}'] * df_copy[f'temp_rate_{zone}']
                created_features.append(f'ac_temp_interaction_{zone}')

                # AC状態変化の検出（ON/OFFが切り替わった時点）
                df_copy[f'ac_state_change_{zone}'] = df_copy[f'AC_valid_{zone}'].diff().abs()
                created_features.append(f'ac_state_change_{zone}')

                # AC OFF後の温度ドリフト検出（自然温度変化）
                ac_off_mask = (df_copy[f'AC_valid_{zone}'] == 0)
                df_copy[f'natural_temp_drift_{zone}'] = 0.0
                if ac_off_mask.sum() > 0:
                    df_copy.loc[ac_off_mask, f'natural_temp_drift_{zone}'] = df_copy.loc[ac_off_mask, f'temp_rate_{zone}']
                created_features.append(f'natural_temp_drift_{zone}')

            # 1.3 外気温との関係
            if any('atmospheric' in col and 'temperature' in col for col in df.columns):
                atmos_col = [col for col in df.columns if 'atmospheric' in col and 'temperature' in col][0]

                # 外気温との温度差（熱伝導の駆動力）
                df_copy[f'temp_diff_to_outside_{zone}'] = df_copy[f'sens_temp_{zone}'] - df_copy[atmos_col]
                created_features.append(f'temp_diff_to_outside_{zone}')

                # 外気温との温度差の変化率
                df_copy[f'temp_diff_to_outside_rate_{zone}'] = df_copy[f'temp_diff_to_outside_{zone}'].diff()
                created_features.append(f'temp_diff_to_outside_rate_{zone}')

                # 外気温の変化率
                df_copy[f'atmos_temp_rate'] = df_copy[atmos_col].diff()
                created_features.append('atmos_temp_rate')

            # 1.4 日射量との関係
            if any('solar' in col and 'radiation' in col for col in df.columns):
                solar_col = [col for col in df.columns if 'solar' in col and 'radiation' in col][0]

                # 日射量の変化率
                df_copy[f'solar_radiation_rate'] = df_copy[solar_col].diff()
                created_features.append('solar_radiation_rate')

                # 日射量と温度変化の交互作用
                df_copy[f'solar_temp_interaction_{zone}'] = df_copy[solar_col] * df_copy[f'temp_rate_{zone}']
                created_features.append(f'solar_temp_interaction_{zone}')

    print(f"作成した物理的特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_future_explanatory_features(df, base_features_config, horizons_minutes, time_diff_seconds):
    """
    制御可能なパラメータの未来値を説明変数として作成する関数
    改善: データリークを防ぐため、制御可能なパラメータのみを使用

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    base_features_config : list of dict
        基本特徴量の設定（例: [{'name': 'thermo_state_1', 'type': 'zone_specific', 'zone': 1}]）
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : int または float
        データのサンプリング間隔（秒）

    Returns:
    --------
    DataFrame
        未来の説明変数を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    print("制御可能なパラメータの未来値を特徴量として作成中...")
    df_copy = df.copy()
    created_features = []

    # 制御可能なパラメータのプレフィックスのリスト
    controllable_params_prefixes = [
        'thermo_state_',  # サーモスタット状態
        'AC_mode_',       # 空調モード
        'AC_valid_',      # 空調有効状態
        'AC_set_'         # 設定温度
    ]

    # 各ホライゾンに対して
    for horizon in horizons_minutes:
        # シフト量を計算（データサンプリング間隔に基づく）
        shift_steps = int(horizon * 60 / time_diff_seconds)

        # 各特徴量の設定を処理
        for config in base_features_config:
            base_col_name = config['name']

            # 特徴量がデータフレームに存在するか確認
            if base_col_name not in df_copy.columns:
                continue

            # 制御可能なパラメータかどうかを確認
            is_controllable = any(base_col_name.startswith(prefix) for prefix in controllable_params_prefixes)

            # 制御可能なパラメータのみ未来値を作成
            if is_controllable:
                future_col = f"{base_col_name}_future_{horizon}"
                df_copy[future_col] = df_copy[base_col_name].shift(-shift_steps)
                created_features.append(future_col)

    print(f"作成した未来の説明変数: {len(created_features)}個")
    return df_copy, created_features


def create_thermo_state_features(df, zone_nums):
    """
    サーモ状態の特徴量を作成
    センサー温度とAC設定温度の差を計算

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト

    Returns:
    --------
    DataFrame
        サーモ状態特徴量を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    df_copy = df.copy()
    thermo_features = []

    for zone in zone_nums:
        if f'sens_temp_{zone}' in df_copy.columns and f'AC_set_{zone}' in df_copy.columns:
            # サーモ状態 = センサー温度 - 設定温度
            thermo_col = f'thermo_state_{zone}'
            # 平滑化されたセンサー温度が存在すればそちらを優先してサーモ状態を計算
            base_temp_col_for_thermo = f'sens_temp_{zone}_smoothed' if f'sens_temp_{zone}_smoothed' in df_copy.columns else f'sens_temp_{zone}'
            df_copy[thermo_col] = df_copy[base_temp_col_for_thermo] - df_copy[f'AC_set_{zone}']
            thermo_features.append(thermo_col)
            print(f"ゾーン{zone}のサーモ状態特徴量を作成しました: {thermo_col} (ベース温度: {base_temp_col_for_thermo})")

    return df_copy, thermo_features


def select_important_features(X_train, y_train, X_test, feature_names, threshold='25%'):
    """
    LightGBMを使った特徴量選択により、重要な特徴量のみを選択
    閾値を変更：物理特徴量を優先的に選択するため

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数
    X_test : DataFrame
        テスト用特徴量
    feature_names : list
        特徴量名のリスト
    threshold : str or float
        特徴量選択の閾値

    Returns:
    --------
    X_train_selected, X_test_selected : 選択された特徴量のみを含むデータフレーム
    selected_features : 選択された特徴量名のリスト
    """
    # 特徴量選択用の軽量なモデル
    selection_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.05,
        random_state=42,
        importance_type='gain'  # 利得ベースの重要度を使用
    )

    # SelectFromModelで重要な特徴量を選択
    selector = SelectFromModel(selection_model, threshold=threshold)

    try:
        # 物理特徴量を優先して選択するために重みを付ける
        # 温度変化の傾向と関連する特徴量
        temp_changes = y_train.diff().abs().fillna(0)
        weights = 1 + temp_changes / temp_changes.mean()

        # スパイクの影響を制限
        max_weight = 3.0  # 最大ウェイト値を制限
        weights = weights.clip(upper=max_weight)

        # モデルをトレーニング
        selector.fit(X_train, y_train, sample_weight=weights)

        # 選択された特徴量のマスクを取得
        feature_mask = selector.get_support()
        selected_feature_indices = [i for i, selected in enumerate(feature_mask) if selected]
        selected_features = [feature_names[i] for i in selected_feature_indices]

        # LAG特徴量の依存度を減らす追加処理
        # すべての選択された特徴量の中でLAG特徴量の割合を確認
        lag_features = [f for f in selected_features if 'lag' in f.lower()]
        if len(lag_features) > len(selected_features) * 0.25:  # LAG特徴量が25%を超える場合
            # LAG特徴量を減らし、物理ベースの特徴量を優先
            physical_features = [f for f in selected_features if any(
                key in f.lower() for key in ['diff', 'rate', 'accel', 'momentum', 'exchange', 'efficiency', 'amplitude']
            )]
            # 環境特徴量
            env_features = [f for f in selected_features if any(
                key in f.lower() for key in ['atmospheric', 'solar', 'radiation', 'outside']
            )]
            # 制御特徴量
            control_features = [f for f in selected_features if any(
                key in f.lower() for key in ['ac_', 'thermo_state', 'valid']
            )]
            # 時間特徴量
            time_features = [f for f in selected_features if any(
                key in f.lower() for key in ['hour', 'day', 'sin', 'cos']
            )]

            # LAG特徴量を減らして他の特徴量を優先
            max_lag_features = max(1, int(len(selected_features) * 0.15))  # 最大でも15%までに制限
            if len(lag_features) > max_lag_features:
                # 長期のLAGを優先して選択（短期LAGを除外）
                lag_periods = [int(f.split('_')[-1]) for f in lag_features if f.split('_')[-1].isdigit()]
                sorted_lag_features = [f for _, f in sorted(zip(lag_periods, lag_features), key=lambda x: x[0], reverse=True)]
                selected_lag_features = sorted_lag_features[:max_lag_features]

                # 最終的な特徴量リストを作成
                final_selected_features = physical_features + env_features + control_features + time_features + selected_lag_features
                final_selected_features = list(dict.fromkeys(final_selected_features))  # 重複を除去

                # 特徴量マスクを更新
                feature_mask = [f in final_selected_features for f in feature_names]
                selected_features = final_selected_features

                print(f"LAG特徴量の依存度を下げるため、特徴量を調整しました: {len(selected_features)}個")

        # 選択された特徴量でデータセットをフィルタリング
        X_train_selected = X_train.loc[:, feature_mask]
        X_test_selected = X_test.loc[:, feature_mask]

        print(f"特徴量選択により{len(selected_features)}/{len(feature_names)}個の特徴量を選択しました")

        return X_train_selected, X_test_selected, selected_features

    except Exception as e:
        print(f"特徴量選択中にエラーが発生しました: {e}")
        # エラーの場合は全特徴量を返す
        return X_train, X_test, feature_names
