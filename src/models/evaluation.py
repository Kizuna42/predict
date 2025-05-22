#!/usr/bin/env python
# coding: utf-8

"""
モデル評価モジュール
予測モデルの評価指標の計算や評価結果の分析関数を提供
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def calculate_metrics(y_true, y_pred):
    """
    回帰モデルの評価指標を計算する関数

    Parameters:
    -----------
    y_true : Series
        実際の値
    y_pred : Series or array
        予測値

    Returns:
    --------
    dict
        各種評価指標を含む辞書
    """
    # NaN値の処理：両方のデータで対応する位置にNaNがないデータのみを使用
    valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        print("警告: 有効なデータがありません。すべての値がNaNです。")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'mape': float('nan'),
            'r2': float('nan')
        }

    # NaN値を除外したデータで評価指標を計算
    return {
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid),
        'mape': mean_absolute_percentage_error(y_true_valid, y_pred_valid) * 100,  # パーセント表示
        'r2': r2_score(y_true_valid, y_pred_valid)
    }


def print_metrics(metrics, zone=None, horizon=None):
    """
    評価指標を整形して表示する関数

    Parameters:
    -----------
    metrics : dict
        評価指標の辞書
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    header = "評価指標"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    モデルの特徴量重要度を分析する関数

    Parameters:
    -----------
    model : LGBMRegressor or similar model
        評価対象のモデル（feature_importances_属性を持つモデル）
    feature_names : list
        特徴量名のリスト
    top_n : int, optional
        表示する上位特徴量の数

    Returns:
    --------
    DataFrame
        特徴量重要度を降順にソートしたデータフレーム
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    # 重要度順にソート
    sorted_importance = feature_importance.sort_values('importance', ascending=False)

    # 上位n個の特徴量を表示
    top_features = sorted_importance.head(top_n)
    print(f"\n上位{top_n}個の重要な特徴量:")
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")

    return sorted_importance


def analyze_lag_dependency(feature_importance, zone=None, horizon=None, zone_system=None):
    """
    特徴量重要度を元にLAG依存度を分析する関数

    Parameters:
    -----------
    feature_importance : DataFrame
        特徴量名と重要度を含むデータフレーム
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    zone_system : str, optional
        ゾーンの系統 (L, M, R など)

    Returns:
    --------
    dict
        LAG依存度分析結果の辞書
    """
    # 特徴量カテゴリの定義と分類
    current_sensor_temp_features = []
    current_sensor_humid_features = []
    lag_temp_features = []
    rolling_temp_features = []
    thermo_state_current_features = []
    ac_control_current_features = []
    env_current_features = []

    future_thermo_state_features = []
    future_ac_control_features = []
    future_env_features = []

    poly_interaction_features = []
    time_features = []
    other_identified_features = []

    all_model_features = feature_importance['feature'].tolist()

    # atmosとsolarのカラム名を抽出（実際のデータに基づいて変更する必要あり）
    atmos_temp_pattern = 'atmospheric'
    solar_rad_pattern = 'solar'

    for f_name in all_model_features:
        is_future = '_future_' in f_name
        is_lag = '_lag_' in f_name
        is_rolling = '_rolling_' in f_name or '_rate_mean_' in f_name or '_accel_mean_' in f_name
        is_poly = 'poly_' in f_name
        is_thermo = 'thermo_state_' in f_name
        is_sens_temp = 'sens_temp_' in f_name and not is_future and not is_lag and not is_rolling and not is_poly
        is_sens_humid = 'sens_humid_' in f_name and not is_future and not is_lag and not is_rolling and not is_poly
        is_ac_valid = 'AC_valid_' in f_name
        is_ac_mode = 'AC_mode_' in f_name
        is_atmo_temp = atmos_temp_pattern in f_name.lower()
        is_solar_rad = solar_rad_pattern in f_name.lower()
        is_hour = f_name == 'hour' or 'hour_sin' in f_name or 'hour_cos' in f_name

        if is_poly:
            poly_interaction_features.append(f_name)
        elif is_lag and is_sens_temp:
            lag_temp_features.append(f_name)
        elif is_rolling and is_sens_temp:
            rolling_temp_features.append(f_name)
        elif is_sens_temp:
            current_sensor_temp_features.append(f_name)
        elif is_sens_humid:
            current_sensor_humid_features.append(f_name)
        elif is_hour:
            time_features.append(f_name)
        elif is_thermo:
            if is_future:
                future_thermo_state_features.append(f_name)
            else:
                thermo_state_current_features.append(f_name)
        elif is_ac_valid or is_ac_mode:
            if is_future:
                future_ac_control_features.append(f_name)
            else:
                ac_control_current_features.append(f_name)
        elif is_atmo_temp or is_solar_rad:
            if is_future:
                future_env_features.append(f_name)
            else:
                env_current_features.append(f_name)
        else:
            other_identified_features.append(f_name)

    # 各カテゴリの重要度合計を計算
    def get_sum_importance(features_list):
        return feature_importance[feature_importance['feature'].isin(features_list)]['importance'].sum()

    current_sensor_temp_importance = get_sum_importance(current_sensor_temp_features)
    current_sensor_humid_importance = get_sum_importance(current_sensor_humid_features)
    lag_temp_importance = get_sum_importance(lag_temp_features)
    rolling_temp_importance = get_sum_importance(rolling_temp_features)
    thermo_state_current_importance = get_sum_importance(thermo_state_current_features)
    ac_control_current_importance = get_sum_importance(ac_control_current_features)
    env_current_importance = get_sum_importance(env_current_features)

    future_thermo_state_importance = get_sum_importance(future_thermo_state_features)
    future_ac_control_importance = get_sum_importance(future_ac_control_features)
    future_env_importance = get_sum_importance(future_env_features)

    poly_interaction_importance = get_sum_importance(poly_interaction_features)
    time_importance = get_sum_importance(time_features)
    other_importance = get_sum_importance(other_identified_features)

    total_importance_calculated = sum([
        current_sensor_temp_importance, current_sensor_humid_importance, lag_temp_importance, rolling_temp_importance,
        thermo_state_current_importance, ac_control_current_importance, env_current_importance,
        future_thermo_state_importance, future_ac_control_importance, future_env_importance,
        poly_interaction_importance, time_importance, other_importance
    ])

    # パーセンテージに変換
    def to_percent(value):
        return (value / total_importance_calculated * 100) if total_importance_calculated > 0 else 0

    result = {
        'zone': zone if zone is not None else 'all',
        'horizon': horizon if horizon is not None else 'all',
        'system': zone_system if zone_system is not None else 'unknown',
        'current_sensor_temp_percent': to_percent(current_sensor_temp_importance),
        'current_sensor_humid_percent': to_percent(current_sensor_humid_importance),
        'lag_temp_percent': to_percent(lag_temp_importance),
        'rolling_temp_percent': to_percent(rolling_temp_importance),
        'thermo_state_current_percent': to_percent(thermo_state_current_importance),
        'ac_control_current_percent': to_percent(ac_control_current_importance),
        'env_current_percent': to_percent(env_current_importance),
        'future_thermo_state_percent': to_percent(future_thermo_state_importance),
        'future_ac_control_percent': to_percent(future_ac_control_importance),
        'future_env_percent': to_percent(future_env_importance),
        'poly_interaction_percent': to_percent(poly_interaction_importance),
        'time_percent': to_percent(time_importance),
        'other_percent': to_percent(other_importance),
        'total_past_time_series_percent': to_percent(current_sensor_temp_importance + current_sensor_humid_importance +
                                                    lag_temp_importance + rolling_temp_importance),
        'total_current_non_sensor_percent': to_percent(thermo_state_current_importance + ac_control_current_importance +
                                                      env_current_importance + time_importance),
        'total_future_explanatory_percent': to_percent(future_thermo_state_importance + future_ac_control_importance +
                                                       future_env_importance)
    }

    return result


def calculate_physical_validity_metrics(y_true, y_pred, ac_state, ac_mode, horizon):
    """
    予測の物理的妥当性を評価する指標を計算する関数

    Parameters:
    -----------
    y_true : Series or array-like
        実測値
    y_pred : Series or array-like
        予測値
    ac_state : Series or array-like
        空調の状態（0: OFF, 1: ON）
    ac_mode : Series or array-like
        空調のモード（0: 冷房, 1: 暖房）
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        物理的妥当性の評価指標
    """
    # 温度変化量の計算
    temp_change_true = y_true.diff()
    temp_change_pred = y_pred.diff()

    # 物理的妥当性の評価
    validity_metrics = {
        'cooling_validity': 0.0,  # 冷房時の妥当性
        'heating_validity': 0.0,  # 暖房時の妥当性
        'natural_validity': 0.0,  # 空調OFF時の妥当性
        'direction_accuracy': 0.0,  # 温度変化方向の一致率
        'response_delay': 0.0,  # 応答遅れの評価
    }

    # 冷房時の妥当性評価
    cooling_mask = (ac_state == 1) & (ac_mode == 0)
    if cooling_mask.sum() > 0:
        cooling_valid = (temp_change_pred[cooling_mask] < 0).mean()
        validity_metrics['cooling_validity'] = cooling_valid

    # 暖房時の妥当性評価
    heating_mask = (ac_state == 1) & (ac_mode == 1)
    if heating_mask.sum() > 0:
        heating_valid = (temp_change_pred[heating_mask] > 0).mean()
        validity_metrics['heating_validity'] = heating_valid

    # 空調OFF時の妥当性評価
    off_mask = (ac_state == 0)
    if off_mask.sum() > 0:
        # 外気温との関係を考慮した自然温度変化の妥当性
        natural_valid = (temp_change_pred[off_mask] * temp_change_true[off_mask] > 0).mean()
        validity_metrics['natural_validity'] = natural_valid

    # 温度変化方向の一致率
    direction_accuracy = (temp_change_pred * temp_change_true > 0).mean()
    validity_metrics['direction_accuracy'] = direction_accuracy

    # 応答遅れの評価
    # 空調状態変化後の温度変化の遅れを評価
    ac_state_change = ac_state.diff().abs()
    if ac_state_change.sum() > 0:
        # 状態変化後の温度変化の遅れを計算
        response_delays = []
        for i in range(1, len(ac_state_change)):
            if ac_state_change[i] == 1:
                # 状態変化後の温度変化を観察
                true_change = temp_change_true[i:i+horizon].sum()
                pred_change = temp_change_pred[i:i+horizon].sum()
                if true_change != 0:
                    delay = abs(pred_change - true_change) / abs(true_change)
                    response_delays.append(delay)

        if response_delays:
            validity_metrics['response_delay'] = np.mean(response_delays)

    return validity_metrics


def print_physical_validity_metrics(metrics, zone=None, horizon=None):
    """
    物理的妥当性の評価指標を整形して表示する関数

    Parameters:
    -----------
    metrics : dict
        物理的妥当性の評価指標
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    header = "物理的妥当性の評価"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"冷房時の妥当性: {metrics['cooling_validity']:.4f}")
    print(f"暖房時の妥当性: {metrics['heating_validity']:.4f}")
    print(f"自然温度変化の妥当性: {metrics['natural_validity']:.4f}")
    print(f"温度変化方向の一致率: {metrics['direction_accuracy']:.4f}")
    print(f"応答遅れの評価: {metrics['response_delay']:.4f}")
