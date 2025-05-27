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


def print_lag_dependency_warning(lag_dependency, threshold=30.0, zone=None, horizon=None):
    """
    LAG特徴量への依存度が高い場合に警告を表示する関数

    Parameters:
    -----------
    lag_dependency : dict
        LAG依存度分析の結果辞書
    threshold : float, optional
        警告を表示するLAG依存度の閾値（パーセント）
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    # 総LAG依存度の計算（lag_temp_percentとrolling_temp_percentの合計）
    total_lag_dependency = lag_dependency['lag_temp_percent'] + lag_dependency['rolling_temp_percent']

    # 現在のセンサー値への依存度（過去データの代替指標として使用）
    current_temp_dependency = lag_dependency.get('current_temp_percent', 0)
    future_temp_dependency = lag_dependency.get('future_temp_percent', 0)

    header = "LAG特徴量依存度分析"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"直接的LAG特徴量依存度: {lag_dependency['lag_temp_percent']:.2f}%")
    print(f"移動平均特徴量依存度: {lag_dependency['rolling_temp_percent']:.2f}%")
    print(f"総LAG依存度: {total_lag_dependency:.2f}%")
    print(f"現在センサー値依存度: {current_temp_dependency:.2f}%")
    print(f"未来制御情報依存度: {future_temp_dependency:.2f}%")

    # 依存度が閾値を超える場合に警告を表示
    if total_lag_dependency > threshold:
        print(f"\n⚠️ 警告: LAG特徴量への依存度が高すぎます ({total_lag_dependency:.2f}% > {threshold:.2f}%)")
        print("  モデルが過去の温度値に過度に依存している可能性があります。")
        print("  以下の対策を検討してください:")
        print("  - より物理的な意味を持つ特徴量を追加")
        print("  - LAG特徴量の重みを下げる")
        print("  - より長期のLAGのみを使用")
    elif current_temp_dependency > threshold * 1.5:
        print(f"\n⚠️ 注意: 現在センサー値への依存度が高めです ({current_temp_dependency:.2f}%)")
        print("  モデルが現在のデータに依存している可能性があります。")
    else:
        print(f"\n✅ LAG依存度は許容範囲内です ({total_lag_dependency:.2f}% <= {threshold:.2f}%)")
        print("  モデルは適切に物理特徴量や未来の説明変数を活用しています。")


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


# この関数は src/diagnostics/lag_analysis.py に移動されました
# 重複を避けるため、ここでは削除されています


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
