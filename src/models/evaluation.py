#!/usr/bin/env python
# coding: utf-8

"""
モデル評価モジュール（簡素化版）
基本的な評価指標の計算と表示機能を提供
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def calculate_metrics(y_true, y_pred):
    """
    回帰モデルの基本評価指標を計算

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
    # NaN値の処理
    valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        print("警告: 有効なデータがありません")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'mape': float('nan'),
            'r2': float('nan')
        }

    return {
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid),
        'mape': mean_absolute_percentage_error(y_true_valid, y_pred_valid) * 100,
        'r2': r2_score(y_true_valid, y_pred_valid)
    }


def print_metrics(metrics, zone=None, horizon=None):
    """評価指標を表示"""
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
    """特徴量重要度を分析"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    sorted_importance = feature_importance.sort_values('importance', ascending=False)
    top_features = sorted_importance.head(top_n)
    
    print(f"\n上位{top_n}個の重要な特徴量:")
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")

    return sorted_importance


def evaluate_temperature_difference_model(y_true_diff, y_pred_diff, current_temps=None):
    """
    差分予測モデルの評価

    Parameters:
    -----------
    y_true_diff : array-like
        実際の温度差分
    y_pred_diff : array-like
        予測された温度差分
    current_temps : array-like, optional
        現在の温度（復元評価用）

    Returns:
    --------
    dict
        差分予測の評価指標
    """
    # 基本的な差分評価指標
    diff_metrics = calculate_metrics(y_true_diff, y_pred_diff)
    
    # 差分特有の指標
    diff_metrics['diff_mae'] = diff_metrics['mae']
    diff_metrics['diff_rmse'] = diff_metrics['rmse']
    diff_metrics['diff_r2'] = diff_metrics['r2']

    # 温度復元評価（現在温度が提供された場合）
    if current_temps is not None:
        y_true_restored = current_temps + y_true_diff
        y_pred_restored = current_temps + y_pred_diff
        
        restored_metrics = calculate_metrics(y_true_restored, y_pred_restored)
        diff_metrics['restored_mae'] = restored_metrics['mae']
        diff_metrics['restored_rmse'] = restored_metrics['rmse']
        diff_metrics['restored_r2'] = restored_metrics['r2']

    return diff_metrics


def print_difference_metrics(metrics, zone=None, horizon=None):
    """差分予測の評価指標を表示"""
    header = "差分予測評価指標"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"差分RMSE: {metrics['diff_rmse']:.4f}℃")
    print(f"差分MAE: {metrics['diff_mae']:.4f}℃")
    print(f"差分R²: {metrics['diff_r2']:.4f}")
    
    if 'restored_rmse' in metrics:
        print(f"復元RMSE: {metrics['restored_rmse']:.4f}℃")
        print(f"復元MAE: {metrics['restored_mae']:.4f}℃")
        print(f"復元R²: {metrics['restored_r2']:.4f}")


def restore_temperature_from_difference(current_temp, predicted_diff):
    """
    差分予測から温度を復元

    Parameters:
    -----------
    current_temp : float or array-like
        現在の温度
    predicted_diff : float or array-like
        予測された温度差分

    Returns:
    --------
    float or array-like
        復元された温度
    """
    return current_temp + predicted_diff


def compare_difference_vs_direct_prediction(direct_metrics, diff_metrics, current_temps, y_true_future):
    """
    直接予測と差分予測の比較

    Parameters:
    -----------
    direct_metrics : dict
        直接予測の評価指標
    diff_metrics : dict
        差分予測の評価指標
    current_temps : array-like
        現在の温度
    y_true_future : array-like
        実際の将来温度

    Returns:
    --------
    dict
        比較結果
    """
    comparison = {
        'direct_rmse': direct_metrics['rmse'],
        'direct_mae': direct_metrics['mae'],
        'direct_r2': direct_metrics['r2'],
        'difference_rmse': diff_metrics.get('restored_rmse', diff_metrics['rmse']),
        'difference_mae': diff_metrics.get('restored_mae', diff_metrics['mae']),
        'difference_r2': diff_metrics.get('restored_r2', diff_metrics['r2'])
    }

    # 改善率の計算
    comparison['rmse_improvement'] = ((comparison['direct_rmse'] - comparison['difference_rmse']) / comparison['direct_rmse']) * 100
    comparison['mae_improvement'] = ((comparison['direct_mae'] - comparison['difference_mae']) / comparison['direct_mae']) * 100
    comparison['r2_improvement'] = ((comparison['difference_r2'] - comparison['direct_r2']) / abs(comparison['direct_r2'])) * 100

    return comparison


def print_prediction_comparison(comparison, zone=None, horizon=None):
    """予測手法の比較結果を表示"""
    header = "予測手法比較"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"直接予測 - RMSE: {comparison['direct_rmse']:.4f}, MAE: {comparison['direct_mae']:.4f}, R²: {comparison['direct_r2']:.4f}")
    print(f"差分予測 - RMSE: {comparison['difference_rmse']:.4f}, MAE: {comparison['difference_mae']:.4f}, R²: {comparison['difference_r2']:.4f}")
    print(f"改善率 - RMSE: {comparison['rmse_improvement']:.2f}%, MAE: {comparison['mae_improvement']:.2f}%, R²: {comparison['r2_improvement']:.2f}%")
