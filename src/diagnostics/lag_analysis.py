#!/usr/bin/env python
# coding: utf-8

"""
LAG依存度分析モジュール
LAG特徴量への依存度分析と後追いパターン検出
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple


def analyze_lag_dependency(feature_importance: pd.DataFrame, zone: int, horizon: int,
                          zone_system: str) -> Dict[str, Any]:
    """
    LAG特徴量への依存度を分析

    Parameters:
    -----------
    feature_importance : pd.DataFrame
        特徴量重要度データフレーム
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    zone_system : str
        ゾーンシステム（L/M/R）

    Returns:
    --------
    dict
        LAG依存度分析結果
    """
    total_importance = feature_importance['importance'].sum()

    if total_importance == 0:
        return {
            'zone': zone,
            'horizon': horizon,
            'system': zone_system,
            'lag_temp_percent': 0.0,
            'rolling_temp_percent': 0.0,
            'future_temp_percent': 0.0,
            'current_temp_percent': 0.0,
            'other_percent': 0.0,
            'total_lag_dependency': 0.0,
            'warning_level': 'none'
        }

    # LAG特徴量の重要度を計算
    lag_temp_importance = feature_importance[
        feature_importance['feature'].str.contains('_lag_', na=False)
    ]['importance'].sum()

    # 移動平均特徴量の重要度を計算
    rolling_temp_importance = feature_importance[
        feature_importance['feature'].str.contains('rolling_', na=False)
    ]['importance'].sum()

    # 未来温度特徴量の重要度を計算
    future_temp_importance = feature_importance[
        feature_importance['feature'].str.contains('_future_', na=False)
    ]['importance'].sum()

    # 現在温度特徴量の重要度を計算
    current_temp_importance = feature_importance[
        (feature_importance['feature'].str.contains('sens_temp', na=False)) &
        (~feature_importance['feature'].str.contains('_lag_', na=False)) &
        (~feature_importance['feature'].str.contains('_future_', na=False)) &
        (~feature_importance['feature'].str.contains('rolling_', na=False))
    ]['importance'].sum()

    # その他の特徴量
    other_importance = total_importance - (lag_temp_importance + rolling_temp_importance +
                                         future_temp_importance + current_temp_importance)

    # パーセンテージ計算
    lag_temp_percent = (lag_temp_importance / total_importance) * 100
    rolling_temp_percent = (rolling_temp_importance / total_importance) * 100
    future_temp_percent = (future_temp_importance / total_importance) * 100
    current_temp_percent = (current_temp_importance / total_importance) * 100
    other_percent = (other_importance / total_importance) * 100

    # 総LAG依存度
    total_lag_dependency = lag_temp_percent + rolling_temp_percent

    # 警告レベルの決定
    if total_lag_dependency > 30:
        warning_level = 'high'
    elif total_lag_dependency > 15:
        warning_level = 'medium'
    else:
        warning_level = 'low'

    return {
        'zone': zone,
        'horizon': horizon,
        'system': zone_system,
        'lag_temp_percent': lag_temp_percent,
        'rolling_temp_percent': rolling_temp_percent,
        'future_temp_percent': future_temp_percent,
        'current_temp_percent': current_temp_percent,
        'other_percent': other_percent,
        'total_lag_dependency': total_lag_dependency,
        'warning_level': warning_level
    }


def detect_lag_following_pattern(timestamps, actual, predicted, horizon: int) -> Dict[str, Any]:
    """
    LAG特徴量による後追いパターンの検出

    Parameters:
    -----------
    timestamps : array-like
        タイムスタンプ
    actual : array-like
        実測値
    predicted : array-like
        予測値
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        検出結果
    """
    detection_results = {
        'is_lag_following': False,
        'lag_correlation': 0.0,
        'optimal_lag_steps': 0,
        'confidence': 'low',
        'recommendations': []
    }

    if len(actual) < 100:
        detection_results['recommendations'].append("データ不足のため分析不可（最低100データポイント必要）")
        return detection_results

    # 正規化
    actual_norm = (actual - np.mean(actual)) / (np.std(actual) + 1e-8)
    predicted_norm = (predicted - np.mean(predicted)) / (np.std(predicted) + 1e-8)

    # 相互相関の計算
    max_lag = min(horizon // 5 + 10, len(actual) // 4)  # 予測ホライゾンに基づく最大遅れ
    correlations = []
    lags = range(0, max_lag + 1)

    for lag in lags:
        try:
            if lag == 0:
                corr = np.corrcoef(actual_norm, predicted_norm)[0, 1]
            else:
                corr = np.corrcoef(actual_norm[:-lag], predicted_norm[lag:])[0, 1]

            if not np.isnan(corr):
                correlations.append((lag, corr))
        except:
            continue

    if correlations:
        # 最大相関とその遅れを特定
        max_corr_lag, max_corr_value = max(correlations, key=lambda x: abs(x[1]))
        detection_results['lag_correlation'] = max_corr_value
        detection_results['optimal_lag_steps'] = max_corr_lag

        # 後追いパターンの判定
        if max_corr_lag > 0 and abs(max_corr_value) > 0.8:
            detection_results['is_lag_following'] = True
            detection_results['confidence'] = 'high' if abs(max_corr_value) > 0.9 else 'medium'

            detection_results['recommendations'].extend([
                f"予測が実測値より{max_corr_lag}ステップ({max_corr_lag*5}分)遅れています",
                "LAG特徴量への依存度を下げてください",
                "未来情報（制御パラメータ、環境データ）の活用を強化してください",
                "物理法則ベースの特徴量を追加してください"
            ])
        elif max_corr_lag == 0 and abs(max_corr_value) > 0.95:
            detection_results['recommendations'].append(
                "予測精度は高いですが、過学習の可能性があります。検証データでの性能を確認してください"
            )

    return detection_results


def print_lag_dependency_warning(lag_dependency: Dict[str, Any], threshold: float = 30.0,
                                zone: int = None, horizon: int = None) -> None:
    """
    LAG依存度警告の表示

    Parameters:
    -----------
    lag_dependency : dict
        LAG依存度分析結果
    threshold : float, optional
        警告閾値（デフォルト: 30.0%）
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    total_lag = lag_dependency.get('total_lag_dependency', 0.0)

    if total_lag > threshold:
        print(f"\n⚠️ 警告: LAG特徴量への依存度が高すぎます！")
        if zone and horizon:
            print(f"   ゾーン {zone}, {horizon}分予測: {total_lag:.1f}% (閾値: {threshold}%)")
        else:
            print(f"   LAG依存度: {total_lag:.1f}% (閾値: {threshold}%)")

        print(f"   直接LAG特徴量: {lag_dependency.get('lag_temp_percent', 0):.1f}%")
        print(f"   移動平均特徴量: {lag_dependency.get('rolling_temp_percent', 0):.1f}%")
        print(f"\n💡 推奨対策:")
        print(f"   1. 未来情報（制御パラメータ）の活用を強化")
        print(f"   2. 物理法則ベースの特徴量を追加")
        print(f"   3. LAG特徴量の重要度を制限")
        print(f"   4. 特徴量選択の閾値を調整")
    elif total_lag > threshold / 2:
        print(f"\n⚠️ 注意: LAG特徴量への依存度が中程度です")
        if zone and horizon:
            print(f"   ゾーン {zone}, {horizon}分予測: {total_lag:.1f}%")
        else:
            print(f"   LAG依存度: {total_lag:.1f}%")
        print(f"   監視を継続してください")
    else:
        print(f"\n✅ LAG依存度は適切な範囲内です ({total_lag:.1f}%)")


def calculate_lag_dependency_summary(results_dict: Dict) -> Dict[str, Any]:
    """
    全ゾーン・ホライゾンのLAG依存度サマリーを計算

    Parameters:
    -----------
    results_dict : dict
        全結果辞書

    Returns:
    --------
    dict
        LAG依存度サマリー
    """
    summary = {
        'total_models': 0,
        'high_lag_models': [],
        'medium_lag_models': [],
        'low_lag_models': [],
        'average_lag_dependency': 0.0,
        'max_lag_dependency': 0.0,
        'min_lag_dependency': 100.0
    }

    all_dependencies = []

    for zone, zone_results in results_dict.items():
        for horizon, horizon_results in zone_results.items():
            if 'lag_dependency' in horizon_results:
                lag_dep = horizon_results['lag_dependency']
                total_lag = lag_dep.get('total_lag_dependency', 0.0)

                summary['total_models'] += 1
                all_dependencies.append(total_lag)

                model_info = {
                    'zone': zone,
                    'horizon': horizon,
                    'lag_dependency': total_lag
                }

                if total_lag > 30:
                    summary['high_lag_models'].append(model_info)
                elif total_lag > 15:
                    summary['medium_lag_models'].append(model_info)
                else:
                    summary['low_lag_models'].append(model_info)

    if all_dependencies:
        summary['average_lag_dependency'] = np.mean(all_dependencies)
        summary['max_lag_dependency'] = np.max(all_dependencies)
        summary['min_lag_dependency'] = np.min(all_dependencies)

    return summary
