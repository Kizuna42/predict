#!/usr/bin/env python
# coding: utf-8

"""
パフォーマンス診断モジュール
予測性能の包括的な評価と診断
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_comprehensive_metrics(actual: np.ndarray, predicted: np.ndarray,
                                   zone: int, horizon: int) -> Dict[str, Any]:
    """
    包括的な性能指標を計算

    Parameters:
    -----------
    actual : array-like
        実測値
    predicted : array-like
        予測値
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        包括的な性能指標
    """
    # 有効データのフィルタリング
    valid_mask = ~(pd.isna(actual) | pd.isna(predicted) |
                   np.isinf(actual) | np.isinf(predicted))

    if not np.any(valid_mask):
        return {
            'zone': zone,
            'horizon': horizon,
            'data_points': 0,
            'error': 'No valid data points'
        }

    actual_valid = actual[valid_mask]
    predicted_valid = predicted[valid_mask]

    # 基本統計
    n_points = len(actual_valid)

    # 基本性能指標
    mae = mean_absolute_error(actual_valid, predicted_valid)
    mse = mean_squared_error(actual_valid, predicted_valid)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_valid, predicted_valid)

    # 追加の性能指標
    mape = np.mean(np.abs((actual_valid - predicted_valid) / (actual_valid + 1e-8))) * 100

    # 残差分析
    residuals = actual_valid - predicted_valid
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)

    # 予測精度の分布
    abs_errors = np.abs(residuals)
    error_percentiles = {
        'p50': np.percentile(abs_errors, 50),
        'p75': np.percentile(abs_errors, 75),
        'p90': np.percentile(abs_errors, 90),
        'p95': np.percentile(abs_errors, 95),
        'p99': np.percentile(abs_errors, 99)
    }

    # 温度範囲分析
    temp_range = np.max(actual_valid) - np.min(actual_valid)
    temp_mean = np.mean(actual_valid)
    temp_std = np.std(actual_valid)

    # 相対性能指標
    relative_rmse = rmse / temp_std if temp_std > 0 else np.inf
    relative_mae = mae / temp_std if temp_std > 0 else np.inf

    # 予測品質評価
    quality_score = _calculate_quality_score(r2, mae, temp_std)
    quality_grade = _get_quality_grade(quality_score)

    # 異常値検出
    outlier_threshold = 3 * residual_std
    outliers = np.abs(residuals) > outlier_threshold
    outlier_ratio = np.sum(outliers) / n_points * 100

    return {
        'zone': zone,
        'horizon': horizon,
        'data_points': n_points,

        # 基本性能指標
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,

        # 残差分析
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'residual_bias': abs(residual_mean),

        # 誤差分布
        'error_percentiles': error_percentiles,

        # 温度統計
        'temp_range': temp_range,
        'temp_mean': temp_mean,
        'temp_std': temp_std,

        # 相対性能
        'relative_rmse': relative_rmse,
        'relative_mae': relative_mae,

        # 品質評価
        'quality_score': quality_score,
        'quality_grade': quality_grade,

        # 異常値
        'outlier_ratio': outlier_ratio,
        'outlier_count': np.sum(outliers)
    }


def _calculate_quality_score(r2: float, mae: float, temp_std: float) -> float:
    """
    予測品質スコアを計算（内部関数）

    Parameters:
    -----------
    r2 : float
        決定係数
    mae : float
        平均絶対誤差
    temp_std : float
        温度の標準偏差

    Returns:
    --------
    float
        品質スコア（0-100）
    """
    # R²スコア（0-40点）
    r2_score = max(0, min(40, r2 * 40))

    # MAE相対スコア（0-40点）
    if temp_std > 0:
        relative_mae = mae / temp_std
        mae_score = max(0, min(40, (1 - relative_mae) * 40))
    else:
        mae_score = 0

    # 安定性スコア（0-20点）
    stability_score = 20 if r2 > 0.7 and mae < temp_std * 0.3 else 10

    return r2_score + mae_score + stability_score


def _get_quality_grade(score: float) -> str:
    """
    品質スコアからグレードを決定（内部関数）
    """
    if score >= 90:
        return 'A+'
    elif score >= 80:
        return 'A'
    elif score >= 70:
        return 'B+'
    elif score >= 60:
        return 'B'
    elif score >= 50:
        return 'C+'
    elif score >= 40:
        return 'C'
    elif score >= 30:
        return 'D'
    else:
        return 'F'


def analyze_performance_trends(results_dict: Dict, horizons: List[int]) -> Dict[str, Any]:
    """
    ホライゾン間での性能トレンドを分析

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizons : list
        分析対象のホライゾンリスト

    Returns:
    --------
    dict
        性能トレンド分析結果
    """
    trend_analysis = {
        'horizons': horizons,
        'zone_trends': {},
        'overall_trends': {},
        'performance_degradation': {},
        'recommendations': []
    }

    # 各ゾーンの性能トレンド
    for zone, zone_results in results_dict.items():
        zone_metrics = []

        for horizon in horizons:
            if horizon in zone_results:
                horizon_results = zone_results[horizon]

                # データの取得
                if all(k in horizon_results for k in ['test_y', 'test_predictions']):
                    actual = horizon_results['test_y']
                    predicted = horizon_results['test_predictions']

                    metrics = calculate_comprehensive_metrics(actual, predicted, zone, horizon)
                    zone_metrics.append(metrics)

        if zone_metrics:
            trend_analysis['zone_trends'][zone] = _analyze_zone_trend(zone_metrics)

    # 全体トレンドの分析
    if trend_analysis['zone_trends']:
        trend_analysis['overall_trends'] = _analyze_overall_trends(trend_analysis['zone_trends'])
        trend_analysis['performance_degradation'] = _analyze_performance_degradation(trend_analysis['zone_trends'])
        trend_analysis['recommendations'] = _generate_trend_recommendations(trend_analysis)

    return trend_analysis


def _analyze_zone_trend(zone_metrics: List[Dict]) -> Dict[str, Any]:
    """
    単一ゾーンの性能トレンドを分析（内部関数）
    """
    if len(zone_metrics) < 2:
        return {'insufficient_data': True}

    horizons = [m['horizon'] for m in zone_metrics]
    r2_values = [m['r2'] for m in zone_metrics]
    mae_values = [m['mae'] for m in zone_metrics]
    rmse_values = [m['rmse'] for m in zone_metrics]

    # トレンド計算
    r2_trend = np.polyfit(horizons, r2_values, 1)[0] if len(horizons) > 1 else 0
    mae_trend = np.polyfit(horizons, mae_values, 1)[0] if len(horizons) > 1 else 0
    rmse_trend = np.polyfit(horizons, rmse_values, 1)[0] if len(horizons) > 1 else 0

    # 性能劣化の検出
    r2_degradation = r2_values[0] - r2_values[-1] if len(r2_values) > 1 else 0
    mae_degradation = mae_values[-1] - mae_values[0] if len(mae_values) > 1 else 0

    return {
        'horizons': horizons,
        'r2_values': r2_values,
        'mae_values': mae_values,
        'rmse_values': rmse_values,
        'r2_trend': r2_trend,
        'mae_trend': mae_trend,
        'rmse_trend': rmse_trend,
        'r2_degradation': r2_degradation,
        'mae_degradation': mae_degradation,
        'best_horizon': horizons[np.argmax(r2_values)],
        'worst_horizon': horizons[np.argmin(r2_values)]
    }


def _analyze_overall_trends(zone_trends: Dict) -> Dict[str, Any]:
    """
    全体的なトレンドを分析（内部関数）
    """
    all_r2_trends = []
    all_mae_trends = []
    all_r2_degradations = []
    all_mae_degradations = []

    for zone, trend in zone_trends.items():
        if not trend.get('insufficient_data', False):
            all_r2_trends.append(trend['r2_trend'])
            all_mae_trends.append(trend['mae_trend'])
            all_r2_degradations.append(trend['r2_degradation'])
            all_mae_degradations.append(trend['mae_degradation'])

    if not all_r2_trends:
        return {'insufficient_data': True}

    return {
        'average_r2_trend': np.mean(all_r2_trends),
        'average_mae_trend': np.mean(all_mae_trends),
        'average_r2_degradation': np.mean(all_r2_degradations),
        'average_mae_degradation': np.mean(all_mae_degradations),
        'consistent_degradation': np.mean(all_r2_degradations) > 0.05,
        'severe_degradation_zones': sum(1 for d in all_r2_degradations if d > 0.1)
    }


def _analyze_performance_degradation(zone_trends: Dict) -> Dict[str, Any]:
    """
    性能劣化の詳細分析（内部関数）
    """
    degradation_analysis = {
        'zones_with_degradation': [],
        'severe_degradation_zones': [],
        'stable_zones': [],
        'improving_zones': []
    }

    for zone, trend in zone_trends.items():
        if trend.get('insufficient_data', False):
            continue

        r2_degradation = trend['r2_degradation']
        mae_degradation = trend['mae_degradation']

        if r2_degradation > 0.1 or mae_degradation > 0.5:
            degradation_analysis['severe_degradation_zones'].append({
                'zone': zone,
                'r2_degradation': r2_degradation,
                'mae_degradation': mae_degradation
            })
        elif r2_degradation > 0.05 or mae_degradation > 0.2:
            degradation_analysis['zones_with_degradation'].append({
                'zone': zone,
                'r2_degradation': r2_degradation,
                'mae_degradation': mae_degradation
            })
        elif r2_degradation < -0.05:
            degradation_analysis['improving_zones'].append({
                'zone': zone,
                'r2_improvement': -r2_degradation,
                'mae_improvement': -mae_degradation
            })
        else:
            degradation_analysis['stable_zones'].append(zone)

    return degradation_analysis


def _generate_trend_recommendations(trend_analysis: Dict) -> List[str]:
    """
    トレンド分析に基づく推奨事項を生成（内部関数）
    """
    recommendations = []

    overall_trends = trend_analysis.get('overall_trends', {})
    degradation = trend_analysis.get('performance_degradation', {})

    # 全体的な劣化傾向
    if overall_trends.get('consistent_degradation', False):
        recommendations.append(
            "全体的に予測ホライゾンが長くなるにつれて性能が劣化しています。"
        )
        recommendations.append(
            "長期予測の特徴量エンジニアリングを強化してください。"
        )

    # 重大な劣化
    severe_zones = degradation.get('severe_degradation_zones', [])
    if severe_zones:
        zone_list = [str(z['zone']) for z in severe_zones]
        recommendations.append(
            f"ゾーン {', '.join(zone_list)} で重大な性能劣化が検出されました。"
        )
        recommendations.append(
            "これらのゾーンのモデルを優先的に改善してください。"
        )

    # 改善傾向
    improving_zones = degradation.get('improving_zones', [])
    if improving_zones:
        zone_list = [str(z['zone']) for z in improving_zones]
        recommendations.append(
            f"ゾーン {', '.join(zone_list)} で性能改善が見られます。"
        )
        recommendations.append(
            "これらのゾーンの成功要因を他のゾーンに適用してください。"
        )

    return recommendations


def generate_performance_summary(results_dict: Dict, horizons: List[int]) -> Dict[str, Any]:
    """
    性能の包括的サマリーを生成

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizons : list
        分析対象のホライゾンリスト

    Returns:
    --------
    dict
        性能サマリー
    """
    summary = {
        'total_models': 0,
        'performance_by_horizon': {},
        'performance_by_zone': {},
        'overall_statistics': {},
        'quality_distribution': {},
        'recommendations': []
    }

    all_metrics = []

    # 各モデルの性能を収集
    for zone, zone_results in results_dict.items():
        zone_metrics = []

        for horizon in horizons:
            if horizon in zone_results:
                horizon_results = zone_results[horizon]

                if all(k in horizon_results for k in ['test_y', 'test_predictions']):
                    actual = horizon_results['test_y']
                    predicted = horizon_results['test_predictions']

                    metrics = calculate_comprehensive_metrics(actual, predicted, zone, horizon)
                    if 'error' not in metrics:
                        all_metrics.append(metrics)
                        zone_metrics.append(metrics)
                        summary['total_models'] += 1

        if zone_metrics:
            summary['performance_by_zone'][zone] = _summarize_zone_performance(zone_metrics)

    # ホライゾン別性能
    for horizon in horizons:
        horizon_metrics = [m for m in all_metrics if m['horizon'] == horizon]
        if horizon_metrics:
            summary['performance_by_horizon'][horizon] = _summarize_horizon_performance(horizon_metrics)

    # 全体統計
    if all_metrics:
        summary['overall_statistics'] = _calculate_overall_statistics(all_metrics)
        summary['quality_distribution'] = _analyze_quality_distribution(all_metrics)
        summary['recommendations'] = _generate_performance_recommendations(summary)

    return summary


def _summarize_zone_performance(zone_metrics: List[Dict]) -> Dict[str, Any]:
    """
    ゾーン性能のサマリー（内部関数）
    """
    r2_values = [m['r2'] for m in zone_metrics]
    mae_values = [m['mae'] for m in zone_metrics]
    quality_scores = [m['quality_score'] for m in zone_metrics]

    return {
        'model_count': len(zone_metrics),
        'average_r2': np.mean(r2_values),
        'average_mae': np.mean(mae_values),
        'average_quality_score': np.mean(quality_scores),
        'best_horizon': zone_metrics[np.argmax(r2_values)]['horizon'],
        'worst_horizon': zone_metrics[np.argmin(r2_values)]['horizon'],
        'quality_grade': _get_quality_grade(np.mean(quality_scores))
    }


def _summarize_horizon_performance(horizon_metrics: List[Dict]) -> Dict[str, Any]:
    """
    ホライゾン性能のサマリー（内部関数）
    """
    r2_values = [m['r2'] for m in horizon_metrics]
    mae_values = [m['mae'] for m in horizon_metrics]
    quality_scores = [m['quality_score'] for m in horizon_metrics]

    return {
        'zone_count': len(horizon_metrics),
        'average_r2': np.mean(r2_values),
        'std_r2': np.std(r2_values),
        'average_mae': np.mean(mae_values),
        'std_mae': np.std(mae_values),
        'average_quality_score': np.mean(quality_scores),
        'best_zone': horizon_metrics[np.argmax(r2_values)]['zone'],
        'worst_zone': horizon_metrics[np.argmin(r2_values)]['zone'],
        'quality_grade': _get_quality_grade(np.mean(quality_scores))
    }


def _calculate_overall_statistics(all_metrics: List[Dict]) -> Dict[str, Any]:
    """
    全体統計の計算（内部関数）
    """
    r2_values = [m['r2'] for m in all_metrics]
    mae_values = [m['mae'] for m in all_metrics]
    quality_scores = [m['quality_score'] for m in all_metrics]

    return {
        'total_models': len(all_metrics),
        'r2_statistics': {
            'mean': np.mean(r2_values),
            'std': np.std(r2_values),
            'min': np.min(r2_values),
            'max': np.max(r2_values),
            'median': np.median(r2_values)
        },
        'mae_statistics': {
            'mean': np.mean(mae_values),
            'std': np.std(mae_values),
            'min': np.min(mae_values),
            'max': np.max(mae_values),
            'median': np.median(mae_values)
        },
        'quality_statistics': {
            'mean': np.mean(quality_scores),
            'std': np.std(quality_scores),
            'min': np.min(quality_scores),
            'max': np.max(quality_scores)
        }
    }


def _analyze_quality_distribution(all_metrics: List[Dict]) -> Dict[str, Any]:
    """
    品質分布の分析（内部関数）
    """
    quality_grades = [m['quality_grade'] for m in all_metrics]
    grade_counts = {}

    for grade in ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F']:
        grade_counts[grade] = quality_grades.count(grade)

    total_models = len(all_metrics)
    grade_percentages = {grade: (count / total_models * 100) for grade, count in grade_counts.items()}

    return {
        'grade_counts': grade_counts,
        'grade_percentages': grade_percentages,
        'high_quality_ratio': (grade_counts.get('A+', 0) + grade_counts.get('A', 0)) / total_models * 100,
        'low_quality_ratio': (grade_counts.get('D', 0) + grade_counts.get('F', 0)) / total_models * 100
    }


def _generate_performance_recommendations(summary: Dict) -> List[str]:
    """
    性能サマリーに基づく推奨事項を生成（内部関数）
    """
    recommendations = []

    overall_stats = summary.get('overall_statistics', {})
    quality_dist = summary.get('quality_distribution', {})

    # 全体的な性能評価
    avg_r2 = overall_stats.get('r2_statistics', {}).get('mean', 0)
    if avg_r2 < 0.7:
        recommendations.append(
            f"全体的な予測精度が低いです（平均R²: {avg_r2:.3f}）。モデルの改善が必要です。"
        )
    elif avg_r2 > 0.9:
        recommendations.append(
            f"優秀な予測精度です（平均R²: {avg_r2:.3f}）。現在の手法を維持してください。"
        )

    # 品質分布の評価
    low_quality_ratio = quality_dist.get('low_quality_ratio', 0)
    if low_quality_ratio > 20:
        recommendations.append(
            f"低品質モデルの割合が高いです（{low_quality_ratio:.1f}%）。優先的に改善してください。"
        )

    high_quality_ratio = quality_dist.get('high_quality_ratio', 0)
    if high_quality_ratio > 70:
        recommendations.append(
            f"高品質モデルの割合が高いです（{high_quality_ratio:.1f}%）。良好な状態です。"
        )

    return recommendations
