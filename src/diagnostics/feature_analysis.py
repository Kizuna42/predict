#!/usr/bin/env python
# coding: utf-8

"""
特徴量分析モジュール
特徴量パターンの詳細分析と診断
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def analyze_feature_patterns(feature_importance: pd.DataFrame, zone: int, horizon: int) -> Dict[str, Any]:
    """
    特徴量パターンの詳細分析

    Parameters:
    -----------
    feature_importance : DataFrame
        特徴量重要度データフレーム
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        分析結果
    """
    analysis_results = {
        'zone': zone,
        'horizon': horizon,
        'lag_features': [],
        'future_features': [],
        'current_features': [],
        'poly_features': [],
        'thermo_features': [],
        'control_features': [],
        'suspicious_patterns': [],
        'feature_distribution': {},
        'recommendations': []
    }

    total_importance = feature_importance['importance'].sum()

    if total_importance == 0:
        analysis_results['suspicious_patterns'].append({
            'type': 'zero_importance',
            'description': '全特徴量の重要度が0です',
            'severity': 'critical'
        })
        return analysis_results

    # 特徴量を分類
    for _, row in feature_importance.iterrows():
        feature_name = row['feature']
        importance = row['importance']
        importance_percent = (importance / total_importance) * 100

        feature_info = {
            'name': feature_name,
            'importance': importance,
            'importance_percent': importance_percent
        }

        if '_lag_' in feature_name:
            analysis_results['lag_features'].append(feature_info)
        elif '_future_' in feature_name:
            analysis_results['future_features'].append(feature_info)
        elif 'poly_' in feature_name:
            analysis_results['poly_features'].append(feature_info)
        elif any(term in feature_name.lower() for term in ['thermo', 'temp_diff', 'atmospheric']):
            analysis_results['thermo_features'].append(feature_info)
        elif any(term in feature_name.lower() for term in ['ac_', 'control', 'mode', 'valid']):
            analysis_results['control_features'].append(feature_info)
        else:
            analysis_results['current_features'].append(feature_info)

    # 特徴量分布の計算
    analysis_results['feature_distribution'] = {
        'lag_features': len(analysis_results['lag_features']),
        'future_features': len(analysis_results['future_features']),
        'current_features': len(analysis_results['current_features']),
        'poly_features': len(analysis_results['poly_features']),
        'thermo_features': len(analysis_results['thermo_features']),
        'control_features': len(analysis_results['control_features'])
    }

    # 疑わしいパターンの検出
    _detect_suspicious_patterns(analysis_results, total_importance)

    # 推奨事項の生成
    _generate_feature_recommendations(analysis_results)

    return analysis_results


def _detect_suspicious_patterns(analysis_results: Dict[str, Any], total_importance: float) -> None:
    """
    疑わしいパターンの検出（内部関数）
    """
    # LAG特徴量への過度な依存
    lag_importance = sum([f['importance'] for f in analysis_results['lag_features']])
    lag_percentage = (lag_importance / total_importance * 100) if total_importance > 0 else 0

    if lag_percentage > 30:
        analysis_results['suspicious_patterns'].append({
            'type': 'high_lag_dependency',
            'description': f'LAG特徴量への依存度が高すぎます ({lag_percentage:.1f}%)',
            'severity': 'high',
            'value': lag_percentage
        })
    elif lag_percentage > 15:
        analysis_results['suspicious_patterns'].append({
            'type': 'medium_lag_dependency',
            'description': f'LAG特徴量への依存度が中程度です ({lag_percentage:.1f}%)',
            'severity': 'medium',
            'value': lag_percentage
        })

    # 単一特徴量への過度な依存
    if analysis_results['lag_features'] or analysis_results['future_features'] or analysis_results['current_features']:
        all_features = (analysis_results['lag_features'] +
                       analysis_results['future_features'] +
                       analysis_results['current_features'] +
                       analysis_results['poly_features'] +
                       analysis_results['thermo_features'] +
                       analysis_results['control_features'])

        if all_features:
            max_feature = max(all_features, key=lambda x: x['importance'])
            max_percentage = max_feature['importance_percent']

            if max_percentage > 80:
                analysis_results['suspicious_patterns'].append({
                    'type': 'single_feature_dominance',
                    'description': f'単一特徴量 "{max_feature["name"]}" への依存度が極端に高いです ({max_percentage:.1f}%)',
                    'severity': 'high',
                    'feature': max_feature['name'],
                    'value': max_percentage
                })
            elif max_percentage > 60:
                analysis_results['suspicious_patterns'].append({
                    'type': 'high_single_feature_dependency',
                    'description': f'特徴量 "{max_feature["name"]}" への依存度が高いです ({max_percentage:.1f}%)',
                    'severity': 'medium',
                    'feature': max_feature['name'],
                    'value': max_percentage
                })

    # 未来情報の不足
    future_importance = sum([f['importance'] for f in analysis_results['future_features']])
    control_importance = sum([f['importance'] for f in analysis_results['control_features']])
    future_control_importance = future_importance + control_importance
    future_percentage = (future_control_importance / total_importance * 100) if total_importance > 0 else 0

    if future_percentage < 30:
        analysis_results['suspicious_patterns'].append({
            'type': 'insufficient_future_info',
            'description': f'未来情報の活用が不足しています ({future_percentage:.1f}%)',
            'severity': 'medium',
            'value': future_percentage
        })

    # 特徴量の多様性不足
    feature_types = sum([1 for count in analysis_results['feature_distribution'].values() if count > 0])
    if feature_types < 3:
        analysis_results['suspicious_patterns'].append({
            'type': 'low_feature_diversity',
            'description': f'特徴量の多様性が不足しています（{feature_types}種類のみ）',
            'severity': 'medium',
            'value': feature_types
        })


def _generate_feature_recommendations(analysis_results: Dict[str, Any]) -> None:
    """
    推奨事項の生成（内部関数）
    """
    recommendations = []

    # LAG依存度に基づく推奨事項
    lag_patterns = [p for p in analysis_results['suspicious_patterns']
                   if p['type'] in ['high_lag_dependency', 'medium_lag_dependency']]
    if lag_patterns:
        recommendations.extend([
            "LAG特徴量への依存度を下げてください",
            "未来制御情報の特徴量を強化してください",
            "物理法則ベースの特徴量を追加してください"
        ])

    # 単一特徴量依存に基づく推奨事項
    dominance_patterns = [p for p in analysis_results['suspicious_patterns']
                         if 'dominance' in p['type'] or 'single_feature' in p['type']]
    if dominance_patterns:
        recommendations.extend([
            "特徴量の多様性を向上させてください",
            "特徴量選択の閾値を調整してください",
            "正則化パラメータを調整してください"
        ])

    # 未来情報不足に基づく推奨事項
    future_patterns = [p for p in analysis_results['suspicious_patterns']
                      if p['type'] == 'insufficient_future_info']
    if future_patterns:
        recommendations.extend([
            "未来制御パラメータの特徴量を追加してください",
            "予定された制御スケジュールを活用してください",
            "環境予測データを組み込んでください"
        ])

    # 多様性不足に基づく推奨事項
    diversity_patterns = [p for p in analysis_results['suspicious_patterns']
                         if p['type'] == 'low_feature_diversity']
    if diversity_patterns:
        recommendations.extend([
            "異なる種類の特徴量を追加してください",
            "熱力学的特徴量を強化してください",
            "時間特徴量を追加してください"
        ])

    analysis_results['recommendations'] = recommendations


def compare_feature_patterns_across_zones(results_dict: Dict, horizon: int) -> Dict[str, Any]:
    """
    ゾーン間での特徴量パターンの比較分析

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        ゾーン間比較結果
    """
    comparison_results = {
        'horizon': horizon,
        'zone_analyses': {},
        'common_patterns': [],
        'unique_patterns': {},
        'recommendations': []
    }

    # 各ゾーンの分析
    for zone, zone_results in results_dict.items():
        if horizon not in zone_results or 'feature_importance' not in zone_results[horizon]:
            continue

        feature_importance = zone_results[horizon]['feature_importance']
        analysis = analyze_feature_patterns(feature_importance, zone, horizon)
        comparison_results['zone_analyses'][zone] = analysis

    # 共通パターンの特定
    if len(comparison_results['zone_analyses']) > 1:
        _identify_common_patterns(comparison_results)
        _identify_unique_patterns(comparison_results)
        _generate_cross_zone_recommendations(comparison_results)

    return comparison_results


def _identify_common_patterns(comparison_results: Dict[str, Any]) -> None:
    """
    共通パターンの特定（内部関数）
    """
    all_patterns = {}

    for zone, analysis in comparison_results['zone_analyses'].items():
        for pattern in analysis['suspicious_patterns']:
            pattern_type = pattern['type']
            if pattern_type not in all_patterns:
                all_patterns[pattern_type] = []
            all_patterns[pattern_type].append({
                'zone': zone,
                'severity': pattern['severity'],
                'value': pattern.get('value', 0)
            })

    # 複数ゾーンで共通するパターンを特定
    for pattern_type, occurrences in all_patterns.items():
        if len(occurrences) > 1:
            comparison_results['common_patterns'].append({
                'type': pattern_type,
                'affected_zones': [occ['zone'] for occ in occurrences],
                'severity': max([occ['severity'] for occ in occurrences],
                              key=lambda x: {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}[x]),
                'average_value': np.mean([occ['value'] for occ in occurrences if occ['value'] > 0])
            })


def _identify_unique_patterns(comparison_results: Dict[str, Any]) -> None:
    """
    ユニークパターンの特定（内部関数）
    """
    all_pattern_types = set()
    zone_patterns = {}

    for zone, analysis in comparison_results['zone_analyses'].items():
        zone_pattern_types = set([p['type'] for p in analysis['suspicious_patterns']])
        zone_patterns[zone] = zone_pattern_types
        all_pattern_types.update(zone_pattern_types)

    # 各ゾーンのユニークパターンを特定
    for zone, patterns in zone_patterns.items():
        unique_patterns = []
        for pattern_type in patterns:
            # 他のゾーンにはないパターンかチェック
            is_unique = True
            for other_zone, other_patterns in zone_patterns.items():
                if other_zone != zone and pattern_type in other_patterns:
                    is_unique = False
                    break

            if is_unique:
                # 詳細情報を取得
                pattern_detail = next(p for p in comparison_results['zone_analyses'][zone]['suspicious_patterns']
                                    if p['type'] == pattern_type)
                unique_patterns.append(pattern_detail)

        if unique_patterns:
            comparison_results['unique_patterns'][zone] = unique_patterns


def _generate_cross_zone_recommendations(comparison_results: Dict[str, Any]) -> None:
    """
    ゾーン間比較に基づく推奨事項の生成（内部関数）
    """
    recommendations = []

    # 共通パターンに基づく推奨事項
    if comparison_results['common_patterns']:
        high_severity_common = [p for p in comparison_results['common_patterns']
                               if p['severity'] in ['high', 'critical']]
        if high_severity_common:
            recommendations.append(
                f"複数ゾーンで共通する重大な問題が検出されました。システム全体の見直しが必要です。"
            )

        lag_common = [p for p in comparison_results['common_patterns']
                     if 'lag' in p['type']]
        if lag_common:
            recommendations.append(
                "全ゾーンでLAG依存度の問題があります。特徴量エンジニアリング戦略を見直してください。"
            )

    # ユニークパターンに基づく推奨事項
    if comparison_results['unique_patterns']:
        recommendations.append(
            "ゾーン固有の問題が検出されました。個別の対策が必要です。"
        )

    comparison_results['recommendations'] = recommendations


def generate_feature_analysis_report(results_dict: Dict, horizons: List[int]) -> Dict[str, Any]:
    """
    特徴量分析の包括的レポートを生成

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizons : list
        分析対象のホライゾンリスト

    Returns:
    --------
    dict
        特徴量分析レポート
    """
    report = {
        'total_horizons': len(horizons),
        'horizon_comparisons': {},
        'overall_patterns': {
            'most_common_issues': [],
            'severity_distribution': {},
            'feature_type_usage': {}
        },
        'recommendations': []
    }

    all_patterns = []
    all_feature_distributions = []

    for horizon in horizons:
        comparison = compare_feature_patterns_across_zones(results_dict, horizon)
        report['horizon_comparisons'][horizon] = comparison

        # 全パターンを収集
        for zone_analysis in comparison['zone_analyses'].values():
            all_patterns.extend(zone_analysis['suspicious_patterns'])
            all_feature_distributions.append(zone_analysis['feature_distribution'])

    # 全体的なパターン分析
    if all_patterns:
        _analyze_overall_patterns(report, all_patterns)

    if all_feature_distributions:
        _analyze_feature_type_usage(report, all_feature_distributions)

    # 全体的な推奨事項
    _generate_overall_recommendations(report)

    return report


def _analyze_overall_patterns(report: Dict[str, Any], all_patterns: List[Dict]) -> None:
    """
    全体的なパターン分析（内部関数）
    """
    # 最も一般的な問題の特定
    pattern_counts = {}
    severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}

    for pattern in all_patterns:
        pattern_type = pattern['type']
        severity = pattern['severity']

        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        severity_counts[severity] += 1

    # 上位の問題を特定
    sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
    report['overall_patterns']['most_common_issues'] = [
        {'type': pattern_type, 'count': count}
        for pattern_type, count in sorted_patterns[:5]
    ]

    report['overall_patterns']['severity_distribution'] = severity_counts


def _analyze_feature_type_usage(report: Dict[str, Any], all_distributions: List[Dict]) -> None:
    """
    特徴量タイプ使用状況の分析（内部関数）
    """
    feature_type_totals = {}

    for distribution in all_distributions:
        for feature_type, count in distribution.items():
            if feature_type not in feature_type_totals:
                feature_type_totals[feature_type] = []
            feature_type_totals[feature_type].append(count)

    # 平均使用数を計算
    feature_type_averages = {
        feature_type: np.mean(counts)
        for feature_type, counts in feature_type_totals.items()
    }

    report['overall_patterns']['feature_type_usage'] = feature_type_averages


def _generate_overall_recommendations(report: Dict[str, Any]) -> None:
    """
    全体的な推奨事項の生成（内部関数）
    """
    recommendations = []

    # 最も一般的な問題に基づく推奨事項
    if report['overall_patterns']['most_common_issues']:
        top_issue = report['overall_patterns']['most_common_issues'][0]
        if 'lag' in top_issue['type']:
            recommendations.append(
                "LAG依存度の問題が最も一般的です。特徴量エンジニアリング戦略の全面的な見直しが必要です。"
            )

    # 重大度分布に基づく推奨事項
    severity_dist = report['overall_patterns']['severity_distribution']
    if severity_dist.get('high', 0) + severity_dist.get('critical', 0) > 0:
        recommendations.append(
            "重大な問題が検出されています。優先的に対処してください。"
        )

    # 特徴量タイプ使用状況に基づく推奨事項
    feature_usage = report['overall_patterns']['feature_type_usage']
    if feature_usage.get('future_features', 0) < feature_usage.get('lag_features', 0):
        recommendations.append(
            "未来特徴量よりもLAG特徴量の使用が多いです。バランスを見直してください。"
        )

    report['recommendations'] = recommendations
