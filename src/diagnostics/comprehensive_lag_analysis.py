#!/usr/bin/env python
# coding: utf-8

"""
包括的LAG分析モジュール
LAG特徴量による後追い問題の詳細診断と原因特定
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def analyze_lag_following_comprehensive(timestamps: pd.DatetimeIndex,
                                      actual_values: np.ndarray,
                                      predicted_values: np.ndarray,
                                      feature_importance: pd.DataFrame,
                                      zone: int,
                                      horizon: int) -> Dict[str, Any]:
    """
    LAG特徴量による後追い問題の包括的分析

    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        タイムスタンプ
    actual_values : np.ndarray
        実測値
    predicted_values : np.ndarray
        予測値
    feature_importance : pd.DataFrame
        特徴量重要度
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        包括的分析結果
    """

    analysis_results = {
        'zone': zone,
        'horizon': horizon,
        'timestamp_analysis': {},
        'pattern_analysis': {},
        'feature_analysis': {},
        'lag_dependency': {},
        'recommendations': [],
        'severity': 'low'
    }

    # 1. タイムスタンプ分析
    analysis_results['timestamp_analysis'] = _analyze_timestamp_alignment(
        timestamps, actual_values, predicted_values, horizon
    )

    # 2. パターン分析（谷や山の後追い検出）
    analysis_results['pattern_analysis'] = _analyze_pattern_following(
        actual_values, predicted_values, horizon
    )

    # 3. 特徴量分析
    analysis_results['feature_analysis'] = _analyze_feature_dependency(
        feature_importance
    )

    # 4. LAG依存度分析
    analysis_results['lag_dependency'] = _analyze_lag_dependency_detailed(
        feature_importance
    )

    # 5. 総合評価と推奨事項
    analysis_results = _generate_comprehensive_recommendations(analysis_results)

    return analysis_results


def _analyze_timestamp_alignment(timestamps: pd.DatetimeIndex,
                               actual_values: np.ndarray,
                               predicted_values: np.ndarray,
                               horizon: int) -> Dict[str, Any]:
    """
    タイムスタンプの整合性分析
    """
    timestamp_analysis = {
        'is_correct_alignment': True,
        'detected_lag_minutes': 0,
        'correlation_by_lag': {},
        'issues': []
    }

    if len(actual_values) < 50:
        timestamp_analysis['issues'].append("データ不足のため分析不可")
        return timestamp_analysis

    # 相互相関分析
    max_lag_steps = min(horizon // 5 + 10, len(actual_values) // 4)
    correlations = {}

    for lag in range(-max_lag_steps, max_lag_steps + 1):
        try:
            if lag < 0:
                corr = np.corrcoef(actual_values[-lag:], predicted_values[:lag])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(actual_values[:-lag], predicted_values[lag:])[0, 1]
            else:
                corr = np.corrcoef(actual_values, predicted_values)[0, 1]

            if not np.isnan(corr):
                correlations[lag] = corr
        except:
            continue

    timestamp_analysis['correlation_by_lag'] = correlations

    if correlations:
        max_corr_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]))
        max_corr_value = correlations[max_corr_lag]

        if max_corr_lag > 0 and abs(max_corr_value) > 0.8:
            timestamp_analysis['is_correct_alignment'] = False
            timestamp_analysis['detected_lag_minutes'] = max_corr_lag * 5
            timestamp_analysis['issues'].append(
                f"予測値が実測値より{max_corr_lag * 5}分遅れて表示されている可能性"
            )

    return timestamp_analysis


def _analyze_pattern_following(actual_values: np.ndarray,
                             predicted_values: np.ndarray,
                             horizon: int) -> Dict[str, Any]:
    """
    パターン後追いの詳細分析（谷や山の検出）
    """
    pattern_analysis = {
        'valley_following': False,
        'peak_following': False,
        'pattern_lag_minutes': 0,
        'pattern_correlation': 0.0,
        'detected_patterns': []
    }

    if len(actual_values) < 100:
        return pattern_analysis

    # 極値検出（谷と山）
    actual_peaks = _detect_peaks_and_valleys(actual_values)
    predicted_peaks = _detect_peaks_and_valleys(predicted_values)

    # パターンマッチング分析
    pattern_matches = _analyze_pattern_matches(actual_peaks, predicted_peaks, horizon)

    pattern_analysis.update(pattern_matches)

    return pattern_analysis


def _detect_peaks_and_valleys(values: np.ndarray, prominence=0.5) -> Dict[str, List[int]]:
    """
    極値（山と谷）の検出
    """
    from scipy.signal import find_peaks

    # 山の検出
    peaks, _ = find_peaks(values, prominence=prominence)

    # 谷の検出（値を反転して山を検出）
    valleys, _ = find_peaks(-values, prominence=prominence)

    return {
        'peaks': peaks.tolist(),
        'valleys': valleys.tolist()
    }


def _analyze_pattern_matches(actual_patterns: Dict[str, List[int]],
                           predicted_patterns: Dict[str, List[int]],
                           horizon: int) -> Dict[str, Any]:
    """
    パターンマッチングの分析
    """
    matches = {
        'valley_following': False,
        'peak_following': False,
        'pattern_lag_minutes': 0,
        'detected_patterns': []
    }

    # 谷のマッチング分析
    valley_lag = _calculate_pattern_lag(
        actual_patterns['valleys'],
        predicted_patterns['valleys']
    )

    # 山のマッチング分析
    peak_lag = _calculate_pattern_lag(
        actual_patterns['peaks'],
        predicted_patterns['peaks']
    )

    if valley_lag > 0:
        matches['valley_following'] = True
        matches['pattern_lag_minutes'] = valley_lag * 5
        matches['detected_patterns'].append(f"谷パターンが{valley_lag * 5}分遅れて出現")

    if peak_lag > 0:
        matches['peak_following'] = True
        matches['pattern_lag_minutes'] = max(matches['pattern_lag_minutes'], peak_lag * 5)
        matches['detected_patterns'].append(f"山パターンが{peak_lag * 5}分遅れて出現")

    return matches


def _calculate_pattern_lag(actual_indices: List[int],
                         predicted_indices: List[int]) -> int:
    """
    パターンの遅れを計算
    """
    if not actual_indices or not predicted_indices:
        return 0

    min_lag = float('inf')

    for actual_idx in actual_indices:
        for pred_idx in predicted_indices:
            lag = pred_idx - actual_idx
            if 0 < lag < min_lag:
                min_lag = lag

    return min_lag if min_lag != float('inf') else 0


def _analyze_feature_dependency(feature_importance: pd.DataFrame) -> Dict[str, Any]:
    """
    特徴量依存度の詳細分析
    """
    feature_analysis = {
        'total_features': len(feature_importance),
        'lag_features': [],
        'smoothed_features': [],
        'future_features': [],
        'current_temp_features': [],
        'other_features': [],
        'dependency_percentages': {}
    }

    total_importance = feature_importance['importance'].sum()

    if total_importance == 0:
        return feature_analysis

    # 特徴量の分類
    for _, row in feature_importance.iterrows():
        feature_name = row['feature']
        importance = row['importance']

        if '_lag_' in feature_name:
            feature_analysis['lag_features'].append({
                'name': feature_name,
                'importance': importance,
                'percentage': (importance / total_importance) * 100
            })
        elif 'smoothed' in feature_name or 'rolling' in feature_name:
            feature_analysis['smoothed_features'].append({
                'name': feature_name,
                'importance': importance,
                'percentage': (importance / total_importance) * 100
            })
        elif 'future' in feature_name:
            feature_analysis['future_features'].append({
                'name': feature_name,
                'importance': importance,
                'percentage': (importance / total_importance) * 100
            })
        elif 'sens_temp' in feature_name and 'lag' not in feature_name and 'future' not in feature_name:
            feature_analysis['current_temp_features'].append({
                'name': feature_name,
                'importance': importance,
                'percentage': (importance / total_importance) * 100
            })
        else:
            feature_analysis['other_features'].append({
                'name': feature_name,
                'importance': importance,
                'percentage': (importance / total_importance) * 100
            })

    # 依存度パーセンテージの計算
    feature_analysis['dependency_percentages'] = {
        'lag_dependency': sum([f['percentage'] for f in feature_analysis['lag_features']]),
        'smoothed_dependency': sum([f['percentage'] for f in feature_analysis['smoothed_features']]),
        'future_dependency': sum([f['percentage'] for f in feature_analysis['future_features']]),
        'current_temp_dependency': sum([f['percentage'] for f in feature_analysis['current_temp_features']]),
        'other_dependency': sum([f['percentage'] for f in feature_analysis['other_features']])
    }

    return feature_analysis


def _analyze_lag_dependency_detailed(feature_importance: pd.DataFrame) -> Dict[str, Any]:
    """
    LAG依存度の詳細分析
    """
    lag_analysis = {
        'has_explicit_lag_features': False,
        'implicit_lag_sources': [],
        'total_lag_like_dependency': 0.0,
        'risk_level': 'low'
    }

    total_importance = feature_importance['importance'].sum()

    if total_importance == 0:
        return lag_analysis

    # 明示的なLAG特徴量の確認
    explicit_lag_features = feature_importance[
        feature_importance['feature'].str.contains('_lag_', na=False)
    ]

    if len(explicit_lag_features) > 0:
        lag_analysis['has_explicit_lag_features'] = True
        lag_analysis['total_lag_like_dependency'] += (
            explicit_lag_features['importance'].sum() / total_importance * 100
        )

    # 暗黙的なLAG効果を持つ特徴量の確認
    implicit_lag_patterns = [
        ('smoothed', '平滑化特徴量（過去の値の移動平均）'),
        ('rolling', 'ローリング統計量（過去の値の統計）'),
        ('rate', '変化率（前の時点との差分）'),
        ('diff', '差分特徴量（前の時点との差）')
    ]

    for pattern, description in implicit_lag_patterns:
        matching_features = feature_importance[
            feature_importance['feature'].str.contains(pattern, na=False)
        ]

        if len(matching_features) > 0:
            dependency = matching_features['importance'].sum() / total_importance * 100
            lag_analysis['implicit_lag_sources'].append({
                'pattern': pattern,
                'description': description,
                'dependency_percentage': dependency,
                'feature_count': len(matching_features)
            })
            lag_analysis['total_lag_like_dependency'] += dependency

    # リスクレベルの判定
    if lag_analysis['total_lag_like_dependency'] > 50:
        lag_analysis['risk_level'] = 'high'
    elif lag_analysis['total_lag_like_dependency'] > 30:
        lag_analysis['risk_level'] = 'medium'
    else:
        lag_analysis['risk_level'] = 'low'

    return lag_analysis


def _generate_comprehensive_recommendations(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    包括的な推奨事項の生成
    """
    recommendations = []
    severity = 'low'

    # タイムスタンプ問題の確認
    if not analysis_results['timestamp_analysis']['is_correct_alignment']:
        recommendations.append({
            'category': 'visualization',
            'priority': 'high',
            'issue': '時間軸表示の問題',
            'description': '予測値が間違った時間軸で表示されています',
            'action': '予測値の表示時刻を「入力時刻 + 予測ホライゾン」に修正してください'
        })
        severity = 'high'

    # パターン後追いの確認
    pattern_analysis = analysis_results['pattern_analysis']
    if pattern_analysis['valley_following'] or pattern_analysis['peak_following']:
        recommendations.append({
            'category': 'model',
            'priority': 'high',
            'issue': 'パターン後追い',
            'description': f"実測値のパターンが{pattern_analysis['pattern_lag_minutes']}分遅れて予測に現れています",
            'action': 'LAG特徴量への依存度を下げ、未来情報の活用を強化してください'
        })
        severity = 'high'

    # LAG依存度の確認
    lag_dependency = analysis_results['lag_dependency']
    if lag_dependency['risk_level'] == 'high':
        recommendations.append({
            'category': 'features',
            'priority': 'high',
            'issue': '高LAG依存度',
            'description': f"LAG様特徴量への依存度が{lag_dependency['total_lag_like_dependency']:.1f}%と高すぎます",
            'action': '物理法則ベースの特徴量や未来情報の比重を増やしてください'
        })
        severity = 'high'
    elif lag_dependency['risk_level'] == 'medium':
        recommendations.append({
            'category': 'features',
            'priority': 'medium',
            'issue': '中程度LAG依存度',
            'description': f"LAG様特徴量への依存度が{lag_dependency['total_lag_like_dependency']:.1f}%です",
            'action': '特徴量のバランスを見直してください'
        })
        if severity == 'low':
            severity = 'medium'

    # 特徴量バランスの確認
    feature_analysis = analysis_results['feature_analysis']
    future_dependency = feature_analysis['dependency_percentages'].get('future_dependency', 0)
    current_temp_dependency = feature_analysis['dependency_percentages'].get('current_temp_dependency', 0)

    if future_dependency < 20:
        recommendations.append({
            'category': 'features',
            'priority': 'medium',
            'issue': '未来情報の活用不足',
            'description': f"未来特徴量への依存度が{future_dependency:.1f}%と低いです",
            'action': '制御パラメータや環境データの未来情報を強化してください'
        })

    if current_temp_dependency > 40:
        recommendations.append({
            'category': 'features',
            'priority': 'medium',
            'issue': '現在温度への過度な依存',
            'description': f"現在温度特徴量への依存度が{current_temp_dependency:.1f}%と高いです",
            'action': '他の物理的特徴量の重要度を高めてください'
        })

    analysis_results['recommendations'] = recommendations
    analysis_results['severity'] = severity

    return analysis_results


def generate_lag_analysis_report(results_dict: Dict, horizon: int, save_dir: str = None) -> Dict[str, Any]:
    """
    全ゾーンのLAG分析レポート生成
    """
    report = {
        'horizon': horizon,
        'zones_analyzed': 0,
        'high_risk_zones': [],
        'medium_risk_zones': [],
        'low_risk_zones': [],
        'common_issues': [],
        'overall_recommendations': []
    }

    zone_analyses = {}

    for zone, zone_results in results_dict.items():
        if horizon not in zone_results:
            continue

        horizon_results = zone_results[horizon]

        # 必要なデータの確認
        required_keys = ['test_data', 'test_y', 'test_predictions', 'feature_importance']
        if not all(k in horizon_results for k in required_keys):
            continue

        # データの取得
        test_df = horizon_results['test_data']
        if not isinstance(test_df, pd.DataFrame) or not hasattr(test_df, 'index'):
            continue

        timestamps = test_df.index
        actual = horizon_results['test_y'].values
        predicted = horizon_results['test_predictions']
        feature_importance = horizon_results['feature_importance']

        # 包括的分析の実行
        analysis = analyze_lag_following_comprehensive(
            timestamps, actual, predicted, feature_importance, zone, horizon
        )

        zone_analyses[zone] = analysis
        report['zones_analyzed'] += 1

        # リスクレベル別の分類
        if analysis['severity'] == 'high':
            report['high_risk_zones'].append(zone)
        elif analysis['severity'] == 'medium':
            report['medium_risk_zones'].append(zone)
        else:
            report['low_risk_zones'].append(zone)

    # 共通問題の特定
    report['common_issues'] = _identify_common_issues(zone_analyses)

    # 全体的な推奨事項
    report['overall_recommendations'] = _generate_overall_recommendations(zone_analyses)

    # レポートの保存
    if save_dir:
        _save_lag_analysis_report(report, zone_analyses, save_dir, horizon)

    return report


def _identify_common_issues(zone_analyses: Dict) -> List[Dict[str, Any]]:
    """
    共通問題の特定
    """
    common_issues = []

    # 各問題カテゴリの出現頻度を計算
    issue_counts = {}
    total_zones = len(zone_analyses)

    for zone, analysis in zone_analyses.items():
        for rec in analysis['recommendations']:
            category = rec['category']
            issue = rec['issue']
            key = f"{category}_{issue}"

            if key not in issue_counts:
                issue_counts[key] = {
                    'category': category,
                    'issue': issue,
                    'count': 0,
                    'zones': []
                }

            issue_counts[key]['count'] += 1
            issue_counts[key]['zones'].append(zone)

    # 50%以上のゾーンで発生している問題を共通問題とする
    for key, data in issue_counts.items():
        if data['count'] / total_zones >= 0.5:
            common_issues.append({
                'category': data['category'],
                'issue': data['issue'],
                'affected_zones': data['zones'],
                'frequency': data['count'] / total_zones
            })

    return common_issues


def _generate_overall_recommendations(zone_analyses: Dict) -> List[str]:
    """
    全体的な推奨事項の生成
    """
    recommendations = []

    high_risk_count = sum(1 for analysis in zone_analyses.values() if analysis['severity'] == 'high')
    total_zones = len(zone_analyses)

    if high_risk_count / total_zones > 0.3:
        recommendations.append(
            "⚠️ 高リスクゾーンが30%を超えています。特徴量エンジニアリング戦略の全面的な見直しが必要です。"
        )

    # LAG依存度の全体的な傾向
    high_lag_zones = sum(1 for analysis in zone_analyses.values()
                        if analysis['lag_dependency']['risk_level'] == 'high')

    if high_lag_zones > 0:
        recommendations.append(
            f"🔄 {high_lag_zones}個のゾーンで高LAG依存度が検出されました。未来情報の活用を強化してください。"
        )

    # 時間軸問題の確認
    timestamp_issues = sum(1 for analysis in zone_analyses.values()
                          if not analysis['timestamp_analysis']['is_correct_alignment'])

    if timestamp_issues > 0:
        recommendations.append(
            f"📅 {timestamp_issues}個のゾーンで時間軸表示の問題が検出されました。可視化の修正が必要です。"
        )

    return recommendations


def _save_lag_analysis_report(report: Dict, zone_analyses: Dict, save_dir: str, horizon: int):
    """
    LAG分析レポートの保存
    """
    import os
    import json

    os.makedirs(save_dir, exist_ok=True)

    # JSONレポートの保存
    report_path = os.path.join(save_dir, f'lag_analysis_report_horizon_{horizon}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({
            'report': report,
            'zone_analyses': zone_analyses
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"LAG分析レポート保存: {report_path}")
