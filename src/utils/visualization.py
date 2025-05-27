#!/usr/bin/env python
# coding: utf-8

"""
可視化統合モジュール
各種可視化機能の統合インターフェースを提供
"""

# 基本的な可視化機能をインポート
from .basic_plots import (
    plot_feature_importance
)

# 高度な可視化機能をインポート
from .advanced_visualization import (
    plot_corrected_time_series_by_horizon,
    plot_ultra_detailed_minute_analysis
)

# 診断機能をインポート
from ..diagnostics import (
    analyze_lag_dependency,
    detect_lag_following_pattern,
    validate_prediction_timing,
    create_correct_prediction_timestamps,
    analyze_feature_patterns,
    calculate_comprehensive_metrics
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any, List, Optional


def create_detailed_analysis_for_zone(results_dict: Dict, zone: int, horizon: int,
                                    save_dir: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    """
    特定ゾーンの詳細分析を実行（簡素化版）

    Parameters:
    -----------
    results_dict : dict
        結果辞書
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        保存ディレクトリ
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    dict
        詳細分析結果
    """
    analysis_results = {
        'zone': zone,
        'horizon': horizon,
        'analysis_completed': False,
        'error_message': None
    }

    try:
        # データの取得
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        if not horizon_results:
            analysis_results['error_message'] = f"ゾーン {zone}, ホライゾン {horizon} のデータが見つかりません"
            return analysis_results

        # 必要なデータの確認
        required_keys = ['test_data', 'test_y', 'test_predictions', 'feature_importance']
        missing_keys = [key for key in required_keys if key not in horizon_results]

        if missing_keys:
            analysis_results['error_message'] = f"必要なデータが不足しています: {missing_keys}"
            return analysis_results

        # データの取得
        test_df = horizon_results['test_data']
        test_y = horizon_results['test_y']
        test_predictions = horizon_results['test_predictions']
        feature_importance = horizon_results['feature_importance']

        # LAG依存度分析
        zone_system = zone_results.get('system', 'Unknown')
        lag_dependency = analyze_lag_dependency(feature_importance, zone, horizon, zone_system)
        analysis_results['lag_dependency'] = lag_dependency

        # 特徴量パターン分析
        feature_patterns = analyze_feature_patterns(feature_importance, zone, horizon)
        analysis_results['feature_patterns'] = feature_patterns

        # 性能指標計算
        performance_metrics = calculate_comprehensive_metrics(test_y.values, test_predictions, zone, horizon)
        analysis_results['performance_metrics'] = performance_metrics

        # 時間軸検証
        if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
            timestamps = test_df.index
            time_validation = validate_prediction_timing(timestamps, test_y.values, test_predictions, horizon, zone)
            analysis_results['time_validation'] = time_validation

            # 後追いパターン検出
            lag_following = detect_lag_following_pattern(timestamps, test_y.values, test_predictions, horizon)
            analysis_results['lag_following'] = lag_following

        # 可視化の生成（簡素化：特徴量重要度のみ）
        if save_dir:
            # 特徴量重要度プロット
            plot_feature_importance(feature_importance, zone, horizon, save_dir, save=save)

        analysis_results['analysis_completed'] = True

    except Exception as e:
        analysis_results['error_message'] = f"分析中にエラーが発生しました: {str(e)}"

    return analysis_results


def create_comprehensive_analysis_report(results_dict: Dict, horizons: List[int],
                                       save_dir: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
    """
    包括的な分析レポートを作成（簡素化版）

    Parameters:
    -----------
    results_dict : dict
        結果辞書
    horizons : list
        分析対象のホライゾンリスト
    save_dir : str, optional
        保存ディレクトリ
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    dict
        包括的分析レポート
    """
    report = {
        'analysis_timestamp': pd.Timestamp.now(),
        'horizons_analyzed': horizons,
        'zones_analyzed': list(results_dict.keys()),
        'summary': {},
        'detailed_results': {},
        'recommendations': []
    }

    try:
        # 各ホライゾンの分析
        for horizon in horizons:
            print(f"\n🔍 {horizon}分予測の分析を開始...")

            # 1. 時間軸修正済み時系列プロット（全ゾーン）
            print(f"📊 {horizon}分予測の時間軸修正済み時系列プロットを生成中...")
            plot_corrected_time_series_by_horizon(results_dict, horizon, save_dir, save=save)

            # 2. 超高解像度分刻み可視化（複数の時間スケール）
            print(f"🔍 {horizon}分予測の超高解像度分刻み分析を開始...")
            ultra_detailed_figures = plot_ultra_detailed_minute_analysis(
                results_dict, horizon, save_dir, save=save
            )

            # ホライゾン別の詳細分析（特徴量重要度のみ）
            horizon_analysis = {}
            for zone in results_dict.keys():
                zone_analysis = create_detailed_analysis_for_zone(results_dict, zone, horizon, save_dir, save)
                if zone_analysis['analysis_completed']:
                    horizon_analysis[zone] = zone_analysis

            report['detailed_results'][horizon] = horizon_analysis

        # 全体サマリーの生成
        from ..diagnostics.performance_metrics import generate_performance_summary
        from ..diagnostics.lag_analysis import calculate_lag_dependency_summary
        from ..diagnostics.time_validation import generate_time_validation_report
        from ..diagnostics.feature_analysis import generate_feature_analysis_report

        # 性能サマリー
        performance_summary = generate_performance_summary(results_dict, horizons)
        report['summary']['performance'] = performance_summary

        # LAG依存度サマリー
        lag_summary = calculate_lag_dependency_summary(results_dict)
        report['summary']['lag_dependency'] = lag_summary

        # 時間軸検証レポート
        time_validation_report = generate_time_validation_report(results_dict, horizons)
        report['summary']['time_validation'] = time_validation_report

        # 特徴量分析レポート
        feature_analysis_report = generate_feature_analysis_report(results_dict, horizons)
        report['summary']['feature_analysis'] = feature_analysis_report

        # 統合推奨事項の生成
        report['recommendations'] = _generate_integrated_recommendations(report['summary'])

        print(f"\n✅ 包括的分析が完了しました")
        print(f"📊 分析対象: {len(horizons)}ホライゾン × {len(results_dict)}ゾーン")
        print(f"💾 結果保存先: {save_dir if save_dir else '保存なし'}")

    except Exception as e:
        report['error'] = f"分析中にエラーが発生しました: {str(e)}"
        print(f"❌ エラー: {str(e)}")

    return report


def _generate_integrated_recommendations(summary: Dict[str, Any]) -> List[str]:
    """
    統合推奨事項を生成（内部関数）
    """
    recommendations = []

    # 性能に基づく推奨事項
    performance = summary.get('performance', {})
    if performance.get('recommendations'):
        recommendations.extend(performance['recommendations'])

    # LAG依存度に基づく推奨事項
    lag_dependency = summary.get('lag_dependency', {})
    if lag_dependency.get('high_lag_models'):
        recommendations.append(
            f"高LAG依存度モデルが{len(lag_dependency['high_lag_models'])}個検出されました。優先的に改善してください。"
        )

    # 時間軸に基づく推奨事項
    time_validation = summary.get('time_validation', {})
    if time_validation.get('recommendations'):
        recommendations.extend(time_validation['recommendations'])

    # 特徴量分析に基づく推奨事項
    feature_analysis = summary.get('feature_analysis', {})
    if feature_analysis.get('recommendations'):
        recommendations.extend(feature_analysis['recommendations'])

    # 重複を除去
    unique_recommendations = list(dict.fromkeys(recommendations))

    return unique_recommendations


def print_analysis_summary(report: Dict[str, Any]) -> None:
    """
    分析サマリーを表示

    Parameters:
    -----------
    report : dict
        分析レポート
    """
    print("\n" + "="*80)
    print("📊 包括的分析サマリー")
    print("="*80)

    # 基本情報
    print(f"🕐 分析実行時刻: {report.get('analysis_timestamp', 'Unknown')}")
    print(f"🎯 分析対象: {len(report.get('horizons_analyzed', []))}ホライゾン × {len(report.get('zones_analyzed', []))}ゾーン")

    # 性能サマリー
    performance = report.get('summary', {}).get('performance', {})
    if performance:
        overall_stats = performance.get('overall_statistics', {})
        r2_stats = overall_stats.get('r2_statistics', {})
        if r2_stats:
            print(f"\n📈 全体性能:")
            print(f"   平均R²: {r2_stats.get('mean', 0):.3f} (±{r2_stats.get('std', 0):.3f})")
            print(f"   R²範囲: {r2_stats.get('min', 0):.3f} - {r2_stats.get('max', 0):.3f}")

    # LAG依存度サマリー
    lag_summary = report.get('summary', {}).get('lag_dependency', {})
    if lag_summary:
        print(f"\n⚠️ LAG依存度分析:")
        print(f"   高依存度モデル: {len(lag_summary.get('high_lag_models', []))}個")
        print(f"   中依存度モデル: {len(lag_summary.get('medium_lag_models', []))}個")
        print(f"   低依存度モデル: {len(lag_summary.get('low_lag_models', []))}個")
        print(f"   平均依存度: {lag_summary.get('average_lag_dependency', 0):.1f}%")

    # 時間軸検証サマリー
    time_validation = report.get('summary', {}).get('time_validation', {})
    if time_validation:
        overall_summary = time_validation.get('overall_summary', {})
        if overall_summary:
            print(f"\n🕐 時間軸検証:")
            print(f"   正確な時間軸: {overall_summary.get('average_correct_ratio', 0)*100:.1f}%")
            print(f"   検証済みモデル: {overall_summary.get('total_models_checked', 0)}個")

    # 推奨事項
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"\n💡 主要推奨事項:")
        for i, rec in enumerate(recommendations[:5], 1):  # 上位5つを表示
            print(f"   {i}. {rec}")

    print("\n" + "="*80)


# 公開API（簡素化）
__all__ = [
    # 基本プロット
    'plot_feature_importance',

    # 高度な可視化
    'plot_corrected_time_series_by_horizon',
    'plot_ultra_detailed_minute_analysis',

    # 統合分析
    'create_detailed_analysis_for_zone',
    'create_comprehensive_analysis_report',
    'print_analysis_summary',

    # 診断機能
    'analyze_lag_dependency',
    'detect_lag_following_pattern',
    'validate_prediction_timing',
    'create_correct_prediction_timestamps',
    'analyze_feature_patterns',
    'calculate_comprehensive_metrics'
]
