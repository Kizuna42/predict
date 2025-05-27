#!/usr/bin/env python
# coding: utf-8

"""
時間軸整合性検証モジュール
予測値と実測値の時間軸対応関係を詳細に検証
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')


def verify_time_axis_alignment(df: pd.DataFrame,
                              zone: int,
                              horizon: int,
                              test_predictions: np.ndarray,
                              test_y: pd.Series,
                              test_data: pd.DataFrame,
                              save_dir: str = None) -> Dict[str, Any]:
    """
    時間軸整合性の詳細検証

    Parameters:
    -----------
    df : pd.DataFrame
        元のデータフレーム（目的変数作成前）
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    test_predictions : np.ndarray
        予測値
    test_y : pd.Series
        テストデータの目的変数（シフト済み）
    test_data : pd.DataFrame
        テストデータ
    save_dir : str, optional
        結果保存ディレクトリ

    Returns:
    --------
    dict
        検証結果
    """

    verification_results = {
        'zone': zone,
        'horizon': horizon,
        'data_structure_analysis': {},
        'time_axis_mapping': {},
        'alignment_verification': {},
        'visualization_correctness': {},
        'recommendations': []
    }

    print(f"\n{'='*60}")
    print(f"ゾーン {zone} - {horizon}分予測の時間軸整合性検証")
    print(f"{'='*60}")

    # 1. データ構造の分析
    verification_results['data_structure_analysis'] = _analyze_data_structure(
        df, zone, horizon, test_y, test_data
    )

    # 2. 時間軸マッピングの分析
    verification_results['time_axis_mapping'] = _analyze_time_axis_mapping(
        test_data, test_y, horizon
    )

    # 3. 整合性の検証
    verification_results['alignment_verification'] = _verify_alignment(
        test_data, test_y, test_predictions, horizon
    )

    # 4. 可視化の正確性検証
    verification_results['visualization_correctness'] = _verify_visualization_correctness(
        test_data, test_y, test_predictions, zone, horizon, save_dir
    )

    # 5. 推奨事項の生成
    verification_results['recommendations'] = _generate_time_axis_recommendations(
        verification_results
    )

    return verification_results


def _analyze_data_structure(df: pd.DataFrame,
                           zone: int,
                           horizon: int,
                           test_y: pd.Series,
                           test_data: pd.DataFrame) -> Dict[str, Any]:
    """
    データ構造の詳細分析
    """
    print("\n🔍 1. データ構造の分析")

    analysis = {
        'original_temp_column': f'sens_temp_{zone}',
        'target_column': f'sens_temp_{zone}_future_{horizon}',
        'original_data_available': False,
        'target_data_available': False,
        'test_y_source': 'unknown',
        'time_index_info': {}
    }

    # 元の温度列の確認
    original_col = f'sens_temp_{zone}'
    if original_col in df.columns:
        analysis['original_data_available'] = True
        print(f"✅ 元の温度データ列が利用可能: {original_col}")
    else:
        print(f"❌ 元の温度データ列が見つかりません: {original_col}")

    # 目的変数列の確認
    target_col = f'sens_temp_{zone}_future_{horizon}'
    if target_col in test_data.columns:
        analysis['target_data_available'] = True
        print(f"✅ 目的変数列が利用可能: {target_col}")

        # test_yの出所を確認
        if isinstance(test_y, pd.Series) and test_y.name == target_col:
            analysis['test_y_source'] = 'target_column'
            print(f"✅ test_yは目的変数列 {target_col} から取得")
        else:
            analysis['test_y_source'] = 'other'
            print(f"⚠️ test_yの出所が不明: {test_y.name if hasattr(test_y, 'name') else 'unnamed'}")
    else:
        print(f"❌ 目的変数列が見つかりません: {target_col}")

    # 時間インデックス情報
    analysis['time_index_info'] = {
        'test_data_start': test_data.index.min(),
        'test_data_end': test_data.index.max(),
        'test_data_length': len(test_data),
        'test_y_length': len(test_y),
        'time_interval': _estimate_time_interval(test_data.index)
    }

    print(f"📅 テストデータ期間: {analysis['time_index_info']['test_data_start']} ～ {analysis['time_index_info']['test_data_end']}")
    print(f"📊 データ長: test_data={analysis['time_index_info']['test_data_length']}, test_y={analysis['time_index_info']['test_y_length']}")
    print(f"⏱️ 推定時間間隔: {analysis['time_index_info']['time_interval']}")

    return analysis


def _analyze_time_axis_mapping(test_data: pd.DataFrame,
                              test_y: pd.Series,
                              horizon: int) -> Dict[str, Any]:
    """
    時間軸マッピングの詳細分析
    """
    print("\n🕐 2. 時間軸マッピングの分析")

    mapping_analysis = {
        'input_timestamps': [],
        'target_timestamps': [],
        'expected_prediction_timestamps': [],
        'shift_verification': {},
        'mapping_examples': []
    }

    # 有効なデータのインデックスを取得
    valid_indices = test_y.dropna().index

    if len(valid_indices) > 0:
        # サンプルとして最初の5つのタイムスタンプを分析
        sample_indices = valid_indices[:5]

        print(f"📋 サンプル分析（最初の5つのタイムスタンプ）:")

        for i, timestamp in enumerate(sample_indices):
            # 入力タイムスタンプ
            input_time = timestamp

            # 期待される予測タイムスタンプ（入力時刻 + horizon）
            expected_pred_time = timestamp + pd.Timedelta(minutes=horizon)

            # 実際の目的変数の値（これは既にシフト済み）
            target_value = test_y.loc[timestamp]

            mapping_example = {
                'input_timestamp': input_time,
                'expected_prediction_timestamp': expected_pred_time,
                'target_value': target_value,
                'explanation': f"入力時刻 {input_time} → 予測対象時刻 {expected_pred_time}"
            }

            mapping_analysis['mapping_examples'].append(mapping_example)

            print(f"  {i+1}. 入力: {input_time} → 予測対象: {expected_pred_time} (値: {target_value:.2f})")

        # シフト検証
        mapping_analysis['shift_verification'] = _verify_shift_correctness(
            test_data, test_y, horizon
        )

    return mapping_analysis


def _verify_shift_correctness(test_data: pd.DataFrame,
                             test_y: pd.Series,
                             horizon: int) -> Dict[str, Any]:
    """
    シフトの正確性を検証
    """
    print("\n🔄 シフト検証:")

    shift_verification = {
        'is_correct_shift': False,
        'detected_shift_minutes': 0,
        'verification_method': 'correlation_analysis',
        'details': {}
    }

    # 元の温度データを探す
    temp_cols = [col for col in test_data.columns if 'sens_temp' in col and 'future' not in col]

    if temp_cols:
        original_col = temp_cols[0]  # 最初の温度列を使用
        original_temp = test_data[original_col].dropna()

        # 共通のタイムスタンプを取得
        common_timestamps = original_temp.index.intersection(test_y.index)

        if len(common_timestamps) > 50:
            # 相関分析による検証
            correlations = {}

            for shift_min in range(0, horizon + 20, 5):  # 0分から horizon+20分まで5分刻み
                try:
                    # 元データをshift_min分後にシフト
                    shifted_original = original_temp.shift(-shift_min // 5)  # 5分間隔と仮定

                    # 共通インデックスで相関を計算
                    common_idx = shifted_original.index.intersection(test_y.index)
                    if len(common_idx) > 10:
                        corr = np.corrcoef(
                            shifted_original.loc[common_idx].values,
                            test_y.loc[common_idx].values
                        )[0, 1]

                        if not np.isnan(corr):
                            correlations[shift_min] = corr

                except Exception as e:
                    continue

            if correlations:
                # 最高相関のシフト量を特定
                best_shift = max(correlations.keys(), key=lambda k: abs(correlations[k]))
                best_corr = correlations[best_shift]

                shift_verification['detected_shift_minutes'] = best_shift
                shift_verification['is_correct_shift'] = abs(best_shift - horizon) <= 5  # 5分の誤差許容
                shift_verification['details'] = {
                    'best_correlation': best_corr,
                    'expected_shift': horizon,
                    'detected_shift': best_shift,
                    'all_correlations': correlations
                }

                print(f"  期待シフト: {horizon}分")
                print(f"  検出シフト: {best_shift}分")
                print(f"  最高相関: {best_corr:.3f}")
                print(f"  シフト正確性: {'✅ 正確' if shift_verification['is_correct_shift'] else '❌ 不正確'}")

    return shift_verification


def _verify_alignment(test_data: pd.DataFrame,
                     test_y: pd.Series,
                     test_predictions: np.ndarray,
                     horizon: int) -> Dict[str, Any]:
    """
    予測値と実測値の整合性検証
    """
    print("\n🎯 3. 予測値と実測値の整合性検証")

    alignment_verification = {
        'data_length_match': False,
        'timestamp_alignment': False,
        'value_range_consistency': False,
        'details': {}
    }

    # データ長の確認
    pred_length = len(test_predictions)
    target_length = len(test_y)

    alignment_verification['data_length_match'] = pred_length == target_length
    print(f"📏 データ長: 予測値={pred_length}, 実測値={target_length} {'✅' if alignment_verification['data_length_match'] else '❌'}")

    # タイムスタンプの整合性
    if hasattr(test_y, 'index'):
        alignment_verification['timestamp_alignment'] = True
        print(f"📅 タイムスタンプ整合性: ✅ test_yにインデックスあり")
    else:
        print(f"📅 タイムスタンプ整合性: ❌ test_yにインデックスなし")

    # 値の範囲の一貫性
    if len(test_predictions) > 0 and len(test_y) > 0:
        pred_range = (np.min(test_predictions), np.max(test_predictions))
        target_range = (np.min(test_y), np.max(test_y))

        range_diff = abs((pred_range[1] - pred_range[0]) - (target_range[1] - target_range[0]))
        alignment_verification['value_range_consistency'] = range_diff < 10  # 10度以内の差

        alignment_verification['details'] = {
            'prediction_range': pred_range,
            'target_range': target_range,
            'range_difference': range_diff
        }

        print(f"📊 値の範囲: 予測値={pred_range[0]:.1f}～{pred_range[1]:.1f}, 実測値={target_range[0]:.1f}～{target_range[1]:.1f}")
        print(f"📊 範囲一貫性: {'✅' if alignment_verification['value_range_consistency'] else '❌'}")

    return alignment_verification


def _verify_visualization_correctness(test_data: pd.DataFrame,
                                    test_y: pd.Series,
                                    test_predictions: np.ndarray,
                                    zone: int,
                                    horizon: int,
                                    save_dir: str = None) -> Dict[str, Any]:
    """
    可視化の正確性検証とデモンストレーション
    """
    print("\n📈 4. 可視化の正確性検証")

    visualization_verification = {
        'correct_plotting_method': 'demonstrated',
        'common_mistakes': [],
        'demonstration_created': False
    }

    try:
        # 可視化デモンストレーションの作成
        fig = _create_time_axis_demonstration(
            test_data, test_y, test_predictions, zone, horizon, save_dir
        )

        if fig is not None:
            visualization_verification['demonstration_created'] = True
            print("✅ 時間軸デモンストレーション作成完了")

    except Exception as e:
        print(f"❌ 可視化デモンストレーション作成エラー: {e}")

    # 一般的な間違いの特定
    common_mistakes = [
        "予測値を入力と同じタイムスタンプでプロット",
        "目的変数（シフト済み）を元の時間軸でプロット",
        "予測値の時間軸を調整せずにそのままプロット"
    ]

    visualization_verification['common_mistakes'] = common_mistakes

    print("⚠️ 一般的な可視化の間違い:")
    for mistake in common_mistakes:
        print(f"  - {mistake}")

    return visualization_verification


def _create_time_axis_demonstration(test_data: pd.DataFrame,
                                  test_y: pd.Series,
                                  test_predictions: np.ndarray,
                                  zone: int,
                                  horizon: int,
                                  save_dir: str = None) -> plt.Figure:
    """
    時間軸の正しい表示方法のデモンストレーション
    """
    from ..utils.font_config import get_font_properties

    # フォント設定
    font_prop = get_font_properties()

    # データの準備
    valid_indices = test_y.dropna().index
    if len(valid_indices) == 0:
        return None

    # サンプルデータの選択（最新100ポイント）
    sample_size = min(100, len(valid_indices))
    sample_indices = valid_indices[-sample_size:]

    # データの抽出
    input_timestamps = sample_indices
    actual_values = test_y.loc[sample_indices].values
    predicted_values = test_predictions[-sample_size:] if len(test_predictions) >= sample_size else test_predictions

    # 正しい予測タイムスタンプの計算
    correct_prediction_timestamps = input_timestamps + pd.Timedelta(minutes=horizon)

    # プロット作成
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. 間違った表示方法
    axes[0].plot(input_timestamps, actual_values, 'b-', linewidth=2, label='実測値', alpha=0.8)
    axes[0].plot(input_timestamps, predicted_values, 'r--', linewidth=2, label='予測値（間違った時間軸）', alpha=0.8)
    axes[0].set_title(f'❌ 間違った表示方法: 予測値が入力と同じ時刻に表示',
                     fontproperties=font_prop, fontsize=14, color='red', fontweight='bold')
    axes[0].set_ylabel('温度 (°C)', fontproperties=font_prop)
    axes[0].legend(prop=font_prop)
    axes[0].grid(True, alpha=0.3)

    # 2. 正しい表示方法
    axes[1].plot(input_timestamps, actual_values, 'b-', linewidth=2, label='実測値（入力時刻）', alpha=0.8)
    axes[1].plot(correct_prediction_timestamps, predicted_values, 'r--', linewidth=2,
                label=f'予測値（正しい時間軸: +{horizon}分）', alpha=0.8)
    axes[1].set_title(f'✅ 正しい表示方法: 予測値が未来の時刻（+{horizon}分）に表示',
                     fontproperties=font_prop, fontsize=14, color='green', fontweight='bold')
    axes[1].set_ylabel('温度 (°C)', fontproperties=font_prop)
    axes[1].legend(prop=font_prop)
    axes[1].grid(True, alpha=0.3)

    # 3. 比較用：実測値の未来値との比較
    future_actual_timestamps = input_timestamps + pd.Timedelta(minutes=horizon)

    # 未来の実測値を取得（利用可能な場合）
    future_actual_values = []
    for ts in future_actual_timestamps:
        if ts in test_y.index:
            future_actual_values.append(test_y.loc[ts])
        else:
            future_actual_values.append(np.nan)

    future_actual_values = np.array(future_actual_values)
    valid_future = ~np.isnan(future_actual_values)

    if np.sum(valid_future) > 0:
        axes[2].plot(future_actual_timestamps[valid_future], future_actual_values[valid_future],
                    'g-', linewidth=2, label=f'実測値（+{horizon}分後）', alpha=0.8)
        axes[2].plot(correct_prediction_timestamps[valid_future], predicted_values[valid_future],
                    'r--', linewidth=2, label=f'予測値（+{horizon}分後）', alpha=0.8)
        axes[2].set_title(f'📊 比較検証: 予測値 vs 実際の{horizon}分後の実測値',
                         fontproperties=font_prop, fontsize=14, color='blue', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, f'{horizon}分後の実測値データが不足',
                    transform=axes[2].transAxes, ha='center', va='center',
                    fontproperties=font_prop, fontsize=12)
        axes[2].set_title(f'📊 比較検証: データ不足のため表示不可',
                         fontproperties=font_prop, fontsize=14, color='orange')

    axes[2].set_xlabel('日時', fontproperties=font_prop)
    axes[2].set_ylabel('温度 (°C)', fontproperties=font_prop)
    axes[2].legend(prop=font_prop)
    axes[2].grid(True, alpha=0.3)

    # X軸の書式設定
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'time_axis_verification_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📁 時間軸検証プロット保存: {save_path}")

    return fig


def _generate_time_axis_recommendations(verification_results: Dict[str, Any]) -> List[str]:
    """
    時間軸に関する推奨事項の生成
    """
    recommendations = []

    # データ構造の問題
    data_analysis = verification_results['data_structure_analysis']
    if not data_analysis['original_data_available']:
        recommendations.append("元の温度データが利用できません。データ構造を確認してください。")

    if not data_analysis['target_data_available']:
        recommendations.append("目的変数データが利用できません。特徴量エンジニアリングを確認してください。")

    # シフトの問題
    mapping_analysis = verification_results['time_axis_mapping']
    if 'shift_verification' in mapping_analysis:
        shift_verification = mapping_analysis['shift_verification']
        if not shift_verification['is_correct_shift']:
            recommendations.append(
                f"シフトが不正確です。期待値: {verification_results['horizon']}分, "
                f"検出値: {shift_verification['detected_shift_minutes']}分"
            )

    # 整合性の問題
    alignment = verification_results['alignment_verification']
    if not alignment['data_length_match']:
        recommendations.append("予測値と実測値のデータ長が一致しません。")

    if not alignment['timestamp_alignment']:
        recommendations.append("タイムスタンプの整合性に問題があります。")

    # 可視化の推奨事項
    recommendations.extend([
        "✅ 正しい可視化方法: 予測値は「入力時刻 + 予測ホライゾン」でプロットしてください",
        "✅ 実測値は目的変数（シフト済み）ではなく、対応する未来時刻の実測値と比較してください",
        "✅ 時間軸のラベルを明確にして、どの時刻の値かを明示してください"
    ])

    return recommendations


def _estimate_time_interval(time_index: pd.DatetimeIndex) -> str:
    """
    時間間隔の推定
    """
    if len(time_index) < 2:
        return "不明"

    intervals = time_index[1:] - time_index[:-1]
    most_common_interval = intervals.value_counts().index[0]

    return str(most_common_interval)


def run_comprehensive_time_axis_verification(results_dict: Dict,
                                           original_df: pd.DataFrame,
                                           save_dir: str = None) -> Dict[str, Any]:
    """
    全ゾーン・ホライゾンの時間軸検証を実行

    Parameters:
    -----------
    results_dict : dict
        モデル結果辞書
    original_df : pd.DataFrame
        元のデータフレーム
    save_dir : str, optional
        結果保存ディレクトリ

    Returns:
    --------
    dict
        検証結果サマリー
    """
    print("\n" + "="*80)
    print("🕐 包括的時間軸整合性検証")
    print("="*80)

    verification_summary = {
        'total_verifications': 0,
        'correct_alignments': 0,
        'incorrect_alignments': 0,
        'zone_horizon_results': {},
        'common_issues': [],
        'overall_recommendations': []
    }

    for zone, zone_results in results_dict.items():
        for horizon, horizon_results in zone_results.items():
            # 必要なデータの確認
            required_keys = ['test_data', 'test_y', 'test_predictions']
            if not all(k in horizon_results for k in required_keys):
                continue

            verification_summary['total_verifications'] += 1

            # 個別検証の実行
            verification_result = verify_time_axis_alignment(
                df=original_df,
                zone=zone,
                horizon=horizon,
                test_predictions=horizon_results['test_predictions'],
                test_y=horizon_results['test_y'],
                test_data=horizon_results['test_data'],
                save_dir=save_dir
            )

            verification_summary['zone_horizon_results'][f'zone_{zone}_horizon_{horizon}'] = verification_result

            # 整合性の判定
            alignment_ok = (
                verification_result['alignment_verification']['data_length_match'] and
                verification_result['alignment_verification']['timestamp_alignment']
            )

            if alignment_ok:
                verification_summary['correct_alignments'] += 1
            else:
                verification_summary['incorrect_alignments'] += 1

    # 共通問題の特定
    verification_summary['common_issues'] = _identify_common_time_axis_issues(
        verification_summary['zone_horizon_results']
    )

    # 全体的な推奨事項
    verification_summary['overall_recommendations'] = _generate_overall_time_axis_recommendations(
        verification_summary
    )

    # サマリーの表示
    print(f"\n📊 検証サマリー:")
    print(f"  総検証数: {verification_summary['total_verifications']}")
    print(f"  正しい整合性: {verification_summary['correct_alignments']}")
    print(f"  問題のある整合性: {verification_summary['incorrect_alignments']}")

    if verification_summary['overall_recommendations']:
        print(f"\n💡 全体的な推奨事項:")
        for rec in verification_summary['overall_recommendations']:
            print(f"  - {rec}")

    return verification_summary


def _identify_common_time_axis_issues(zone_horizon_results: Dict) -> List[str]:
    """
    共通の時間軸問題を特定
    """
    issues = []
    total_results = len(zone_horizon_results)

    if total_results == 0:
        return issues

    # 各問題の出現頻度を計算
    shift_issues = 0
    alignment_issues = 0
    data_length_issues = 0

    for result in zone_horizon_results.values():
        if 'shift_verification' in result['time_axis_mapping']:
            if not result['time_axis_mapping']['shift_verification']['is_correct_shift']:
                shift_issues += 1

        if not result['alignment_verification']['data_length_match']:
            data_length_issues += 1

        if not result['alignment_verification']['timestamp_alignment']:
            alignment_issues += 1

    # 50%以上で発生している問題を共通問題とする
    if shift_issues / total_results >= 0.5:
        issues.append(f"シフト処理の問題が{shift_issues}/{total_results}のケースで検出されました")

    if data_length_issues / total_results >= 0.5:
        issues.append(f"データ長の不整合が{data_length_issues}/{total_results}のケースで検出されました")

    if alignment_issues / total_results >= 0.5:
        issues.append(f"タイムスタンプ整合性の問題が{alignment_issues}/{total_results}のケースで検出されました")

    return issues


def _generate_overall_time_axis_recommendations(verification_summary: Dict) -> List[str]:
    """
    全体的な時間軸推奨事項の生成
    """
    recommendations = []

    total = verification_summary['total_verifications']
    incorrect = verification_summary['incorrect_alignments']

    if total == 0:
        recommendations.append("検証対象データがありません。")
        return recommendations

    if incorrect / total > 0.5:
        recommendations.append(
            "⚠️ 50%以上のケースで時間軸の問題が検出されました。システム全体の見直しが必要です。"
        )

    if verification_summary['common_issues']:
        recommendations.append(
            "🔧 共通問題が検出されました。以下の点を重点的に修正してください："
        )
        recommendations.extend([f"  - {issue}" for issue in verification_summary['common_issues']])

    # 基本的な推奨事項
    recommendations.extend([
        "📈 可視化時は必ず予測値を「入力時刻 + 予測ホライゾン」でプロットしてください",
        "🔍 実測値との比較は同じ時刻の値同士で行ってください",
        "📝 プロットにタイムスタンプの説明を明記してください"
    ])

    return recommendations
