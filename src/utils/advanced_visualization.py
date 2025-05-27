#!/usr/bin/env python
# coding: utf-8

"""
高度な可視化機能
詳細時系列分析、時間軸修正、診断可視化などの高度な可視化を提供
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import math
import os
from sklearn.metrics import r2_score
from .font_config import get_font_properties
from ..diagnostics.time_validation import create_correct_prediction_timestamps, validate_prediction_timing

# フォント設定
font_prop = get_font_properties()


def plot_corrected_time_series(timestamps, actual, predicted, zone, horizon, save_dir=None,
                              points=100, save=True, validate_timing=True):
    """
    時間軸を修正した時系列プロット

    Parameters:
    -----------
    timestamps : array-like
        入力データのタイムスタンプ
    actual : Series
        実測値（目的変数）
    predicted : array-like
        予測値
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    points : int, optional
        最大表示データ点数
    save : bool, optional
        グラフを保存するか
    validate_timing : bool, optional
        時間軸の検証を行うか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # NaN値をフィルタ
    valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
    timestamps_valid = timestamps[valid_indices]
    actual_valid = actual[valid_indices]
    predicted_valid = predicted[valid_indices]

    if len(timestamps_valid) == 0:
        print(f"ゾーン {zone}: 有効なデータがありません")
        return None

    # 予測値用の正しいタイムスタンプを作成
    prediction_timestamps = create_correct_prediction_timestamps(timestamps_valid, horizon)

    # 時間軸の検証
    if validate_timing:
        validation_results = validate_prediction_timing(
            timestamps_valid, actual_valid, predicted_valid, horizon, zone
        )

        if not validation_results['is_correct_timing']:
            print(f"\n⚠️ ゾーン {zone} の時間軸に問題が検出されました:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
            print("推奨事項:")
            for rec in validation_results['recommendations']:
                print(f"  - {rec}")

    # データのサンプリング
    sample_size = min(len(timestamps_valid), points)
    if len(timestamps_valid) > sample_size:
        indices = np.linspace(0, len(timestamps_valid) - 1, sample_size, dtype=int)
        timestamps_sample = timestamps_valid[indices]
        actual_sample = actual_valid[indices]
        predicted_sample = predicted_valid[indices]
        prediction_timestamps_sample = prediction_timestamps[indices]
    else:
        timestamps_sample = timestamps_valid
        actual_sample = actual_valid
        predicted_sample = predicted_valid
        prediction_timestamps_sample = prediction_timestamps

    # プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # 時系列の長さを一致させるため、重複する時間範囲のみを使用
    actual_start = timestamps_sample.min()
    actual_end = timestamps_sample.max()
    pred_start = prediction_timestamps_sample.min()
    pred_end = prediction_timestamps_sample.max()

    # 重複する時間範囲を計算
    overlap_start = max(actual_start, pred_start)
    overlap_end = min(actual_end, pred_end)

    # 重複範囲内のデータのみを抽出
    actual_mask = (timestamps_sample >= overlap_start) & (timestamps_sample <= overlap_end)
    pred_mask = (prediction_timestamps_sample >= overlap_start) & (prediction_timestamps_sample <= overlap_end)

    timestamps_aligned = timestamps_sample[actual_mask]
    actual_aligned = actual_sample[actual_mask]
    prediction_timestamps_aligned = prediction_timestamps_sample[pred_mask]
    predicted_aligned = predicted_sample[pred_mask]

    # 長さを確認して調整
    min_length = min(len(timestamps_aligned), len(prediction_timestamps_aligned))
    if min_length > 0:
        timestamps_aligned = timestamps_aligned[:min_length]
        actual_aligned = actual_aligned[:min_length]
        prediction_timestamps_aligned = prediction_timestamps_aligned[:min_length]
        predicted_aligned = predicted_aligned[:min_length]

    # 上段: 従来の表示方法（問題のある表示）
    ax1.plot(timestamps_aligned, actual_aligned, 'b-', linewidth=2, label='実測値', alpha=0.8)
    ax1.plot(timestamps_aligned, predicted_aligned, 'r--', linewidth=2, label='予測値（間違った時間軸）', alpha=0.8)
    ax1.set_title(f'ゾーン {zone} - 従来の表示方法（問題あり）: 予測値が入力と同じ時刻に表示',
                 fontproperties=font_prop, fontsize=14, color='red', fontweight='bold')
    ax1.set_ylabel('温度 (°C)', fontproperties=font_prop, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop=font_prop)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    # 下段: 修正された表示方法（正しい表示）
    ax2.plot(timestamps_aligned, actual_aligned, 'b-', linewidth=2, label='実測値', alpha=0.8)
    ax2.plot(prediction_timestamps_aligned, predicted_aligned, 'r--', linewidth=2,
            label=f'予測値（正しい時間軸: +{horizon}分）', alpha=0.8)
    ax2.set_title(f'ゾーン {zone} - 修正された表示方法（正しい）: 予測値が未来の時刻に表示',
                 fontproperties=font_prop, fontsize=14, color='green', fontweight='bold')
    ax2.set_xlabel('日時', fontproperties=font_prop, fontsize=12)
    ax2.set_ylabel('温度 (°C)', fontproperties=font_prop, fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(prop=font_prop)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    # X軸の回転
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'corrected_timeseries_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"修正された時系列プロット保存: {output_path}")

    return fig


def plot_corrected_time_series_by_horizon(results_dict, horizon, save_dir=None,
                                         points=100, save=True, validate_timing=True):
    """
    全ゾーンの時間軸修正済み時系列プロット

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    points : int, optional
        最大表示データ点数
    save : bool, optional
        グラフを保存するか
    validate_timing : bool, optional
        時間軸の検証を行うか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # データが利用可能なゾーンを収集
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分予測のデータが利用できません。")
        return None

    # サブプロットのレイアウト計算
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)
    n_rows = math.ceil(n_zones / n_cols)

    # サブプロット作成
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)
    axs = axs.flatten()

    validation_summary = []

    for i, zone in enumerate(sorted(zones_with_data)):
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        # データの取得
        timestamps = None
        actual = None
        predicted = None

        if all(k in horizon_results for k in ['test_data', 'test_y', 'test_predictions']):
            test_df = horizon_results['test_data']
            if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
                timestamps = test_df.index
                actual = horizon_results['test_y']
                predicted = horizon_results['test_predictions']

        if timestamps is None or actual is None or predicted is None:
            axs[i].text(0.5, 0.5, 'データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop, fontsize=12)
            continue

        # 有効データのフィルタリング
        try:
            valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
            timestamps_valid = timestamps[valid_indices]
            actual_valid = actual[valid_indices]
            predicted_valid = predicted[valid_indices]
        except Exception as e:
            axs[i].text(0.5, 0.5, 'データ処理エラー', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop, fontsize=12)
            continue

        if len(actual_valid) == 0:
            axs[i].text(0.5, 0.5, '有効データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop, fontsize=12)
            continue

        # 予測値用の正しいタイムスタンプを作成
        prediction_timestamps = create_correct_prediction_timestamps(timestamps_valid, horizon)

        # 時間軸の検証
        if validate_timing:
            validation_results = validate_prediction_timing(
                timestamps_valid, actual_valid, predicted_valid, horizon, zone
            )
            validation_summary.append({
                'zone': zone,
                'horizon': horizon,
                'is_correct': validation_results['is_correct_timing'],
                'issues': validation_results['issues']
            })

        # データのサンプリング
        sample_size = min(len(timestamps_valid), points)
        if len(timestamps_valid) > sample_size:
            indices = np.linspace(0, len(timestamps_valid) - 1, sample_size, dtype=int)
            timestamps_sample = timestamps_valid[indices]
            actual_sample = actual_valid[indices]
            predicted_sample = predicted_valid[indices]
            prediction_timestamps_sample = prediction_timestamps[indices]
        else:
            timestamps_sample = timestamps_valid
            actual_sample = actual_valid
            predicted_sample = predicted_valid
            prediction_timestamps_sample = prediction_timestamps

        # 時系列の長さを一致させるため、重複する時間範囲のみを使用
        try:
            # 実測値と予測値の時間範囲の重複部分を計算
            actual_start = timestamps_sample.min()
            actual_end = timestamps_sample.max()
            pred_start = prediction_timestamps_sample.min()
            pred_end = prediction_timestamps_sample.max()

            # 重複する時間範囲を計算
            overlap_start = max(actual_start, pred_start)
            overlap_end = min(actual_end, pred_end)

            # 重複範囲内のデータのみを抽出
            actual_mask = (timestamps_sample >= overlap_start) & (timestamps_sample <= overlap_end)
            pred_mask = (prediction_timestamps_sample >= overlap_start) & (prediction_timestamps_sample <= overlap_end)

            timestamps_aligned = timestamps_sample[actual_mask]
            actual_aligned = actual_sample[actual_mask]
            prediction_timestamps_aligned = prediction_timestamps_sample[pred_mask]
            predicted_aligned = predicted_sample[pred_mask]

            # 長さを確認して調整
            min_length = min(len(timestamps_aligned), len(prediction_timestamps_aligned))
            if min_length > 0:
                timestamps_aligned = timestamps_aligned[:min_length]
                actual_aligned = actual_aligned[:min_length]
                prediction_timestamps_aligned = prediction_timestamps_aligned[:min_length]
                predicted_aligned = predicted_aligned[:min_length]

            # プロット（正しい時間軸で表示、誤差面積なし）
            # 実測値は目的変数の時刻で表示
            axs[i].plot(timestamps_aligned, actual_aligned, 'b-', linewidth=2,
                       label='実測値', alpha=0.8)
            # 予測値は入力時刻 + 予測ホライゾンで表示
            axs[i].plot(prediction_timestamps_aligned, predicted_aligned, 'r--', linewidth=2,
                       label=f'予測値（+{horizon}分）', alpha=0.8)

            # タイトルに検証結果を反映
            title_color = 'green' if validate_timing and validation_results['is_correct_timing'] else 'red'
            status = '✓' if validate_timing and validation_results['is_correct_timing'] else '⚠'
            axs[i].set_title(f'{status} ゾーン {zone}', fontproperties=font_prop,
                           color=title_color, fontweight='bold')

            axs[i].set_ylabel('温度 (°C)', fontproperties=font_prop)
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(prop=font_prop, fontsize=9)
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)

        except Exception as e:
            axs[i].text(0.5, 0.5, f'プロットエラー: {str(e)[:20]}...',
                       ha='center', va='center', transform=axs[i].transAxes,
                       fontproperties=font_prop, fontsize=10)

    # 未使用のサブプロットを非表示
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # 全体タイトル
    fig.suptitle(f'{horizon}分後予測の時間軸修正済み時系列プロット',
                fontproperties=font_prop, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 検証結果の表示
    if validate_timing and validation_summary:
        print(f"\n📊 {horizon}分予測の時間軸検証結果:")
        correct_count = sum(1 for v in validation_summary if v['is_correct'])
        total_count = len(validation_summary)
        print(f"  正しい時間軸: {correct_count}/{total_count} ゾーン")

        for v in validation_summary:
            if not v['is_correct']:
                print(f"  ⚠️ ゾーン {v['zone']}: {', '.join(v['issues'])}")

    # 保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'corrected_timeseries_all_zones_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"修正された時系列プロット（全ゾーン）保存: {output_path}")

    return fig








def plot_ultra_detailed_minute_analysis(results_dict, horizon, save_dir=None, save=True):
    """
    超高解像度分刻み時系列分析（複数の時間スケール）

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    list
        生成されたFigureオブジェクトのリスト
    """
    figures = []

    # 複数の詳細時間スケールを定義
    time_scales = [
        {'name': 'ultra_minute', 'hours': 2, 'description': '2時間詳細（分刻み）'},
        {'name': 'detailed_minute', 'hours': 6, 'description': '6時間詳細（分刻み）'},
        {'name': 'extended_minute', 'hours': 12, 'description': '12時間詳細（分刻み）'},
        {'name': 'daily_minute', 'hours': 24, 'description': '24時間詳細（分刻み）'},
        {'name': 'multi_day_minute', 'hours': 48, 'description': '48時間詳細（分刻み）'}
    ]

    for scale_config in time_scales:
        print(f"📊 {scale_config['description']}の可視化を生成中...")

        fig = _create_minute_scale_visualization(
            results_dict, horizon, scale_config, save_dir, save
        )

        if fig is not None:
            figures.append(fig)

    return figures


def _create_minute_scale_visualization(results_dict, horizon, scale_config, save_dir, save):
    """
    分刻みスケール可視化の内部実装
    """
    # データが利用可能なゾーンを収集
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分予測のデータが利用できません。")
        return None

    # レイアウト計算（分刻み表示用に最適化）
    n_zones = len(zones_with_data)
    if n_zones == 1:
        n_cols, n_rows = 1, 1
        fig_size = (32, 12)  # 超横長
    elif n_zones <= 2:
        n_cols, n_rows = 1, 2  # 縦に配置
        fig_size = (32, 20)
    elif n_zones <= 4:
        n_cols, n_rows = 2, 2
        fig_size = (40, 20)
    else:
        n_cols = 2
        n_rows = math.ceil(n_zones / n_cols)
        fig_size = (40, n_rows * 10)

    # サブプロット作成
    fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(sorted(zones_with_data)):
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        # データの取得
        timestamps = None
        actual = None
        predicted = None

        if all(k in horizon_results for k in ['test_data', 'test_y', 'test_predictions']):
            test_df = horizon_results['test_data']
            if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
                timestamps = test_df.index
                actual = horizon_results['test_y']
                predicted = horizon_results['test_predictions']

        if timestamps is None or actual is None or predicted is None:
            axs[i].text(0.5, 0.5, 'データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop,
                       fontsize=16, fontweight='bold')
            continue

        # 有効データのフィルタリング
        try:
            valid_indices = ~(pd.isna(actual) | pd.isna(predicted) |
                             np.isinf(actual) | np.isinf(predicted))

            timestamps_valid = timestamps[valid_indices]
            actual_valid = actual[valid_indices]
            predicted_valid = predicted[valid_indices]
        except Exception as e:
            print(f"ゾーン {zone} のデータ処理エラー: {e}")
            continue

        if len(actual_valid) == 0:
            continue

        # 指定時間範囲のデータを抽出
        hours = scale_config['hours']
        if len(timestamps_valid) > 0:
            end_time = timestamps_valid[-1]
            start_time = end_time - pd.Timedelta(hours=hours)

            period_mask = (timestamps_valid >= start_time) & (timestamps_valid <= end_time)
            timestamps_period = timestamps_valid[period_mask]
            actual_period = actual_valid[period_mask]
            predicted_period = predicted_valid[period_mask]

            if len(timestamps_period) == 0:
                # データが少ない場合は利用可能な全データを使用
                timestamps_period = timestamps_valid[-min(len(timestamps_valid), hours*60):]
                actual_period = actual_valid[-min(len(actual_valid), hours*60):]
                predicted_period = predicted_valid[-min(len(predicted_valid), hours*60):]

        # 正しい時間軸での予測値タイムスタンプ
        prediction_timestamps = create_correct_prediction_timestamps(timestamps_period, horizon)

        # 時系列の長さを一致させるため、重複する時間範囲のみを使用
        try:
            # 実測値と予測値の時間範囲の重複部分を計算
            actual_start = timestamps_period.min()
            actual_end = timestamps_period.max()
            pred_start = prediction_timestamps.min()
            pred_end = prediction_timestamps.max()

            # 重複する時間範囲を計算
            overlap_start = max(actual_start, pred_start)
            overlap_end = min(actual_end, pred_end)

            # 重複範囲内のデータのみを抽出
            actual_mask = (timestamps_period >= overlap_start) & (timestamps_period <= overlap_end)
            pred_mask = (prediction_timestamps >= overlap_start) & (prediction_timestamps <= overlap_end)

            timestamps_aligned = timestamps_period[actual_mask]
            actual_aligned = actual_period[actual_mask]
            prediction_timestamps_aligned = prediction_timestamps[pred_mask]
            predicted_aligned = predicted_period[pred_mask]

            # 長さを確認して調整
            min_length = min(len(timestamps_aligned), len(prediction_timestamps_aligned))
            if min_length > 0:
                timestamps_aligned = timestamps_aligned[:min_length]
                actual_aligned = actual_aligned[:min_length]
                prediction_timestamps_aligned = prediction_timestamps_aligned[:min_length]
                predicted_aligned = predicted_aligned[:min_length]

            # 超詳細プロット（正しい時間軸で表示、誤差面積なし）
            # 実測値（太い青線、マーカー付き）- 目的変数の時刻で表示
            axs[i].plot(timestamps_aligned, actual_aligned, 'b-', linewidth=3,
                       marker='o', markersize=4, markevery=max(1, len(timestamps_aligned)//50),
                       label='実測値', alpha=0.9, zorder=4)

            # 予測値（正しい時間軸、赤い破線、マーカー付き）- 入力時刻 + 予測ホライゾンで表示
            axs[i].plot(prediction_timestamps_aligned, predicted_aligned, 'r--', linewidth=2.5,
                       marker='s', markersize=3, markevery=max(1, len(prediction_timestamps_aligned)//50),
                       label=f'予測値 (+{horizon}分)', alpha=0.8, zorder=3)

            # 時間軸の詳細設定（分刻み表示）
            if hours <= 2:
                # 2時間以下：5分間隔
                axs[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                axs[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif hours <= 6:
                # 6時間以下：15分間隔
                axs[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
                axs[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            elif hours <= 12:
                # 12時間以下：30分間隔
                axs[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                axs[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            elif hours <= 24:
                # 24時間以下：1時間間隔
                axs[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))
                axs[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            else:
                # 24時間超：2時間間隔
                axs[i].xaxis.set_major_locator(mdates.HourLocator(interval=2))
                axs[i].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

            axs[i].tick_params(axis='x', rotation=45, labelsize=10)
            axs[i].tick_params(axis='both', which='major', labelsize=10)
            axs[i].tick_params(axis='both', which='minor', labelsize=8)

        except Exception as e:
            print(f"ゾーン {zone} のプロットエラー: {e}")
            continue

        # タイトルと詳細統計
        lag_dependency = horizon_results.get('lag_dependency', {})
        total_lag = lag_dependency.get('lag_temp_percent', 0) + lag_dependency.get('rolling_temp_percent', 0)

        # 詳細統計計算（長さ一致データを使用）
        if min_length > 0:
            mae = np.mean(np.abs(actual_aligned - predicted_aligned))
            rmse = np.sqrt(np.mean((actual_aligned - predicted_aligned)**2))
            r2 = r2_score(actual_aligned, predicted_aligned)
            max_error = np.max(np.abs(actual_aligned - predicted_aligned))
        else:
            mae = rmse = r2 = max_error = 0

        title = f'ゾーン {zone} - {scale_config["description"]}'
        if total_lag > 30:
            title += f' [高LAG依存: {total_lag:.1f}%]'
            title_color = 'red'
        elif total_lag > 15:
            title += f' [中LAG依存: {total_lag:.1f}%]'
            title_color = 'orange'
        else:
            title += f' [低LAG依存: {total_lag:.1f}%]'
            title_color = 'green'

        axs[i].set_title(title, fontproperties=font_prop, fontsize=14,
                        fontweight='bold', color=title_color)
        axs[i].set_ylabel('温度 (°C)', fontproperties=font_prop, fontsize=12, fontweight='bold')

        # 詳細グリッド
        axs[i].grid(True, linestyle='-', alpha=0.3, which='major')
        axs[i].grid(True, linestyle=':', alpha=0.2, which='minor')

        # 凡例
        legend = axs[i].legend(loc='upper right', fontsize=11, framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

        # 詳細統計情報ボックス
        stats_text = f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C\nR²: {r2:.3f}\n最大誤差: {max_error:.3f}°C'
        axs[i].text(0.02, 0.98, stats_text, transform=axs[i].transAxes,
                   fontproperties=font_prop,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
                   verticalalignment='top', fontsize=10, fontweight='bold')

        # データ点数情報（長さ一致データを使用）
        data_info = f'データ点数: {min_length}\n時間範囲: {hours}時間'
        axs[i].text(0.98, 0.02, data_info, transform=axs[i].transAxes,
                   fontproperties=font_prop,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                   verticalalignment='bottom', horizontalalignment='right', fontsize=9)

    # 未使用のサブプロットを非表示
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # 全体タイトル
    main_title = f'{horizon}分後予測 - {scale_config["description"]}\n'
    main_title += f'対象ゾーン: {len(zones_with_data)}個 | 時間解像度: 分刻み表示'
    fig.suptitle(main_title, fontproperties=font_prop, fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存
    if save and save_dir:
        output_path = os.path.join(save_dir,
                                  f'ultra_detailed_{scale_config["name"]}_horizon_{horizon}_{scale_config["hours"]}h.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"超詳細分刻み可視化保存: {output_path}")

    return fig



