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

    # 上段: 従来の表示方法（問題のある表示）
    ax1.plot(timestamps_sample, actual_sample, 'b-', linewidth=2, label='実測値', alpha=0.8)
    ax1.plot(timestamps_sample, predicted_sample, 'r--', linewidth=2, label='予測値（間違った時間軸）', alpha=0.8)
    ax1.set_title(f'ゾーン {zone} - 従来の表示方法（問題あり）: 予測値が入力と同じ時刻に表示',
                 fontproperties=font_prop, fontsize=14, color='red', fontweight='bold')
    ax1.set_ylabel('温度 (°C)', fontproperties=font_prop, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(prop=font_prop)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    # 下段: 修正された表示方法（正しい表示）
    ax2.plot(timestamps_sample, actual_sample, 'b-', linewidth=2, label='実測値', alpha=0.8)
    ax2.plot(prediction_timestamps_sample, predicted_sample, 'r--', linewidth=2,
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

        # プロット
        try:
            axs[i].plot(timestamps_sample, actual_sample, 'b-', linewidth=2,
                       label='実測値', alpha=0.8)
            axs[i].plot(prediction_timestamps_sample, predicted_sample, 'r--', linewidth=2,
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


def plot_enhanced_detailed_time_series_by_horizon(results_dict, horizon, save_dir=None,
                                                 time_scale='day', data_period_days=7,
                                                 show_lag_analysis=True, save=True):
    """
    改善された全ゾーンの詳細時系列プロット

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    time_scale : str, optional
        時間軸のスケール ('hour', 'day', 'week')
    data_period_days : int, optional
        表示するデータ期間（日数）
    show_lag_analysis : bool, optional
        LAG依存度分析を表示するか
    save : bool, optional
        グラフを保存するか

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
        print(f"警告: {horizon}分予測のデータが利用できません。スキップします。")
        return None

    # サブプロットのレイアウト計算（改善版・横軸拡大）
    n_zones = len(zones_with_data)
    if n_zones == 1:
        n_cols, n_rows = 1, 1
        fig_size = (24, 10)  # 横軸拡大
    elif n_zones <= 2:
        n_cols, n_rows = 2, 1
        fig_size = (28, 10)  # 横軸拡大
    elif n_zones <= 4:
        n_cols, n_rows = 2, 2
        fig_size = (28, 16)  # 横軸拡大
    else:
        n_cols = 3
        n_rows = math.ceil(n_zones / n_cols)
        fig_size = (32, n_rows * 8)  # 横軸拡大

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
        lag_dependency = horizon_results.get('lag_dependency', {})

        # データ取得の試行
        if all(k in horizon_results for k in ['test_data', 'test_y', 'test_predictions']):
            test_df = horizon_results['test_data']
            if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
                timestamps = test_df.index
                actual = horizon_results['test_y']
                predicted = horizon_results['test_predictions']

        if timestamps is None or actual is None or predicted is None:
            print(f"ゾーン {zone} のデータが見つかりません")
            axs[i].text(0.5, 0.5, 'データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop,
                       fontsize=14, fontweight='bold')
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
            axs[i].text(0.5, 0.5, 'データ処理エラー', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop,
                       fontsize=14, fontweight='bold')
            continue

        if len(actual_valid) == 0:
            axs[i].text(0.5, 0.5, '有効データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop,
                       fontsize=14, fontweight='bold')
            continue

        # データ期間の制限
        if len(timestamps_valid) > 0:
            end_time = timestamps_valid[-1]
            start_time = end_time - pd.Timedelta(days=data_period_days)

            period_mask = (timestamps_valid >= start_time) & (timestamps_valid <= end_time)
            timestamps_period = timestamps_valid[period_mask]
            actual_period = actual_valid[period_mask]
            predicted_period = predicted_valid[period_mask]

            if len(timestamps_period) == 0:
                timestamps_period = timestamps_valid
                actual_period = actual_valid
                predicted_period = predicted_valid

        # 改善された時系列プロット
        try:
            # 実測値（太い青線）
            axs[i].plot(timestamps_period, actual_period, 'b-', linewidth=2.5,
                       label='実測値', alpha=0.9, zorder=3)
            # 予測値（赤い破線）
            axs[i].plot(timestamps_period, predicted_period, 'r--', linewidth=2.0,
                       label='予測値', alpha=0.8, zorder=2)

            # 誤差の帯グラフ
            axs[i].fill_between(timestamps_period, actual_period, predicted_period,
                               alpha=0.3, color='gray', label='予測誤差', zorder=1)

            # X軸の時間フォーマット設定（横軸拡大に対応してより細かく）
            if time_scale == 'minute':
                axs[i].xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # 15分間隔
                axs[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))  # 5分間隔
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            elif time_scale == 'hour':
                axs[i].xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1時間間隔
                axs[i].xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # 30分間隔
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
            elif time_scale == 'day':
                axs[i].xaxis.set_major_locator(mdates.HourLocator(interval=4))  # 4時間間隔
                axs[i].xaxis.set_minor_locator(mdates.HourLocator(interval=2))  # 2時間間隔
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
            elif time_scale == 'week':
                axs[i].xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 12時間間隔
                axs[i].xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # 6時間間隔
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

            axs[i].tick_params(axis='x', rotation=45, labelsize=9)
        except Exception as e:
            print(f"ゾーン {zone} のプロットエラー: {e}")
            axs[i].text(0.5, 0.5, 'プロットエラー', ha='center', va='center',
                       transform=axs[i].transAxes, fontproperties=font_prop,
                       fontsize=14, fontweight='bold')
            continue

        # タイトルとグリッド（改善版）
        title = f'ゾーン {zone}'
        if show_lag_analysis and lag_dependency:
            total_lag = lag_dependency.get('lag_temp_percent', 0) + lag_dependency.get('rolling_temp_percent', 0)
            if total_lag > 30:
                title += f' LAG依存度高: {total_lag:.1f}%'
                title_color = 'red'
            elif total_lag > 15:
                title += f' LAG依存度中: {total_lag:.1f}%'
                title_color = 'orange'
            else:
                title += f' LAG依存度低: {total_lag:.1f}%'
                title_color = 'green'
        else:
            title_color = 'black'

        axs[i].set_title(title, fontproperties=font_prop, fontsize=12,
                        fontweight='bold', color=title_color)
        axs[i].set_ylabel('温度 (°C)', fontproperties=font_prop, fontsize=11, fontweight='bold')
        axs[i].grid(True, linestyle='--', alpha=0.7)
        legend = axs[i].legend(loc='upper right', fontsize=10, framealpha=0.9)
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)

        # 統計情報をテキストボックスで表示
        mae = np.mean(np.abs(actual_period - predicted_period))
        r2 = r2_score(actual_period, predicted_period)
        stats_text = f'MAE: {mae:.3f}°C\nR²: {r2:.3f}'
        axs[i].text(0.02, 0.98, stats_text, transform=axs[i].transAxes,
                   fontproperties=font_prop,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=9, fontweight='bold')

    # 未使用のサブプロットを非表示
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # 全体タイトル（改善版）
    main_title = f'{horizon}分後予測の詳細時系列分析\n'
    main_title += f'時間軸: {time_scale} | 表示期間: {data_period_days}日間 | 対象ゾーン: {len(zones_with_data)}個'
    fig.suptitle(main_title, fontproperties=font_prop, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # グラフの保存
    if save and save_dir:
        filename = f'enhanced_detailed_timeseries_all_zones_horizon_{horizon}_{time_scale}_{data_period_days}days.png'
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"改善版詳細時系列プロット（全ゾーン）保存: {output_path}")

    return fig


def plot_lag_dependency_visualization(lag_dependency_data, zone, horizon, save_dir=None, save=True):
    """
    LAG依存度の可視化

    Parameters:
    -----------
    lag_dependency_data : dict
        LAG依存度データ
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        保存ディレクトリ
    save : bool, optional
        保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # LAG依存度の円グラフ
    categories = ['直接LAG\n特徴量', '移動平均\n特徴量', '未来\n特徴量', '現在\n特徴量', 'その他']
    values = [
        lag_dependency_data.get('lag_temp_percent', 0),
        lag_dependency_data.get('rolling_temp_percent', 0),
        lag_dependency_data.get('future_temp_percent', 0),
        lag_dependency_data.get('current_temp_percent', 0),
        lag_dependency_data.get('other_percent', 0)
    ]

    colors = ['red', 'orange', 'green', 'blue', 'gray']
    wedges, texts, autotexts = ax1.pie(values, labels=categories, colors=colors, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontproperties': font_prop})

    ax1.set_title(f'ゾーン {zone} - {horizon}分予測\n特徴量重要度分布',
                 fontproperties=font_prop, fontsize=14, fontweight='bold')

    # LAG依存度の棒グラフ
    lag_categories = ['直接LAG\n特徴量', '移動平均\n特徴量', '総LAG\n依存度']
    lag_values = [
        lag_dependency_data.get('lag_temp_percent', 0),
        lag_dependency_data.get('rolling_temp_percent', 0),
        lag_dependency_data.get('total_lag_dependency', 0)
    ]

    colors_bar = ['red' if val > 30 else 'orange' if val > 15 else 'green' for val in lag_values]
    bars = ax2.bar(lag_categories, lag_values, color=colors_bar, alpha=0.8, edgecolor='black')

    # 値をバーの上に表示
    for bar, val in zip(bars, lag_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # ラベルにフォントを適用
    for label in ax2.get_xticklabels():
        label.set_fontproperties(font_prop)

    ax2.set_ylabel('依存度 (%)', fontproperties=font_prop, fontsize=12)
    ax2.set_title('LAG依存度分析', fontproperties=font_prop, fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(50, max(lag_values) * 1.3))
    ax2.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='警告閾値')
    ax2.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='注意閾値')
    legend_lag = ax2.legend(prop=font_prop, fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'lag_dependency_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"LAG依存度可視化保存: {output_path}")

    return fig
