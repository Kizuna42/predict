#!/usr/bin/env python
# coding: utf-8

"""
基本的な可視化機能
散布図、基本時系列プロットなどの基本的な可視化を提供
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
from .font_config import setup_japanese_font

# フォント設定を実行
setup_japanese_font()

# グラフ設定
sns.set_theme(style="whitegrid", palette="colorblind")

# 追加のmatplotlib設定
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})


def plot_feature_importance(feature_importance, zone, horizon, save_dir=None, top_n=15, save=True):
    """
    特徴量重要度のプロット

    Parameters:
    -----------
    feature_importance : DataFrame
        特徴量と重要度を含むDataFrame
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    top_n : int, optional
        表示する特徴量数
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # 重要度で降順ソート
    importance_sorted = feature_importance.sort_values('importance', ascending=False)

    # 上位N個の特徴量を抽出
    top_features = importance_sorted.head(top_n)

    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax)

    ax.set_title(f'Zone {zone} - Feature Importance for {horizon}-min Prediction (Top {top_n})', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    # グラフ保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'feature_importance_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"特徴量重要度グラフ保存: {output_path}")

    return fig


def plot_scatter_actual_vs_predicted(actual, predicted, zone, horizon, save_dir=None, save=True):
    """
    実測値 vs 予測値の散布図

    Parameters:
    -----------
    actual : Series
        実測値
    predicted : array-like
        予測値
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # 散布図作成
    fig, ax = plt.subplots(figsize=(10, 6))

    # NaN値をフィルタ
    valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
    actual_valid = actual[valid_indices]
    predicted_valid = predicted[valid_indices]

    # 散布図
    ax.scatter(actual_valid, predicted_valid, alpha=0.5)

    # 理想予測線（y=x）を追加
    min_val = min(actual_valid.min(), predicted_valid.min())
    max_val = max(actual_valid.max(), predicted_valid.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    # 軸ラベルとタイトル
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'Zone {zone} - Temperature Prediction for {horizon}-min Horizon (Scatter Plot)', fontsize=14)

    # R²を計算・表示
    r2 = r2_score(actual_valid, predicted_valid)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=12)

    # グリッド線
    ax.grid(True, linestyle='--', alpha=0.6)

    # グラフ保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'scatter_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"散布図保存: {output_path}")

    return fig


def plot_scatter_actual_vs_predicted_by_horizon(results_dict, horizon, save_dir=None, save=True):
    """
    特定ホライゾンの全ゾーン散布図サブプロット

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
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # 予測ホライゾンのデータがあるゾーンを収集
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分予測のデータが利用できません。スキップします。")
        return None

    # サブプロットの行・列数を計算
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)  # 最大3列に制限
    n_rows = math.ceil(n_zones / n_cols)

    # サブプロット作成
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(sorted(zones_with_data)):
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        # データ取得のヘルパー関数
        def get_data_from_results(results, key_pairs):
            """結果辞書からキーペアに基づいてデータを取得するヘルパー関数"""
            for actual_key, pred_key in key_pairs:
                if actual_key in results and pred_key in results:
                    return results[actual_key], results[pred_key]
            return None, None

        # 異なるキー組み合わせを試行
        key_pairs = [
            ('test_y', 'test_predictions'),
            ('y_test', 'y_pred'),
            ('actual', 'predicted'),
            ('train_y', 'train_predictions'),
            ('y_train', 'y_train_pred')
        ]

        actual, predicted = get_data_from_results(horizon_results, key_pairs)

        if actual is None or predicted is None:
            print(f"Zone {zone}, Horizon {horizon}: データが見つかりません。利用可能キー: {list(horizon_results.keys())}")
            axs[i].text(0.5, 0.5, 'データなし',
                      ha='center', va='center', transform=axs[i].transAxes)
            continue

        # NaN値と無限値をフィルタ
        try:
            valid_indices = ~(pd.isna(actual) | pd.isna(predicted) |
                             np.isinf(actual) | np.isinf(predicted))

            if isinstance(actual, pd.Series):
                actual_valid = actual[valid_indices]
            else:
                actual_valid = np.array(actual)[valid_indices]

            if isinstance(predicted, pd.Series):
                predicted_valid = predicted[valid_indices]
            else:
                predicted_valid = np.array(predicted)[valid_indices]
        except Exception as e:
            print(f"Zone {zone} のデータ処理エラー: {e}")
            axs[i].text(0.5, 0.5, f'データ処理エラー: {str(e)[:30]}...',
                      ha='center', va='center', transform=axs[i].transAxes)
            continue

        if len(actual_valid) == 0:
            axs[i].text(0.5, 0.5, '有効データなし',
                      ha='center', va='center', transform=axs[i].transAxes)
            continue

        # 散布図
        axs[i].scatter(actual_valid, predicted_valid, alpha=0.5)

        # 理想予測線（y=x）を追加
        try:
            min_val = min(np.min(actual_valid), np.min(predicted_valid))
            max_val = max(np.max(actual_valid), np.max(predicted_valid))
            axs[i].plot([min_val, max_val], [min_val, max_val], 'r--')

            # R²を計算・表示
            r2 = r2_score(actual_valid, predicted_valid)
            axs[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axs[i].transAxes,
                      verticalalignment='top', fontsize=12)
        except Exception as e:
            print(f"Zone {zone} のR²計算エラー: {e}")
            axs[i].text(0.05, 0.95, "R²計算エラー", transform=axs[i].transAxes,
                      verticalalignment='top', fontsize=12)

        # タイトルとグリッド
        axs[i].set_title(f'Zone {zone}')
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Predicted')
        axs[i].grid(True, linestyle='--', alpha=0.6)

    # 未使用のサブプロットを非表示
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # 全体タイトル
    fig.suptitle(f'Actual vs Predicted for {horizon}-min Horizon', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # グラフ保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'scatter_all_zones_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"全ゾーン散布図保存: {output_path}")

    return fig


def plot_time_series(timestamps, actual, predicted, zone, horizon, save_dir=None, points=100, save=True):
    """
    温度データと予測の時系列プロット

    Parameters:
    -----------
    timestamps : array-like
        時系列のタイムスタンプ
    actual : Series
        実測値
    predicted : array-like
        予測値
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    points : int, optional
        プロットする最大データ点数
    save : bool, optional
        グラフを保存するか

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

    # データ点数が多い場合はサンプリング
    sample_size = min(len(timestamps_valid), points)
    if len(timestamps_valid) > sample_size:
        # 端点からサンプリング
        step = len(timestamps_valid) // sample_size
        indices = list(range(0, len(timestamps_valid), step))[:sample_size]

        # データをサンプリング
        if isinstance(timestamps_valid, pd.DatetimeIndex):
            timestamps_sample = timestamps_valid[indices]
        else:
            timestamps_sample = timestamps_valid.iloc[indices] if hasattr(timestamps_valid, 'iloc') else timestamps_valid[indices]

        actual_sample = actual_valid.iloc[indices] if hasattr(actual_valid, 'iloc') else actual_valid[indices]
        predicted_sample = predicted_valid[indices] if hasattr(predicted_valid, 'iloc') else predicted_valid[indices]
    else:
        timestamps_sample = timestamps_valid
        actual_sample = actual_valid
        predicted_sample = predicted_valid

    # 時系列プロット作成
    fig, ax = plt.subplots(figsize=(12, 6))

    # 実測値と予測値をプロット
    ax.plot(timestamps_sample, actual_sample, 'b-', label='Actual')
    ax.plot(timestamps_sample, predicted_sample, 'r--', label='Predicted')

    # X軸フォーマット
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    plt.xticks(rotation=45)

    # 軸ラベルとタイトル
    ax.set_xlabel('Date/Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Zone {zone} - Temperature Prediction for {horizon}-min Horizon (Time Series)', fontsize=14)

    # グリッドと凡例
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # レイアウト調整
    plt.tight_layout()

    # グラフ保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'timeseries_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"時系列プロット保存: {output_path}")

    return fig


def plot_time_series_by_horizon(results_dict, horizon, save_dir=None, points=100, save=True):
    """
    特定ホライゾンの全ゾーン時系列データのサブプロット

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    points : int, optional
        プロットする最大データ点数
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # 予測ホライゾンのデータがあるゾーンを収集
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分予測のデータが利用できません。スキップします。")
        return None

    # サブプロットの行・列数を計算
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)  # 最大3列に制限
    n_rows = math.ceil(n_zones / n_cols)

    # サブプロット作成
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(sorted(zones_with_data)):
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        # 時系列データを取得する異なる方法を試行
        timestamps = None
        actual = None
        predicted = None

        # 方法1: test_data, test_y, test_predictionsを使用
        if all(k in horizon_results for k in ['test_data', 'test_y', 'test_predictions']):
            test_df = horizon_results['test_data']
            if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
                timestamps = test_df.index
                actual = horizon_results['test_y']
                predicted = horizon_results['test_predictions']

        # 方法2: y_test, y_predを使用し、時系列インデックスを持つDataFrameを探す
        elif all(k in horizon_results for k in ['y_test', 'y_pred']):
            actual = horizon_results['y_test']
            predicted = horizon_results['y_pred']

            # datetimeインデックスを持つDataFrameを探す
            for key, value in horizon_results.items():
                if isinstance(value, pd.DataFrame) and hasattr(value, 'index') and isinstance(value.index, pd.DatetimeIndex):
                    timestamps = value.index
                    break

        # 方法3: タイムスタンプキーを探す
        if timestamps is None:
            for key in ['test_timestamps', 'timestamps', 'time_index', 'date_index']:
                if key in horizon_results:
                    timestamps = horizon_results[key]
                    break

        # タイムスタンプが見つからない場合はダミーを生成
        if timestamps is None:
            if actual is not None:
                length = len(actual)
                timestamps = pd.date_range(start='2023-01-01', periods=length, freq='1min')
                print(f"Zone {zone} のタイムインデックスが見つかりません。ダミーインデックスを生成しました")
            else:
                # データが見つからない
                print(f"Zone {zone}, Horizon {horizon} のデータが見つかりません")
                axs[i].text(0.5, 0.5, 'データなし',
                           ha='center', va='center', transform=axs[i].transAxes)
                continue

        # 実測値/予測値データの不足をチェック
        if actual is None or predicted is None:
            # 異なるキーパターンを試行
            key_pairs = [
                ('test_y', 'test_predictions'),
                ('y_test', 'y_pred'),
                ('actual', 'predicted'),
                ('train_y', 'train_predictions')
            ]

            for actual_key, pred_key in key_pairs:
                if actual_key in horizon_results and pred_key in horizon_results:
                    actual = horizon_results[actual_key]
                    predicted = horizon_results[pred_key]
                    break

        if actual is None or predicted is None:
            print(f"Zone {zone}, Horizon {horizon} のデータキーが見つかりません。利用可能キー: {list(horizon_results.keys())}")
            axs[i].text(0.5, 0.5, 'データなし',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        # 長さの不一致をチェック・修正
        try:
            min_length = min(len(timestamps), len(actual), len(predicted))
            if len(timestamps) > min_length:
                timestamps = timestamps[:min_length]
            if len(actual) > min_length:
                actual = actual[:min_length]
            if len(predicted) > min_length:
                predicted = predicted[:min_length]
        except Exception as e:
            print(f"Zone {zone} のデータ長調整エラー: {e}")

        # NaN値と無効値をフィルタ
        try:
            valid_indices = ~(pd.isna(actual) | pd.isna(predicted) |
                              np.isinf(actual) | np.isinf(predicted))

            timestamps_valid = timestamps[valid_indices]

            if isinstance(actual, pd.Series):
                actual_valid = actual[valid_indices]
            else:
                actual_valid = np.array(actual)[valid_indices]

            if isinstance(predicted, pd.Series):
                predicted_valid = predicted[valid_indices]
            else:
                predicted_valid = np.array(predicted)[valid_indices]
        except Exception as e:
            print(f"Zone {zone} のデータ処理エラー: {e}")
            axs[i].text(0.5, 0.5, f'データ処理エラー: {str(e)[:30]}...',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        if len(actual_valid) == 0 or len(timestamps_valid) == 0:
            axs[i].text(0.5, 0.5, '有効データなし',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        # データ点数が多い場合はサンプリング
        sample_size = min(len(timestamps_valid), points)
        if len(timestamps_valid) > sample_size:
            try:
                # 端点を含むインデックスをサンプリング
                indices = np.linspace(0, len(timestamps_valid) - 1, sample_size, dtype=int)

                # データをサンプリング
                timestamps_sample = timestamps_valid[indices]
                actual_sample = actual_valid[indices]
                predicted_sample = predicted_valid[indices]
            except Exception as e:
                print(f"Zone {zone} のデータサンプリングエラー: {e}")
                # エラーが発生した場合は全データを使用
                timestamps_sample = timestamps_valid
                actual_sample = actual_valid
                predicted_sample = predicted_valid
        else:
            timestamps_sample = timestamps_valid
            actual_sample = actual_valid
            predicted_sample = predicted_valid

        # 実測値と予測値をプロット
        try:
            axs[i].plot(timestamps_sample, actual_sample, 'b-', label='Actual')
            axs[i].plot(timestamps_sample, predicted_sample, 'r--', label='Predicted')

            # X軸フォーマット
            if isinstance(timestamps_sample, pd.DatetimeIndex) or isinstance(timestamps_sample[0], (pd.Timestamp, np.datetime64)):
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
                axs[i].tick_params(axis='x', rotation=45)
        except Exception as e:
            print(f"Zone {zone} のプロットエラー: {e}")
            axs[i].text(0.5, 0.5, f'プロットエラー: {str(e)[:30]}...',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        # タイトルとグリッド
        axs[i].set_title(f'Zone {zone}')
        axs[i].set_ylabel('Temperature (°C)')
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].legend(loc='upper right')

    # 未使用のサブプロットを非表示
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # 全体タイトル
    fig.suptitle(f'Time Series Data for {horizon}-min Horizon', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # グラフ保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'timeseries_all_zones_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"全ゾーン時系列プロット保存: {output_path}")

    return fig
