#!/usr/bin/env python
# coding: utf-8

"""
可視化モジュール
温度予測モデルの結果を可視化するための関数を提供
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import math
import os
from src.config import OUTPUT_DIR


# グラフ設定
sns.set_theme(style="whitegrid", palette="colorblind")
# フォント設定 - matplotlibに日本語フォントがない場合にも対応
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False


def plot_feature_importance(feature_importance, zone, zone_system, horizon, top_n=15, save=True):
    """
    特徴量重要度を棒グラフで可視化する関数

    Parameters:
    -----------
    feature_importance : DataFrame
        特徴量名と重要度を含むデータフレーム
    zone : int
        ゾーン番号
    zone_system : str
        ゾーンの系統 (L, M, R など)
    horizon : int
        予測ホライゾン（分）
    top_n : int, optional
        表示する上位特徴量の数
    save : bool, optional
        グラフを保存するかどうか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットしたFigureオブジェクト
    """
    # 重要度順にソート
    top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'LightGBM Feature Importance (Zone {zone} - {zone_system} System, {horizon}min ahead)')
    plt.tight_layout()

    if save:
        output_path = os.path.join(OUTPUT_DIR, f'feature_importance_zone_{zone}.png')
        plt.savefig(output_path)
        print(f"ゾーン{zone}({zone_system}系統)の特徴量重要度グラフを保存しました: {output_path}")

    return plt.gcf()


def plot_scatter_actual_vs_predicted(results_dict, horizon, save=True):
    """
    実測値と予測値の散布図を作成する関数

    Parameters:
    -----------
    results_dict : dict
        ゾーンごとの結果を含む辞書
    horizon : int
        予測ホライゾン（分）
    save : bool, optional
        グラフを保存するかどうか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットしたFigureオブジェクト
    """
    # 予測ホライゾンに対応するゾーンとデータを収集
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分後予測のデータがありません。スキップします。")
        return None

    # サブプロットの行数と列数を計算
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)
    n_rows = math.ceil(n_zones / n_cols)

    # 散布図（実測値 vs 予測値）
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(zones_with_data):
        results = results_dict[zone][horizon]
        y_test = results['y_test']
        y_pred = results['y_pred']
        r2 = results['r2']
        zone_system = results['system']

        axs[i].scatter(y_test, y_pred, alpha=0.5)
        axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axs[i].set_title(f'Zone {zone} - {zone_system} System (R² = {r2:.4f})')
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Predicted')
        axs[i].grid(True)

    # 使わないサブプロットを非表示
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f'{horizon}min Ahead Temperature Prediction - Actual vs Predicted', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # suptitleのスペースを確保

    if save:
        output_path = os.path.join(OUTPUT_DIR, f'prediction_vs_actual_horizon_{horizon}.png')
        plt.savefig(output_path)
        print(f"{horizon}分後予測の散布図を保存しました: {output_path}")

    return fig


def plot_time_series(results_dict, horizon, points=100, save=True):
    """
    時系列データの実測値と予測値を線グラフで可視化する関数

    Parameters:
    -----------
    results_dict : dict
        ゾーンごとの結果を含む辞書
    horizon : int
        予測ホライゾン（分）
    points : int, optional
        表示するデータポイント数
    save : bool, optional
        グラフを保存するかどうか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットしたFigureオブジェクト
    """
    # 予測ホライゾンに対応するゾーンとデータを収集
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分後予測のデータがありません。スキップします。")
        return None

    # サブプロットの行数と列数を計算
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)
    n_rows = math.ceil(n_zones / n_cols)

    try:
        # sharexをFalseに設定して各プロットが独自のx軸を持つようにする
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), squeeze=False)
        axs = axs.flatten()

        for i, zone in enumerate(zones_with_data):
            try:
                results = results_dict[zone][horizon]
                y_test = results['y_test']
                y_pred = results['y_pred']
                zone_system = results['system']

                # インデックスの確認
                if isinstance(y_test.index, pd.DatetimeIndex):
                    # 日付インデックスを確認
                    has_valid_dates = True
                    try:
                        # 有効な日付範囲かチェック
                        for idx in y_test.index[-points:]:
                            if idx.year < 1 or idx.year > 9999:
                                has_valid_dates = False
                                break
                    except Exception:
                        has_valid_dates = False

                    if has_valid_dates:
                        # 有効な日付インデックスを使用
                        test_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred
                        }, index=y_test.index)
                        # 時間順にソート
                        test_df = test_df.sort_index()
                    else:
                        # 連番インデックスを使用
                        range_index = pd.RangeIndex(start=0, stop=len(y_test))
                        test_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred
                        }, index=range_index)
                else:
                    # 日付インデックスでない場合は連番を使用
                    range_index = pd.RangeIndex(start=0, stop=len(y_test))
                    test_df = pd.DataFrame({
                        'Actual': y_test.values,
                        'Predicted': y_pred
                    }, index=range_index)

                # 最新のpointsポイントを使用
                plot_data = test_df.iloc[-points:]

                # データがない場合、メッセージを表示して次へ
                if len(plot_data) == 0:
                    axs[i].text(0.5, 0.5, "データがありません",
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[i].transAxes)
                    continue

                # 実測値と予測値をカラー分けして明確に表示
                axs[i].plot(plot_data.index, plot_data['Actual'], 'b-', label='Actual', linewidth=2)
                axs[i].plot(plot_data.index, plot_data['Predicted'], 'r--', label='Predicted', linewidth=2)

                # 判読しやすいように凡例を表示
                axs[i].legend(loc='best')

                # インデックスがDatetimeIndexの場合のみ日付フォーマットを設定
                if isinstance(plot_data.index, pd.DatetimeIndex):
                    try:
                        # 時間軸のフォーマットを改善
                        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
                        axs[i].set_xlabel('Time')
                    except Exception:
                        # 日付フォーマットが失敗したら通常のインデックス表示
                        axs[i].set_xlabel('Sample')
                else:
                    # データポイント数を表示
                    axs[i].set_xlabel('Sample')
                    # 読みやすいようにx軸のティック数を制限
                    if len(plot_data) > 10:
                        step = max(1, len(plot_data) // 10)
                        ticks = plot_data.index[::step].tolist()
                        axs[i].set_xticks(ticks)

                axs[i].set_title(f'Zone {zone} - {zone_system} System ({horizon}min Ahead)')
                axs[i].set_ylabel('Temperature (°C)')
                axs[i].grid(True)

                # エラーの可視化 (RMSE)
                rmse = np.sqrt(np.mean((plot_data['Actual'] - plot_data['Predicted'])**2))
                axs[i].annotate(f'RMSE: {rmse:.3f}°C',
                             xy=(0.02, 0.95),
                             xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            except Exception as e:
                print(f"警告: ゾーン{zone}の時系列プロット作成中にエラーが発生しました: {e}")
                axs[i].text(0.5, 0.5, f"プロットエラー",
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[i].transAxes)

        # 使わないサブプロットを非表示
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f'{horizon}min Ahead Temperature Prediction - Time Series', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # suptitleのスペースを確保

        if save:
            output_path = os.path.join(OUTPUT_DIR, f'time_series_horizon_{horizon}.png')
            plt.savefig(output_path)
            print(f"{horizon}分後予測の時系列プロットを保存しました: {output_path}")

        return fig

    except Exception as e:
        print(f"エラー: {horizon}分後予測の時系列プロット作成中にエラーが発生しました: {e}")
        return None


def plot_lag_dependency_analysis(lag_dependency_df, save=True):
    """
    LAG依存度分析結果を視覚化する関数

    Parameters:
    -----------
    lag_dependency_df : DataFrame
        LAG依存度分析結果を含むデータフレーム
    save : bool, optional
        グラフを保存するかどうか

    Returns:
    --------
    list
        プロットしたFigureオブジェクトのリスト
    """
    figures = []

    # 1. 時系列依存度とその内訳
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # ゾーン番号でソート
    sorted_df = lag_dependency_df.sort_values('ゾーン')

    # 時系列依存度とその内訳をプロット
    x = np.arange(len(sorted_df))
    width = 0.2

    ax1.bar(x - width, sorted_df['現在温度依存度(%)'], width, label='現在温度')
    ax1.bar(x, sorted_df['LAG温度依存度(%)'], width, label='LAG温度')
    ax1.bar(x + width, sorted_df['移動平均温度依存度(%)'], width, label='移動平均温度')

    ax1.set_xlabel('ゾーン')
    ax1.set_ylabel('依存度 (%)')
    ax1.set_title('温度時系列依存度の内訳')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{z}\n({s}系統)" for z, s in zip(sorted_df['ゾーン'], sorted_df['系統'])])
    ax1.legend()

    plt.tight_layout()
    if save:
        output_path = os.path.join(OUTPUT_DIR, 'lag_dependency_timeseries.png')
        plt.savefig(output_path)
        print(f"時系列依存度分析グラフを保存しました: {output_path}")

    figures.append(fig1)

    # 2. 大カテゴリ分析
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # 大カテゴリの依存度をプロット
    ax2.bar(x - width, sorted_df['過去時系列合計(%)'], width, label='過去時系列')
    ax2.bar(x, sorted_df['現在非センサー合計(%)'], width, label='現在非センサー')
    ax2.bar(x + width, sorted_df['未来説明変数合計(%)'], width, label='未来説明変数')

    ax2.set_xlabel('ゾーン')
    ax2.set_ylabel('依存度 (%)')
    ax2.set_title('大カテゴリ別予測依存度')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{z}\n({s}系統)" for z, s in zip(sorted_df['ゾーン'], sorted_df['系統'])])
    ax2.legend()

    plt.tight_layout()
    if save:
        output_path = os.path.join(OUTPUT_DIR, 'lag_dependency_categories.png')
        plt.savefig(output_path)
        print(f"カテゴリ別依存度分析グラフを保存しました: {output_path}")

    figures.append(fig2)

    return figures
