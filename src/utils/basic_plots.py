#!/usr/bin/env python
# coding: utf-8

"""
包括的な可視化システム
予測精度を視覚的に理解するための様々なプロットを提供
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import r2_score
from scipy import stats
from .font_config import setup_japanese_font, get_font_properties

# フォント設定を実行
japanese_font_prop = setup_japanese_font()

# グラフ設定
sns.set_theme(style="whitegrid", palette="husl")

# matplotlib設定（文字化け対策含む）
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.unicode_minus': False,  # マイナス記号の文字化け対策
    'figure.autolayout': True,    # レイアウト自動調整
})


def plot_feature_importance(model, feature_names, zone, horizon, save_path=None,
                          top_n=20, model_type="予測", save=True):
    """
    特徴量重要度のプロット

    Parameters:
    -----------
    model : LGBMRegressor or similar model
        学習済みモデル（feature_importances_属性を持つモデル）
    feature_names : list
        特徴量名のリスト
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_path : str, optional
        保存先パス（Noneの場合は保存しない）
    top_n : int, optional
        表示する特徴量数
    model_type : str, optional
        モデルタイプ（グラフタイトル用）
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # モデルから特徴量重要度を取得
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"警告: モデルに feature_importances_ 属性がありません")
        return None

    # DataFrameを作成
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # 重要度で降順ソート
    importance_sorted = feature_importance_df.sort_values('importance', ascending=False)

    # 上位N個の特徴量を抽出
    top_features = importance_sorted.head(top_n)

    # プロット作成
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(range(len(top_features)), top_features['importance'])

    # 色分け（重要度に応じて）
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 軸の設定
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # 上位から表示

    ax.set_title(f'Zone {zone} - {model_type}Model Feature Importance ({horizon}min Prediction)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)

    # グリッド追加
    ax.grid(True, alpha=0.3, axis='x')

    # 値をバーの上に表示
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()

    # グラフ保存
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {save_path}")

    return fig


def analyze_lag_dependency(model, feature_names):
    """
    モデルの特徴量重要度からLAG依存度を分析

    Parameters:
    -----------
    model : trained model
        学習済みモデル
    feature_names : list
        特徴量名のリスト

    Returns:
    --------
    dict
        LAG依存度分析結果
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        return {'lag_temp_percent': 0, 'rolling_temp_percent': 0, 'total_lag_percent': 0}

    # 特徴量重要度の合計
    total_importance = np.sum(importances)

    if total_importance == 0:
        return {'lag_temp_percent': 0, 'rolling_temp_percent': 0, 'total_lag_percent': 0}

    # LAG系特徴量の重要度を計算
    lag_importance = 0
    rolling_importance = 0

    for i, feature_name in enumerate(feature_names):
        if 'lag' in feature_name.lower() and 'temp' in feature_name.lower():
            lag_importance += importances[i]
        elif 'rolling' in feature_name.lower() and 'temp' in feature_name.lower():
            rolling_importance += importances[i]

    # パーセンテージ計算
    lag_temp_percent = (lag_importance / total_importance) * 100
    rolling_temp_percent = (rolling_importance / total_importance) * 100
    total_lag_percent = lag_temp_percent + rolling_temp_percent

    return {
        'lag_temp_percent': lag_temp_percent,
        'rolling_temp_percent': rolling_temp_percent,
        'total_lag_percent': total_lag_percent
    }


def plot_time_series_comparison(y_true, y_pred, timestamps, zone, horizon,
                               save_path=None, model_type="Prediction", save=True,
                               show_period_hours=24, detailed_mode=True, model=None, feature_names=None):
    """
    シンプルな時系列での実際値と予測値の比較プロット

    Parameters:
    -----------
    y_true : array-like
        実際値
    y_pred : array-like
        予測値
    timestamps : array-like
        時刻データ（実際値の時刻）
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_path : str, optional
        保存先パス
    model_type : str
        モデルタイプ
    save : bool
        保存するか
    show_period_hours : int
        表示期間（時間）
    detailed_mode : bool
        詳細モード（未使用、互換性のため残存）
    model : trained model, optional
        学習済みモデル
    feature_names : list, optional
        特徴量名のリスト

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # データの前処理
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    timestamps = pd.to_datetime(timestamps)

    # 有効データのフィルタリング
    valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred) |
                     np.isinf(y_true) | np.isinf(y_pred))

    timestamps_valid = timestamps[valid_indices]
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(timestamps_valid) == 0:
        print(f"Warning: Zone {zone} has no valid data")
        return None

    # 表示期間の設定
    if len(timestamps_valid) > 0:
        end_time = timestamps_valid[-1]
        start_time = end_time - pd.Timedelta(hours=show_period_hours)

        period_mask = (timestamps_valid >= start_time) & (timestamps_valid <= end_time)
        timestamps_period = timestamps_valid[period_mask]
        y_true_period = y_true_valid[period_mask]
        y_pred_period = y_pred_valid[period_mask]

        if len(timestamps_period) == 0:
            # データが少ない場合は利用可能な全データを使用
            max_points = min(len(timestamps_valid), show_period_hours * 60)
            timestamps_period = timestamps_valid[-max_points:]
            y_true_period = y_true_valid[-max_points:]
            y_pred_period = y_pred_valid[-max_points:]

    # 正確な予測時間軸を作成（入力時刻 + 予測ホライゾン）
    prediction_timestamps = timestamps_period + pd.Timedelta(minutes=horizon)

    # シンプルなプロット作成
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # 実測値（青線、マーカー付き）
    ax.plot(timestamps_period, y_true_period, 'b-', linewidth=2.5,
            marker='o', markersize=3, markevery=max(1, len(timestamps_period)//100),
            label='Actual', alpha=0.9)

    # 予測値（正確な時間軸、赤線、マーカー付き）
    ax.plot(prediction_timestamps, y_pred_period, 'r-', linewidth=2.5,
            marker='s', markersize=2, markevery=max(1, len(prediction_timestamps)//100),
            label=f'Predicted (+{horizon}min)', alpha=0.9)

    # LAG依存度分析（モデルが提供された場合）
    lag_analysis = {'total_lag_percent': 0}
    if model is not None and feature_names is not None:
        lag_analysis = analyze_lag_dependency(model, feature_names)

    # タイトルにLAG依存度情報を含める
    total_lag = lag_analysis['total_lag_percent']
    if total_lag > 30:
        lag_info = f' [High LAG: {total_lag:.1f}%]'
        title_color = 'darkred'
    elif total_lag > 15:
        lag_info = f' [Med LAG: {total_lag:.1f}%]'
        title_color = 'darkorange'
    elif total_lag > 0:
        lag_info = f' [Low LAG: {total_lag:.1f}%]'
        title_color = 'darkgreen'
    else:
        lag_info = ''
        title_color = 'black'

    title = f'Zone {zone} - {model_type} ({horizon}min){lag_info}'
    ax.set_title(title, fontsize=16, fontweight='bold', color=title_color)
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_xlabel('DateTime', fontsize=12, fontweight='bold')

    # 時間軸の設定
    if show_period_hours <= 2:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    elif show_period_hours <= 6:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    elif show_period_hours <= 12:
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    elif show_period_hours <= 24:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=8)

    # グリッド
    ax.grid(True, linestyle='-', alpha=0.3, which='major')
    ax.grid(True, linestyle=':', alpha=0.2, which='minor')

    ax.legend(fontsize=12, framealpha=0.9)

    # 統計情報（重複する時間範囲で計算）
    # 重複する時間範囲を計算
    actual_start = timestamps_period.min()
    actual_end = timestamps_period.max()
    pred_start = prediction_timestamps.min()
    pred_end = prediction_timestamps.max()

    overlap_start = max(actual_start, pred_start)
    overlap_end = min(actual_end, pred_end)

    # 重複範囲内のデータのみを抽出
    actual_mask = (timestamps_period >= overlap_start) & (timestamps_period <= overlap_end)
    pred_mask = (prediction_timestamps >= overlap_start) & (prediction_timestamps <= overlap_end)

    timestamps_aligned = timestamps_period[actual_mask]
    y_true_aligned = y_true_period[actual_mask]
    prediction_timestamps_aligned = prediction_timestamps[pred_mask]
    y_pred_aligned = y_pred_period[pred_mask]

    # 長さを確認して調整
    min_length = min(len(timestamps_aligned), len(prediction_timestamps_aligned))
    if min_length > 0:
        timestamps_aligned = timestamps_aligned[:min_length]
        y_true_aligned = y_true_aligned[:min_length]
        prediction_timestamps_aligned = prediction_timestamps_aligned[:min_length]
        y_pred_aligned = y_pred_aligned[:min_length]

        # 詳細統計計算
        mae = np.mean(np.abs(y_pred_aligned - y_true_aligned))
        rmse = np.sqrt(np.mean((y_pred_aligned - y_true_aligned)**2))
        r2 = r2_score(y_true_aligned, y_pred_aligned)

        # 統計情報をグラフに表示
        stats_text = f'RMSE: {rmse:.3f}°C | MAE: {mae:.3f}°C | R²: {r2:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=11, fontweight='bold')

        # データ点数情報
        data_info = f'{min_length} points | {show_period_hours}h'
        ax.text(0.98, 0.02, data_info, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right', fontsize=10)

    plt.tight_layout()

    # グラフ保存
    if save and save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Simple timeseries comparison saved: {save_path}")

    return fig


def plot_scatter_analysis(y_true, y_pred, zone, horizon, save_path=None,
                         model_type="Prediction", save=True):
    """
    散布図による予測精度分析

    Parameters:
    -----------
    y_true : array-like
        実際値
    y_pred : array-like
        予測値
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_path : str, optional
        保存先パス
    model_type : str
        モデルタイプ
    save : bool
        保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 散布図
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')

    # 理想線（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax1.set_title(f'Zone {zone} - {model_type} Accuracy Scatter Plot', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 統計情報
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    mae = np.mean(np.abs(y_pred - y_true))
    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2

    textstr = f'RMSE: {rmse:.3f}°C\nMAE: {mae:.3f}°C\nR²: {r2:.3f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    # 残差プロット
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Residuals (°C)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 残差のヒストグラム
    ax3.hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals (°C)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Q-Qプロット
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # グラフ保存
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter analysis plot saved: {save_path}")

    return fig


def plot_performance_summary(metrics_dict, zone, horizon, save_path=None, save=True):
    """
    性能指標のサマリープロット

    Parameters:
    -----------
    metrics_dict : dict
        評価指標の辞書
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_path : str, optional
        保存先パス
    save : bool
        保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # メトリクスの準備
    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())

    # バープロット
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')

    # 値をバーの上に表示
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title(f'Zone {zone} - Performance Metrics ({horizon}min Prediction)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Metric Value', fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # グラフ保存
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance summary plot saved: {save_path}")

    return fig


def plot_comparison_analysis(direct_metrics, diff_metrics, zone, horizon,
                           save_path=None, save=True):
    """
    直接予測と差分予測の比較分析プロット

    Parameters:
    -----------
    direct_metrics : dict
        直接予測の評価指標
    diff_metrics : dict
        差分予測の評価指標
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_path : str, optional
        保存先パス
    save : bool
        保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # 共通メトリクスを抽出
    common_metrics = ['rmse', 'mae', 'r2']
    direct_values = [direct_metrics.get(metric, 0) for metric in common_metrics]
    diff_values = [diff_metrics.get('restored_' + metric, diff_metrics.get(metric, 0)) for metric in common_metrics]

    x = np.arange(len(common_metrics))
    width = 0.35

    # バー比較
    bars1 = ax1.bar(x - width/2, direct_values, width, label='Direct Prediction', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, diff_values, width, label='Difference Prediction', alpha=0.8, color='lightcoral')

    ax1.set_xlabel('Metrics', fontsize=14)
    ax1.set_ylabel('Values', fontsize=14)
    ax1.set_title(f'Zone {zone} - Prediction Method Comparison ({horizon}min)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in common_metrics])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 値をバーの上に表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # 改善率の計算と表示
    improvements = []
    for i, metric in enumerate(common_metrics):
        if direct_values[i] != 0:
            if metric == 'r2':  # R²は大きいほど良い
                improvement = ((diff_values[i] - direct_values[i]) / abs(direct_values[i])) * 100
            else:  # RMSE、MAEは小さいほど良い
                improvement = ((direct_values[i] - diff_values[i]) / direct_values[i]) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)

    # 改善率バープロット
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars3 = ax2.bar(common_metrics, improvements, color=colors, alpha=0.8)

    ax2.set_xlabel('Metrics', fontsize=14)
    ax2.set_ylabel('Improvement (%)', fontsize=14)
    ax2.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3)

    # 改善率の値を表示
    for bar, improvement in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{improvement:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')

    plt.tight_layout()

    # グラフ保存
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison analysis plot saved: {save_path}")

    return fig


def create_comprehensive_visualization_report(model, feature_names, y_true, y_pred,
                                            timestamps, metrics, zone, horizon,
                                            model_type="Prediction", save_dir="Output/visualizations"):
    """
    包括的な可視化レポートを作成

    Parameters:
    -----------
    model : trained model
        学習済みモデル
    feature_names : list
        特徴量名
    y_true : array-like
        実際値
    y_pred : array-like
        予測値
    timestamps : array-like
        時刻データ
    metrics : dict
        評価指標
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン
    model_type : str
        モデルタイプ
    save_dir : str
        保存ディレクトリ

    Returns:
    --------
    dict
        作成された可視化ファイルのパス
    """
    os.makedirs(save_dir, exist_ok=True)

    created_files = {}

    # 1. 特徴量重要度
    importance_path = os.path.join(save_dir, f"{model_type.lower()}_feature_importance_zone_{zone}_horizon_{horizon}.png")
    plot_feature_importance(model, feature_names, zone, horizon, importance_path, model_type=model_type)
    created_files['feature_importance'] = importance_path

    # 2. 時系列比較
    timeseries_path = os.path.join(save_dir, f"{model_type.lower()}_timeseries_zone_{zone}_horizon_{horizon}.png")
    plot_time_series_comparison(y_true, y_pred, timestamps, zone, horizon,
                              timeseries_path, model_type=model_type,
                              show_period_hours=24, detailed_mode=True,
                              model=model, feature_names=feature_names)
    created_files['timeseries'] = timeseries_path

    # 3. 散布図分析
    scatter_path = os.path.join(save_dir, f"{model_type.lower()}_scatter_analysis_zone_{zone}_horizon_{horizon}.png")
    plot_scatter_analysis(y_true, y_pred, zone, horizon, scatter_path, model_type=model_type)
    created_files['scatter'] = scatter_path

    # 4. 性能サマリー（削除）
    # summary_path = os.path.join(save_dir, f"{model_type.lower()}_performance_summary_zone_{zone}_horizon_{horizon}.png")
    # plot_performance_summary(metrics, zone, horizon, summary_path)
    # created_files['summary'] = summary_path

    print(f"\n📊 {model_type} model comprehensive minute analysis report completed:")
    for viz_type, path in created_files.items():
        print(f"  - {viz_type}: {path}")

    return created_files


def create_correct_prediction_timestamps(timestamps, horizon_minutes):
    """
    正しい予測タイムスタンプを作成

    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        入力データのタイムスタンプ
    horizon_minutes : int
        予測ホライゾン（分）

    Returns:
    --------
    pd.DatetimeIndex
        正しい予測タイムスタンプ
    """
    return timestamps + pd.Timedelta(minutes=horizon_minutes)


def plot_corrected_time_series(timestamps, actual, predicted, zone, horizon, save_dir=None,
                              points=100, save=True, validate_timing=True):
    """
    シンプルな時系列プロット（正しい時間軸のみ）

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
        時間軸の検証を行うか（未使用、互換性のため残存）

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
        print(f"Warning: Zone {zone} has no valid data")
        return None

    # 予測値用の正しいタイムスタンプを作成
    prediction_timestamps = create_correct_prediction_timestamps(timestamps_valid, horizon)

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

    # シンプルなプロット作成（1つのプロットのみ）
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

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

    # 正しい時間軸での表示のみ
    ax.plot(timestamps_aligned, actual_aligned, 'b-', linewidth=2.5, label='Actual', alpha=0.9)
    ax.plot(prediction_timestamps_aligned, predicted_aligned, 'r--', linewidth=2.5,
            label=f'Predicted (+{horizon}min)', alpha=0.9)

    ax.set_title(f'Zone {zone} - Temperature Prediction ({horizon}min ahead)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('DateTime', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'simple_timeseries_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Simple timeseries plot saved: {output_path}")

    return fig


def plot_ultra_detailed_minute_analysis(y_true, y_pred, timestamps, zone, horizon,
                                       save_dir="Output/visualizations", save=True,
                                       model=None, feature_names=None):
    """
    超高解像度分刻み時系列分析（複数の時間スケール）

    Parameters:
    -----------
    y_true : array-like
        実際値
    y_pred : array-like
        予測値
    timestamps : array-like
        時刻データ
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str
        保存ディレクトリ
    save : bool
        保存するか
    model : trained model, optional
        学習済みモデル
    feature_names : list, optional
        特徴量名のリスト

    Returns:
    --------
    list
        生成されたFigureオブジェクトのリスト
    """
    figures = []

    # 複数の詳細時間スケールを定義
    time_scales = [
        {'name': 'ultra_minute', 'hours': 2, 'description': '2h Detailed (Minute-scale)'},
        {'name': 'detailed_minute', 'hours': 6, 'description': '6h Detailed (Minute-scale)'},
        {'name': 'extended_minute', 'hours': 12, 'description': '12h Detailed (Minute-scale)'},
        {'name': 'daily_minute', 'hours': 24, 'description': '24h Detailed (Minute-scale)'},
        {'name': 'multi_day_minute', 'hours': 48, 'description': '48h Detailed (Minute-scale)'}
    ]

    for scale_config in time_scales:
        print(f"📊 Generating {scale_config['description']} visualization...")

        fig = _create_minute_scale_visualization(
            y_true, y_pred, timestamps, zone, horizon, scale_config,
            save_dir, save, model, feature_names
        )

        if fig is not None:
            figures.append(fig)

    return figures


def _create_minute_scale_visualization(y_true, y_pred, timestamps, zone, horizon,
                                     scale_config, save_dir, save, model, feature_names):
    """
    シンプルな分刻みスケール可視化
    """
    # データの前処理
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    timestamps = pd.to_datetime(timestamps)

    # 有効データのフィルタリング
    valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred) |
                     np.isinf(y_true) | np.isinf(y_pred))

    timestamps_valid = timestamps[valid_indices]
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(timestamps_valid) == 0:
        print(f"Warning: Zone {zone} has no valid data")
        return None

    # 指定時間範囲のデータを抽出
    hours = scale_config['hours']
    if len(timestamps_valid) > 0:
        end_time = timestamps_valid[-1]
        start_time = end_time - pd.Timedelta(hours=hours)

        period_mask = (timestamps_valid >= start_time) & (timestamps_valid <= end_time)
        timestamps_period = timestamps_valid[period_mask]
        y_true_period = y_true_valid[period_mask]
        y_pred_period = y_pred_valid[period_mask]

        if len(timestamps_period) == 0:
            # データが少ない場合は利用可能な全データを使用
            max_points = min(len(timestamps_valid), hours * 60)
            timestamps_period = timestamps_valid[-max_points:]
            y_true_period = y_true_valid[-max_points:]
            y_pred_period = y_pred_valid[-max_points:]

    # 正しい時間軸での予測値タイムスタンプ
    prediction_timestamps = create_correct_prediction_timestamps(timestamps_period, horizon)

    # シンプルなプロット作成（1つのプロットのみ）
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

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
        y_true_aligned = y_true_period[actual_mask]
        prediction_timestamps_aligned = prediction_timestamps[pred_mask]
        y_pred_aligned = y_pred_period[pred_mask]

        # 長さを確認して調整
        min_length = min(len(timestamps_aligned), len(prediction_timestamps_aligned))
        if min_length > 0:
            timestamps_aligned = timestamps_aligned[:min_length]
            y_true_aligned = y_true_aligned[:min_length]
            prediction_timestamps_aligned = prediction_timestamps_aligned[:min_length]
            y_pred_aligned = y_pred_aligned[:min_length]

            # シンプルなプロット
            ax.plot(timestamps_aligned, y_true_aligned, 'b-', linewidth=3,
                    marker='o', markersize=3, markevery=max(1, len(timestamps_aligned)//100),
                    label='Actual', alpha=0.9)

            ax.plot(prediction_timestamps_aligned, y_pred_aligned, 'r-', linewidth=2.5,
                    marker='s', markersize=2, markevery=max(1, len(prediction_timestamps_aligned)//100),
                    label=f'Predicted (+{horizon}min)', alpha=0.9)

            # LAG依存度分析（モデルが提供された場合）
            lag_analysis = {'total_lag_percent': 0}
            if model is not None and feature_names is not None:
                lag_analysis = analyze_lag_dependency(model, feature_names)

            # 詳細統計計算
            mae = np.mean(np.abs(y_true_aligned - y_pred_aligned))
            rmse = np.sqrt(np.mean((y_true_aligned - y_pred_aligned)**2))
            r2 = r2_score(y_true_aligned, y_pred_aligned)

            # タイトル
            total_lag = lag_analysis['total_lag_percent']
            if total_lag > 30:
                lag_info = f' [High LAG: {total_lag:.1f}%]'
                title_color = 'darkred'
            elif total_lag > 15:
                lag_info = f' [Med LAG: {total_lag:.1f}%]'
                title_color = 'darkorange'
            elif total_lag > 0:
                lag_info = f' [Low LAG: {total_lag:.1f}%]'
                title_color = 'darkgreen'
            else:
                lag_info = ''
                title_color = 'black'

            title = f'Zone {zone} - {scale_config["description"]} ({horizon}min){lag_info}'
            ax.set_title(title, fontsize=18, fontweight='bold', color=title_color)
            ax.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
            ax.set_xlabel('DateTime', fontsize=14, fontweight='bold')

            # 時間軸の詳細設定（分刻み表示）
            if hours <= 2:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            elif hours <= 6:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            elif hours <= 12:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            elif hours <= 24:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            else:
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

            ax.tick_params(axis='x', rotation=45, labelsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='both', which='minor', labelsize=10)

            # グリッド
            ax.grid(True, linestyle='-', alpha=0.3, which='major')
            ax.grid(True, linestyle=':', alpha=0.2, which='minor')

            # 凡例
            ax.legend(fontsize=14, framealpha=0.9, loc='upper right')

            # 統計情報ボックス（シンプル化）
            stats_text = f'RMSE: {rmse:.3f}°C | MAE: {mae:.3f}°C | R²: {r2:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top', fontsize=12, fontweight='bold')

            # データ情報（シンプル化）
            data_info = f'{min_length} points | {hours}h range'
            ax.text(0.98, 0.02, data_info, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                    verticalalignment='bottom', horizontalalignment='right', fontsize=11)

    except Exception as e:
        print(f"Zone {zone} plotting error: {e}")
        return None

    plt.tight_layout()

    # 保存
    if save and save_dir:
        output_path = os.path.join(save_dir,
                                  f'minute_{scale_config["name"]}_zone_{zone}_horizon_{horizon}_{scale_config["hours"]}h.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Minute visualization saved: {output_path}")

    return fig


def create_comprehensive_minute_analysis_report(model, feature_names, y_true, y_pred,
                                              timestamps, metrics, zone, horizon,
                                              model_type="Prediction", save_dir="Output/visualizations"):
    """
    包括的な分刻み可視化レポートを作成

    Parameters:
    -----------
    model : trained model
        学習済みモデル
    feature_names : list
        特徴量名
    y_true : array-like
        実際値
    y_pred : array-like
        予測値
    timestamps : array-like
        時刻データ
    metrics : dict
        評価指標
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン
    model_type : str
        モデルタイプ
    save_dir : str
        保存ディレクトリ

    Returns:
    --------
    dict
        作成された可視化ファイルのパス
    """
    os.makedirs(save_dir, exist_ok=True)

    created_files = {}

    # 1. 既存の包括的可視化レポート
    standard_report = create_comprehensive_visualization_report(
        model, feature_names, y_true, y_pred, timestamps, metrics,
        zone, horizon, model_type, save_dir
    )
    created_files.update(standard_report)

    # 2. シンプルな時間軸プロット
    simple_path = os.path.join(save_dir, f"{model_type.lower()}_simple_timeseries_zone_{zone}_horizon_{horizon}.png")
    simple_fig = plot_corrected_time_series(
        timestamps, y_true, y_pred, zone, horizon, save_dir, save=True
    )
    if simple_fig is not None:
        created_files['simple_timeseries'] = simple_path

    # 3. 分刻み時系列分析
    minute_figures = plot_ultra_detailed_minute_analysis(
        y_true, y_pred, timestamps, zone, horizon, save_dir, save=True,
        model=model, feature_names=feature_names
    )

    # 分刻み可視化のパスを記録
    time_scales = ['ultra_minute', 'detailed_minute', 'extended_minute', 'daily_minute', 'multi_day_minute']
    hours_list = [2, 6, 12, 24, 48]

    for i, (scale_name, hours) in enumerate(zip(time_scales, hours_list)):
        if i < len(minute_figures) and minute_figures[i] is not None:
            path = os.path.join(save_dir, f'minute_{scale_name}_zone_{zone}_horizon_{horizon}_{hours}h.png')
            created_files[f'minute_{scale_name}'] = path

    print(f"\n📊 {model_type} model comprehensive minute analysis report completed:")
    for viz_type, path in created_files.items():
        print(f"  - {viz_type}: {path}")

    return created_files


# 公開APIに新しい関数を追加
__all__ = [
    'plot_feature_importance',
    'plot_time_series_comparison',
    'plot_scatter_analysis',
    'plot_performance_summary',
    'plot_comparison_analysis',
    'create_comprehensive_visualization_report',
    'analyze_lag_dependency',
    'create_correct_prediction_timestamps',
    'plot_corrected_time_series',
    'plot_ultra_detailed_minute_analysis',
    'create_comprehensive_minute_analysis_report'
]
