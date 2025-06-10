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


# ===== ユーティリティ関数 =====

def _validate_data(y_true, y_pred, timestamps=None):
    """データの検証と前処理"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if timestamps is not None:
        timestamps = pd.to_datetime(timestamps)
        valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred) | 
                         np.isinf(y_true) | np.isinf(y_pred))
        return y_true[valid_indices], y_pred[valid_indices], timestamps[valid_indices]
    else:
        valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred) | 
                         np.isinf(y_true) | np.isinf(y_pred))
        return y_true[valid_indices], y_pred[valid_indices]


def _calculate_metrics(y_true, y_pred):
    """基本統計指標の計算"""
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def _setup_time_axis(ax, show_period_hours):
    """時間軸の設定"""
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
    ax.grid(True, linestyle='-', alpha=0.3, which='major')
    ax.grid(True, linestyle=':', alpha=0.2, which='minor')


def _create_prediction_timestamps(timestamps, horizon_minutes):
    """正しい予測タイムスタンプを作成"""
    return timestamps + pd.Timedelta(minutes=horizon_minutes)


def _align_time_series(timestamps_actual, y_actual, timestamps_pred, y_pred):
    """時系列データの時間軸を整列"""
    actual_start = timestamps_actual.min()
    actual_end = timestamps_actual.max()
    pred_start = timestamps_pred.min()
    pred_end = timestamps_pred.max()
    
    overlap_start = max(actual_start, pred_start)
    overlap_end = min(actual_end, pred_end)
    
    actual_mask = (timestamps_actual >= overlap_start) & (timestamps_actual <= overlap_end)
    pred_mask = (timestamps_pred >= overlap_start) & (timestamps_pred <= overlap_end)
    
    timestamps_aligned_actual = timestamps_actual[actual_mask]
    y_aligned_actual = y_actual[actual_mask]
    timestamps_aligned_pred = timestamps_pred[pred_mask]
    y_aligned_pred = y_pred[pred_mask]
    
    min_length = min(len(timestamps_aligned_actual), len(timestamps_aligned_pred))
    if min_length > 0:
        return (timestamps_aligned_actual[:min_length], y_aligned_actual[:min_length],
                timestamps_aligned_pred[:min_length], y_aligned_pred[:min_length])
    return None, None, None, None


def analyze_lag_dependency(model, feature_names):
    """モデルの特徴量重要度からLAG依存度を分析"""
    try:
        importances = model.feature_importances_
    except AttributeError:
        return {'lag_temp_percent': 0, 'rolling_temp_percent': 0, 'total_lag_percent': 0}

    total_importance = np.sum(importances)
    if total_importance == 0:
        return {'lag_temp_percent': 0, 'rolling_temp_percent': 0, 'total_lag_percent': 0}

    lag_importance = sum(importances[i] for i, name in enumerate(feature_names) 
                        if 'lag' in name.lower() and 'temp' in name.lower())
    rolling_importance = sum(importances[i] for i, name in enumerate(feature_names) 
                           if 'rolling' in name.lower() and 'temp' in name.lower())

    lag_temp_percent = (lag_importance / total_importance) * 100
    rolling_temp_percent = (rolling_importance / total_importance) * 100
    total_lag_percent = lag_temp_percent + rolling_temp_percent

    return {
        'lag_temp_percent': lag_temp_percent,
        'rolling_temp_percent': rolling_temp_percent,
        'total_lag_percent': total_lag_percent
    }


# ===== メインプロット関数 =====

def plot_feature_importance(model, feature_names, zone, horizon, save_path=None,
                          top_n=20, model_type="予測", save=True):
    """特徴量重要度のプロット"""
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"警告: モデルに feature_importances_ 属性がありません")
        return None

    # DataFrameを作成してソート
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # プロット作成
    fig, ax = plt.subplots(figsize=(14, 10))
    bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'])

    # 色分け
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 軸設定
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df['feature'])
    ax.invert_yaxis()

    ax.set_title(f'Zone {zone} - {model_type}Model Feature Importance ({horizon}min Prediction)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance Score', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # 値を表示
    for bar, importance in zip(bars, feature_importance_df['importance']):
        ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=10)

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {save_path}")

    return fig


def plot_time_series_comparison(y_true, y_pred, timestamps, zone, horizon,
                               save_path=None, model_type="Prediction", save=True,
                               show_period_hours=24, detailed_mode=True, model=None, feature_names=None):
    """時系列での実際値と予測値の比較プロット"""
    # データ検証
    y_true, y_pred, timestamps = _validate_data(y_true, y_pred, timestamps)
    
    if len(timestamps) == 0:
        print(f"Warning: Zone {zone} has no valid data")
        return None

    # 表示期間の設定
    end_time = timestamps[-1]
        start_time = end_time - pd.Timedelta(hours=show_period_hours)
    period_mask = (timestamps >= start_time) & (timestamps <= end_time)
    
    if not period_mask.any():
        max_points = min(len(timestamps), show_period_hours * 60)
        timestamps_period = timestamps[-max_points:]
        y_true_period = y_true[-max_points:]
        y_pred_period = y_pred[-max_points:]
    else:
        timestamps_period = timestamps[period_mask]
        y_true_period = y_true[period_mask]
        y_pred_period = y_pred[period_mask]

    # 予測タイムスタンプの作成
    prediction_timestamps = _create_prediction_timestamps(timestamps_period, horizon)

    # プロット作成
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # 実測値と予測値をプロット
    ax.plot(timestamps_period, y_true_period, 'b-', linewidth=2.5,
            marker='o', markersize=3, markevery=max(1, len(timestamps_period)//100),
            label='Actual', alpha=0.9)
    ax.plot(prediction_timestamps, y_pred_period, 'r-', linewidth=2.5,
            marker='s', markersize=2, markevery=max(1, len(prediction_timestamps)//100),
            label=f'Predicted (+{horizon}min)', alpha=0.9)

    # LAG依存度分析
    lag_analysis = {'total_lag_percent': 0}
    if model is not None and feature_names is not None:
        lag_analysis = analyze_lag_dependency(model, feature_names)

    # タイトル設定
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

    ax.set_title(f'Zone {zone} - {model_type} ({horizon}min){lag_info}',
                fontsize=16, fontweight='bold', color=title_color)
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_xlabel('DateTime', fontsize=12, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)

    # 時間軸設定
    _setup_time_axis(ax, show_period_hours)

    # 統計情報の計算と表示
    timestamps_aligned_actual, y_aligned_actual, timestamps_aligned_pred, y_aligned_pred = _align_time_series(
        timestamps_period, y_true_period, prediction_timestamps, y_pred_period
    )
    
    if timestamps_aligned_actual is not None:
        metrics = _calculate_metrics(y_aligned_actual, y_aligned_pred)
        stats_text = f'RMSE: {metrics["rmse"]:.3f}°C | MAE: {metrics["mae"]:.3f}°C | R²: {metrics["r2"]:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=11, fontweight='bold')

        data_info = f'{len(y_aligned_actual)} points | {show_period_hours}h'
        ax.text(0.98, 0.02, data_info, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right', fontsize=10)

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Time series comparison saved: {save_path}")

    return fig


def plot_scatter_analysis(y_true, y_pred, zone, horizon, save_path=None,
                         model_type="Prediction", save=True):
    """散布図による予測精度分析"""
    y_true, y_pred = _validate_data(y_true, y_pred)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 散布図
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax1.set_xlabel('Actual Temperature (°C)', fontsize=12)
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=12)
    ax1.set_title(f'Zone {zone} - {model_type} Accuracy Scatter Plot', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 統計情報
    metrics = _calculate_metrics(y_true, y_pred)
    textstr = f'RMSE: {metrics["rmse"]:.3f}°C\nMAE: {metrics["mae"]:.3f}°C\nR²: {metrics["r2"]:.3f}'
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 2. 残差プロット
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Temperature (°C)', fontsize=12)
    ax2.set_ylabel('Residuals (°C)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 残差ヒストグラム
    ax3.hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuals (°C)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Q-Qプロット
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter analysis plot saved: {save_path}")

    return fig


def plot_performance_summary(metrics_dict, zone, horizon, save_path=None, save=True):
    """性能指標のサマリープロット"""
    fig, ax = plt.subplots(figsize=(12, 8))

    metrics_names = list(metrics_dict.keys())
    metrics_values = list(metrics_dict.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_names)))
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')

    # 値を表示
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

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance summary plot saved: {save_path}")

    return fig


def plot_comparison_analysis(direct_metrics, diff_metrics, zone, horizon,
                           save_path=None, save=True):
    """直接予測と差分予測の比較分析プロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

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

    # 値を表示
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)

    # 改善率計算
    improvements = []
    for i, metric in enumerate(common_metrics):
        if direct_values[i] != 0:
            if metric == 'r2':
                improvement = ((diff_values[i] - direct_values[i]) / abs(direct_values[i])) * 100
            else:
                improvement = ((direct_values[i] - diff_values[i]) / direct_values[i]) * 100
            improvements.append(improvement)
        else:
            improvements.append(0)

    # 改善率プロット
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

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison analysis plot saved: {save_path}")

    return fig


# ===== 高レベル可視化関数 =====

def create_comprehensive_visualization_report(model, feature_names, y_true, y_pred,
                                            timestamps, metrics, zone, horizon,
                                            model_type="Prediction", save_dir="Output/visualizations"):
    """包括的な可視化レポートを作成"""
    os.makedirs(save_dir, exist_ok=True)
    created_files = {}

    # 各種プロットを作成
    plots_config = [
        ('feature_importance', plot_feature_importance, 
         (model, feature_names, zone, horizon), {'model_type': model_type}),
        ('timeseries', plot_time_series_comparison, 
         (y_true, y_pred, timestamps, zone, horizon), 
         {'model_type': model_type, 'show_period_hours': 24, 'model': model, 'feature_names': feature_names}),
        ('scatter', plot_scatter_analysis, 
         (y_true, y_pred, zone, horizon), {'model_type': model_type})
    ]

    for plot_name, plot_func, args, kwargs in plots_config:
        save_path = os.path.join(save_dir, f"{model_type.lower()}_{plot_name}_zone_{zone}_horizon_{horizon}.png")
        kwargs['save_path'] = save_path
        
        fig = plot_func(*args, **kwargs)
        if fig is not None:
            created_files[plot_name] = save_path
            plt.close(fig)

    print(f"\n📊 {model_type} model comprehensive visualization report completed:")
    for viz_type, path in created_files.items():
        print(f"  - {viz_type}: {path}")

    return created_files


def _create_minute_scale_visualization(y_true, y_pred, timestamps, zone, horizon,
                                     scale_config, save_dir, save, model, feature_names):
    """分刻みスケール可視化の作成"""
    y_true, y_pred, timestamps = _validate_data(y_true, y_pred, timestamps)
    
    if len(timestamps) == 0:
        print(f"Warning: Zone {zone} has no valid data")
        return None

    # 指定時間範囲のデータを抽出
    hours = scale_config['hours']
    end_time = timestamps[-1]
        start_time = end_time - pd.Timedelta(hours=hours)
    period_mask = (timestamps >= start_time) & (timestamps <= end_time)
    
    if not period_mask.any():
        max_points = min(len(timestamps), hours * 60)
        timestamps_period = timestamps[-max_points:]
        y_true_period = y_true[-max_points:]
        y_pred_period = y_pred[-max_points:]
    else:
        timestamps_period = timestamps[period_mask]
        y_true_period = y_true[period_mask]
        y_pred_period = y_pred[period_mask]

    prediction_timestamps = _create_prediction_timestamps(timestamps_period, horizon)

    # プロット作成
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    try:
        # 時系列データの整列
        timestamps_aligned_actual, y_aligned_actual, timestamps_aligned_pred, y_aligned_pred = _align_time_series(
            timestamps_period, y_true_period, prediction_timestamps, y_pred_period
        )

        if timestamps_aligned_actual is not None:
            # プロット
            ax.plot(timestamps_aligned_actual, y_aligned_actual, 'b-', linewidth=3,
                    marker='o', markersize=3, markevery=max(1, len(timestamps_aligned_actual)//100),
                    label='Actual', alpha=0.9)
            ax.plot(timestamps_aligned_pred, y_aligned_pred, 'r-', linewidth=2.5,
                    marker='s', markersize=2, markevery=max(1, len(timestamps_aligned_pred)//100),
                    label=f'Predicted (+{horizon}min)', alpha=0.9)

            # LAG依存度分析
            lag_analysis = {'total_lag_percent': 0}
            if model is not None and feature_names is not None:
                lag_analysis = analyze_lag_dependency(model, feature_names)

            # タイトル設定
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

            # 時間軸設定
            _setup_time_axis(ax, hours)
            ax.legend(fontsize=14, framealpha=0.9, loc='upper right')

            # 統計情報
            metrics = _calculate_metrics(y_aligned_actual, y_aligned_pred)
            stats_text = f'RMSE: {metrics["rmse"]:.3f}°C | MAE: {metrics["mae"]:.3f}°C | R²: {metrics["r2"]:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                    verticalalignment='top', fontsize=12, fontweight='bold')

            data_info = f'{len(y_aligned_actual)} points | {hours}h range'
            ax.text(0.98, 0.02, data_info, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                    verticalalignment='bottom', horizontalalignment='right', fontsize=11)

    except Exception as e:
        print(f"Zone {zone} plotting error: {e}")
        return None

    plt.tight_layout()

    if save and save_dir:
        output_path = os.path.join(save_dir,
                                  f'minute_{scale_config["name"]}_zone_{zone}_horizon_{horizon}_{scale_config["hours"]}h.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Minute visualization saved: {output_path}")

    return fig


def create_comprehensive_minute_analysis_report(model, feature_names, y_true, y_pred,
                                              timestamps, metrics, zone, horizon,
                                              model_type="Prediction", save_dir="Output/visualizations"):
    """包括的な分刻み可視化レポートを作成"""
    os.makedirs(save_dir, exist_ok=True)
    created_files = {}

    # 1. 標準可視化レポート
    standard_report = create_comprehensive_visualization_report(
        model, feature_names, y_true, y_pred, timestamps, metrics,
        zone, horizon, model_type, save_dir
    )
    created_files.update(standard_report)

    # 2. 分刻み時系列分析
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
            save_dir, True, model, feature_names
        )

        if fig is not None:
            path = os.path.join(save_dir, f'minute_{scale_config["name"]}_zone_{zone}_horizon_{horizon}_{scale_config["hours"]}h.png')
            created_files[f'minute_{scale_config["name"]}'] = path
            plt.close(fig)

    print(f"\n📊 {model_type} model comprehensive minute analysis report completed:")
    for viz_type, path in created_files.items():
        print(f"  - {viz_type}: {path}")

    return created_files


# ===== 後方互換性のための関数 =====

def create_correct_prediction_timestamps(timestamps, horizon_minutes):
    """後方互換性のための関数"""
    return _create_prediction_timestamps(timestamps, horizon_minutes)


def plot_corrected_time_series(timestamps, actual, predicted, zone, horizon, save_dir=None,
                              points=100, save=True, validate_timing=True):
    """後方互換性のための簡単な時系列プロット"""
    return plot_time_series_comparison(
        actual, predicted, timestamps, zone, horizon,
        save_path=os.path.join(save_dir, f'simple_timeseries_zone_{zone}_horizon_{horizon}.png') if save_dir else None,
        model_type="Prediction", save=save, show_period_hours=24
    )


def plot_ultra_detailed_minute_analysis(y_true, y_pred, timestamps, zone, horizon,
                                       save_dir="Output/visualizations", save=True,
                                       model=None, feature_names=None):
    """後方互換性のための超詳細分析"""
    time_scales = [
        {'name': 'ultra_minute', 'hours': 2, 'description': '2h Detailed (Minute-scale)'},
        {'name': 'detailed_minute', 'hours': 6, 'description': '6h Detailed (Minute-scale)'},
        {'name': 'extended_minute', 'hours': 12, 'description': '12h Detailed (Minute-scale)'},
        {'name': 'daily_minute', 'hours': 24, 'description': '24h Detailed (Minute-scale)'},
        {'name': 'multi_day_minute', 'hours': 48, 'description': '48h Detailed (Minute-scale)'}
    ]

    figures = []
    for scale_config in time_scales:
        fig = _create_minute_scale_visualization(
            y_true, y_pred, timestamps, zone, horizon, scale_config,
            save_dir, save, model, feature_names
        )
        if fig is not None:
            figures.append(fig)

    return figures


# ===== 公開API =====

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
