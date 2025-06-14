#!/usr/bin/env python
# coding: utf-8

"""
統合可視化システム (リファクタリング版)
予測精度を視覚的に理解するための最適化されたプロット機能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import r2_score
from scipy import stats
import platform
import warnings

# === フォント設定（完全修正版） ===

def setup_japanese_font_system():
    """日本語フォント設定（絵文字非使用版）"""
    system = platform.system()
    
    # システム別フォント候補（日本語のみ）
    if system == "Darwin":  # macOS
        japanese_fonts = [
            "Hiragino Sans",
            "Hiragino Kaku Gothic Pro", 
            "Hiragino Sans GB",
            "Yu Gothic Medium",
            "BIZ UDGothic",
            "Apple SD Gothic Neo"
        ]
        fallback_fonts = [
            "Arial Unicode MS",
            "Lucida Grande",
            "Geneva",
            "DejaVu Sans"
        ]
    elif system == "Windows":
        japanese_fonts = [
            "Yu Gothic UI",
            "BIZ UDGothic", 
            "Meiryo UI",
            "MS UI Gothic"
        ]
        fallback_fonts = [
            "Arial Unicode MS",
            "Arial",
            "DejaVu Sans"
        ]
    else:  # Linux
        japanese_fonts = [
            "Noto Sans CJK JP",
            "TakaoGothic",
            "IPAexGothic"
        ]
        fallback_fonts = [
            "DejaVu Sans",
            "Liberation Sans"
        ]
    
    # 利用可能フォント検索
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    print(f"[INFO] フォント検索開始 - 日本語候補: {len(japanese_fonts)}")
    
    # 日本語フォント選択
    selected_japanese_font = None
    for font_name in japanese_fonts:
        if font_name in available_fonts:
            selected_japanese_font = font_name
            print(f"[OK] 日本語フォント発見: {selected_japanese_font}")
            break
        else:
            print(f"[SKIP] {font_name} は利用不可")
    
    # フォントリスト構築
    font_list = []
    if selected_japanese_font:
        font_list.append(selected_japanese_font)
    font_list.extend(fallback_fonts)
    
    # 重複削除
    font_list = list(dict.fromkeys(font_list))
    print(f"[OK] フォントリスト構築完了: {font_list[:3]}...")
    
    # matplotlib完全初期化とフォント設定
    plt.rcdefaults()
    
    # シンプル設定（絵文字非対応）
plt.rcParams.update({
        'font.family': ['sans-serif'],
        'font.sans-serif': font_list,
        'axes.unicode_minus': False,
        'figure.autolayout': True,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.format': 'png',
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white'
    })
    
    # 日本語テスト（絵文字非使用）
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, '日本語テスト: 温度予測システム', ha='center', va='center', 
               fontsize=12, fontfamily=font_list[0])
        ax.set_title('フォント表示テスト', fontsize=14, fontfamily=font_list[0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.close(fig)
        print("[OK] 日本語表示テスト成功")
    except Exception as e:
        print(f"[WARNING] 表示テスト失敗: {e}")
    
    # フォントキャッシュクリア
    try:
        fm._get_fontconfig_fonts.cache_clear()
    except:
        pass
        
    primary_font = font_list[0] if font_list else 'DejaVu Sans'
    print(f"[COMPLETE] プライマリフォント設定: {primary_font}")
    
    return primary_font, font_list

# フォント設定実行
JAPANESE_FONT, FONT_LIST = setup_japanese_font_system()

# seaborn設定
sns.set_theme(style="whitegrid", palette="husl")


# ===== ユーティリティ関数 =====

def _safe_text_render(ax, x, y, text, fontsize=12, ha='center', va='center', **kwargs):
    """安全なテキスト描画（絵文字・記号対応）"""
    try:
        # 主要フォントで描画試行
        return ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va, 
                      fontfamily=JAPANESE_FONT, **kwargs)
    except:
        try:
            # フォールバック1: システムフォント
            return ax.text(x, y, text, fontsize=fontsize, ha=ha, va=va, **kwargs)
        except:
            # フォールバック2: ASCII安全バージョン
            safe_text = text.encode('ascii', 'ignore').decode('ascii')
            return ax.text(x, y, safe_text, fontsize=fontsize, ha=ha, va=va, **kwargs)

def _safe_title_render(ax, title, fontsize=14, **kwargs):
    """安全なタイトル描画（絵文字・記号対応）"""
    try:
        return ax.set_title(title, fontsize=fontsize, fontfamily=JAPANESE_FONT, **kwargs)
    except:
        try:
            return ax.set_title(title, fontsize=fontsize, **kwargs)
        except:
            safe_title = title.encode('ascii', 'ignore').decode('ascii')
            return ax.set_title(safe_title, fontsize=fontsize, **kwargs)

def _safe_label_render(ax, xlabel=None, ylabel=None, fontsize=12):
    """安全なラベル描画（絵文字・記号対応）"""
    if xlabel:
        try:
            ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily=JAPANESE_FONT)
        except:
            ax.set_xlabel(xlabel, fontsize=fontsize)
    
    if ylabel:
        try:
            ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily=JAPANESE_FONT)
        except:
            ax.set_ylabel(ylabel, fontsize=fontsize)

def _safe_legend_render(ax, handles=None, labels=None, **kwargs):
    """安全な凡例描画（絵文字・記号対応）"""
    try:
        if handles:
            legend = ax.legend(handles=handles, **kwargs)
        else:
            legend = ax.legend(**kwargs)
        
        # 凡例フォント設定
        for text in legend.get_texts():
            try:
                text.set_fontfamily(JAPANESE_FONT)
            except:
                pass
        
        return legend
    except:
        # フォールバック
        return ax.legend(**kwargs)

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
    ax.grid(True, linestyle='-', alpha=0.3, which='major')
    ax.grid(True, linestyle=':', alpha=0.2, which='minor')


def analyze_lag_dependency(model, feature_names):
    """モデルの特徴量重要度からLAG依存度を分析"""
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"[WARNING] モデルに feature_importances_ 属性がありません")
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


# ===== コアプロット関数（重複除去・最適化済み） =====

def plot_feature_importance(model, feature_names, zone, horizon, save_path=None,
                          top_n=20, model_type="予測", save=True):
    """特徴量重要度プロット（統合版）"""
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"[WARNING] モデルに feature_importances_ 属性がありません")
        return None

    # 重要度分析
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    lag_analysis = analyze_lag_dependency(model, feature_names)

    # プロット作成
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # LAG特徴量の色分け
    colors = []
    for feature in feature_importance_df['feature']:
        if 'lag' in feature.lower() or 'rolling' in feature.lower():
            colors.append('red' if 'temp' in feature.lower() else 'orange')
        else:
            colors.append('skyblue')
    
    bars = ax.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 軸設定
    ax.set_yticks(range(len(feature_importance_df)))
    ax.set_yticklabels(feature_importance_df['feature'], fontsize=10)
    ax.invert_yaxis()

    # LAG情報をタイトルに含める
    total_lag = lag_analysis['total_lag_percent']
    lag_info = f" [LAG依存度: {total_lag:.1f}%]" if total_lag > 0 else ""
    
    # 安全なタイトル・ラベル描画
    _safe_title_render(ax, f'Zone {zone} - 特徴量重要度分析 ({horizon}分予測){lag_info}',
                      fontsize=16, fontweight='bold', pad=20)
    _safe_label_render(ax, xlabel='重要度スコア', ylabel='特徴量', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # 値を表示（安全描画）
    for bar, importance in zip(bars, feature_importance_df['importance']):
        _safe_text_render(ax, bar.get_width() + bar.get_width()*0.01, 
                         bar.get_y() + bar.get_height()/2,
                         f'{importance:.3f}', ha='left', va='center', fontsize=10)

    # 凡例（安全描画）
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='LAG温度特徴量'),
        Patch(facecolor='orange', label='その他LAG特徴量'),
        Patch(facecolor='skyblue', label='通常特徴量')
    ]
    _safe_legend_render(ax, handles=legend_elements, loc='lower right', fontsize=11)

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVE] 特徴量重要度プロット保存: {save_path}")

    return fig


def plot_comprehensive_time_series(y_true, y_pred, timestamps, zone, horizon,
                                  save_path=None, model_type="予測", save=True,
                                  show_period_hours=24, model=None, feature_names=None):
    """包括的時系列プロット（統合版）"""
    # データ検証
    y_true, y_pred, timestamps = _validate_data(y_true, y_pred, timestamps)
    
    if len(timestamps) == 0:
        print(f"⚠️ Zone {zone} のデータがありません")
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

    # 正しい予測タイムスタンプ（未来時刻）
    prediction_timestamps = timestamps_period + pd.Timedelta(minutes=horizon)
    current_time = timestamps_period[-1]

    # プロット作成
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # 実測値プロット
    ax.plot(timestamps_period, y_true_period, 'b-', linewidth=2.5,
            marker='o', markersize=3, markevery=max(1, len(timestamps_period)//100),
            label='実測値', alpha=0.9)

    # 予測値プロット（未来時刻）
    ax.plot(prediction_timestamps, y_pred_period, 'r-', linewidth=2.5,
            marker='s', markersize=2, markevery=max(1, len(prediction_timestamps)//100),
            label=f'予測値 (+{horizon}分)', alpha=0.9)

    # 現在時刻ライン
    ax.axvline(x=current_time, color='green', linestyle='--', 
               linewidth=2, label='現在時刻', alpha=0.8)
    
    # 未来予測領域をハイライト
    future_start = current_time
    future_end = prediction_timestamps[-1]
    ax.axvspan(future_start, future_end, alpha=0.2, color='yellow', 
               label='未来予測領域')

    # LAG依存度分析
    lag_analysis = {'total_lag_percent': 0}
    if model is not None and feature_names is not None:
        lag_analysis = analyze_lag_dependency(model, feature_names)

    # タイトル設定（LAG警告含む）
    total_lag = lag_analysis['total_lag_percent']
    if total_lag > 30:
        lag_info = f' [高LAG依存: {total_lag:.1f}% 警告]'
        title_color = 'darkred'
    elif total_lag > 15:
        lag_info = f' [中LAG依存: {total_lag:.1f}%]'
        title_color = 'darkorange'
    elif total_lag > 0:
        lag_info = f' [低LAG依存: {total_lag:.1f}%]'
        title_color = 'darkgreen'
    else:
        lag_info = ''
        title_color = 'black'

    # 安全なタイトル・ラベル描画
    _safe_title_render(ax, f'Zone {zone} - {model_type}時系列比較 ({horizon}分予測){lag_info}',
                      fontsize=16, fontweight='bold', color=title_color)
    _safe_label_render(ax, xlabel='日時', ylabel='温度 (°C)', fontsize=12)
    _safe_legend_render(ax, fontsize=12, framealpha=0.9)

    # 時間軸設定
    _setup_time_axis(ax, show_period_hours + horizon/60)

    # 統計情報の表示（安全描画）
    if len(y_true_period) > 0 and len(y_pred_period) > 0:
        metrics = _calculate_metrics(y_true_period, y_pred_period)
        stats_text = f'RMSE: {metrics["rmse"]:.3f}°C | MAE: {metrics["mae"]:.3f}°C | R²: {metrics["r2"]:.3f}'
        _safe_text_render(ax, 0.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                         verticalalignment='top', fontsize=11, fontweight='bold', ha='left')

        data_info = f'{len(y_true_period)} データ点 | {show_period_hours}時間'
        _safe_text_render(ax, 0.98, 0.02, data_info, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                         verticalalignment='bottom', ha='right', fontsize=10)

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"[SAVE] 時系列比較プロット保存: {save_path}")

    return fig


def plot_accuracy_analysis(y_true, y_pred, zone, horizon, save_path=None,
                          model_type="予測", save=True):
    """予測精度分析プロット（統合版）"""
    y_true, y_pred = _validate_data(y_true, y_pred)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 散布図
    ax1.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完全予測線')

    # 安全なラベル・タイトル描画
    _safe_label_render(ax1, xlabel='実測温度 (°C)', ylabel='予測温度 (°C)', fontsize=12)
    _safe_title_render(ax1, f'Zone {zone} - 予測精度散布図', fontsize=14, fontweight='bold')
    _safe_legend_render(ax1)
    ax1.grid(True, alpha=0.3)

    # 統計情報
    metrics = _calculate_metrics(y_true, y_pred)
    textstr = f'RMSE: {metrics["rmse"]:.3f}°C\nMAE: {metrics["mae"]:.3f}°C\nR²: {metrics["r2"]:.3f}'
    _safe_text_render(ax1, 0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=11,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), ha='left')

    # 2. 残差プロット
    residuals = y_pred - y_true
    ax2.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    _safe_label_render(ax2, xlabel='予測温度 (°C)', ylabel='残差 (°C)', fontsize=12)
    _safe_title_render(ax2, '残差分析', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 残差分布
    ax3.hist(residuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(residuals), color='blue', linestyle='-', linewidth=2,
               label=f'平均: {np.mean(residuals):.3f}°C')
    _safe_label_render(ax3, xlabel='残差 (°C)', ylabel='頻度', fontsize=12)
    _safe_title_render(ax3, '残差分布', fontsize=14, fontweight='bold')
    _safe_legend_render(ax3)
    ax3.grid(True, alpha=0.3)

    # 4. Q-Qプロット
    stats.probplot(residuals, dist="norm", plot=ax4)
    _safe_title_render(ax4, 'Q-Q プロット（正規性確認）', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 全体タイトル（安全描画）
    try:
        fig.suptitle(f'Zone {zone} - {model_type}精度分析 ({horizon}分予測)', 
                    fontsize=18, fontweight='bold', fontfamily=JAPANESE_FONT)
    except:
        fig.suptitle(f'Zone {zone} - {model_type}精度分析 ({horizon}分予測)', 
                    fontsize=18, fontweight='bold')

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVE] 精度分析プロット保存: {save_path}")

    return fig


def plot_detailed_time_series_analysis(y_true, y_pred, timestamps, zone, horizon,
                                      save_path=None, model_type="予測", save=True,
                                      analysis_hours=2, model=None, feature_names=None):
    """詳細時系列分析プロット（分単位での精密分析）"""
    # データ検証
    y_true, y_pred, timestamps = _validate_data(y_true, y_pred, timestamps)
    
    if len(timestamps) == 0:
        print(f"[WARNING] Zone {zone} のデータがありません")
        return None

    # 分析期間の設定（デフォルト2時間）
    end_time = timestamps[-1]
    start_time = end_time - pd.Timedelta(hours=analysis_hours)
    period_mask = (timestamps >= start_time) & (timestamps <= end_time)
    
    if not period_mask.any():
        max_points = min(len(timestamps), analysis_hours * 60)
        timestamps_period = timestamps[-max_points:]
        y_true_period = y_true[-max_points:]
        y_pred_period = y_pred[-max_points:]
    else:
        timestamps_period = timestamps[period_mask]
        y_true_period = y_true[period_mask]
        y_pred_period = y_pred[period_mask]

    # 正しい予測タイムスタンプ（未来時刻に配置）
    prediction_timestamps = timestamps_period + pd.Timedelta(minutes=horizon)
    current_time = timestamps_period[-1]

    # プロット作成（大きめサイズで詳細表示）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

    # === 上段：詳細時系列比較 ===
    # 実測値プロット（全点表示）
    ax1.plot(timestamps_period, y_true_period, 'b-', linewidth=2,
            marker='o', markersize=4, label='実測値', alpha=0.9)

    # 予測値プロット（未来時刻、全点表示）
    ax1.plot(prediction_timestamps, y_pred_period, 'r-', linewidth=2,
            marker='s', markersize=3, label=f'予測値 (+{horizon}分)', alpha=0.9)

    # 現在時刻ライン
    ax1.axvline(x=current_time, color='green', linestyle='--', 
               linewidth=2, label='現在時刻', alpha=0.8)
    
    # 未来予測領域をハイライト
    future_start = current_time
    future_end = prediction_timestamps[-1]
    ax1.axvspan(future_start, future_end, alpha=0.15, color='yellow', 
               label='未来予測領域')

    # LAG依存度分析
    lag_analysis = {'total_lag_percent': 0}
    if model is not None and feature_names is not None:
        lag_analysis = analyze_lag_dependency(model, feature_names)

    # タイトル設定
    total_lag = lag_analysis['total_lag_percent']
    if total_lag > 30:
        lag_info = f' [高LAG依存: {total_lag:.1f}% 警告]'
        title_color = 'darkred'
    elif total_lag > 15:
        lag_info = f' [中LAG依存: {total_lag:.1f}%]'
        title_color = 'darkorange'
    elif total_lag > 0:
        lag_info = f' [低LAG依存: {total_lag:.1f}%]'
        title_color = 'darkgreen'
    else:
        lag_info = ''
        title_color = 'black'

    _safe_title_render(ax1, f'Zone {zone} - 詳細時系列分析 ({horizon}分予測, {analysis_hours}時間){lag_info}',
                      fontsize=16, fontweight='bold', color=title_color)
    _safe_label_render(ax1, ylabel='温度 (°C)', fontsize=12)
    _safe_legend_render(ax1, fontsize=12, framealpha=0.9)

    # 詳細時間軸設定（分単位）
    if analysis_hours <= 1:
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    elif analysis_hours <= 2:
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    else:
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
        ax1.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.grid(True, linestyle='-', alpha=0.3, which='major')
    ax1.grid(True, linestyle=':', alpha=0.2, which='minor')

    # 統計情報表示
    if len(y_true_period) > 0 and len(y_pred_period) > 0:
        metrics = _calculate_metrics(y_true_period, y_pred_period)
        stats_text = f'RMSE: {metrics["rmse"]:.3f}°C | MAE: {metrics["mae"]:.3f}°C | R²: {metrics["r2"]:.3f}'
        _safe_text_render(ax1, 0.02, 0.98, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                         verticalalignment='top', fontsize=12, fontweight='bold', ha='left')

        data_info = f'{len(y_true_period)} データ点 | 分析期間: {analysis_hours}時間'
        _safe_text_render(ax1, 0.98, 0.02, data_info, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                         verticalalignment='bottom', ha='right', fontsize=11)

    # === 下段：予測誤差の時系列分析 ===
    # 時刻を合わせるため、実測値の時刻を予測時刻に調整して誤差計算
    if len(y_true_period) == len(y_pred_period):
        residuals = y_pred_period - y_true_period
        
        # 誤差の時系列プロット
        ax2.plot(prediction_timestamps, residuals, 'purple', linewidth=2,
                marker='o', markersize=3, label='予測誤差', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # 誤差の統計的境界
        mean_error = np.mean(residuals)
        std_error = np.std(residuals)
        ax2.axhline(y=mean_error, color='red', linestyle='--', linewidth=1, 
                   label=f'平均誤差: {mean_error:.3f}°C')
        ax2.axhline(y=mean_error + 2*std_error, color='orange', linestyle=':', 
                   alpha=0.7, label=f'+2σ: {mean_error + 2*std_error:.3f}°C')
        ax2.axhline(y=mean_error - 2*std_error, color='orange', linestyle=':', 
                   alpha=0.7, label=f'-2σ: {mean_error - 2*std_error:.3f}°C')
        
        # 現在時刻ライン
        ax2.axvline(x=current_time, color='green', linestyle='--', 
                   linewidth=2, alpha=0.8)
        
        _safe_title_render(ax2, '予測誤差の時系列分析', fontsize=14, fontweight='bold')
        _safe_label_render(ax2, xlabel='予測時刻', ylabel='予測誤差 (°C)', fontsize=12)
        _safe_legend_render(ax2, fontsize=11)
        
        # 同じ時間軸設定
        if analysis_hours <= 1:
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        elif analysis_hours <= 2:
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
            ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=2))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        else:
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            ax2.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        
        ax2.tick_params(axis='x', rotation=45, labelsize=10)
        ax2.grid(True, linestyle='-', alpha=0.3, which='major')
        ax2.grid(True, linestyle=':', alpha=0.2, which='minor')
        
        # 誤差統計情報
        error_stats = f'誤差統計 | 平均: {mean_error:.3f}°C | 標準偏差: {std_error:.3f}°C | 範囲: [{np.min(residuals):.3f}, {np.max(residuals):.3f}]°C'
        _safe_text_render(ax2, 0.02, 0.98, error_stats, transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lavender", alpha=0.8),
                         verticalalignment='top', fontsize=10, ha='left')

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVE] 詳細時系列分析保存: {save_path}")

    return fig


def plot_thermostat_control_validation(model, feature_names, test_data, zone, horizon,
                                      save_path=None, model_type="予測", save=True,
                                      is_difference_model=False, current_temp_col=None):
    """サーモON/OFF制御による予測温度変化の検証プロット"""
    
    # 必要な制御列の確認
    ac_valid_col = f'AC_valid_{zone}'
    ac_mode_col = f'AC_mode_{zone}'
    ac_set_col = f'AC_set_{zone}'
    
    required_cols = [ac_valid_col, ac_mode_col, ac_set_col]
    available_cols = [col for col in required_cols if col in test_data.columns]
    
    if len(available_cols) < 2:
        print(f"[WARNING] 制御列が不足: {available_cols}")
        return None
    
    # サンプルデータの準備（最新200点）
    sample_data = test_data.tail(200).copy()
    if len(sample_data) == 0:
        return None
    
    print(f"\n🔬 Zone {zone} - サーモ制御検証実行中...")
    print(f"  検証データ: {len(sample_data)}点")
    print(f"  利用可能制御列: {available_cols}")
    
    # プロット作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # === 1. サーモON vs OFF比較 ===
    if ac_valid_col in sample_data.columns:
        # ベースライン（現在の設定）
        baseline_pred = model.predict(sample_data[feature_names])
        
        # サーモON状態
        test_data_on = sample_data.copy()
        test_data_on[ac_valid_col] = 1  # サーモON
        pred_on = model.predict(test_data_on[feature_names])
        
        # サーモOFF状態
        test_data_off = sample_data.copy()
        test_data_off[ac_valid_col] = 0  # サーモOFF
        pred_off = model.predict(test_data_off[feature_names])
        
        # 差分予測の場合は温度に復元
        if is_difference_model and current_temp_col:
            current_temps = sample_data[current_temp_col]
            baseline_temp = current_temps + baseline_pred
            pred_on_temp = current_temps + pred_on
            pred_off_temp = current_temps + pred_off
        else:
            baseline_temp = baseline_pred
            pred_on_temp = pred_on
            pred_off_temp = pred_off
        
        # 時系列プロット
        time_index = range(len(sample_data))
        ax1.plot(time_index, baseline_temp, 'g-', linewidth=2, label='ベースライン', alpha=0.8)
        ax1.plot(time_index, pred_on_temp, 'r-', linewidth=2, label='サーモON', alpha=0.8)
        ax1.plot(time_index, pred_off_temp, 'b-', linewidth=2, label='サーモOFF', alpha=0.8)
        
        _safe_title_render(ax1, f'Zone {zone} - サーモON/OFF制御検証', fontsize=14, fontweight='bold')
        _safe_label_render(ax1, xlabel='時間ステップ', ylabel='予測温度 (°C)', fontsize=12)
        _safe_legend_render(ax1, fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 統計情報
        temp_diff_on_off = np.mean(pred_on_temp - pred_off_temp)
        temp_diff_on_base = np.mean(pred_on_temp - baseline_temp)
        temp_diff_off_base = np.mean(pred_off_temp - baseline_temp)
        
        stats_text = f'平均温度差:\nON vs OFF: {temp_diff_on_off:+.3f}°C\nON vs Base: {temp_diff_on_base:+.3f}°C\nOFF vs Base: {temp_diff_off_base:+.3f}°C'
        _safe_text_render(ax1, 0.02, 0.98, stats_text, transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                         verticalalignment='top', fontsize=10, ha='left')
    
    # === 2. 冷房 vs 暖房モード比較 ===
    if ac_mode_col in sample_data.columns:
        # 冷房モード
        test_data_cool = sample_data.copy()
        test_data_cool[ac_mode_col] = 0  # 冷房
        if ac_valid_col in test_data_cool.columns:
            test_data_cool[ac_valid_col] = 1  # サーモON
        pred_cool = model.predict(test_data_cool[feature_names])
        
        # 暖房モード
        test_data_heat = sample_data.copy()
        test_data_heat[ac_mode_col] = 1  # 暖房
        if ac_valid_col in test_data_heat.columns:
            test_data_heat[ac_valid_col] = 1  # サーモON
        pred_heat = model.predict(test_data_heat[feature_names])
        
        # 差分予測の場合は温度に復元
        if is_difference_model and current_temp_col:
            current_temps = sample_data[current_temp_col]
            pred_cool_temp = current_temps + pred_cool
            pred_heat_temp = current_temps + pred_heat
    else:
            pred_cool_temp = pred_cool
            pred_heat_temp = pred_heat
        
        # 時系列プロット
        ax2.plot(time_index, pred_cool_temp, 'b-', linewidth=2, label='冷房モード', alpha=0.8)
        ax2.plot(time_index, pred_heat_temp, 'r-', linewidth=2, label='暖房モード', alpha=0.8)
        
        _safe_title_render(ax2, f'Zone {zone} - 冷房/暖房モード比較', fontsize=14, fontweight='bold')
        _safe_label_render(ax2, xlabel='時間ステップ', ylabel='予測温度 (°C)', fontsize=12)
        _safe_legend_render(ax2, fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 統計情報
        temp_diff_heat_cool = np.mean(pred_heat_temp - pred_cool_temp)
        
        stats_text = f'平均温度差:\n暖房 vs 冷房: {temp_diff_heat_cool:+.3f}°C'
        _safe_text_render(ax2, 0.02, 0.98, stats_text, transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                         verticalalignment='top', fontsize=10, ha='left')
    
    # === 3. 設定温度変更の影響 ===
    if ac_set_col in sample_data.columns:
        current_setpoint = sample_data[ac_set_col].mean()
        
        # 高設定温度（+3°C）
        test_data_high = sample_data.copy()
        test_data_high[ac_set_col] = current_setpoint + 3
        if ac_valid_col in test_data_high.columns:
            test_data_high[ac_valid_col] = 1  # サーモON
        pred_high = model.predict(test_data_high[feature_names])
        
        # 低設定温度（-3°C）
        test_data_low = sample_data.copy()
        test_data_low[ac_set_col] = current_setpoint - 3
        if ac_valid_col in test_data_low.columns:
            test_data_low[ac_valid_col] = 1  # サーモON
        pred_low = model.predict(test_data_low[feature_names])
        
        # 現在設定温度
        test_data_current = sample_data.copy()
        if ac_valid_col in test_data_current.columns:
            test_data_current[ac_valid_col] = 1  # サーモON
        pred_current = model.predict(test_data_current[feature_names])
        
        # 差分予測の場合は温度に復元
        if is_difference_model and current_temp_col:
            current_temps = sample_data[current_temp_col]
            pred_high_temp = current_temps + pred_high
            pred_low_temp = current_temps + pred_low
            pred_current_temp = current_temps + pred_current
        else:
            pred_high_temp = pred_high
            pred_low_temp = pred_low
            pred_current_temp = pred_current
        
        # 時系列プロット
        ax3.plot(time_index, pred_low_temp, 'b-', linewidth=2, label=f'低設定 ({current_setpoint-3:.1f}°C)', alpha=0.8)
        ax3.plot(time_index, pred_current_temp, 'g-', linewidth=2, label=f'現在設定 ({current_setpoint:.1f}°C)', alpha=0.8)
        ax3.plot(time_index, pred_high_temp, 'r-', linewidth=2, label=f'高設定 ({current_setpoint+3:.1f}°C)', alpha=0.8)
        
        _safe_title_render(ax3, f'Zone {zone} - 設定温度変更の影響', fontsize=14, fontweight='bold')
        _safe_label_render(ax3, xlabel='時間ステップ', ylabel='予測温度 (°C)', fontsize=12)
        _safe_legend_render(ax3, fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 統計情報
        temp_diff_high_low = np.mean(pred_high_temp - pred_low_temp)
        temp_diff_high_current = np.mean(pred_high_temp - pred_current_temp)
        temp_diff_current_low = np.mean(pred_current_temp - pred_low_temp)
        
        stats_text = f'平均温度差:\n高 vs 低: {temp_diff_high_low:+.3f}°C\n高 vs 現在: {temp_diff_high_current:+.3f}°C\n現在 vs 低: {temp_diff_current_low:+.3f}°C'
        _safe_text_render(ax3, 0.02, 0.98, stats_text, transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                         verticalalignment='top', fontsize=10, ha='left')
    
    # === 4. 制御応答性の分析 ===
    # 制御変更に対する予測温度の応答性を分析
    ax4.axis('off')
    
    # 検証結果のサマリー
    validation_results = []
    
    if ac_valid_col in sample_data.columns:
        on_off_valid = temp_diff_on_off > 0  # サーモONの方が高い温度予測
        validation_results.append(f"サーモON/OFF: {'✅ 妥当' if on_off_valid else '❌ 異常'} ({temp_diff_on_off:+.3f}°C)")
    
    if ac_mode_col in sample_data.columns:
        heat_cool_valid = temp_diff_heat_cool > 0  # 暖房の方が高い温度予測
        validation_results.append(f"暖房/冷房: {'✅ 妥当' if heat_cool_valid else '❌ 異常'} ({temp_diff_heat_cool:+.3f}°C)")
    
    if ac_set_col in sample_data.columns:
        setpoint_valid = temp_diff_high_low > 0  # 高設定の方が高い温度予測
        validation_results.append(f"設定温度: {'✅ 妥当' if setpoint_valid else '❌ 異常'} ({temp_diff_high_low:+.3f}°C)")
    
    # 総合判定
    valid_count = sum(1 for result in validation_results if '✅' in result)
    total_count = len(validation_results)
    overall_validity = valid_count == total_count
    
    summary_text = f"""制御応答性検証結果

{chr(10).join(validation_results)}

総合判定: {'✅ 物理的に妥当' if overall_validity else '⚠️ 要確認'}
妥当性スコア: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)

検証データ: {len(sample_data)}点
予測ホライゾン: {horizon}分
モデルタイプ: {model_type}"""
    
    _safe_text_render(ax4, 0.1, 0.9, summary_text, transform=ax4.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen" if overall_validity else "lightyellow", alpha=0.8),
                     verticalalignment='top', fontsize=12, ha='left', fontfamily='monospace')
    
    # 全体タイトル
    try:
        fig.suptitle(f'Zone {zone} - サーモ制御応答性検証 ({horizon}分予測)', 
                    fontsize=18, fontweight='bold', y=0.95, fontfamily=JAPANESE_FONT)
    except:
        fig.suptitle(f'Zone {zone} - サーモ制御応答性検証 ({horizon}分予測)', 
                    fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVE] サーモ制御検証プロット保存: {save_path}")
    
    return fig


def create_optimized_visualization_report(model, feature_names, y_true, y_pred,
                                        timestamps, metrics, zone, horizon,
                                        model_type="予測", save_dir="Output/visualizations"):
    """最適化された可視化レポート（制御検証追加版）"""
    os.makedirs(save_dir, exist_ok=True)
    created_files = {}

    print(f"[INFO] Zone {zone} の可視化レポートを作成中...")

    # 1. 特徴量重要度分析
    save_path = os.path.join(save_dir, f"feature_importance_zone_{zone}_horizon_{horizon}.png")
    fig = plot_feature_importance(model, feature_names, zone, horizon, 
                                 save_path=save_path, model_type=model_type)
    if fig:
        created_files['feature_importance'] = save_path
        plt.close(fig)

    # 2. 包括的時系列分析
    save_path = os.path.join(save_dir, f"time_series_zone_{zone}_horizon_{horizon}.png")
    fig = plot_comprehensive_time_series(y_true, y_pred, timestamps, zone, horizon,
                                        save_path=save_path, model_type=model_type,
                                        model=model, feature_names=feature_names)
    if fig:
        created_files['time_series'] = save_path
        plt.close(fig)

    # 3. 精度分析
    save_path = os.path.join(save_dir, f"accuracy_analysis_zone_{zone}_horizon_{horizon}.png")
    fig = plot_accuracy_analysis(y_true, y_pred, zone, horizon,
                                save_path=save_path, model_type=model_type)
    if fig:
        created_files['accuracy_analysis'] = save_path
        plt.close(fig)

    # 4. 詳細時系列分析
    save_path = os.path.join(save_dir, f"detailed_time_series_zone_{zone}_horizon_{horizon}.png")
    fig = plot_detailed_time_series_analysis(y_true, y_pred, timestamps, zone, horizon,
                                            save_path=save_path, model_type=model_type,
                                            analysis_hours=2, model=model, feature_names=feature_names)
    if fig:
        created_files['detailed_time_series'] = save_path
        plt.close(fig)

    print(f"[OK] Zone {zone} 可視化レポート完了:")
    for viz_type, path in created_files.items():
        print(f"  [SAVE] {viz_type}: {path}")

    return created_files


# ===== 比較分析関数 =====

def plot_method_comparison(direct_metrics, diff_metrics, zone, horizon,
                          save_path=None, save=True):
    """予測手法比較プロット"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    common_metrics = ['rmse', 'mae', 'r2']
    direct_values = [direct_metrics.get(metric, 0) for metric in common_metrics]
    diff_values = [diff_metrics.get('restored_' + metric, diff_metrics.get(metric, 0)) for metric in common_metrics]

    x = np.arange(len(common_metrics))
    width = 0.35

    # バー比較
    bars1 = ax1.bar(x - width/2, direct_values, width, label='直接予測', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, diff_values, width, label='差分予測', alpha=0.8, color='lightcoral')

    _safe_label_render(ax1, xlabel='評価指標', ylabel='値', fontsize=14)
    _safe_title_render(ax1, f'Zone {zone} - 予測手法比較 ({horizon}分予測)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in common_metrics])
    _safe_legend_render(ax1)
    ax1.grid(True, alpha=0.3)

    # 値を表示（安全描画）
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            _safe_text_render(ax1, bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{height:.3f}', fontsize=10, va='bottom')

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

    _safe_label_render(ax2, xlabel='評価指標', ylabel='改善率 (%)', fontsize=14)
    _safe_title_render(ax2, '性能改善率', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3)

    # 改善率の値を表示（安全描画）
    for bar, improvement in zip(bars3, improvements):
        height = bar.get_height()
        _safe_text_render(ax2, bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                        f'{improvement:.1f}%', fontsize=11, fontweight='bold',
                        va='bottom' if height > 0 else 'top')

    plt.tight_layout()

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[SAVE] 手法比較プロット保存: {save_path}")

    return fig


# ===== 公開API（整理済み） =====

__all__ = [
    # コアプロット関数
    'plot_feature_importance',
    'plot_comprehensive_time_series', 
    'plot_accuracy_analysis',
    'plot_detailed_time_series_analysis',
    'plot_thermostat_control_validation',
    'plot_method_comparison',
    
    # レポート生成
    'create_optimized_visualization_report',
    
    # ユーティリティ
    'analyze_lag_dependency'
]
