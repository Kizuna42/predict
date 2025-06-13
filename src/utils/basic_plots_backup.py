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
    """完全な日本語・絵文字・記号対応フォント設定（最終版）"""
    system = platform.system()
    
    # システム別フォント候補（絵文字・記号対応を優先）
    if system == "Darwin":  # macOS
        # 日本語フォント候補
        japanese_fonts = [
            "Hiragino Sans",
            "Hiragino Kaku Gothic Pro", 
            "Hiragino Sans GB",
            "Yu Gothic Medium",
            "BIZ UDGothic",
            "Apple SD Gothic Neo"
        ]
        # 絵文字・記号対応フォント
        emoji_fonts = [
            "Apple Color Emoji",
            "Apple Symbols", 
            "Symbol"
        ]
        # フォールバックフォント
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
        emoji_fonts = [
            "Segoe UI Emoji",
            "Segoe UI Symbol"
        ]
        fallback_fonts = [
            "Arial Unicode MS",
            "Arial",
            "DejaVu Sans"
        ]
    else:  # Linux
        japanese_fonts = [
            "Noto Sans CJK JP",
            "Noto Color Emoji",
            "TakaoGothic",
            "IPAexGothic"
        ]
        emoji_fonts = [
            "Noto Color Emoji",
            "Symbola"
        ]
        fallback_fonts = [
            "DejaVu Sans",
            "Liberation Sans"
        ]
    
    # 利用可能フォント検索
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    print(f"🔍 フォント検索開始...")
    print(f"   日本語候補: {len(japanese_fonts)}, 絵文字候補: {len(emoji_fonts)}")
    
    # 日本語フォント選択
    selected_japanese_font = None
    for font_name in japanese_fonts:
        if font_name in available_fonts:
            selected_japanese_font = font_name
            print(f"✅ 日本語フォント発見: {selected_japanese_font}")
            break
        else:
            print(f"❌ {font_name} は利用不可")
    
    # 絵文字フォント選択
    selected_emoji_font = None
    for font_name in emoji_fonts:
        if font_name in available_fonts:
            selected_emoji_font = font_name
            print(f"✅ 絵文字フォント発見: {selected_emoji_font}")
            break
    
    # フォントリスト構築
    font_list = []
    if selected_japanese_font:
        font_list.append(selected_japanese_font)
    if selected_emoji_font and selected_emoji_font != selected_japanese_font:
        font_list.append(selected_emoji_font)
    font_list.extend(fallback_fonts)
    
    # 重複削除
    font_list = list(dict.fromkeys(font_list))
    
    print(f"✅ 最終フォントリスト: {font_list[:3]}...")
    
    # matplotlib完全初期化とフォント設定
    plt.rcdefaults()
    
    # 包括的matplotlib設定
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
    
    # 包括的表示テスト
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # 日本語テスト
        ax.text(0.5, 0.8, '日本語テスト: 温度予測', ha='center', va='center', 
               fontsize=12, fontfamily=font_list[0])
        
        # 絵文字・記号テスト
        emoji_text = '🌡️📊🎯✅❌⚠️°℃'
        ax.text(0.5, 0.5, f'絵文字テスト: {emoji_text}', ha='center', va='center', 
               fontsize=12, fontfamily=font_list[0])
        
        # 数学記号テスト
        math_text = 'R² MAE RMSE ±'
        ax.text(0.5, 0.2, f'記号テスト: {math_text}', ha='center', va='center', 
               fontsize=12, fontfamily=font_list[0])
        
        ax.set_title('🎨 フォント包括テスト', fontsize=14, fontfamily=font_list[0])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.close(fig)
        print("✅ 包括的表示テスト成功")
        
    except Exception as e:
        print(f"⚠️ 表示テスト失敗: {e}")
    
    # フォントキャッシュクリア（必要に応じて）
    try:
        fm._get_fontconfig_fonts.cache_clear()
    except:
        pass
        
    primary_font = font_list[0] if font_list else 'DejaVu Sans'
    print(f"🎨 プライマリフォント設定完了: {primary_font}")
    
    return primary_font, font_list

# フォント設定実行
JAPANESE_FONT, FONT_LIST = setup_japanese_font_system()

# seaborn設定
sns.set_theme(style="whitegrid", palette="husl")


# ===== ユーティリティ関数 =====

def _ascii_safe_convert(text):
    """絵文字・特殊文字・日本語をASCII安全文字に変換"""
    
    # 日本語→英語変換マップ
    japanese_map = {
        '復元温度モデル信頼性ダッシュボード': 'Restored Temperature Model Reliability Dashboard',
        'モデル信頼性ダッシュボード': 'Model Reliability Dashboard', 
        '総合スコア': 'Overall Score',
        '優秀': 'Excellent',
        '良好': 'Good', 
        '要改善': 'Needs Improvement',
        '詳細メトリクス': 'Detailed Metrics',
        'データ点数': 'Data Points',
        '依存度': 'Dependency',
        '最新24時間の予測性能': 'Latest 24h Prediction Performance',
        '信頼区間': 'Confidence Interval',
        '誤差分布': 'Error Distribution',
        '重要特徴量': 'Important Features',
        '特徴量重要度': 'Feature Importance',
        '取得不可': 'Unavailable',
        '情報なし': 'No Info',
        '平均誤差': 'Mean Error',
        '予測誤差': 'Prediction Error',
        '頻度': 'Frequency',
        '重要度': 'Importance',
        '温度': 'Temperature',
        '実測値': 'Actual',
        '予測値': 'Predicted',
        '現在時刻': 'Current Time',
        '未来予測領域': 'Future Prediction Area',
        '日時': 'DateTime',
        '特徴量重要度分析': 'Feature Importance Analysis',
        '時系列比較': 'Time Series Comparison',
        '分予測': 'min Prediction',
        '精度分析': 'Accuracy Analysis',
        '予測精度散布図': 'Prediction Accuracy Scatter',
        '残差分析': 'Residual Analysis',
        '残差分布': 'Residual Distribution',
        '正規性確認': 'Normality Check',
        '実測温度': 'Actual Temperature',
        '予測温度': 'Predicted Temperature',
        '残差': 'Residual',
        '完全予測線': 'Perfect Prediction Line',
        '平均': 'Mean'
    }
    
    # 絵文字→ASCII変換マップ
    emoji_map = {
        '🌡️': '[TEMP]',
        '📊': '[CHART]', 
        '🎯': '[TARGET]',
        '✅': '[OK]',
        '❌': '[NG]',
        '⚠️': '[WARN]',
        '🟢': '[GREEN]',
        '🟡': '[YELLOW]', 
        '🔴': '[RED]',
        '📈': '[TREND]',
        '🔝': '[TOP]',
        '°': 'deg',
        '℃': 'C',
        '²': '2',
        '±': '+/-'
    }
    
    result = str(text)
    
    # 日本語変換
    for japanese, english in japanese_map.items():
        result = result.replace(japanese, english)
    
    # 絵文字変換
    for emoji, replacement in emoji_map.items():
        result = result.replace(emoji, replacement)
    
    # 残った非ASCII文字を除去
    try:
        result = result.encode('ascii', 'ignore').decode('ascii')
    except:
        pass
    
    return result

def _safe_text_render(ax, x, y, text, fontsize=12, ha='center', va='center', **kwargs):
    """安全なテキスト描画（完全ASCII対応）"""
    try:
        # ASCII安全変換
        safe_text = _ascii_safe_convert(str(text))
        return ax.text(x, y, safe_text, fontsize=fontsize, ha=ha, va=va, 
                      fontfamily='Arial', **kwargs)
    except Exception as e:
        # 最終フォールバック
        fallback_text = str(text).encode('ascii', 'ignore').decode('ascii')
        return ax.text(x, y, fallback_text, fontsize=fontsize, ha=ha, va=va, **kwargs)

def _safe_title_render(ax, title, fontsize=14, **kwargs):
    """安全なタイトル描画（完全ASCII対応）"""
    try:
        safe_title = _ascii_safe_convert(str(title))
        return ax.set_title(safe_title, fontsize=fontsize, fontfamily='Arial', **kwargs)
    except Exception as e:
        fallback_title = str(title).encode('ascii', 'ignore').decode('ascii')
        return ax.set_title(fallback_title, fontsize=fontsize, **kwargs)

def _safe_label_render(ax, xlabel=None, ylabel=None, fontsize=12):
    """安全なラベル描画（完全ASCII対応）"""
    if xlabel:
        try:
            safe_xlabel = _ascii_safe_convert(str(xlabel))
            ax.set_xlabel(safe_xlabel, fontsize=fontsize, fontfamily='Arial')
        except:
            ax.set_xlabel(str(xlabel).encode('ascii', 'ignore').decode('ascii'), fontsize=fontsize)
    
    if ylabel:
        try:
            safe_ylabel = _ascii_safe_convert(str(ylabel))
            ax.set_ylabel(safe_ylabel, fontsize=fontsize, fontfamily='Arial')
        except:
            ax.set_ylabel(str(ylabel).encode('ascii', 'ignore').decode('ascii'), fontsize=fontsize)

def _safe_legend_render(ax, handles=None, labels=None, **kwargs):
    """安全な凡例描画（完全ASCII対応）"""
    try:
        if handles:
            legend = ax.legend(handles=handles, **kwargs)
        else:
            legend = ax.legend(**kwargs)
        
        # 凡例テキストをASCII安全に変換
        for text in legend.get_texts():
            try:
                original_text = text.get_text()
                safe_text = _ascii_safe_convert(original_text)
                text.set_text(safe_text)
                text.set_fontfamily('Arial')
            except:
                pass
        
        return legend
    except:
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
        print(f"⚠️ モデルに feature_importances_ 属性がありません")
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
        print(f"📊 特徴量重要度プロット保存: {save_path}")

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
        lag_info = f' [高LAG依存: {total_lag:.1f}%⚠️]'
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
        print(f"📈 時系列比較プロット保存: {save_path}")

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
        print(f"📊 精度分析プロット保存: {save_path}")

    return fig


def plot_model_reliability_dashboard(y_true, y_pred, timestamps, zone, horizon,
                                   model=None, feature_names=None,
                                   save_path=None, model_type="予測", save=True):
    """モデル信頼性ダッシュボード（意思決定者向け）"""
    # データ検証
    y_true, y_pred, timestamps = _validate_data(y_true, y_pred, timestamps)
    
    if len(timestamps) == 0:
        print(f"⚠️ Zone {zone} のデータがありません")
        return None

    # メトリクス計算
    metrics = _calculate_metrics(y_true, y_pred)
    
    # LAG依存度分析
    lag_analysis = {'total_lag_percent': 0}
    if model is not None and feature_names is not None:
        lag_analysis = analyze_lag_dependency(model, feature_names)
    
    # プロット作成
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # === 1. 総合スコア表示 ===
    ax_score = fig.add_subplot(gs[0, :2])
    ax_score.axis('off')
    
    # 信頼性スコア計算
    r2_score = max(0, metrics['r2'])
    mae_score = max(0, 1 - metrics['mae'] / 5.0)
    lag_score = max(0, 1 - lag_analysis['total_lag_percent'] / 50.0)
    
    overall_score = (r2_score * 0.5 + mae_score * 0.3 + lag_score * 0.2) * 100
    
    # スコア色分け
    if overall_score >= 80:
        score_color = 'green'
        score_status = '優秀'
        score_emoji = '🟢'
    elif overall_score >= 60:
        score_color = 'orange'
        score_status = '良好'
        score_emoji = '🟡'
    else:
        score_color = 'red'
        score_status = '要改善'
        score_emoji = '🔴'
    
    # 大きなスコア表示（安全描画）
    _safe_text_render(ax_score, 0.5, 0.7, f'{overall_score:.0f}', 
                     fontsize=80, fontweight='bold', color=score_color, 
                     transform=ax_score.transAxes)
    _safe_text_render(ax_score, 0.5, 0.3, f'{score_emoji} 総合スコア ({score_status})', 
                     fontsize=16, fontweight='bold', transform=ax_score.transAxes)
    _safe_text_render(ax_score, 0.5, 0.1, f'Zone {zone} - {horizon}分予測', 
                     fontsize=14, transform=ax_score.transAxes)
    
    # === 2. 詳細メトリクス ===
    ax_metrics = fig.add_subplot(gs[0, 2:])
    ax_metrics.axis('off')
    
    metrics_text = f"""📊 詳細メトリクス
├─ R² Score:     {metrics['r2']:.3f}
├─ RMSE:         {metrics['rmse']:.3f}°C
├─ MAE:          {metrics['mae']:.3f}°C
├─ LAG依存度:     {lag_analysis['total_lag_percent']:.1f}%
└─ データ点数:    {len(y_true):,}"""
    
    _safe_text_render(ax_metrics, 0.1, 0.9, metrics_text, fontsize=14, 
                     fontfamily='monospace', transform=ax_metrics.transAxes, ha='left', va='top')
    
    # === 3. 時系列比較 ===
    ax_ts = fig.add_subplot(gs[1, :])
    
    # 最新24時間のデータ
    show_hours = 24
    end_time = timestamps[-1]
    start_time = end_time - pd.Timedelta(hours=show_hours)
    period_mask = (timestamps >= start_time) & (timestamps <= end_time)
    
    if period_mask.any():
        ts_period = timestamps[period_mask]
        y_true_period = y_true[period_mask]
        y_pred_period = y_pred[period_mask]
    else:
        max_points = min(len(timestamps), show_hours * 60)
        ts_period = timestamps[-max_points:]
        y_true_period = y_true[-max_points:]
        y_pred_period = y_pred[-max_points:]
    
    # 予測タイムスタンプ（未来）
    pred_timestamps = ts_period + pd.Timedelta(minutes=horizon)
    
    # プロット
    ax_ts.plot(ts_period, y_true_period, 'b-', linewidth=3, 
              label='実測値', alpha=0.9, marker='o', markersize=2)
    ax_ts.plot(pred_timestamps, y_pred_period, 'r-', linewidth=2.5,
              label=f'予測値（+{horizon}分）', alpha=0.8, marker='s', markersize=2)
    
    # 信頼区間
    residuals = y_pred_period - y_true_period
    std_residual = np.std(residuals)
    upper_bound = y_pred_period + 2 * std_residual
    lower_bound = y_pred_period - 2 * std_residual
    
    ax_ts.fill_between(pred_timestamps, lower_bound, upper_bound, 
                      alpha=0.2, color='red', label='95%信頼区間')
    
    _safe_title_render(ax_ts, '📈 最新24時間の予測性能', fontsize=16, fontweight='bold')
    _safe_label_render(ax_ts, ylabel='温度 (°C)', fontsize=12)
    _safe_legend_render(ax_ts, fontsize=12)
    ax_ts.grid(True, alpha=0.3)
    _setup_time_axis(ax_ts, show_hours + horizon/60)
    
    # === 4. 誤差分布 ===
    ax_error = fig.add_subplot(gs[2, :2])
    
    residuals = y_pred - y_true
    ax_error.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax_error.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax_error.axvline(x=np.mean(residuals), color='orange', linestyle='-', linewidth=2,
                    label=f'平均誤差: {np.mean(residuals):.3f}°C')
    
    _safe_title_render(ax_error, '📊 誤差分布', fontsize=14, fontweight='bold')
    _safe_label_render(ax_error, xlabel='予測誤差 (°C)', ylabel='頻度', fontsize=12)
    _safe_legend_render(ax_error, fontsize=10)
    ax_error.grid(True, alpha=0.3)
    
    # === 5. 特徴量重要度トップ5 ===
    ax_feat = fig.add_subplot(gs[2, 2:])
    
    if model is not None and feature_names is not None:
        try:
            importances = model.feature_importances_
            feat_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(5)
            
            bars = ax_feat.barh(range(len(feat_df)), feat_df['importance'])
            ax_feat.set_yticks(range(len(feat_df)))
            ax_feat.set_yticklabels(feat_df['feature'], fontsize=10)
            ax_feat.invert_yaxis()
            
            # LAG特徴量を色分け
            colors = ['red' if 'lag' in feat.lower() or 'rolling' in feat.lower() 
                     else 'skyblue' for feat in feat_df['feature']]
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            _safe_title_render(ax_feat, '🔝 重要特徴量 Top5', fontsize=14, fontweight='bold')
            _safe_label_render(ax_feat, xlabel='重要度', fontsize=12)
            ax_feat.grid(True, alpha=0.3, axis='x')
            
        except:
            _safe_text_render(ax_feat, 0.5, 0.5, '特徴量重要度\n取得不可',
                            fontsize=14, transform=ax_feat.transAxes)
            ax_feat.axis('off')
    else:
        _safe_text_render(ax_feat, 0.5, 0.5, '特徴量重要度\n情報なし',
                        fontsize=14, transform=ax_feat.transAxes)
        ax_feat.axis('off')
    
    # 全体タイトル（安全描画）
    try:
        fig.suptitle(f'🎯 {model_type}モデル信頼性ダッシュボード', 
                    fontsize=24, fontweight='bold', y=0.95, fontfamily=JAPANESE_FONT)
    except:
        fig.suptitle(f'🎯 {model_type}モデル信頼性ダッシュボード', 
                    fontsize=24, fontweight='bold', y=0.95)

    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"🎯 モデル信頼性ダッシュボード保存: {save_path}")

    return fig


# ===== 統合レポート生成関数 =====

def create_optimized_visualization_report(model, feature_names, y_true, y_pred,
                                        timestamps, metrics, zone, horizon,
                                        model_type="予測", save_dir="Output/visualizations"):
    """最適化された可視化レポート（重複除去版）"""
    os.makedirs(save_dir, exist_ok=True)
    created_files = {}

    print(f"🎨 Zone {zone} の最適化可視化レポートを作成中...")

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

    # 4. 信頼性ダッシュボード
    save_path = os.path.join(save_dir, f"reliability_dashboard_zone_{zone}_horizon_{horizon}.png")
    fig = plot_model_reliability_dashboard(y_true, y_pred, timestamps, zone, horizon,
                                          model=model, feature_names=feature_names,
                                          save_path=save_path, model_type=model_type)
    if fig:
        created_files['reliability_dashboard'] = save_path
        plt.close(fig)

    print(f"✅ Zone {zone} 可視化レポート完了:")
    for viz_type, path in created_files.items():
        print(f"  📊 {viz_type}: {path}")

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
        print(f"📊 手法比較プロット保存: {save_path}")

    return fig


# ===== 公開API（整理済み） =====

__all__ = [
    # コアプロット関数
    'plot_feature_importance',
    'plot_comprehensive_time_series', 
    'plot_accuracy_analysis',
    'plot_model_reliability_dashboard',
    'plot_method_comparison',
    
    # レポート生成
    'create_optimized_visualization_report',
    
    # ユーティリティ
    'analyze_lag_dependency'
]
