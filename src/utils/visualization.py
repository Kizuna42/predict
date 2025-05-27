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
import matplotlib.font_manager as fm
import math
import os
from src.config import OUTPUT_DIR
from sklearn.metrics import r2_score
import warnings


# 日本語フォント設定の強化
def setup_japanese_font():
    """
    日本語フォントの設定を行う（強化版）
    """
    # 利用可能な日本語フォントを検索（優先順位付き）
    japanese_fonts = [
        'Hiragino Sans',
        'Hiragino Kaku Gothic Pro',
        'Yu Gothic',
        'Meiryo',
        'IPAexGothic',
        'IPAGothic',
        'MS Gothic',
        'Takao Gothic',
        'VL Gothic',
        'Noto Sans CJK JP'
    ]

    # システムで利用可能なフォント一覧を取得
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 利用可能な日本語フォントを見つける
    selected_font = None
    for font in japanese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    # フォント設定を強制的に適用
    if selected_font:
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['font.sans-serif'] = [selected_font] + japanese_fonts
        print(f"日本語フォント設定: {selected_font}")
    else:
        # macOSの場合のフォールバック
        import platform
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Hiragino Sans']
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
            print("macOS用フォント設定: Hiragino Sans")
        else:
            # その他のシステム用フォールバック
            plt.rcParams['font.family'] = ['DejaVu Sans']
            plt.rcParams['font.sans-serif'] = japanese_fonts + ['DejaVu Sans', 'Arial']
            print("フォールバック設定を使用")

    # 重要な設定を強制適用
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True

    # 設定確認のためのテスト
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, 'テスト', fontsize=12)
        plt.close(fig)
        print("✅ 日本語フォント設定テスト成功")
    except Exception as e:
        print(f"⚠️ フォント設定テストで警告: {e}")

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
    Plot feature importance

    Parameters:
    -----------
    feature_importance : DataFrame
        DataFrame containing features and their importance values
    zone : int
        Zone number
    horizon : int
        Prediction horizon (minutes)
    save_dir : str, optional
        Directory to save the graph
    top_n : int, optional
        Number of features to display
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Sort by importance in descending order
    importance_sorted = feature_importance.sort_values('importance', ascending=False)

    # Extract top N features
    top_features = importance_sorted.head(top_n)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax)

    ax.set_title(f'Zone {zone} - Feature Importance for {horizon}-min Prediction (Top {top_n})', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    # Save graph
    if save and save_dir:
        output_path = os.path.join(save_dir, f'feature_importance_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Feature importance graph saved: {output_path}")

    return fig


def plot_scatter_actual_vs_predicted(actual, predicted, zone, horizon, save_dir=None, save=True):
    """
    Generate scatter plot of actual vs predicted values

    Parameters:
    -----------
    actual : Series
        Actual values
    predicted : array-like
        Predicted values
    zone : int
        Zone number
    horizon : int
        Prediction horizon (minutes)
    save_dir : str, optional
        Directory to save the graph
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out NaN values
    valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
    actual_valid = actual[valid_indices]
    predicted_valid = predicted[valid_indices]

    # Scatter plot
    ax.scatter(actual_valid, predicted_valid, alpha=0.5)

    # Add ideal prediction line (y=x)
    min_val = min(actual_valid.min(), predicted_valid.min())
    max_val = max(actual_valid.max(), predicted_valid.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Axis labels and title
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'Zone {zone} - Temperature Prediction for {horizon}-min Horizon (Scatter Plot)', fontsize=14)

    # Calculate and display R²
    r2 = r2_score(actual_valid, predicted_valid)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=12)

    # Grid lines
    ax.grid(True, linestyle='--', alpha=0.6)

    # Save graph
    if save and save_dir:
        output_path = os.path.join(save_dir, f'scatter_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Scatter plot saved: {output_path}")

    return fig


def plot_scatter_actual_vs_predicted_by_horizon(results_dict, horizon, save_dir=None, save=True):
    """
    Generate subplots of scatter plots for all zones at a specific horizon

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each zone
    horizon : int
        Prediction horizon (minutes)
    save_dir : str, optional
        Directory to save the graph
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Collect zones with data for the prediction horizon
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"Warning: No data available for {horizon}-min prediction. Skipping.")
        return None

    # Calculate number of rows and columns for subplots
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)  # Limit to maximum 3 columns
    n_rows = math.ceil(n_zones / n_cols)

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(sorted(zones_with_data)):
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        # Helper function to get data from results
        def get_data_from_results(results, key_pairs):
            """Helper function to retrieve data from results dictionary based on key pairs"""
            for actual_key, pred_key in key_pairs:
                if actual_key in results and pred_key in results:
                    return results[actual_key], results[pred_key]
            return None, None

        # Try different key combinations
        key_pairs = [
            ('test_y', 'test_predictions'),
            ('y_test', 'y_pred'),
            ('actual', 'predicted'),
            ('train_y', 'train_predictions'),
            ('y_train', 'y_train_pred')
        ]

        actual, predicted = get_data_from_results(horizon_results, key_pairs)

        if actual is None or predicted is None:
            print(f"Zone {zone}, Horizon {horizon}: Data not found. Available keys: {list(horizon_results.keys())}")
            axs[i].text(0.5, 0.5, 'No data',
                      ha='center', va='center', transform=axs[i].transAxes)
            continue

        # Filter out NaN values and infinities
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
            print(f"Error processing data for Zone {zone}: {e}")
            axs[i].text(0.5, 0.5, f'Data processing error: {str(e)[:30]}...',
                      ha='center', va='center', transform=axs[i].transAxes)
            continue

        if len(actual_valid) == 0:
            axs[i].text(0.5, 0.5, 'No valid data',
                      ha='center', va='center', transform=axs[i].transAxes)
            continue

        # Scatter plot
        axs[i].scatter(actual_valid, predicted_valid, alpha=0.5)

        # Add ideal prediction line (y=x)
        try:
            min_val = min(np.min(actual_valid), np.min(predicted_valid))
            max_val = max(np.max(actual_valid), np.max(predicted_valid))
            axs[i].plot([min_val, max_val], [min_val, max_val], 'r--')

            # Calculate and display R²
            r2 = r2_score(actual_valid, predicted_valid)
            axs[i].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axs[i].transAxes,
                      verticalalignment='top', fontsize=12)
        except Exception as e:
            print(f"Error calculating R² for zone {zone}: {e}")
            axs[i].text(0.05, 0.95, "R² calculation error", transform=axs[i].transAxes,
                      verticalalignment='top', fontsize=12)

        # Title and grid
        axs[i].set_title(f'Zone {zone}')
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Predicted')
        axs[i].grid(True, linestyle='--', alpha=0.6)

    # Hide unused subplots
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # Overall title
    fig.suptitle(f'Actual vs Predicted for {horizon}-min Horizon', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save graph
    if save and save_dir:
        output_path = os.path.join(save_dir, f'scatter_all_zones_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Scatter plot for all zones saved: {output_path}")

    return fig


def plot_time_series(timestamps, actual, predicted, zone, horizon, save_dir=None, points=100, save=True):
    """
    Create time series plot of temperature data and predictions

    Parameters:
    -----------
    timestamps : array-like
        Timestamps for the time series
    actual : Series
        Actual values
    predicted : array-like
        Predicted values
    zone : int
        Zone number
    horizon : int
        Prediction horizon (minutes)
    save_dir : str, optional
        Directory to save the graph
    points : int, optional
        Maximum number of data points to plot
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Filter out NaN values
    valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
    timestamps_valid = timestamps[valid_indices]
    actual_valid = actual[valid_indices]
    predicted_valid = predicted[valid_indices]

    # Sample data if there are too many points
    sample_size = min(len(timestamps_valid), points)
    if len(timestamps_valid) > sample_size:
        # Sample from endpoints
        step = len(timestamps_valid) // sample_size
        indices = list(range(0, len(timestamps_valid), step))[:sample_size]

        # Sample data
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

    # Create time series plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual and predicted values
    ax.plot(timestamps_sample, actual_sample, 'b-', label='Actual')
    ax.plot(timestamps_sample, predicted_sample, 'r--', label='Predicted')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    plt.xticks(rotation=45)

    # Axis labels and title
    ax.set_xlabel('Date/Time', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Zone {zone} - Temperature Prediction for {horizon}-min Horizon (Time Series)', fontsize=14)

    # Grid and legend
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    # Adjust layout
    plt.tight_layout()

    # Save graph
    if save and save_dir:
        output_path = os.path.join(save_dir, f'timeseries_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Time series plot saved: {output_path}")

    return fig


def plot_time_series_by_horizon(results_dict, horizon, save_dir=None, points=100, save=True):
    """
    Generate subplots of time series data for all zones at a specific horizon

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each zone
    horizon : int
        Prediction horizon (minutes)
    save_dir : str, optional
        Directory to save the graph
    points : int, optional
        Maximum number of data points to plot
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Collect zones with data for the prediction horizon
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"Warning: No data available for {horizon}-min prediction. Skipping.")
        return None

    # Calculate number of rows and columns for subplots
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)  # Limit to maximum 3 columns
    n_rows = math.ceil(n_zones / n_cols)

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(sorted(zones_with_data)):
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        # Try different methods to get time series data
        timestamps = None
        actual = None
        predicted = None

        # Method 1: Use test_data, test_y, test_predictions
        if all(k in horizon_results for k in ['test_data', 'test_y', 'test_predictions']):
            test_df = horizon_results['test_data']
            if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
                timestamps = test_df.index
                actual = horizon_results['test_y']
                predicted = horizon_results['test_predictions']
                print(f"Zone {zone}: データ取得成功 - timestamps: {len(timestamps)}, actual: {len(actual)}, predicted: {len(predicted)}")

        # Method 2: Use y_test, y_pred and find DataFrame with time series index
        elif all(k in horizon_results for k in ['y_test', 'y_pred']):
            actual = horizon_results['y_test']
            predicted = horizon_results['y_pred']

            # Look for DataFrame with datetime index
            for key, value in horizon_results.items():
                if isinstance(value, pd.DataFrame) and hasattr(value, 'index') and isinstance(value.index, pd.DatetimeIndex):
                    timestamps = value.index
                    break

        # Method 3: Look for timestamp keys
        if timestamps is None:
            for key in ['test_timestamps', 'timestamps', 'time_index', 'date_index']:
                if key in horizon_results:
                    timestamps = horizon_results[key]
                    break

        # Generate dummy timestamps if none found
        if timestamps is None:
            if actual is not None:
                length = len(actual)
                timestamps = pd.date_range(start='2023-01-01', periods=length, freq='1min')
                print(f"No time index found for Zone {zone}, generating dummy index")
            else:
                # No data found
                print(f"No data found for Zone {zone}, Horizon {horizon}")
                axs[i].text(0.5, 0.5, 'No data',
                           ha='center', va='center', transform=axs[i].transAxes)
                continue

        # Check for missing actual/predicted data
        if actual is None or predicted is None:
            # Try different key patterns
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
            print(f"Data keys not found for Zone {zone}, Horizon {horizon}. Available keys: {list(horizon_results.keys())}")
            axs[i].text(0.5, 0.5, 'No data',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        # Check and fix length mismatches
        try:
            min_length = min(len(timestamps), len(actual), len(predicted))
            if len(timestamps) > min_length:
                timestamps = timestamps[:min_length]
            if len(actual) > min_length:
                actual = actual[:min_length]
            if len(predicted) > min_length:
                predicted = predicted[:min_length]
        except Exception as e:
            print(f"Error adjusting data lengths for Zone {zone}: {e}")

        # Filter out NaN values and invalid values
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
            print(f"Error processing data for Zone {zone}: {e}")
            axs[i].text(0.5, 0.5, f'Data processing error: {str(e)[:30]}...',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        if len(actual_valid) == 0 or len(timestamps_valid) == 0:
            axs[i].text(0.5, 0.5, 'No valid data',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        # Sample data if there are too many points
        sample_size = min(len(timestamps_valid), points)
        if len(timestamps_valid) > sample_size:
            try:
                # Sample indices including endpoints
                indices = np.linspace(0, len(timestamps_valid) - 1, sample_size, dtype=int)

                # Sample data
                timestamps_sample = timestamps_valid[indices]
                actual_sample = actual_valid[indices]
                predicted_sample = predicted_valid[indices]
            except Exception as e:
                print(f"Error sampling data for Zone {zone}: {e}")
                # Use all data if error occurs
                timestamps_sample = timestamps_valid
                actual_sample = actual_valid
                predicted_sample = predicted_valid
        else:
            timestamps_sample = timestamps_valid
            actual_sample = actual_valid
            predicted_sample = predicted_valid

        # Plot actual and predicted values
        try:
            axs[i].plot(timestamps_sample, actual_sample, 'b-', label='Actual')
            axs[i].plot(timestamps_sample, predicted_sample, 'r--', label='Predicted')

            # Format x-axis
            if isinstance(timestamps_sample, pd.DatetimeIndex) or isinstance(timestamps_sample[0], (pd.Timestamp, np.datetime64)):
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
                axs[i].tick_params(axis='x', rotation=45)
        except Exception as e:
            print(f"Error plotting data for Zone {zone}: {e}")
            axs[i].text(0.5, 0.5, f'Plot error: {str(e)[:30]}...',
                       ha='center', va='center', transform=axs[i].transAxes)
            continue

        # Title and grid
        axs[i].set_title(f'Zone {zone}')
        axs[i].set_ylabel('Temperature (°C)')
        axs[i].grid(True, linestyle='--', alpha=0.6)
        axs[i].legend(loc='upper right')

    # Hide unused subplots
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # Overall title
    fig.suptitle(f'Time Series Data for {horizon}-min Horizon', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save graph
    if save and save_dir:
        output_path = os.path.join(save_dir, f'timeseries_all_zones_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Time series plot for all zones saved: {output_path}")

    return fig


def plot_lag_dependency_analysis(lag_dependency, zone=None, horizon=None, save_dir=None):
    """
    Visualize LAG dependency analysis results
    Note: Graph output is disabled

    Parameters:
    -----------
    lag_dependency : dict
        Dictionary containing LAG dependency analysis results
    zone : int, optional
        Zone number
    horizon : int, optional
        Prediction horizon (minutes)
    save_dir : str, optional
        Directory path to save graphs

    Returns:
    --------
    list
        List of Figure objects (empty list as output is disabled)
    """
    # Graph saving is disabled
    save = False

    # Return empty dummy list (no output)
    print("LAG dependency visualization output is disabled")
    return []


def plot_physical_validity_analysis(results_dict, horizon, save=True):
    """
    Visualize physical validity analysis results

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each zone
    horizon : int
        Prediction horizon (minutes)
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Collect zones with data for the prediction horizon
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"Warning: No data available for {horizon}-min prediction. Skipping.")
        return None

    # Calculate number of rows and columns for subplots
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)
    n_rows = math.ceil(n_zones / n_cols)

    # Visualize physical validity analysis results
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(zones_with_data):
        results = results_dict[zone][horizon]
        y_test = results['y_test']
        y_pred = results['y_pred']
        ac_state = results['ac_state']
        ac_mode = results['ac_mode']
        zone_system = results['system']

        # Calculate temperature changes
        temp_change_true = y_test.diff()
        temp_change_pred = y_pred.diff()

        # Create masks for cooling, heating, and AC off states
        cooling_mask = (ac_state == 1) & (ac_mode == 0)
        heating_mask = (ac_state == 1) & (ac_mode == 1)
        off_mask = (ac_state == 0)

        # Create scatter plot
        axs[i].scatter(temp_change_true[cooling_mask], temp_change_pred[cooling_mask],
                      alpha=0.5, label='Cooling', color='blue')
        axs[i].scatter(temp_change_true[heating_mask], temp_change_pred[heating_mask],
                      alpha=0.5, label='Heating', color='red')
        axs[i].scatter(temp_change_true[off_mask], temp_change_pred[off_mask],
                      alpha=0.5, label='AC Off', color='gray')

        # Ideal prediction line (y=x)
        min_change = min(temp_change_true.min(), temp_change_pred.min())
        max_change = max(temp_change_true.max(), temp_change_pred.max())
        axs[i].plot([min_change, max_change], [min_change, max_change], 'k--', label='Ideal Prediction')

        # Expected change regions for cooling and heating
        axs[i].axhspan(min_change, 0, alpha=0.1, color='blue', label='Expected Cooling Region')
        axs[i].axhspan(0, max_change, alpha=0.1, color='red', label='Expected Heating Region')

        axs[i].set_title(f'Zone {zone} - {zone_system} System')
        axs[i].set_xlabel('Actual Temperature Change')
        axs[i].set_ylabel('Predicted Temperature Change')
        axs[i].grid(True)
        axs[i].legend()

    # Hide unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f'Physical Validity Analysis for {horizon}-min Horizon', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        output_path = os.path.join(OUTPUT_DIR, f'physical_validity_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Physical validity analysis graph saved for {horizon}-min horizon: {output_path}")

    return fig


def plot_response_delay_analysis(results_dict, horizon, save=True):
    """
    Analyze response delay in predictions

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each zone
    horizon : int
        Prediction horizon (minutes)
    save : bool, optional
        Whether to save the graph

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object of the plot
    """
    # Collect zones with data for the prediction horizon
    zones_with_data = []
    for zone, zone_results in results_dict.items():
        if horizon in zone_results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"Warning: No data available for {horizon}-min prediction. Skipping.")
        return None

    # Calculate number of rows and columns for subplots
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)
    n_rows = math.ceil(n_zones / n_cols)

    # Visualize response delay analysis results
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(zones_with_data):
        results = results_dict[zone][horizon]
        y_test = results['y_test']
        y_pred = results['y_pred']
        ac_state = results['ac_state']
        zone_system = results['system']

        # Detect AC state changes
        ac_state_change = ac_state.diff().abs()
        change_indices = ac_state_change[ac_state_change == 1].index

        # Plot temperature changes after state changes
        for idx in change_indices[:5]:  # Only show first 5 state changes
            # Get data before and after state change
            start_idx = idx - pd.Timedelta(minutes=5)
            end_idx = idx + pd.Timedelta(minutes=horizon)
            mask = (y_test.index >= start_idx) & (y_test.index <= end_idx)

            # Convert to relative time (minutes)
            relative_time = (y_test.index[mask] - idx).total_seconds() / 60

            # Plot actual and predicted values
            axs[i].plot(relative_time, y_test[mask], 'b-', label='Actual' if idx == change_indices[0] else "")
            axs[i].plot(relative_time, y_pred[mask], 'r--', label='Predicted' if idx == change_indices[0] else "")

        axs[i].axvline(x=0, color='k', linestyle='--', label='State Change Point')
        axs[i].set_title(f'Zone {zone} - {zone_system} System')
        axs[i].set_xlabel('Time Since State Change (minutes)')
        axs[i].set_ylabel('Temperature (°C)')
        axs[i].grid(True)
        axs[i].legend()

    # Hide unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f'Response Delay Analysis for {horizon}-min Horizon', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        output_path = os.path.join(OUTPUT_DIR, f'response_delay_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150)
        print(f"Response delay analysis graph saved for {horizon}-min horizon: {output_path}")

    return fig


def plot_enhanced_detailed_time_series(timestamps, actual, predicted, zone, horizon,
                                      save_dir=None, time_scale='day', data_period_days=7,
                                      show_lag_analysis=True, lag_dependency=None, save=True):
    """
    改善された詳細時系列プロット（文字化け修正・直感的レイアウト）

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
    time_scale : str, optional
        時間軸のスケール ('hour', 'day', 'week')
    data_period_days : int, optional
        表示するデータ期間（日数）
    show_lag_analysis : bool, optional
        LAG依存度分析を表示するか
    lag_dependency : dict, optional
        LAG依存度情報
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # フォントプロパティを定義
    font_prop = fm.FontProperties()
    try:
        # 利用可能な日本語フォントを取得
        japanese_fonts = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'Meiryo']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        selected_font = None
        for font in japanese_fonts:
            if font in available_fonts:
                selected_font = font
                break

        if selected_font:
            font_prop.set_family(selected_font)
        else:
            # macOSのデフォルト
            font_prop.set_family('Hiragino Sans')
    except:
        pass

    # NaN値をフィルタ
    valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
    timestamps_valid = timestamps[valid_indices]
    actual_valid = actual[valid_indices]
    predicted_valid = predicted[valid_indices]

    if len(timestamps_valid) == 0:
        print(f"ゾーン {zone}: 有効なデータがありません")
        return None

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
    else:
        timestamps_period = timestamps_valid
        actual_period = actual_valid
        predicted_period = predicted_valid

    # 誤差の計算
    error_period = actual_period - predicted_period
    abs_error_period = np.abs(error_period)

    # レイアウト設定（予測誤差プロットを削除し、横軸を拡大）
    if show_lag_analysis and lag_dependency:
        fig = plt.figure(figsize=(24, 10))  # 横幅を大幅に拡大
        gs = fig.add_gridspec(3, 4, height_ratios=[5, 1.5, 1.5], width_ratios=[3, 1, 1, 1], hspace=0.4, wspace=0.3)

        # メイン時系列プロット（上段全体）
        ax_main = fig.add_subplot(gs[0, :])
        # LAG依存度分析（2段目左）
        ax_lag = fig.add_subplot(gs[1, 0])
        # 散布図（2段目中央）
        ax_scatter = fig.add_subplot(gs[1, 1])
        # 統計情報（2段目右）
        ax_stats = fig.add_subplot(gs[1, 2:])
        # 時間遅れ分析（3段目全体）
        ax_delay = fig.add_subplot(gs[2, :])
    else:
        fig, ax_main = plt.subplots(figsize=(20, 8))  # 横幅を拡大

    # メイン時系列プロット（改善版）
    # 実測値を太い青線で
    ax_main.plot(timestamps_period, actual_period, 'b-', linewidth=2.5,
                label='実測値', alpha=0.9, zorder=3)
    # 予測値を赤い破線で
    ax_main.plot(timestamps_period, predicted_period, 'r--', linewidth=2.0,
                label='予測値', alpha=0.8, zorder=2)

    # 誤差の帯グラフ（薄いグレー）
    ax_main.fill_between(timestamps_period, actual_period, predicted_period,
                        alpha=0.3, color='gray', label='予測誤差', zorder=1)

    # X軸の時間フォーマット設定（横軸拡大に対応してより細かく）
    if time_scale == 'minute':
        ax_main.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))  # 15分間隔
        ax_main.xaxis.set_minor_locator(mdates.MinuteLocator(interval=5))  # 5分間隔
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    elif time_scale == 'hour':
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=1))  # 1時間間隔
        ax_main.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))  # 30分間隔
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    elif time_scale == 'day':
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # 3時間間隔
        ax_main.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # 1時間間隔
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    elif time_scale == 'week':
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # 12時間間隔
        ax_main.xaxis.set_minor_locator(mdates.HourLocator(interval=6))  # 6時間間隔
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))

    ax_main.tick_params(axis='x', rotation=45, labelsize=10)
    ax_main.set_xlabel('日時', fontproperties=font_prop, fontsize=12, fontweight='bold')
    ax_main.set_ylabel('温度 (°C)', fontproperties=font_prop, fontsize=12, fontweight='bold')

    # タイトルの改善
    title = f'ゾーン {zone} - {horizon}分後予測の詳細分析\n'
    title += f'期間: {data_period_days}日間 | 時間軸: {time_scale} | データ点数: {len(timestamps_period)}'
    ax_main.set_title(title, fontproperties=font_prop, fontsize=14, fontweight='bold', pad=20)

    ax_main.grid(True, linestyle='--', alpha=0.7)
    legend = ax_main.legend(fontsize=11, loc='upper right', framealpha=0.9)
    # 凡例のフォントも設定
    for text in legend.get_texts():
        text.set_fontproperties(font_prop)

    # 詳細分析プロット（LAG分析が有効な場合）
    if show_lag_analysis and lag_dependency:

        # 統計情報の計算
        mae = np.mean(abs_error_period)
        rmse = np.sqrt(np.mean(error_period**2))

        # 1. LAG依存度分析（改善版）
        lag_categories = ['直接LAG\n特徴量', '移動平均\n特徴量', '総LAG\n依存度']
        lag_values = [
            lag_dependency.get('lag_temp_percent', 0),
            lag_dependency.get('rolling_temp_percent', 0),
            lag_dependency.get('lag_temp_percent', 0) + lag_dependency.get('rolling_temp_percent', 0)
        ]

        colors = ['red' if val > 30 else 'orange' if val > 15 else 'green' for val in lag_values]
        bars = ax_lag.bar(lag_categories, lag_values, color=colors, alpha=0.8, edgecolor='black')

        # 値をバーの上に表示
        for bar, val in zip(bars, lag_values):
            ax_lag.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # ラベルにフォントを適用
        for label in ax_lag.get_xticklabels():
            label.set_fontproperties(font_prop)

        ax_lag.set_ylabel('依存度 (%)', fontproperties=font_prop, fontsize=10)
        ax_lag.set_title('LAG依存度分析', fontproperties=font_prop, fontsize=11, fontweight='bold')
        ax_lag.set_ylim(0, max(50, max(lag_values) * 1.3))
        ax_lag.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='警告閾値')
        ax_lag.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='注意閾値')
        legend_lag = ax_lag.legend(fontsize=8)
        for text in legend_lag.get_texts():
            text.set_fontproperties(font_prop)
        ax_lag.grid(True, alpha=0.3)

        # 2. 散布図（実測値 vs 予測値）
        ax_scatter.scatter(actual_period, predicted_period, alpha=0.6, s=20, c='blue', edgecolors='black', linewidth=0.5)

        # 理想線（y=x）
        min_val = min(actual_period.min(), predicted_period.min())
        max_val = max(actual_period.max(), predicted_period.max())
        ax_scatter.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想線')

        ax_scatter.set_xlabel('実測値 (°C)', fontproperties=font_prop, fontsize=10)
        ax_scatter.set_ylabel('予測値 (°C)', fontproperties=font_prop, fontsize=10)
        ax_scatter.set_title('実測値 vs 予測値', fontproperties=font_prop, fontsize=11, fontweight='bold')
        ax_scatter.grid(True, alpha=0.3)
        legend_scatter = ax_scatter.legend(fontsize=9)
        for text in legend_scatter.get_texts():
            text.set_fontproperties(font_prop)

        # R²を表示
        r2 = r2_score(actual_period, predicted_period)
        ax_scatter.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax_scatter.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       fontsize=10, fontweight='bold')

        # 3. 統計情報表示
        ax_stats.axis('off')
        stats_text = f"""統計情報:

データ点数: {len(timestamps_period):,}
平均絶対誤差: {mae:.3f}°C
二乗平均平方根誤差: {rmse:.3f}°C
決定係数: {r2:.4f}

実測値:
  平均: {np.mean(actual_period):.2f}°C
  標準偏差: {np.std(actual_period):.2f}°C

予測値:
  平均: {np.mean(predicted_period):.2f}°C
  標準偏差: {np.std(predicted_period):.2f}°C"""

        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontproperties=font_prop, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

        # 4. 時間遅れ分析（改善版）
        _plot_enhanced_lag_analysis(timestamps_period, actual_period, predicted_period, ax_delay, horizon, font_prop)

    plt.tight_layout()

    # グラフの保存
    if save and save_dir:
        filename = f'enhanced_detailed_timeseries_zone_{zone}_horizon_{horizon}_{time_scale}_{data_period_days}days.png'
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"改善版詳細時系列プロット保存: {output_path}")

    return fig


def _plot_enhanced_lag_analysis(timestamps, actual, predicted, ax, horizon, font_prop):
    """
    改善された時間遅れ分析のプロット

    Parameters:
    -----------
    timestamps : array-like
        タイムスタンプ
    actual : array-like
        実測値
    predicted : array-like
        予測値
    ax : matplotlib.axes.Axes
        プロット用のAxes
    horizon : int
        予測ホライゾン（分）
    font_prop : matplotlib.font_manager.FontProperties
        フォントプロパティ
    """
    if len(actual) > 100:
        # データの正規化
        actual_norm = (actual - np.mean(actual)) / (np.std(actual) + 1e-8)
        predicted_norm = (predicted - np.mean(predicted)) / (np.std(predicted) + 1e-8)

        # 相互相関の計算
        max_lag = min(50, len(actual) // 4)
        correlations = []
        lags = range(-max_lag, max_lag + 1)

        for lag in lags:
            try:
                if lag < 0:
                    corr = np.corrcoef(actual_norm[-lag:], predicted_norm[:lag])[0, 1]
                elif lag > 0:
                    corr = np.corrcoef(actual_norm[:-lag], predicted_norm[lag:])[0, 1]
                else:
                    corr = np.corrcoef(actual_norm, predicted_norm)[0, 1]

                if not np.isnan(corr):
                    correlations.append(corr)
                else:
                    correlations.append(0)
            except:
                correlations.append(0)

        # 相互相関のプロット（改善版）
        ax.plot(lags, correlations, 'b-', linewidth=2.5, alpha=0.8)
        ax.fill_between(lags, 0, correlations, alpha=0.3, color='blue')

        # 重要な線を追加
        ax.axvline(x=0, color='red', linestyle='-', linewidth=2, alpha=0.8, label='遅れなし')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # 最大相関を持つ遅れを検出
        max_corr_idx = np.argmax(correlations)
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlations[max_corr_idx]

        ax.axvline(x=optimal_lag, color='green', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'最適遅れ: {optimal_lag}ステップ')

        # ラベルとタイトル
        ax.set_xlabel('遅れ（データポイント）', fontproperties=font_prop, fontsize=11, fontweight='bold')
        ax.set_ylabel('相互相関係数', fontproperties=font_prop, fontsize=11, fontweight='bold')
        ax.set_title(f'時間遅れ分析 (最大相関: {max_correlation:.3f}, 最適遅れ: {optimal_lag}ステップ)',
                    fontproperties=font_prop, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        legend_delay = ax.legend(fontsize=10)
        for text in legend_delay.get_texts():
            text.set_fontproperties(font_prop)

        # 警告メッセージの改善
        if optimal_lag > 0:
            warning_text = f"⚠️ 予測が{optimal_lag}ステップ({optimal_lag*5}分)遅れています"
            ax.text(0.02, 0.98, warning_text, transform=ax.transAxes,
                   fontproperties=font_prop,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.9),
                   verticalalignment='top', fontsize=10, fontweight='bold')
        elif optimal_lag < 0:
            warning_text = f"📈 予測が{abs(optimal_lag)}ステップ({abs(optimal_lag)*5}分)先行しています"
            ax.text(0.02, 0.98, warning_text, transform=ax.transAxes,
                   fontproperties=font_prop,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
                   verticalalignment='top', fontsize=10, fontweight='bold')
        else:
            success_text = "✅ 時間遅れは検出されませんでした"
            ax.text(0.02, 0.98, success_text, transform=ax.transAxes,
                   fontproperties=font_prop,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.9),
                   verticalalignment='top', fontsize=10, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'データ不足のため分析不可\n(最低100データポイント必要)',
               ha='center', va='center', transform=ax.transAxes,
               fontproperties=font_prop, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


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
    # フォントプロパティを定義
    font_prop = fm.FontProperties()
    try:
        # 利用可能な日本語フォントを取得
        japanese_fonts = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'Meiryo']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        selected_font = None
        for font in japanese_fonts:
            if font in available_fonts:
                selected_font = font
                break

        if selected_font:
            font_prop.set_family(selected_font)
        else:
            # macOSのデフォルト
            font_prop.set_family('Hiragino Sans')
    except:
        pass

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
                print(f"Zone {zone}: データ取得成功 - timestamps: {len(timestamps)}, actual: {len(actual)}, predicted: {len(predicted)}")

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
                title += f'LAG依存度高: {total_lag:.1f}%'
                title_color = 'red'
            elif total_lag > 15:
                title += f'LAG依存度中: {total_lag:.1f}%'
                title_color = 'orange'
            else:
                title += f'LAG依存度低: {total_lag:.1f}%'
                title_color = 'green'
        else:
            title_color = 'black'

        axs[i].set_title(title, fontproperties=font_prop, fontsize=12, fontweight='bold', color=title_color)
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


def create_detailed_analysis_for_zone(results_dict, zone, horizon, save_dir=None,
                                    time_scales=['hour', 'day'], data_periods=[3, 7]):
    """
    指定されたゾーンとホライゾンの詳細分析を複数の時間スケールで実行

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    zone : int
        分析対象のゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    time_scales : list, optional
        時間軸のスケールリスト
    data_periods : list, optional
        表示するデータ期間リスト（日数）

    Returns:
    --------
    dict
        生成された図のパスと分析結果
    """
    if zone not in results_dict or horizon not in results_dict[zone]:
        print(f"ゾーン {zone}, ホライゾン {horizon}分 のデータが見つかりません")
        return None

    zone_results = results_dict[zone][horizon]

    # データの取得
    if not all(k in zone_results for k in ['test_data', 'test_y', 'test_predictions']):
        print(f"必要なデータが不足しています: {list(zone_results.keys())}")
        return None

    test_df = zone_results['test_data']
    test_y = zone_results['test_y']
    test_predictions = zone_results['test_predictions']
    lag_dependency = zone_results.get('lag_dependency', {})

    # タイムスタンプの取得
    if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
        timestamps = test_df.index
    else:
        print("タイムスタンプが見つかりません")
        return None

    generated_files = {}
    analysis_summary = {}

    # 各時間スケールでの詳細分析
    for time_scale, data_period in zip(time_scales, data_periods):
        print(f"\n### ゾーン {zone}, {horizon}分後予測 - {time_scale}軸分析（{data_period}日間）")

        # 詳細時系列プロット生成
        fig = plot_enhanced_detailed_time_series(
            timestamps=timestamps,
            actual=test_y,
            predicted=test_predictions,
            zone=zone,
            horizon=horizon,
            save_dir=save_dir,
            time_scale=time_scale,
            data_period_days=data_period,
            show_lag_analysis=True,
            lag_dependency=lag_dependency,
            save=True
        )

        if fig:
            filename = f'enhanced_detailed_timeseries_zone_{zone}_horizon_{horizon}_{time_scale}_{data_period}days.png'
            if save_dir:
                file_path = os.path.join(save_dir, filename)
                generated_files[f'{time_scale}_{data_period}days'] = file_path

            print(f"✓ 詳細分析完了: {time_scale}軸, {data_period}日間")

            # 分析結果の要約
            valid_mask = ~(pd.isna(test_y) | pd.isna(test_predictions))
            if valid_mask.sum() > 0:
                actual_valid = test_y[valid_mask]
                pred_valid = test_predictions[valid_mask]

                mae = np.mean(np.abs(actual_valid - pred_valid))
                rmse = np.sqrt(np.mean((actual_valid - pred_valid)**2))
                r2 = r2_score(actual_valid, pred_valid)

                analysis_summary[f'{time_scale}_{data_period}days'] = {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'data_points': valid_mask.sum()
                }
        else:
            print(f"✗ 詳細分析失敗: {time_scale}軸, {data_period}日間")

    # LAG依存度の詳細分析
    print(f"\n### LAG依存度詳細分析:")
    total_lag = lag_dependency.get('lag_temp_percent', 0) + lag_dependency.get('rolling_temp_percent', 0)
    direct_lag = lag_dependency.get('lag_temp_percent', 0)
    rolling_lag = lag_dependency.get('rolling_temp_percent', 0)

    print(f"  総LAG依存度: {total_lag:.1f}%")
    print(f"  直接LAG特徴量依存度: {direct_lag:.1f}%")
    print(f"  移動平均特徴量依存度: {rolling_lag:.1f}%")

    if total_lag > 30:
        print("警告: LAG依存度が高すぎます（30%超）")
        print("    → 予測が過去データに過度に依存している可能性があります")
    elif total_lag > 15:
        print("注意: LAG依存度が中程度です（15-30%）")
        print("    → 適度な過去情報の利用ですが、監視が必要です")
    else:
        print("良好: LAG依存度は低いレベルです（15%未満）")

    return {
        'generated_files': generated_files,
        'analysis_summary': analysis_summary,
        'lag_dependency': lag_dependency,
        'zone': zone,
        'horizon': horizon
    }


def generate_comprehensive_time_series_report(results_dict, save_dir=None,
                                             focus_zones=None, focus_horizons=None):
    """
    時系列予測の包括的なレポート生成

    Parameters:
    -----------
    results_dict : dict
        各ゾーンの結果を含む辞書
    save_dir : str, optional
        レポート保存ディレクトリ
    focus_zones : list, optional
        重点的に分析するゾーンのリスト
    focus_horizons : list, optional
        重点的に分析するホライゾンのリスト

    Returns:
    --------
    dict
        レポートの要約と生成されたファイル情報
    """
    print("\n" + "="*60)
    print("🕐 時系列予測モデル包括レポート生成開始")
    print("="*60)

    report_summary = {
        'total_zones': 0,
        'total_horizons': 0,
        'high_lag_dependency': [],
        'best_performance': {},
        'generated_files': []
    }

    # 対象ゾーンとホライゾンの決定
    all_zones = list(results_dict.keys())
    all_horizons = []
    for zone_results in results_dict.values():
        all_horizons.extend(zone_results.keys())
    all_horizons = sorted(list(set(all_horizons)))

    target_zones = focus_zones if focus_zones else all_zones
    target_horizons = focus_horizons if focus_horizons else all_horizons

    report_summary['total_zones'] = len(target_zones)
    report_summary['total_horizons'] = len(target_horizons)

    print(f"📊 分析対象: {len(target_zones)}ゾーン × {len(target_horizons)}ホライゾン")
    print(f"対象ゾーン: {target_zones}")
    print(f"対象ホライゾン: {target_horizons}")

    # 各ゾーン・ホライゾンの詳細分析
    for zone in target_zones:
        if zone not in results_dict:
            continue

        print(f"\n🔍 ゾーン {zone} の詳細分析:")

        for horizon in target_horizons:
            if horizon not in results_dict[zone]:
                continue

            # 詳細分析の実行
            analysis_result = create_detailed_analysis_for_zone(
                results_dict, zone, horizon, save_dir
            )

            if analysis_result:
                # LAG依存度チェック
                lag_dep = analysis_result['lag_dependency'].get('lag_temp_percent', 0) + analysis_result['lag_dependency'].get('rolling_temp_percent', 0)
                if lag_dep > 30:
                    report_summary['high_lag_dependency'].append({
                        'zone': zone,
                        'horizon': horizon,
                        'lag_dependency': lag_dep
                    })

                # パフォーマンス評価
                if 'analysis_summary' in analysis_result:
                    for analysis_key, metrics in analysis_result['analysis_summary'].items():
                        perf_key = f"zone_{zone}_horizon_{horizon}_{analysis_key}"
                        report_summary['best_performance'][perf_key] = metrics

                # ファイル記録
                report_summary['generated_files'].extend(
                    list(analysis_result['generated_files'].values())
                )

    # レポート要約の表示
    print(f"\n" + "="*60)
    print("📋 レポート要約")
    print("="*60)

    print(f"✅ 生成されたファイル数: {len(report_summary['generated_files'])}")

    if report_summary['high_lag_dependency']:
        print(f"高LAG依存度モデル ({len(report_summary['high_lag_dependency'])}個):")
        for item in report_summary['high_lag_dependency']:
            print(f"  - ゾーン {item['zone']}, {item['horizon']}分: {item['lag_dependency']:.1f}%")
    else:
        print("✓ 全モデルのLAG依存度は適切な範囲内です")

    # 最高性能モデルの特定
    if report_summary['best_performance']:
        best_r2 = max(report_summary['best_performance'].values(), key=lambda x: x.get('r2', 0))
        best_model_key = [k for k, v in report_summary['best_performance'].items() if v == best_r2][0]
        print(f"🏆 最高性能モデル: {best_model_key}")
        print(f"   R²: {best_r2['r2']:.4f}, RMSE: {best_r2['rmse']:.3f}°C")

    print(f"\n📁 すべてのファイルは {save_dir} に保存されています")

    return report_summary


def create_correct_prediction_timestamps(input_timestamps, horizon_minutes):
    """
    予測値を正しい未来のタイムスタンプで表示するための関数

    Parameters:
    -----------
    input_timestamps : pd.DatetimeIndex or array-like
        入力データのタイムスタンプ
    horizon_minutes : int
        予測ホライゾン（分）

    Returns:
    --------
    pd.DatetimeIndex
        予測値用の正しいタイムスタンプ（入力時刻 + horizon_minutes）
    """
    if isinstance(input_timestamps, pd.DatetimeIndex):
        return input_timestamps + pd.Timedelta(minutes=horizon_minutes)
    else:
        # array-likeの場合はpd.DatetimeIndexに変換
        timestamps_index = pd.DatetimeIndex(input_timestamps)
        return timestamps_index + pd.Timedelta(minutes=horizon_minutes)


def validate_prediction_timing(input_timestamps, actual_values, predicted_values, horizon_minutes, zone):
    """
    予測の時間軸が正しいかを検証する関数

    Parameters:
    -----------
    input_timestamps : pd.DatetimeIndex
        入力データのタイムスタンプ
    actual_values : array-like
        実測値（目的変数）
    predicted_values : array-like
        予測値
    horizon_minutes : int
        予測ホライゾン（分）
    zone : int
        ゾーン番号

    Returns:
    --------
    dict
        検証結果
    """
    validation_results = {
        'is_correct_timing': True,
        'issues': [],
        'recommendations': []
    }

    # 1. 予測値が過去の実測値パターンを単純にコピーしていないかチェック
    if len(actual_values) > horizon_minutes // 5:  # 十分なデータがある場合
        # 実測値と予測値の相関を時間遅れ別に計算
        correlations = []
        max_lag = min(20, len(actual_values) // 4)  # 最大20ステップまで

        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag < 0:
                    # 予測値が実測値より先行している場合
                    corr = np.corrcoef(actual_values[-lag:], predicted_values[:lag])[0, 1]
                elif lag > 0:
                    # 予測値が実測値より遅れている場合
                    corr = np.corrcoef(actual_values[:-lag], predicted_values[lag:])[0, 1]
                else:
                    # 同時刻の相関
                    corr = np.corrcoef(actual_values, predicted_values)[0, 1]

                if not np.isnan(corr):
                    correlations.append((lag, corr))
            except:
                continue

        if correlations:
            # 最大相関とその遅れを特定
            max_corr_lag, max_corr_value = max(correlations, key=lambda x: abs(x[1]))

            # 問題の検出
            if max_corr_lag > 0:
                validation_results['is_correct_timing'] = False
                validation_results['issues'].append(
                    f"予測値が実測値より{max_corr_lag}ステップ({max_corr_lag*5}分)遅れています"
                )
                validation_results['recommendations'].append(
                    "予測値の時間軸を修正してください。予測値は入力時刻+予測ホライゾンで表示されるべきです。"
                )

            if abs(max_corr_value) > 0.95 and max_corr_lag != 0:
                validation_results['issues'].append(
                    f"予測値が過去の実測値パターンを単純にコピーしている可能性があります（相関={max_corr_value:.3f}）"
                )
                validation_results['recommendations'].append(
                    "LAG特徴量への依存度を確認し、未来情報の活用を強化してください。"
                )

    return validation_results


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
                 fontsize=14, color='red', fontweight='bold')
    ax1.set_ylabel('温度 (°C)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    # 下段: 修正された表示方法（正しい表示）
    ax2.plot(timestamps_sample, actual_sample, 'b-', linewidth=2, label='実測値', alpha=0.8)
    ax2.plot(prediction_timestamps_sample, predicted_sample, 'r--', linewidth=2,
            label=f'予測値（正しい時間軸: +{horizon}分）', alpha=0.8)
    ax2.set_title(f'ゾーン {zone} - 修正された表示方法（正しい）: 予測値が未来の時刻に表示',
                 fontsize=14, color='green', fontweight='bold')
    ax2.set_xlabel('日時', fontsize=12)
    ax2.set_ylabel('温度 (°C)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
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
                print(f"Zone {zone}: データ取得成功 - timestamps: {len(timestamps)}, actual: {len(actual)}, predicted: {len(predicted)}")

        if timestamps is None or actual is None or predicted is None:
            axs[i].text(0.5, 0.5, 'データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontsize=12)
            continue

        # 有効データのフィルタリング
        try:
            valid_indices = ~(pd.isna(actual) | pd.isna(predicted))
            timestamps_valid = timestamps[valid_indices]
            actual_valid = actual[valid_indices]
            predicted_valid = predicted[valid_indices]
        except Exception as e:
            axs[i].text(0.5, 0.5, 'データ処理エラー', ha='center', va='center',
                       transform=axs[i].transAxes, fontsize=12)
            continue

        if len(actual_valid) == 0:
            axs[i].text(0.5, 0.5, '有効データなし', ha='center', va='center',
                       transform=axs[i].transAxes, fontsize=12)
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
            axs[i].set_title(f'{status} ゾーン {zone}', color=title_color, fontweight='bold')

            axs[i].set_ylabel('温度 (°C)')
            axs[i].grid(True, alpha=0.3)
            axs[i].legend(fontsize=9)
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)

        except Exception as e:
            axs[i].text(0.5, 0.5, f'プロットエラー: {str(e)[:20]}...',
                       ha='center', va='center', transform=axs[i].transAxes, fontsize=10)

    # 未使用のサブプロットを非表示
    for j in range(n_zones, len(axs)):
        fig.delaxes(axs[j])

    # 全体タイトル
    fig.suptitle(f'{horizon}分後予測の時間軸修正済み時系列プロット', fontsize=16, fontweight='bold')
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


def analyze_feature_patterns(feature_importance, zone, horizon):
    """
    特徴量パターンの詳細分析

    Parameters:
    -----------
    feature_importance : DataFrame
        特徴量重要度データフレーム
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        分析結果
    """
    analysis_results = {
        'zone': zone,
        'horizon': horizon,
        'lag_features': [],
        'future_features': [],
        'current_features': [],
        'poly_features': [],
        'suspicious_patterns': []
    }

    # 特徴量を分類
    for _, row in feature_importance.iterrows():
        feature_name = row['feature']
        importance = row['importance']

        if '_lag_' in feature_name:
            analysis_results['lag_features'].append({
                'name': feature_name,
                'importance': importance
            })
        elif '_future_' in feature_name:
            analysis_results['future_features'].append({
                'name': feature_name,
                'importance': importance
            })
        elif 'poly_' in feature_name:
            analysis_results['poly_features'].append({
                'name': feature_name,
                'importance': importance
            })
        else:
            analysis_results['current_features'].append({
                'name': feature_name,
                'importance': importance
            })

    # 疑わしいパターンの検出
    total_importance = feature_importance['importance'].sum()

    # LAG特徴量への過度な依存
    lag_importance = sum([f['importance'] for f in analysis_results['lag_features']])
    lag_percentage = (lag_importance / total_importance * 100) if total_importance > 0 else 0

    if lag_percentage > 30:
        analysis_results['suspicious_patterns'].append({
            'type': 'high_lag_dependency',
            'description': f'LAG特徴量への依存度が高すぎます ({lag_percentage:.1f}%)',
            'severity': 'high'
        })

    # 単一特徴量への過度な依存
    max_importance = feature_importance['importance'].max()
    max_percentage = (max_importance / total_importance * 100) if total_importance > 0 else 0

    if max_percentage > 80:
        max_feature = feature_importance.loc[feature_importance['importance'].idxmax(), 'feature']
        analysis_results['suspicious_patterns'].append({
            'type': 'single_feature_dominance',
            'description': f'単一特徴量 "{max_feature}" への依存度が極端に高いです ({max_percentage:.1f}%)',
            'severity': 'high'
        })

    # 未来情報の不足
    future_importance = sum([f['importance'] for f in analysis_results['future_features']])
    future_percentage = (future_importance / total_importance * 100) if total_importance > 0 else 0

    if future_percentage < 50:
        analysis_results['suspicious_patterns'].append({
            'type': 'insufficient_future_info',
            'description': f'未来情報の活用が不足しています ({future_percentage:.1f}%)',
            'severity': 'medium'
        })

    return analysis_results


def detect_lag_following_pattern(timestamps, actual, predicted, horizon):
    """
    LAG特徴量による後追いパターンの検出

    Parameters:
    -----------
    timestamps : array-like
        タイムスタンプ
    actual : array-like
        実測値
    predicted : array-like
        予測値
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        検出結果
    """
    detection_results = {
        'is_lag_following': False,
        'lag_correlation': 0.0,
        'optimal_lag_steps': 0,
        'confidence': 'low',
        'recommendations': []
    }

    if len(actual) < 100:
        return detection_results

    # 正規化
    actual_norm = (actual - np.mean(actual)) / (np.std(actual) + 1e-8)
    predicted_norm = (predicted - np.mean(predicted)) / (np.std(predicted) + 1e-8)

    # 相互相関の計算
    max_lag = min(horizon // 5 + 10, len(actual) // 4)  # 予測ホライゾンに基づく最大遅れ
    correlations = []
    lags = range(0, max_lag + 1)

    for lag in lags:
        try:
            if lag == 0:
                corr = np.corrcoef(actual_norm, predicted_norm)[0, 1]
            else:
                corr = np.corrcoef(actual_norm[:-lag], predicted_norm[lag:])[0, 1]

            if not np.isnan(corr):
                correlations.append((lag, corr))
        except:
            continue

    if correlations:
        # 最大相関とその遅れを特定
        max_corr_lag, max_corr_value = max(correlations, key=lambda x: abs(x[1]))
        detection_results['lag_correlation'] = max_corr_value
        detection_results['optimal_lag_steps'] = max_corr_lag

        # 後追いパターンの判定
        if max_corr_lag > 0 and abs(max_corr_value) > 0.8:
            detection_results['is_lag_following'] = True
            detection_results['confidence'] = 'high' if abs(max_corr_value) > 0.9 else 'medium'

            detection_results['recommendations'].extend([
                f"予測が実測値より{max_corr_lag}ステップ({max_corr_lag*5}分)遅れています",
                "LAG特徴量への依存度を下げてください",
                "未来情報（制御パラメータ、環境データ）の活用を強化してください",
                "物理法則ベースの特徴量を追加してください"
            ])
        elif max_corr_lag == 0 and abs(max_corr_value) > 0.95:
            detection_results['recommendations'].append(
                "予測精度は高いですが、過学習の可能性があります。検証データでの性能を確認してください"
            )

    return detection_results
