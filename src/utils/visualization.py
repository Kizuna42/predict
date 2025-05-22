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
from sklearn.metrics import r2_score


# グラフ設定
sns.set_theme(style="whitegrid", palette="colorblind")
# フォント設定 - 日本語フォント対応（警告を減らすため、一般的なフォントのみ指定）
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica', 'Hiragino Sans GB', 'MS Gothic', 'IPAGothic', 'IPAexGothic', 'IPAPGothic']
plt.rcParams['axes.unicode_minus'] = False


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
            print(f"Error processing data for zone {zone}: {e}")
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

        # Method 1: Use test_df, test_y, test_predictions
        if all(k in horizon_results for k in ['test_df', 'test_y', 'test_predictions']):
            test_df = horizon_results['test_df']
            if isinstance(test_df, pd.DataFrame) and hasattr(test_df, 'index'):
                timestamps = test_df.index
                actual = horizon_results['test_y']
                predicted = horizon_results['test_predictions']

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

            timestamps_valid = timestamps[valid_indices] if len(timestamps) > 0 else pd.DatetimeIndex([])

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
