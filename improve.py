#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import time
import logging
from datetime import timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings

import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

def determine_thermo_status(df, deadband=1.0):
    df = df.reset_index(drop=True)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    min_date = df['time_stamp'].min()
    max_date = df['time_stamp'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='1min')
    result_df = pd.DataFrame({'time_stamp': date_range})
    thermo_cols = [f'thermo_{zone}' for zone in range(12)]
    result_df[thermo_cols] = 0
    for zone in range(12):
        valid_col = f'AC_valid_{zone}'
        set_col = f'AC_set_{zone}'
        temp_col = f'AC_temp_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        if all(col in df.columns for col in [valid_col, set_col, temp_col, mode_col]):
            df_zone = df[[valid_col, set_col, temp_col, mode_col, 'time_stamp']].copy()
            df_zone.fillna({valid_col: 0, set_col: 0, temp_col: 0, mode_col: 0}, inplace=True)
            df_zone[thermo_col] = 0
            mask = (df_zone[valid_col] > 0) & (df_zone[mode_col].isin([1, 2]))
            rows = len(df_zone)
            thermo_values = np.zeros(rows)
            for i in range(1, rows):
                if not mask.iloc[i]:
                    thermo_values[i] = 0
                    continue
                current_mode = df_zone[mode_col].iloc[i]
                prev_thermo = thermo_values[i-1]
                if current_mode == 2:
                    if prev_thermo == 0 and df_zone[temp_col].iloc[i] < df_zone[set_col].iloc[i] - deadband:
                        thermo_values[i] = 1
                    elif prev_thermo == 1 and df_zone[temp_col].iloc[i] > df_zone[set_col].iloc[i] + deadband:
                        thermo_values[i] = 0
                    else:
                        thermo_values[i] = prev_thermo
                elif current_mode == 1:
                    if prev_thermo == 0 and df_zone[temp_col].iloc[i] > df_zone[set_col].iloc[i] + deadband:
                        thermo_values[i] = 1
                    elif prev_thermo == 1 and df_zone[temp_col].iloc[i] < df_zone[set_col].iloc[i] - deadband:
                        thermo_values[i] = 0
                    else:
                        thermo_values[i] = prev_thermo
            df_zone[thermo_col] = thermo_values
            result_df = pd.merge(
                result_df,
                df_zone[['time_stamp', thermo_col]],
                on='time_stamp',
                how='left',
                suffixes=('', '_new')
            )
            col_name = f"{thermo_col}_new" if f"{thermo_col}_new" in result_df.columns else thermo_col
            result_df[thermo_col] = result_df[col_name].fillna(0).astype(int)
            if f"{thermo_col}_new" in result_df.columns:
                result_df = result_df.drop(columns=[f'{thermo_col}_new'])
        else:
            print(f"Warning: Required columns for zone {zone} not found. Setting thermo_{zone} to 0.")
    result_df['thermo_L_or'] = (
        result_df['thermo_0'].astype(bool) |
        result_df['thermo_1'].astype(bool) |
        result_df['thermo_6'].astype(bool) |
        result_df['thermo_7'].astype(bool)
    ).astype(int)
    result_df['thermo_M_or'] = (
        result_df['thermo_2'].astype(bool) |
        result_df['thermo_3'].astype(bool) |
        result_df['thermo_8'].astype(bool) |
        result_df['thermo_9'].astype(bool)
    ).astype(int)
    result_df['thermo_R_or'] = (
        result_df['thermo_4'].astype(bool) |
        result_df['thermo_5'].astype(bool) |
        result_df['thermo_10'].astype(bool) |
        result_df['thermo_11'].astype(bool)
    ).astype(int)
    return result_df

def prepare_features_for_sens_temp(df, thermo_df, zone, look_back=60, prediction_horizon=5):
    valid_col = f'AC_valid_{zone}'
    mode_col = f'AC_mode_{zone}'
    thermo_col = f'thermo_{zone}'
    sens_temp_col = f'sens_temp_{zone}'
    L_zones = [0, 1, 6, 7]
    M_zones = [2, 3, 8, 9]
    R_zones = [4, 5, 10, 11]
    if zone in L_zones:
        power_col = 'L'
        thermo_or_col = 'thermo_L_or'
    elif zone in M_zones:
        power_col = 'M'
        thermo_or_col = 'thermo_M_or'
    elif zone in R_zones:
        power_col = 'R'
        thermo_or_col = 'thermo_R_or'
    else:
        print(f"Zone {zone} not assigned to any outdoor unit")
        return None, None, None
    required_thermo_cols = ['time_stamp', thermo_col, thermo_or_col]
    thermo_subset = thermo_df[required_thermo_cols].copy()
    required_df_cols = ['time_stamp', valid_col, mode_col, sens_temp_col, power_col]
    optional_cols = ['outdoor_temp', 'solar_radiation', 'humidity']
    for col in optional_cols:
        if col in df.columns:
            required_df_cols.append(col)
    df_subset = df[required_df_cols].copy()
    merged_df = pd.merge(df_subset, thermo_subset, on='time_stamp', how='left')
    required_cols = [valid_col, mode_col, thermo_col, sens_temp_col, power_col, thermo_or_col]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"Missing required columns for zone {zone}: {missing_cols}")
        return None, None, None
    merged_df['hour'] = merged_df['time_stamp'].dt.hour
    merged_df['day_of_week'] = merged_df['time_stamp'].dt.dayofweek
    merged_df['month'] = merged_df['time_stamp'].dt.month
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
    merged_df['is_night'] = ((merged_df['hour'] >= 19) | (merged_df['hour'] <= 6)).astype(int)
    merged_df['is_morning'] = ((merged_df['hour'] > 6) & (merged_df['hour'] <= 12)).astype(int)
    merged_df['is_afternoon'] = ((merged_df['hour'] > 12) & (merged_df['hour'] < 19)).astype(int)
    hour_rad = 2 * np.pi * merged_df['hour'] / 24
    merged_df['hour_sin'] = np.sin(hour_rad)
    merged_df['hour_cos'] = np.cos(hour_rad)
    day_rad = 2 * np.pi * merged_df['time_stamp'].dt.day / 31
    merged_df['day_sin'] = np.sin(day_rad)
    merged_df['day_cos'] = np.cos(day_rad)
    week_rad = 2 * np.pi * merged_df['day_of_week'] / 7
    merged_df['week_sin'] = np.sin(week_rad)
    merged_df['week_cos'] = np.cos(week_rad)
    lag_cols = {}
    target_col = f'{sens_temp_col}_future_{prediction_horizon}'
    merged_df[target_col] = merged_df[sens_temp_col].shift(prediction_horizon)
    for lag in [1, 2, 5, 15, 30, 60]:
        if lag <= look_back:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            merged_df[lag_col] = merged_df[sens_temp_col].shift(lag)
            lag_cols[lag_col] = True
    for lag in [1, 5, 15]:
        if lag <= look_back:
            change_col = f'{sens_temp_col}_change_{lag}'
            if lag == 1:
                merged_df[change_col] = merged_df[sens_temp_col] - merged_df[f'{sens_temp_col}_lag_1']
            else:
                merged_df[change_col] = (merged_df[sens_temp_col] - merged_df[f'{sens_temp_col}_lag_{lag}']) / lag
            lag_cols[change_col] = True
    for lag in [1, 5, 15, 30]:
        if lag <= look_back:
            lag_col = f'{power_col}_lag_{lag}'
            merged_df[lag_col] = merged_df[power_col].shift(lag)
            lag_cols[lag_col] = True
    windows = [5, 15, 30, 60]
    for window in windows:
        if window <= look_back:
            roll_temp = f'{sens_temp_col}_roll_{window}'
            temp_past = merged_df[sens_temp_col].shift(1)
            merged_df[roll_temp] = temp_past.rolling(window=window, min_periods=1).mean()
            roll_power = f'{power_col}_roll_{window}'
            power_past = merged_df[power_col].shift(1)
            merged_df[roll_power] = power_past.rolling(window=window, min_periods=1).mean()
            std_temp = f'{sens_temp_col}_std_{window}'
            merged_df[std_temp] = temp_past.rolling(window=window, min_periods=1).std()
            lag_cols[roll_temp] = True
            lag_cols[roll_power] = True
            lag_cols[std_temp] = True
    merged_df['thermo_change'] = merged_df[thermo_col].diff(1).fillna(0)
    reset_points = (merged_df[thermo_col] != merged_df[thermo_col].shift(1)).astype(int)
    reset_points.iloc[0] = 1
    group_id = reset_points.cumsum()
    merged_df['thermo_duration'] = merged_df.groupby(group_id).cumcount()
    change_points = (merged_df['thermo_change'] != 0).astype(int)
    change_group_id = change_points.cumsum()
    merged_df['time_since_thermo_change'] = merged_df.groupby(change_group_id).cumcount()
    if 'outdoor_temp' in merged_df.columns:
        merged_df['outdoor_temp_lag_1'] = merged_df['outdoor_temp'].shift(1)
        merged_df['temp_diff_outdoor'] = merged_df[sens_temp_col] - merged_df['outdoor_temp_lag_1']
        lag_cols['outdoor_temp_lag_1'] = True
        lag_cols['temp_diff_outdoor'] = True
        for window in [15, 60]:
            if window <= look_back:
                outdoor_roll = f'outdoor_temp_roll_{window}'
                outdoor_past = merged_df['outdoor_temp'].shift(1)
                merged_df[outdoor_roll] = outdoor_past.rolling(window=window, min_periods=1).mean()
                lag_cols[outdoor_roll] = True
    if 'solar_radiation' in merged_df.columns:
        merged_df['solar_radiation_lag_1'] = merged_df['solar_radiation'].shift(1)
        lag_cols['solar_radiation_lag_1'] = True
        for window in [15, 60]:
            if window <= look_back:
                solar_roll = f'solar_radiation_roll_{window}'
                solar_past = merged_df['solar_radiation'].shift(1)
                merged_df[solar_roll] = solar_past.rolling(window=window, min_periods=1).mean()
                lag_cols[solar_roll] = True
    if 'humidity' in merged_df.columns:
        merged_df['humidity_lag_1'] = merged_df['humidity'].shift(1)
        lag_cols['humidity_lag_1'] = True
        for window in [15, 60]:
            if window <= look_back:
                humidity_roll = f'humidity_roll_{window}'
                humidity_past = merged_df['humidity'].shift(1)
                merged_df[humidity_roll] = humidity_past.rolling(window=window, min_periods=1).mean()
                lag_cols[humidity_roll] = True
    merged_df['thermo_x_temp'] = merged_df[thermo_col] * merged_df[sens_temp_col]
    merged_df['thermo_duration_x_temp'] = merged_df['thermo_duration'] * merged_df[sens_temp_col]
    if f'{sens_temp_col}_change_5' in merged_df.columns:
        merged_df['thermo_on_temp_change'] = merged_df[thermo_col] * merged_df[f'{sens_temp_col}_change_5']
    else:
        merged_df['thermo_on_temp_change'] = 0
    if 'outdoor_temp' in merged_df.columns and 'solar_radiation' in merged_df.columns:
        merged_df['is_sunny_day'] = ((merged_df['hour'] >= 9) &
                                    (merged_df['hour'] <= 17) &
                                    (merged_df['solar_radiation'] >
                                     merged_df['solar_radiation'].mean())).astype(int)
        merged_df['is_cold_night'] = ((merged_df['is_night'] == 1) &
                                     (merged_df['outdoor_temp'] <
                                      merged_df['outdoor_temp'].mean())).astype(int)
    merged_df = merged_df.dropna()
    feature_columns = [
        'hour', 'day_of_week', 'month', 'is_weekend',
        'is_night', 'is_morning', 'is_afternoon', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos', 'week_sin', 'week_cos',
        valid_col, mode_col, thermo_col, thermo_or_col,
        'thermo_duration', 'time_since_thermo_change', 'thermo_change',
        power_col,
        'thermo_x_temp', 'thermo_duration_x_temp', 'thermo_on_temp_change'
    ]
    if 'is_sunny_day' in merged_df.columns:
        feature_columns.append('is_sunny_day')
    if 'is_cold_night' in merged_df.columns:
        feature_columns.append('is_cold_night')
    feature_columns.extend([col for col in lag_cols.keys() if col in merged_df.columns])
    target = target_col
    X = merged_df[feature_columns]
    y = merged_df[target]
    return X, y, merged_df

def train_lgbm_model(X, y, test_size=0.2, random_state=42):
    try:
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'force_col_wise': True,
        }
        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=100, show_stdv=False)
        ]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=callbacks
        )
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        print("\nTop 10 Feature Importance:")
        print(importance_df.head(10))
        return model, X_test, y_test, y_pred, importance_df
    except Exception as e:
        print(f"Error in train_lgbm_model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def visualize_feature_importance(importance_df, zone, output_dir):
    try:
        if importance_df is None or importance_df.empty:
            print(f"Zone {zone}: 特徴量重要度データが空です")
            return None
        top_features = importance_df.head(15).copy()
        total_importance = importance_df['Importance'].sum()
        if total_importance > 0:
            top_features['Importance_Pct'] = top_features['Importance'] / total_importance * 100
        else:
            top_features['Importance_Pct'] = 0
        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            top_features['Feature'],
            top_features['Importance_Pct'],
            color=plt.cm.viridis(np.linspace(0, 0.8, len(top_features))),
            edgecolor='gray',
            alpha=0.8
        )
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{top_features['Importance_Pct'].iloc[i]:.1f}%",
                va='center'
            )
        plt.xlabel('重要度 (%)')
        plt.ylabel('特徴量')
        plt.title(f'ゾーン {zone} のトップ15特徴量重要度')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f'zone_{zone}_feature_importance.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        return fig_path
    except Exception as e:
        print(f"Error in visualize_feature_importance for zone {zone}: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_features_for_prediction_without_dropna(df, zone, power_col):
    try:
        valid_col = f'AC_valid_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        sens_temp_col = f'sens_temp_{zone}'
        thermo_or_map = {'L': 'thermo_L_or', 'M': 'thermo_M_or', 'R': 'thermo_R_or'}
        if power_col not in thermo_or_map:
            raise ValueError(f"Invalid power column: {power_col}")
        thermo_or_col = thermo_or_map[power_col]
        required_cols = [valid_col, mode_col, thermo_col, sens_temp_col, power_col, thermo_or_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for zone {zone}: {', '.join(missing_cols)}")

        # 特徴量のリスト（基本的特徴量）を定義
        feature_columns = [
            # 時間特徴量
            'hour', 'day_of_week', 'month', 'is_weekend',
            'is_night', 'is_morning', 'is_afternoon', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'week_sin', 'week_cos',
            # HVAC操作特徴量
            valid_col, mode_col, thermo_col, thermo_or_col,
            'thermo_duration', 'time_since_thermo_change', 'thermo_change',
            # 電力消費
            power_col,
            # インタラクション特徴量
            'thermo_x_temp', 'thermo_duration_x_temp', 'thermo_on_temp_change'
        ]

        features_df = df
        hour = features_df['time_stamp'].dt.hour
        day_of_week = features_df['time_stamp'].dt.dayofweek
        day_of_month = features_df['time_stamp'].dt.day
        month = features_df['time_stamp'].dt.month
        features_df['hour'] = hour
        features_df['day_of_week'] = day_of_week
        features_df['month'] = month
        features_df['is_weekend'] = (day_of_week >= 5).astype(int)
        features_df['is_night'] = ((hour >= 19) | (hour <= 6)).astype(int)
        features_df['is_morning'] = ((hour > 6) & (hour <= 12)).astype(int)
        features_df['is_afternoon'] = ((hour > 12) & (hour < 19)).astype(int)
        hour_rad = 2 * np.pi * hour / 24
        features_df['hour_sin'] = np.sin(hour_rad)
        features_df['hour_cos'] = np.cos(hour_rad)
        day_rad = 2 * np.pi * day_of_month / 31
        features_df['day_sin'] = np.sin(day_rad)
        features_df['day_cos'] = np.cos(day_rad)
        week_rad = 2 * np.pi * day_of_week / 7
        features_df['week_sin'] = np.sin(week_rad)
        features_df['week_cos'] = np.cos(week_rad)

        for lag in [1, 2, 5, 15, 30, 60]:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            features_df[lag_col] = features_df[sens_temp_col].shift(lag)
            feature_columns.append(lag_col)  # 特徴量リストに追加

        for lag in [1, 5, 15]:
            change_col = f'{sens_temp_col}_change_{lag}'
            if lag == 1:
                features_df[f'{sens_temp_col}_change_{lag}'] = features_df[sens_temp_col] - features_df[f'{sens_temp_col}_lag_1']
            else:
                features_df[f'{sens_temp_col}_change_{lag}'] = (features_df[sens_temp_col] - features_df[f'{sens_temp_col}_lag_{lag}']) / lag
            feature_columns.append(change_col)  # 特徴量リストに追加


        for lag in [1, 5, 15, 30]:
            lag_col = f'{power_col}_lag_{lag}'
            features_df[f'{power_col}_lag_{lag}'] = features_df[power_col].shift(lag)
            feature_columns.append(lag_col)

        windows = [5, 15, 30, 60]
        for window in windows:
            roll_temp = f'{sens_temp_col}_roll_{window}'
            temp_past = features_df[sens_temp_col].shift(1)
            features_df[roll_temp] = temp_past.rolling(window=window, min_periods=1).mean()
            feature_columns.append(roll_temp)

            roll_power = f'{power_col}_roll_{window}'
            power_past = features_df[power_col].shift(1)
            features_df[roll_power] = power_past.rolling(window=window, min_periods=1).mean()
            feature_columns.append(roll_power)

            std_temp = f'{sens_temp_col}_std_{window}'
            features_df[std_temp] = temp_past.rolling(window=window, min_periods=1).std()
            feature_columns.append(std_temp)
        features_df['thermo_change'] = features_df[thermo_col].diff(1).fillna(0)
        reset_points = (features_df[thermo_col] != features_df[thermo_col].shift(1)).astype(int)
        reset_points.iloc[0] = 1
        group_id = reset_points.cumsum()
        features_df['thermo_duration'] = features_df.groupby(group_id).cumcount()

        change_points = (features_df['thermo_change'] != 0).astype(int)
        change_group_id = change_points.cumsum()
        features_df['time_since_thermo_change'] = features_df.groupby(change_group_id).cumcount()

        if 'outdoor_temp' in features_df.columns:
            outdoor_lag = 'outdoor_temp_lag_1'
            features_df[outdoor_lag] = features_df['outdoor_temp'].shift(1)
            features_df['temp_diff_outdoor'] = features_df[sens_temp_col] - features_df[outdoor_lag]
            feature_columns.append(outdoor_lag)
            feature_columns.append('temp_diff_outdoor')

            for window in [15, 60]:
                outdoor_roll = f'outdoor_temp_roll_{window}'
                outdoor_past = features_df['outdoor_temp'].shift(1)
                features_df[outdoor_roll] = outdoor_past.rolling(window=window, min_periods=1).mean()
                feature_columns.append(outdoor_roll)


        if thermo_col in features_df.columns:
            features_df['thermo_x_temp'] = features_df[thermo_col] * features_df[sens_temp_col]
            features_df['thermo_duration_x_temp'] = features_df['thermo_duration'] * features_df[sens_temp_col]

        if 'outdoor_temp' in features_df.columns and 'solar_radiation' in features_df.columns:
            features_df['is_sunny_day'] = ((features_df['hour'] >= 9) &
                                         (features_df['hour'] <= 17) &
                                         (features_df['solar_radiation'] >
                                          features_df['solar_radiation'].mean())).astype(int)
            feature_columns.append('is_sunny_day')

            features_df['is_cold_night'] = ((features_df['is_night'] == 1) &
                                          (features_df['outdoor_temp'] <
                                           features_df['outdoor_temp'].mean())).astype(int)
            feature_columns.append('is_cold_night')

        available_columns = [col for col in feature_columns if col in features_df.columns]
        return features_df[available_columns]
    except Exception as e:
        print(f"Error in prepare_features_for_prediction_without_dropna: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_zone_with_predictions(df, thermo_df, zone, power_col, model, features_df, output_dir,
                                   start_date=None, end_date=None, days_to_show=7):
    try:
        valid_col = f'AC_valid_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        sens_temp_col = f'sens_temp_{zone}'
        set_col = f'AC_set_{zone}'
        if valid_col not in df.columns or sens_temp_col not in df.columns:
            print(f"Missing required columns for visualization of zone {zone}")
            return None
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        else:
            start = df['time_stamp'].min()
            end = start + pd.Timedelta(days=days_to_show)
        time_mask_df = (df['time_stamp'] >= start) & (df['time_stamp'] <= end)
        time_mask_thermo = (thermo_df['time_stamp'] >= start) & (thermo_df['time_stamp'] <= end)
        if time_mask_df.sum() == 0 or time_mask_thermo.sum() == 0:
            print(f"No data available for zone {zone} in the selected period")
            return None
        required_cols_df = ['time_stamp', valid_col, mode_col, sens_temp_col, power_col]
        if set_col in df.columns:
            required_cols_df.append(set_col)
        filtered_df = df.loc[time_mask_df, required_cols_df].copy()
        required_cols_thermo = ['time_stamp', thermo_col]
        thermo_or_col = f'thermo_{power_col}_or'
        if thermo_or_col in thermo_df.columns:
            required_cols_thermo.append(thermo_or_col)
        filtered_thermo = thermo_df.loc[time_mask_thermo, required_cols_thermo].copy()
        merged_df = pd.merge(filtered_df, filtered_thermo, on='time_stamp', how='left')
        model_features = model.feature_name()
        temp_features = prepare_features_for_prediction_without_dropna(merged_df, zone, power_col)
        if temp_features is None:
            print(f"Could not prepare features for prediction for zone {zone}")
            return None
        missing_features = [feat for feat in model_features if feat not in temp_features.columns]
        if missing_features:
            for feat in missing_features:
                temp_features[feat] = 0
            if len(missing_features) <= 3:
                print(f"Added missing features for zone {zone}: {', '.join(missing_features)}")
            else:
                print(f"Added {len(missing_features)} missing features for zone {zone}")
        X_for_pred = temp_features[model_features].copy()
        valid_indices = ~X_for_pred.isna().any(axis=1)
        valid_rows = valid_indices.sum()
        if valid_rows == 0:
            print(f"No valid rows without NaN for prediction in zone {zone}")
            return None
        X_for_pred = X_for_pred[valid_indices].copy()
        try:
            predictions = model.predict(X_for_pred, predict_disable_shape_check=True)
            merged_df[f'{sens_temp_col}_pred'] = np.nan
            merged_df.loc[merged_df.index[valid_indices], f'{sens_temp_col}_pred'] = predictions
        except Exception as e:
            print(f"Error making predictions for zone {zone}: {e}")
            print(f"Model features: {len(model_features)}, Prediction features: {X_for_pred.shape[1]}")
            if len(model_features) != X_for_pred.shape[1]:
                print(f"Feature count mismatch: model={len(model_features)}, data={X_for_pred.shape[1]}")
                max_show = min(5, len(model_features))
                print(f"First {max_show} model features: {model_features[:max_show]}")
                print(f"First {max_show} data features: {list(X_for_pred.columns[:max_show])}")
            return None
        fig, ax = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1], sharex=True)
        ax[0].plot(merged_df['time_stamp'], merged_df[sens_temp_col], 'c-', linewidth=2, label='実測温度')
        ax[0].plot(merged_df['time_stamp'], merged_df[f'{sens_temp_col}_pred'], 'm--', linewidth=2, label='予測温度')
        if set_col in merged_df.columns:
            ax[0].plot(merged_df['time_stamp'], merged_df[set_col], 'g-', linewidth=1.5, label='設定温度')
            ax[0].fill_between(merged_df['time_stamp'], merged_df[set_col] - 1.0, merged_df[set_col] + 1.0,
                          color='gray', alpha=0.2, label='不感帯(±1.0°C)')
        ax0_twin = ax[0].twinx()
        ax0_twin.plot(merged_df['time_stamp'], merged_df[thermo_col], 'r-', linewidth=1.5, label='サーモ状態')
        if mode_col in merged_df.columns:
            modes = merged_df[mode_col].copy()
            cooling_mask = (modes == 1)
            heating_mask = (modes == 2)
            if cooling_mask.any():
                cooling_df = merged_df[cooling_mask].copy()
                ax0_twin.plot(cooling_df['time_stamp'], cooling_df[thermo_col], 'b-', linewidth=1.5, label='冷房ON')
            if heating_mask.any():
                heating_df = merged_df[heating_mask].copy()
                ax0_twin.plot(heating_df['time_stamp'], heating_df[thermo_col], 'r-', linewidth=1.5, label='暖房ON')
        ax0_twin.set_ylim(-0.1, 1.1)
        ax0_twin.set_ylabel('運転状態', fontsize=12)
        ax0_twin.tick_params(axis='y', labelsize=10)
        temp_min = merged_df[sens_temp_col].min()
        temp_max = merged_df[sens_temp_col].max()
        padding = (temp_max - temp_min) * 0.1
        ax[0].set_ylim(temp_min - padding, temp_max + padding)
        lines0, labels0 = ax[0].get_legend_handles_labels()
        lines0_twin, labels0_twin = ax0_twin.get_legend_handles_labels()
        ax[0].legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper right', fontsize=10)
        ax[0].set_ylabel('温度 (°C)', fontsize=12)
        ax[0].tick_params(axis='y', labelsize=10)
        ax[0].set_title(f'ゾーン {zone} - 温度予測と実測値', fontsize=14)
        ax[0].grid(True, alpha=0.3)
        if power_col in merged_df.columns:
            ax[1].plot(merged_df['time_stamp'], merged_df[power_col], 'b-', linewidth=1.5, label='消費電力')
            ax[1].set_ylabel('消費電力 (kW)', fontsize=12)
            ax[1].tick_params(axis='y', labelsize=10)
            if valid_col in merged_df.columns:
                ax1_twin = ax[1].twinx()
                ax1_twin.plot(merged_df['time_stamp'], merged_df[valid_col], 'g--', linewidth=1.5, label='空調ON/OFF')
                ax1_twin.set_ylim(-0.1, 1.1)
                ax1_twin.set_ylabel('空調状態', fontsize=12)
                ax1_twin.tick_params(axis='y', labelsize=10)
                lines1, labels1 = ax[1].get_legend_handles_labels()
                lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
                ax[1].legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper right', fontsize=10)
            else:
                ax[1].legend(loc='upper right', fontsize=10)
            ax[1].set_title(f'ゾーン {zone} - 消費電力と空調状態', fontsize=14)
            ax[1].grid(True, alpha=0.3)
        else:
            ax[1].text(0.5, 0.5, f'ゾーン {zone} の電力データがありません',
                      ha='center', va='center', fontsize=12)
            ax[1].set_title(f'ゾーン {zone}', fontsize=14)
        fig.autofmt_xdate()
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        fig.tight_layout()
        fig_scatter = plt.figure(figsize=(12, 10))
        ax_scatter = fig_scatter.add_subplot(111)
        valid_data = merged_df.dropna(subset=[sens_temp_col, f'{sens_temp_col}_pred'])
        if not valid_data.empty:
            timestamps = pd.to_datetime(valid_data['time_stamp'])
            hours = timestamps.dt.hour
            time_periods = pd.cut(
                hours,
                bins=[0, 6, 12, 18, 24],
                labels=['夜間\n(0-6時)', '午前\n(6-12時)', '午後\n(12-18時)', '夕方/夜\n(18-24時)'],
                include_lowest=True
            )
            cmap = plt.cm.viridis
            colors = {
                '夜間\n(0-6時)': cmap(0.1),
                '午前\n(6-12時)': cmap(0.4),
                '午後\n(12-18時)': cmap(0.7),
                '夕方/夜\n(18-24時)': cmap(0.9)
            }
            for period in np.unique(time_periods):
                mask = (time_periods == period)
                ax_scatter.scatter(
                    valid_data[sens_temp_col][mask],
                    valid_data[f'{sens_temp_col}_pred'][mask],
                    c=[colors[period]],
                    label=period,
                    alpha=0.7,
                    s=50,
                    edgecolor='k',
                    linewidth=0.3
                )
            min_val = min(valid_data[sens_temp_col].min(), valid_data[f'{sens_temp_col}_pred'].min())
            max_val = max(valid_data[sens_temp_col].max(), valid_data[f'{sens_temp_col}_pred'].max())
            padding = (max_val - min_val) * 0.05
            ax_scatter.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'k--', alpha=0.7, label='完全一致線')
            ax_scatter.set_xlim(min_val-padding, max_val+padding)
            ax_scatter.set_ylim(min_val-padding, max_val+padding)
            ax_scatter.set_xlabel('実測温度 (°C)', fontsize=14)
            ax_scatter.set_ylabel('予測温度 (°C)', fontsize=14)
            ax_scatter.tick_params(axis='both', labelsize=12)
            actual = valid_data[sens_temp_col]
            predicted = valid_data[f'{sens_temp_col}_pred']
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
            max_error = np.max(np.abs(actual - predicted))
            stats_text = (
                f'評価指標:\n'
                f'RMSE: {rmse:.3f}°C\n'
                f'MAE: {mae:.3f}°C\n'
                f'R²: {r2:.3f}\n'
                f'最大誤差: {max_error:.3f}°C\n'
                f'データ数: {len(actual)}点'
            )
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax_scatter.text(
                0.05, 0.95, stats_text,
                transform=ax_scatter.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=props
            )
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val-padding, max_val+padding, 100)
            ax_scatter.plot(x_trend, p(x_trend), "r-", linewidth=1.5, alpha=0.6, label=f'傾向線 (y={z[0]:.2f}x+{z[1]:.2f})')
            ax_scatter.legend(loc='lower right', fontsize=12)
            ax_scatter.grid(True, alpha=0.3)
            title = f'ゾーン {zone} - 実測温度 vs 予測温度の比較'
            subtitle = f'期間: {start.strftime("%Y-%m-%d")} から {end.strftime("%Y-%m-%d")}'
            ax_scatter.set_title(f'{title}\n{subtitle}', fontsize=16)
        else:
            ax_scatter.text(0.5, 0.5, '有効なデータがありません', ha='center', va='center', fontsize=14)
            ax_scatter.set_title(f'ゾーン {zone} - データなし', fontsize=16)
        fig_scatter.tight_layout()
        return fig, fig_scatter
    except Exception as e:
        print(f"Error in visualize_zone_with_predictions for zone {zone}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    import pandas as pd
    import numpy as np
    import os
    import time
    import logging
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    start_time = time.time()
    file_path = './AllDayData.csv'
    output_dir = './output/sens_temp_predictions'
    prediction_horizon = 5
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',
        2: 'M', 3: 'M', 8: 'M', 9: 'M',
        4: 'R', 5: 'R', 10: 'R', 11: 'R'
    }
    os.makedirs(output_dir, exist_ok=True)
    try:
        logger.info(f"データを読み込んでいます: {file_path}")
        read_start = time.time()
        try:
            df = pd.read_csv(file_path)
            read_time = time.time() - read_start
            logger.info(f"データ読み込み完了: {len(df)}行 × {len(df.columns)}列 ({read_time:.2f}秒)")
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return
        preprocess_start = time.time()
        if 'algo' in df.columns:
            before_rows = len(df)
            df = df.dropna(subset=['algo'])
            logger.info(f"'algo'列のNaN行を削除しました: {before_rows} → {len(df)}行")
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        start_date = '2024-06-26'
        end_date = '2024-09-20'
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        filtered_df = df[(df['time_stamp'] >= start) & (df['time_stamp'] <= end)]
        logger.info(f"分析期間: {start_date} から {end_date} ({len(filtered_df)}行)")
        logger.info("サーモ状態を計算しています...")
        thermo_start = time.time()
        thermo_df = determine_thermo_status(filtered_df)
        wanted_columns = ['time_stamp'] + [f'thermo_{i}' for i in range(12)] + ['thermo_L_or', 'thermo_M_or', 'thermo_R_or']
        for col in wanted_columns:
            if col not in thermo_df.columns and col != 'time_stamp':
                thermo_df[col] = 0
        thermo_df = thermo_df[wanted_columns]
        thermo_time = time.time() - thermo_start
        logger.info(f"サーモ状態の計算が完了しました ({thermo_time:.2f}秒)")
        preprocess_time = time.time() - preprocess_start
        logger.info(f"前処理完了 ({preprocess_time:.2f}秒)")
        results_df = pd.DataFrame(columns=['Zone', 'RMSE', 'MAE', 'MAPE', 'R2', 'Top_Features', 'Training_Time'])
        for zone in range(12):
            zone_start = time.time()
            logger.info(f"\n===== ゾーン {zone} の処理開始 =====")
            if zone not in zone_to_power:
                logger.warning(f"ゾーン {zone} はどの室外機にも割り当てられていません")
                continue
            power_col = zone_to_power[zone]
            valid_col = f'AC_valid_{zone}'
            mode_col = f'AC_mode_{zone}'
            sens_temp_col = f'sens_temp_{zone}'
            required_cols = [valid_col, mode_col, sens_temp_col, power_col]
            missing_cols = [col for col in required_cols if col not in filtered_df.columns]
            if missing_cols:
                logger.warning(f"ゾーン {zone} に必要なカラムがありません: {', '.join(missing_cols)}")
                continue
            try:
                feature_start = time.time()
                logger.info(f"ゾーン {zone} の特徴量を作成中...")
                X, y, features_df = prepare_features_for_sens_temp(filtered_df, thermo_df, zone, prediction_horizon=prediction_horizon)
                if X is None or y is None or features_df is None:
                    logger.warning(f"ゾーン {zone} の特徴量を作成できませんでした")
                    continue
                feature_time = time.time() - feature_start
                logger.info(f"特徴量作成完了: {X.shape} ({feature_time:.2f}秒)")
                train_start = time.time()
                logger.info(f"ゾーン {zone} の LightGBM モデルをトレーニングしています...")
                model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)
                train_time = time.time() - train_start
                logger.info(f"モデルトレーニング完了 ({train_time:.2f}秒)")
                viz_start = time.time()
                importance_path = visualize_feature_importance(importance_df, zone, output_dir)
                logger.info(f"特徴量重要度グラフを保存しました: {importance_path}")
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = np.mean(np.abs(y_test - y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
                r2 = r2_score(y_test, y_pred)
                top_features = ', '.join(importance_df.head(3)['Feature'].tolist())
                results_df = pd.concat([
                    results_df,
                    pd.DataFrame({
                        'Zone': [zone],
                        'RMSE': [rmse],
                        'MAE': [mae],
                        'MAPE': [mape],
                        'R2': [r2],
                        'Top_Features': [top_features],
                        'Training_Time': [train_time]
                    })
                ], ignore_index=True)
                logger.info(f"評価指標: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")
                first_week_start = start
                first_week_end = first_week_start + timedelta(days=7)
                last_week_end = end
                last_week_start = last_week_end - timedelta(days=7)
                logger.info(f"ゾーン {zone} の最初の週の予測を可視化しています...")
                first_week_figs = visualize_zone_with_predictions(
                    filtered_df, thermo_df, zone, power_col, model, features_df, output_dir,
                    start_date=first_week_start, end_date=first_week_end
                )
                if first_week_figs:
                    first_week_fig, first_week_scatter = first_week_figs
                    if first_week_fig:
                        first_week_path = os.path.join(output_dir, f'zone_{zone}_first_week.png')
                        first_week_fig.savefig(first_week_path, dpi=150)
                        plt.close(first_week_fig)
                        logger.info(f"最初の週の時系列可視化を保存しました: {first_week_path}")
                    if first_week_scatter:
                        first_week_scatter_path = os.path.join(output_dir, f'zone_{zone}_first_week_scatter.png')
                        first_week_scatter.savefig(first_week_scatter_path, dpi=150)
                        plt.close(first_week_scatter)
                        logger.info(f"最初の週の散布図を保存しました: {first_week_scatter_path}")
                logger.info(f"ゾーン {zone} の最後の週の予測を可視化しています...")
                last_week_figs = visualize_zone_with_predictions(
                    filtered_df, thermo_df, zone, power_col, model, features_df, output_dir,
                    start_date=last_week_start, end_date=last_week_end
                )
                if last_week_figs:
                    last_week_fig, last_week_scatter = last_week_figs
                    if last_week_fig:
                        last_week_path = os.path.join(output_dir, f'zone_{zone}_last_week.png')
                        last_week_fig.savefig(last_week_path, dpi=150)
                        plt.close(last_week_fig)
                        logger.info(f"最後の週の時系列可視化を保存しました: {last_week_path}")
                    if last_week_scatter:
                        last_week_scatter_path = os.path.join(output_dir, f'zone_{zone}_last_week_scatter.png')
                        last_week_scatter.savefig(last_week_scatter_path, dpi=150)
                        plt.close(last_week_scatter)
                        logger.info(f"最後の週の散布図を保存しました: {last_week_scatter_path}")
                viz_time = time.time() - viz_start
                logger.info(f"可視化完了 ({viz_time:.2f}秒)")
                zone_time = time.time() - zone_start
                logger.info(f"ゾーン {zone} の処理が完了しました (合計: {zone_time:.2f}秒)")
            except Exception as e:
                logger.error(f"ゾーン {zone} の処理中にエラーが発生しました: {e}")
                import traceback
                logger.error(traceback.format_exc())
        if not results_df.empty:
            results_path = os.path.join(output_dir, 'model_results_summary.csv')
            results_df.to_csv(results_path, index=False)
            logger.info(f"モデル結果サマリーを保存しました: {results_path}")
            logger.info("全体結果を可視化しています...")
            visualize_overall_results(results_df, output_dir)
            logger.info("全体結果の可視化が完了しました")
        else:
            logger.warning("処理結果がありません。結果サマリーは保存されませんでした。")
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"\n全処理が完了しました！ 合計処理時間: {int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒")
    except Exception as e:
        logger.error(f"メイン処理中に予期せぬエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

def visualize_overall_results(results_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sns.barplot(x='Zone', y='RMSE', data=results_df, ax=axes[0, 0])
    axes[0, 0].set_title('RMSE by Zone')
    axes[0, 0].set_xlabel('Zone')
    axes[0, 0].set_ylabel('RMSE (°C)')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    sns.barplot(x='Zone', y='MAE', data=results_df, ax=axes[0, 1])
    axes[0, 1].set_title('MAE by Zone')
    axes[0, 1].set_xlabel('Zone')
    axes[0, 1].set_ylabel('MAE (°C)')
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    sns.barplot(x='Zone', y='MAPE', data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title('MAPE by Zone')
    axes[1, 0].set_xlabel('Zone')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    sns.barplot(x='Zone', y='R2', data=results_df, ax=axes[1, 1])
    axes[1, 1].set_title('R² by Zone')
    axes[1, 1].set_xlabel('Zone')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=150)
    plt.close()
    all_features = []
    for features in results_df['Top_Features']:
        all_features.extend([f.strip() for f in features.split(',')])
    feature_counts = pd.Series(all_features).value_counts().head(15)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_counts.values, y=feature_counts.index)
    plt.title('Most Common Top Features Across All Zones')
    plt.xlabel('Frequency')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_features_overall.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
