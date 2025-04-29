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
import argparse

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

def prepare_features_for_sens_temp(df, thermo_df, zone, look_back=60, prediction_horizon=5, feature_selection=True, importance_threshold=0.05, max_features=6):
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
    optional_cols = ['outdoor_temp', 'solar_radiation']  # 'humidity'は削除
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

    # シンプルな時間特徴量のみを保持
    merged_df['hour'] = merged_df['time_stamp'].dt.hour
    merged_df['day_of_week'] = merged_df['time_stamp'].dt.dayofweek
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
    merged_df['is_night'] = ((merged_df['hour'] >= 19) | (merged_df['hour'] <= 6)).astype(int)

    # 周期的特徴量（サイン・コサイン変換）- 時間のみ保持
    hour_rad = 2 * np.pi * merged_df['hour'] / 24
    merged_df['hour_sin'] = np.sin(hour_rad)
    merged_df['hour_cos'] = np.cos(hour_rad)

    # 目的変数の設定
    target_col = f'{sens_temp_col}_future_{prediction_horizon}'
    merged_df[target_col] = merged_df[sens_temp_col].shift(prediction_horizon)

    lag_cols = {}

    # 温度のラグ特徴量 - 最も重要な3つのみ保持
    lag_values = [1, 5, 15]

    for lag in lag_values:
        if lag <= look_back:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            merged_df[lag_col] = merged_df[sens_temp_col].shift(lag)
            lag_cols[lag_col] = True

    # 温度変化率 - 最も重要な1つのみ保持
    change_col = f'{sens_temp_col}_change_5'
    merged_df[change_col] = (merged_df[sens_temp_col] - merged_df[f'{sens_temp_col}_lag_5']) / 5
    lag_cols[change_col] = True

    # 電力消費のラグ特徴量 - 最も重要な1つのみ保持
    lag_col = f'{power_col}_lag_1'
    merged_df[lag_col] = merged_df[power_col].shift(1)
    lag_cols[lag_col] = True

    # 移動平均 - 重要度の高い1つのウィンドウサイズのみ保持
    window = 15
    roll_temp = f'{sens_temp_col}_roll_{window}'
    temp_past = merged_df[sens_temp_col].shift(1)
    merged_df[roll_temp] = temp_past.rolling(window=window, min_periods=1).mean()

    roll_power = f'{power_col}_roll_{window}'
    power_past = merged_df[power_col].shift(1)
    merged_df[roll_power] = power_past.rolling(window=window, min_periods=1).mean()

    lag_cols[roll_temp] = True
    lag_cols[roll_power] = True

    # サーモ状態の変化と持続時間 - 最も重要なもののみ保持
    merged_df['thermo_change'] = merged_df[thermo_col].diff(1).fillna(0)
    reset_points = (merged_df[thermo_col] != merged_df[thermo_col].shift(1)).astype(int)
    reset_points.iloc[0] = 1
    group_id = reset_points.cumsum()
    merged_df['thermo_duration'] = merged_df.groupby(group_id).cumcount()

    # 外気温特徴量 - 最も重要なもののみ保持
    if 'outdoor_temp' in merged_df.columns:
        merged_df['outdoor_temp_lag_1'] = merged_df['outdoor_temp'].shift(1)
        merged_df['temp_diff_outdoor'] = merged_df[sens_temp_col] - merged_df['outdoor_temp_lag_1']
        lag_cols['outdoor_temp_lag_1'] = True
        lag_cols['temp_diff_outdoor'] = True

    # 交互作用特徴量 - 最も重要なもののみ保持
    merged_df['thermo_x_temp'] = merged_df[thermo_col] * merged_df[sens_temp_col]

    if f'{sens_temp_col}_change_5' in merged_df.columns:
        merged_df['thermo_on_temp_change'] = merged_df[thermo_col] * merged_df[f'{sens_temp_col}_change_5']
    else:
        merged_df['thermo_on_temp_change'] = 0

    # 天候状態特徴量 - 条件付きで保持
    if 'outdoor_temp' in merged_df.columns and 'solar_radiation' in merged_df.columns:
        merged_df['is_sunny_day'] = ((merged_df['hour'] >= 9) &
                                    (merged_df['hour'] <= 17) &
                                    (merged_df['solar_radiation'] >
                                     merged_df['solar_radiation'].mean())).astype(int)

    # NaN値を含む行を削除
    merged_df = merged_df.dropna()

    # 基本特徴量（常に含める最小限の特徴量）
    base_feature_columns = [
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'hour_sin', 'hour_cos',
        valid_col, mode_col, thermo_col, thermo_or_col,
        'thermo_duration', 'thermo_change',
        power_col,
        'thermo_x_temp', f'{sens_temp_col}_lag_1'
    ]

    # 条件付き特徴量
    conditional_features = []
    if 'is_sunny_day' in merged_df.columns:
        conditional_features.append('is_sunny_day')

    # ラグ特徴量
    lag_feature_columns = [col for col in lag_cols.keys() if col in merged_df.columns]

    # 全特徴量の結合
    all_feature_columns = base_feature_columns + conditional_features + lag_feature_columns
    feature_columns = list(set(all_feature_columns))  # 重複を除去

    # 特徴量選択が有効な場合、初期モデルを訓練して重要度に基づいて特徴量を選択
    if feature_selection and len(feature_columns) > max_features:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split

        X_initial = merged_df[feature_columns]
        y_initial = merged_df[target_col]

        # 簡易的なモデルを訓練して特徴量重要度を取得
        X_train, X_val, y_train, y_val = train_test_split(X_initial, y_initial, test_size=0.2, random_state=42)
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'verbosity': -1,
            'force_col_wise': True,
        }

        # callbacksを使用して早期停止を設定
        callbacks = [
            lgb.early_stopping(20),
            lgb.log_evaluation(period=100, show_stdv=False)
        ]

        initial_model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=callbacks
        )

        # 特徴量重要度を計算
        importance = initial_model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # 重要度の合計を計算
        total_importance = importance_df['Importance'].sum()
        importance_df['Importance_Ratio'] = importance_df['Importance'] / total_importance

        # 累積重要度
        importance_df['Cumulative_Importance'] = importance_df['Importance_Ratio'].cumsum()

        # 特徴量選択方法の改善:
        # 1. 累積重要度が90%に達するまでの特徴量を選択（95%→90%に変更）
        cumulative_features = importance_df[importance_df['Cumulative_Importance'] <= 0.90]['Feature'].tolist()

        # 2. 重要度の閾値以上の特徴量を選択（閾値を上げて厳しく選択）
        threshold_features = importance_df[
            importance_df['Importance_Ratio'] >= importance_threshold
        ]['Feature'].tolist()

        # 3. 上位max_features個の特徴量を選択
        top_features = importance_df.head(max_features)['Feature'].tolist()

        # 特徴量選択基準を組み合わせる（優先順位: 閾値 > 上位N個 > 累積90%）
        selected_features = threshold_features

        # 閾値ベースの選択が十分な特徴量を得られない場合、上位N個を追加
        if len(selected_features) < 5:
            remaining_needed = 5 - len(selected_features)
            for feat in top_features:
                if feat not in selected_features and remaining_needed > 0:
                    selected_features.append(feat)
                    remaining_needed -= 1

        # それでも特徴量が足りない場合、累積重要度ベースの特徴量を追加
        if len(selected_features) < 5:
            remaining_needed = 5 - len(selected_features)
            for feat in cumulative_features:
                if feat not in selected_features and remaining_needed > 0:
                    selected_features.append(feat)
                    remaining_needed -= 1

        # 特徴量数制限
        if max_features > 0 and len(selected_features) > max_features:
            # 上位max_features個に制限
            selected_features = importance_df[
                importance_df['Feature'].isin(selected_features)
            ].head(max_features)['Feature'].tolist()

        # 常に含める必須特徴量
        critical_features = [valid_col, mode_col, thermo_col, power_col, f'{sens_temp_col}_lag_1']
        for feat in critical_features:
            if feat in merged_df.columns and feat not in selected_features:
                selected_features.append(feat)

        print(f"Zone {zone}: 特徴量数を {len(feature_columns)} から {len(selected_features)} に削減しました")
        print(f"選択された特徴量: {', '.join(selected_features)}")

        # 上位5個の特徴量と重要度を表示
        top5_importance = importance_df.head(5)
        print("Top 5 特徴量重要度:")
        for i, row in top5_importance.iterrows():
            print(f"  {row['Feature']}: {row['Importance_Ratio']*100:.2f}%")

        feature_columns = selected_features

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

        # 特徴量のリスト（必要最小限の特徴量のみに厳選）
        feature_columns = [
            # 時間特徴量（シンプル化）
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'hour_sin', 'hour_cos',
            # HVAC操作特徴量（重要なもののみ）
            valid_col, mode_col, thermo_col, thermo_or_col,
            'thermo_duration', 'thermo_change',
            # 電力消費
            power_col,
            # 重要度の高いインタラクション特徴量
            'thermo_x_temp'
        ]

        features_df = df
        # 時間関連特徴量の計算
        hour = features_df['time_stamp'].dt.hour
        day_of_week = features_df['time_stamp'].dt.dayofweek
        features_df['hour'] = hour
        features_df['day_of_week'] = day_of_week
        features_df['is_weekend'] = (day_of_week >= 5).astype(int)
        features_df['is_night'] = ((hour >= 19) | (hour <= 6)).astype(int)

        # 周期的時間特徴量（時間のみ）
        hour_rad = 2 * np.pi * hour / 24
        features_df['hour_sin'] = np.sin(hour_rad)
        features_df['hour_cos'] = np.cos(hour_rad)

        # 重要な温度のラグ特徴量のみ追加
        for lag in [1, 5, 15]:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            features_df[lag_col] = features_df[sens_temp_col].shift(lag)
            feature_columns.append(lag_col)

        # 重要な温度変化率のみ
        change_col = f'{sens_temp_col}_change_5'
        features_df[change_col] = (features_df[sens_temp_col] - features_df[f'{sens_temp_col}_lag_5']) / 5
        feature_columns.append(change_col)

        # 電力消費のラグ特徴量（重要なもののみ）
        lag_col = f'{power_col}_lag_1'
        features_df[lag_col] = features_df[power_col].shift(1)
        feature_columns.append(lag_col)

        # 移動平均（重要度の高い一つだけ）
        roll_temp = f'{sens_temp_col}_roll_15'
        temp_past = features_df[sens_temp_col].shift(1)
        features_df[roll_temp] = temp_past.rolling(window=15, min_periods=1).mean()

        roll_power = f'{power_col}_roll_15'
        power_past = features_df[power_col].shift(1)
        features_df[roll_power] = power_past.rolling(window=15, min_periods=1).mean()

        feature_columns.extend([roll_temp, roll_power])

        # サーモ状態の特徴量
        features_df['thermo_change'] = features_df[thermo_col].diff(1).fillna(0)
        reset_points = (features_df[thermo_col] != features_df[thermo_col].shift(1)).astype(int)
        reset_points.iloc[0] = 1
        group_id = reset_points.cumsum()
        features_df['thermo_duration'] = features_df.groupby(group_id).cumcount()

        # 外気温関連の特徴量（重要なもののみ）
        if 'outdoor_temp' in features_df.columns:
            outdoor_lag = 'outdoor_temp_lag_1'
            features_df[outdoor_lag] = features_df['outdoor_temp'].shift(1)
            features_df['temp_diff_outdoor'] = features_df[sens_temp_col] - features_df[outdoor_lag]
            feature_columns.extend([outdoor_lag, 'temp_diff_outdoor'])

        # 重要な交互作用特徴量
        features_df['thermo_x_temp'] = features_df[thermo_col] * features_df[sens_temp_col]

        if f'{sens_temp_col}_change_5' in features_df.columns:
            features_df['thermo_on_temp_change'] = features_df[thermo_col] * features_df[f'{sens_temp_col}_change_5']
            feature_columns.append('thermo_on_temp_change')

        # 気象条件の特徴量（重要なもののみ）
        if 'outdoor_temp' in features_df.columns and 'solar_radiation' in features_df.columns:
            features_df['is_sunny_day'] = ((features_df['hour'] >= 9) &
                                         (features_df['hour'] <= 17) &
                                         (features_df['solar_radiation'] >
                                          features_df['solar_radiation'].mean())).astype(int)
            feature_columns.append('is_sunny_day')

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
    import argparse

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='温度予測モデルの学習と評価')
    parser.add_argument('--file_path', type=str, default='./AllDayData.csv', help='入力データファイルのパス')
    parser.add_argument('--output_dir', type=str, default='./output/sens_temp_predictions', help='出力ディレクトリ')
    parser.add_argument('--no_feature_selection', action='store_true', help='特徴量選択を無効にする（デフォルトは有効）')
    parser.add_argument('--importance_threshold', type=float, default=0.04, help='特徴量選択の重要度閾値')
    parser.add_argument('--max_features', type=int, default=8, help='選択する特徴量の最大数')
    parser.add_argument('--no_baseline_compare', action='store_true', help='ベースラインとの比較を無効にする（デフォルトは有効）')
    parser.add_argument('--start_date', type=str, default='2024-06-26', help='分析開始日')
    parser.add_argument('--end_date', type=str, default='2024-09-20', help='分析終了日')
    parser.add_argument('--prediction_horizon', type=int, default=5, help='予測ホライズン（分）')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    start_time = time.time()
    file_path = args.file_path
    output_dir = args.output_dir
    prediction_horizon = args.prediction_horizon
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',
        2: 'M', 3: 'M', 8: 'M', 9: 'M',
        4: 'R', 5: 'R', 10: 'R', 11: 'R'
    }

    # 特徴量選択を行う場合、出力ディレクトリを分ける
    if not args.no_feature_selection:
        output_dir = os.path.join(output_dir, f'selected_features_max{args.max_features}_thresh{args.importance_threshold}')
    else:
        output_dir = os.path.join(output_dir, 'all_features')

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
        start_date = args.start_date
        end_date = args.end_date
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

        # 結果を保存するためのデータフレーム
        results_df = pd.DataFrame(columns=[
            'Zone', 'RMSE', 'MAE', 'MAPE', 'R2', 'Top_Features',
            'Feature_Count', 'Training_Time', 'Feature_Selection'
        ])

        # ベースラインとの比較が有効な場合
        if not args.no_baseline_compare:
            baseline_results_df = pd.DataFrame(columns=[
                'Zone', 'RMSE', 'MAE', 'MAPE', 'R2', 'Top_Features',
                'Feature_Count', 'Training_Time', 'Feature_Selection'
            ])
            comparison_results = []
            # 重要度データを保存
            importance_data_dict = {}

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
                # ベースラインモデル（特徴量選択なし）を学習
                if not args.no_baseline_compare:
                    feature_start = time.time()
                    logger.info(f"ゾーン {zone} のベースライン特徴量を作成中...")
                    X_baseline, y_baseline, features_df_baseline = prepare_features_for_sens_temp(
                        filtered_df, thermo_df, zone,
                        prediction_horizon=prediction_horizon,
                        feature_selection=False
                    )

                    if X_baseline is None or y_baseline is None or features_df_baseline is None:
                        logger.warning(f"ゾーン {zone} のベースライン特徴量を作成できませんでした")
                        continue

                    feature_time = time.time() - feature_start
                    logger.info(f"ベースライン特徴量作成完了: {X_baseline.shape} ({feature_time:.2f}秒)")

                    train_start = time.time()
                    logger.info(f"ゾーン {zone} のベースライン LightGBM モデルをトレーニングしています...")
                    baseline_model, X_test_baseline, y_test_baseline, y_pred_baseline, importance_df_baseline = train_lgbm_model(X_baseline, y_baseline)

                    train_time = time.time() - train_start
                    logger.info(f"ベースラインモデルトレーニング完了 ({train_time:.2f}秒)")

                    importance_path = visualize_feature_importance(importance_df_baseline, f"{zone}_baseline", output_dir)
                    logger.info(f"ベースライン特徴量重要度グラフを保存しました: {importance_path}")

                    baseline_rmse = np.sqrt(mean_squared_error(y_test_baseline, y_pred_baseline))
                    baseline_mae = np.mean(np.abs(y_test_baseline - y_pred_baseline))
                    baseline_mape = np.mean(np.abs((y_test_baseline - y_pred_baseline) / (y_test_baseline + 1e-6))) * 100
                    baseline_r2 = r2_score(y_test_baseline, y_pred_baseline)

                    baseline_top_features = ', '.join(importance_df_baseline.head(3)['Feature'].tolist())
                    baseline_results_df = pd.concat([
                        baseline_results_df,
                        pd.DataFrame({
                            'Zone': [zone],
                            'RMSE': [baseline_rmse],
                            'MAE': [baseline_mae],
                            'MAPE': [baseline_mape],
                            'R2': [baseline_r2],
                            'Top_Features': [baseline_top_features],
                            'Feature_Count': [len(X_baseline.columns)],
                            'Training_Time': [train_time],
                            'Feature_Selection': ['No']
                        })
                    ], ignore_index=True)

                    logger.info(f"ベースライン評価指標: RMSE={baseline_rmse:.4f}, MAE={baseline_mae:.4f}, MAPE={baseline_mape:.2f}%, R²={baseline_r2:.4f}")
                    logger.info(f"ベースライン特徴量数: {len(X_baseline.columns)}")

                    # 重要度データを保存
                    importance_data_dict[zone] = importance_df_baseline.copy()
                    total_importance = importance_data_dict[zone]['Importance'].sum()
                    importance_data_dict[zone]['Importance_Ratio'] = importance_data_dict[zone]['Importance'] / total_importance
                    importance_data_dict[zone]['Cumulative_Importance'] = importance_data_dict[zone]['Importance_Ratio'].cumsum()

                # 特徴量選択モデルを学習
                feature_start = time.time()
                logger.info(f"ゾーン {zone} の特徴量を作成中...")
                X, y, features_df = prepare_features_for_sens_temp(
                    filtered_df, thermo_df, zone,
                    prediction_horizon=prediction_horizon,
                    feature_selection=not args.no_feature_selection,
                    importance_threshold=args.importance_threshold,
                    max_features=args.max_features
                )

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
                        'Feature_Count': [len(X.columns)],
                        'Training_Time': [train_time],
                        'Feature_Selection': ['Yes' if not args.no_feature_selection else 'No']
                    })
                ], ignore_index=True)

                logger.info(f"評価指標: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")
                logger.info(f"特徴量数: {len(X.columns)}")

                # ベースラインとの比較
                if not args.no_baseline_compare:
                    rmse_change = (rmse - baseline_rmse) / baseline_rmse * 100
                    mae_change = (mae - baseline_mae) / baseline_mae * 100
                    r2_change = (r2 - baseline_r2) / max(baseline_r2, 0.0001) * 100
                    feature_count_change = (len(X.columns) - len(X_baseline.columns)) / len(X_baseline.columns) * 100
                    training_time_change = (train_time - train_time) / train_time * 100

                    comparison_results.append({
                        'Zone': zone,
                        'RMSE_Change_Pct': rmse_change,
                        'MAE_Change_Pct': mae_change,
                        'R2_Change_Pct': r2_change,
                        'Feature_Count_Change_Pct': feature_count_change,
                        'Training_Time_Change_Pct': training_time_change,
                        'RMSE_Baseline': baseline_rmse,
                        'RMSE_Selected': rmse,
                        'MAE_Baseline': baseline_mae,
                        'MAE_Selected': mae,
                        'R2_Baseline': baseline_r2,
                        'R2_Selected': r2,
                        'Features_Baseline': len(X_baseline.columns),
                        'Features_Selected': len(X.columns)
                    })

                    logger.info(f"比較結果: RMSE変化={rmse_change:.2f}%, MAE変化={mae_change:.2f}%, R²変化={r2_change:.2f}%")
                    logger.info(f"特徴量数変化: {feature_count_change:.2f}% ({len(X_baseline.columns)} → {len(X.columns)})")

                    # 重要度データのサマリーを表示
                    if zone in importance_data_dict:
                        important_features_baseline = importance_data_dict[zone].head(5)['Feature'].tolist()
                        important_features_selected = importance_df.head(5)['Feature'].tolist()
                        common_important = set(important_features_baseline) & set(important_features_selected)
                        logger.info(f"重要度上位5個の特徴量の共通数: {len(common_important)}/{min(5, len(important_features_baseline))}")
                        logger.info(f"共通の重要特徴量: {', '.join(common_important)}")

                # 可視化
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

            if not args.no_baseline_compare and len(comparison_results) > 0:
                comparison_df = pd.DataFrame(comparison_results)

                # 重要度データを保存するための属性を追加
                comparison_df.importance_data = []

                # 各ゾーンの重要度データを収集
                for zone in comparison_df['Zone'].unique():
                    zone_idx = comparison_df[comparison_df['Zone'] == zone].index[0]
                    if zone in importance_data_dict:
                        comparison_df.importance_data.append(importance_data_dict[zone])
                    else:
                        comparison_df.importance_data.append(None)

                comparison_path = os.path.join(output_dir, 'baseline_comparison.csv')
                comparison_df.to_csv(comparison_path, index=False)
                logger.info(f"ベースライン比較結果を保存しました: {comparison_path}")

                visualize_comparison_results(comparison_df, output_dir)
                logger.info("比較結果の可視化が完了しました")
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

def visualize_comparison_results(comparison_df, output_dir):
    """ベースラインと特徴量選択モデルの比較結果を可視化します。"""
    if comparison_df.empty:
        print("比較データがありません")
        return

    # メトリクス変化の可視化
    plt.figure(figsize=(14, 8))
    zones = comparison_df['Zone'].astype(str)

    width = 0.3
    x = np.arange(len(zones))

    # RMSEとMAEの変化（負の値が改善）
    rmse_changes = comparison_df['RMSE_Change_Pct']
    mae_changes = comparison_df['MAE_Change_Pct']

    # R2の変化（正の値が改善）
    r2_changes = comparison_df['R2_Change_Pct']

    # 特徴量数の変化（負の値が減少）
    feature_changes = comparison_df['Feature_Count_Change_Pct']

    # 色の設定
    colors = {
        'rmse': '#FF5733',  # 赤
        'mae': '#C70039',   # 暗い赤
        'r2': '#33A1FF',    # 青
        'features': '#33FF57'  # 緑
    }

    bars1 = plt.bar(x - width*1.5, rmse_changes, width, label='RMSE変化率(%)', color=colors['rmse'], alpha=0.7)
    bars2 = plt.bar(x - width/2, mae_changes, width, label='MAE変化率(%)', color=colors['mae'], alpha=0.7)
    bars3 = plt.bar(x + width/2, r2_changes, width, label='R²変化率(%)', color=colors['r2'], alpha=0.7)
    bars4 = plt.bar(x + width*1.5, feature_changes, width, label='特徴量数変化率(%)', color=colors['features'], alpha=0.7)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    # 改善エリアを強調表示
    plt.axhspan(-5, 5, alpha=0.1, color='yellow', label='±5%変化')

    plt.xlabel('ゾーン')
    plt.ylabel('変化率 (%)')
    plt.title('特徴量選択モデルとベースラインモデルの比較')
    plt.xticks(x, zones)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # 値のラベル表示
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 1:  # 変化が±1%以上の場合のみラベル表示
                plt.text(
                    bar.get_x() + bar.get_width()/2.,
                    height + (5 if height > 0 else -10),
                    f'{height:.1f}%',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, rotation=90
                )

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 特徴量数の比較
    plt.figure(figsize=(12, 6))
    bar_width = 0.35

    baseline_features = comparison_df['Features_Baseline']
    selected_features = comparison_df['Features_Selected']

    bar1 = plt.bar(x - bar_width/2, baseline_features, bar_width, label='ベースライン（全特徴量）', color='#3498db', alpha=0.7)
    bar2 = plt.bar(x + bar_width/2, selected_features, bar_width, label='選択済み特徴量', color='#2ecc71', alpha=0.7)

    for i, (b1, b2) in enumerate(zip(bar1, bar2)):
        reduction = ((baseline_features.iloc[i] - selected_features.iloc[i]) / baseline_features.iloc[i]) * 100
        if reduction > 0:
            plt.text(
                i,
                max(baseline_features.iloc[i], selected_features.iloc[i]) + 2,
                f'-{reduction:.1f}%',
                ha='center', va='bottom', fontsize=9
            )

    plt.xlabel('ゾーン')
    plt.ylabel('特徴量数')
    plt.title('特徴量数の比較')
    plt.xticks(x, zones)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_count_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # RMSE比較のテーブル
    plt.figure(figsize=(12, 6))
    table_data = []
    columns = ['ゾーン', 'RMSE (ベースライン)', 'RMSE (特徴量選択)', '変化率 (%)', 'MAE (ベースライン)', 'MAE (特徴量選択)', '変化率 (%)']

    for i, zone in enumerate(zones):
        row = [
            zone,
            f"{comparison_df['RMSE_Baseline'].iloc[i]:.4f}",
            f"{comparison_df['RMSE_Selected'].iloc[i]:.4f}",
            f"{comparison_df['RMSE_Change_Pct'].iloc[i]:.2f}%",
            f"{comparison_df['MAE_Baseline'].iloc[i]:.4f}",
            f"{comparison_df['MAE_Selected'].iloc[i]:.4f}",
            f"{comparison_df['MAE_Change_Pct'].iloc[i]:.2f}%"
        ]
        table_data.append(row)

    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    table = plt.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
        colColours=['#f2f2f2'] * len(columns)
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # セルの色付け（改善、悪化を視覚化）
    for i in range(len(zones)):
        # RMSE変化率の色付け
        rmse_change = comparison_df['RMSE_Change_Pct'].iloc[i]
        if rmse_change < -1:  # 1%以上の改善
            table[(i+1, 3)]._text.set_color('green')
        elif rmse_change > 1:  # 1%以上の悪化
            table[(i+1, 3)]._text.set_color('red')

        # MAE変化率の色付け
        mae_change = comparison_df['MAE_Change_Pct'].iloc[i]
        if mae_change < -1:  # 1%以上の改善
            table[(i+1, 6)]._text.set_color('green')
        elif mae_change > 1:  # 1%以上の悪化
            table[(i+1, 6)]._text.set_color('red')

    plt.title('精度の変化詳細', y=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 散布図（RMSE vs 特徴量数の削減率）
    plt.figure(figsize=(10, 8))

    # 特徴量削減率を計算
    feature_reduction = -comparison_df['Feature_Count_Change_Pct']

    # 散布図のサイズをR2の変化に基づいて調整
    sizes = comparison_df['R2_Change_Pct'].apply(lambda x: 100 + abs(x)*5)

    # 色分け（RMSEが改善したかどうか）
    colors = comparison_df['RMSE_Change_Pct'].apply(lambda x: '#2ecc71' if x < 0 else '#e74c3c')

    scatter = plt.scatter(
        feature_reduction,
        comparison_df['RMSE_Change_Pct'],
        s=sizes,
        c=colors,
        alpha=0.7,
        edgecolors='black'
    )

    # ゾーンラベルの追加
    for i, zone in enumerate(zones):
        plt.annotate(
            f'ゾーン {zone}',
            (feature_reduction.iloc[i], comparison_df['RMSE_Change_Pct'].iloc[i]),
            xytext=(7, 0),
            textcoords='offset points',
            fontsize=9
        )

    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # 象限にラベルを追加
    plt.text(max(feature_reduction)*0.7, min(comparison_df['RMSE_Change_Pct'])*0.7, '最適（特徴量減少・精度向上）', fontsize=10, color='green')
    plt.text(min(feature_reduction)*0.7, min(comparison_df['RMSE_Change_Pct'])*0.7, '要検討（特徴量増加・精度向上）', fontsize=10, color='blue')
    plt.text(max(feature_reduction)*0.7, max(comparison_df['RMSE_Change_Pct'])*0.7, '要検討（特徴量減少・精度低下）', fontsize=10, color='orange')
    plt.text(min(feature_reduction)*0.7, max(comparison_df['RMSE_Change_Pct'])*0.7, '最悪（特徴量増加・精度低下）', fontsize=10, color='red')

    plt.xlabel('特徴量削減率 (%)')
    plt.ylabel('RMSE変化率 (%)')
    plt.title('特徴量削減とRMSE変化の関係')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_vs_feature_reduction.png'), dpi=150, bbox_inches='tight')
    plt.close()

def visualize_feature_reduction_impact(comparison_df, output_dir):
    """特徴量削減とモデル精度の関係を可視化します"""
    if comparison_df.empty or len(comparison_df) < 2:
        print("比較データが不十分です")
        return

    # 特徴量数と精度の関係
    plt.figure(figsize=(14, 8))

    # 精度への影響を示す散布図
    plt.scatter(
        comparison_df['Features_Selected'],
        comparison_df['RMSE_Change_Pct'],
        s=100,
        c=comparison_df['RMSE_Change_Pct'].apply(lambda x: 'green' if x < 0 else 'red'),
        alpha=0.7,
        edgecolors='black'
    )

    # ゾーン番号をプロット
    for i, row in comparison_df.iterrows():
        plt.annotate(
            f'Zone {int(row["Zone"])}',
            (row['Features_Selected'], row['RMSE_Change_Pct']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=10
        )

    # 基準線
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # 特徴量削減率の表示
    reduction_avg = ((comparison_df['Features_Baseline'].mean() - comparison_df['Features_Selected'].mean())
                    / comparison_df['Features_Baseline'].mean() * 100)
    rmse_change_avg = comparison_df['RMSE_Change_Pct'].mean()

    plt.title(f'特徴量削減の影響分析 (平均削減率: {reduction_avg:.1f}%, 平均RMSE変化: {rmse_change_avg:.2f}%)')
    plt.xlabel('選択された特徴量数')
    plt.ylabel('RMSE変化率 (%)')
    plt.grid(True, alpha=0.3)

    # 傾向線の追加
    if len(comparison_df) > 2:
        z = np.polyfit(comparison_df['Features_Selected'], comparison_df['RMSE_Change_Pct'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(comparison_df['Features_Selected'].min(),
                             comparison_df['Features_Selected'].max(), 100)
        plt.plot(x_trend, p(x_trend), "b--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_reduction_impact.png'), dpi=150)
    plt.close()

    # 累積特徴量重要度の可視化
    # 最初のゾーンの重要度データをサンプルとして使用
    if hasattr(comparison_df, 'importance_data') and len(comparison_df['importance_data']) > 0:
        importance_data = comparison_df['importance_data'][0]
        if importance_data is not None and not importance_data.empty:
            plt.figure(figsize=(12, 6))

            # 重要度の累積グラフ
            plt.plot(range(1, len(importance_data) + 1),
                    importance_data['Cumulative_Importance'],
                    'b-', linewidth=2)

            # 95%ラインのマーク
            idx_95 = (importance_data['Cumulative_Importance'] <= 0.95).sum()
            plt.axvline(x=idx_95, color='r', linestyle='--')
            plt.text(idx_95 + 1, 0.5, f'95%の重要度: {idx_95}個の特徴量',
                    color='r', fontsize=10)

            plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)

            plt.title('特徴量の累積重要度')
            plt.xlabel('特徴量数')
            plt.ylabel('累積重要度')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cumulative_importance.png'), dpi=150)
            plt.close()

def evaluate_prediction_horizons(df, thermo_df, zone, horizons=[5, 10, 15, 20, 30], output_dir='./output/horizon_analysis'):
    """異なる予測ホライゾンでのモデル性能を評価"""
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',
        2: 'M', 3: 'M', 8: 'M', 9: 'M',
        4: 'R', 5: 'R', 10: 'R', 11: 'R'
    }

    if zone not in zone_to_power:
        print(f"ゾーン {zone} はどの室外機にも割り当てられていません")
        return None

    power_col = zone_to_power[zone]
    valid_col = f'AC_valid_{zone}'
    mode_col = f'AC_mode_{zone}'
    sens_temp_col = f'sens_temp_{zone}'

    required_cols = [valid_col, mode_col, sens_temp_col, power_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ゾーン {zone} に必要なカラムがありません: {', '.join(missing_cols)}")
        return None

    # 結果格納用
    results = {
        'horizon': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'top_features': [],
        'predictions': {},
        'actual': {},
        'timestamps': {},  # 時系列可視化用にタイムスタンプを保存
        'features_df': {}  # 時系列可視化用に特徴量データフレームを保存
    }

    # 各予測ホライゾンでモデルを訓練・評価
    for horizon in horizons:
        print(f"\nゾーン {zone} の予測ホライゾン {horizon} 分のモデル評価を開始")

        print(f"特徴量を作成中 (予測ホライゾン: {horizon}分)...")
        X, y, features_df = prepare_features_for_sens_temp(df, thermo_df, zone, prediction_horizon=horizon)
        if X is None or y is None or features_df is None:
            print(f"ゾーン {zone} の特徴量を作成できませんでした")
            continue

        print(f"モデルをトレーニングしています...")
        model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)
        if model is None:
            print(f"モデルのトレーニングに失敗しました")
            continue

        # 評価指標を計算
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"予測ホライゾン {horizon}分の結果: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # 上位特徴量
        top_features = importance_df.head(5)['Feature'].tolist() if importance_df is not None else []

        # 結果を保存
        results['horizon'].append(horizon)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['r2'].append(r2)
        results['top_features'].append(top_features)

        # テストデータの最後の部分を保存（可視化用）
        sample_size = min(1000, len(y_test))
        results['predictions'][horizon] = y_pred[-sample_size:]
        results['actual'][horizon] = y_test.iloc[-sample_size:].values

        # 時系列可視化用にタイムスタンプと特徴量データフレームを保存
        if not y_test.empty:
            test_timestamps = pd.Series(X_test.index).iloc[-sample_size:].values
            results['timestamps'][horizon] = test_timestamps
            results['features_df'][horizon] = features_df

    if not results['horizon']:
        print(f"ゾーン {zone} の評価結果がありません")
        return None

    # 結果ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 評価指標の可視化
    metrics_df = pd.DataFrame({
        'Horizon': results['horizon'],
        'RMSE': results['rmse'],
        'MAE': results['mae'],
        'R²': results['r2']
    })

    # RMSEとMAEのグラフ
    fig, ax1 = plt.subplots(figsize=(12, 7))

    line1 = ax1.plot(metrics_df['Horizon'], metrics_df['RMSE'], 'bo-', linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('予測ホライゾン（分）', fontsize=14)
    ax1.set_ylabel('RMSE (°C)', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    line2 = ax2.plot(metrics_df['Horizon'], metrics_df['MAE'], 'ro-', linewidth=2, markersize=8, label='MAE')
    ax2.set_ylabel('MAE (°C)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')

    # R²のグラフ（第二Y軸）
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # 右軸を少し外側に
    line3 = ax3.plot(metrics_df['Horizon'], metrics_df['R²'], 'go-', linewidth=2, markersize=8, label='R²')
    ax3.set_ylabel('R²', color='green', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='green')

    # 凡例を作成
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=12)

    plt.title(f'ゾーン {zone} - 予測ホライゾンに対する評価指標の変化', fontsize=16)
    plt.grid(True, alpha=0.3)

    # X軸の設定（予測ホライゾンの値を直接表示）
    plt.xticks(metrics_df['Horizon'])

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f'zone_{zone}_horizon_metrics.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 異なるホライゾンでの予測結果の時系列比較
    # 複数のホライゾンの予測結果を同じタイムステップで比較するのは難しいため、
    # 各ホライゾンごとに実測値と予測値の関係を可視化

    # 各ホライゾンについて散布図
    ncols = min(2, len(horizons))
    nrows = (len(horizons) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, horizon in enumerate(horizons):
        if i < len(axes) and horizon in results['predictions']:
            ax = axes[i]
            actual = results['actual'][horizon]
            predicted = results['predictions'][horizon]

            ax.scatter(actual, predicted, alpha=0.5, color=plt.cm.viridis(i/len(horizons)))

            # 完全一致線
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

            # 回帰線
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val, max_val, 100)
            ax.plot(x_trend, p(x_trend), 'r-', linewidth=1.5, alpha=0.6,
                    label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

            rmse = results['rmse'][results['horizon'].index(horizon)]
            r2 = results['r2'][results['horizon'].index(horizon)]

            ax.set_title(f'予測ホライゾン: {horizon}分\nRMSE: {rmse:.3f}, R²: {r2:.3f}', fontsize=12)
            ax.set_xlabel('実測値 (°C)', fontsize=10)
            ax.set_ylabel('予測値 (°C)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=9)

    # 使用されていない軸を非表示
    for i in range(len(horizons), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f'zone_{zone}_horizon_scatter.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 時系列折れ線グラフの作成 - 新機能追加
    # 各ホライゾンごとに時系列で実測値と予測値を比較
    for horizon in horizons:
        if horizon in results['predictions'] and horizon in results['timestamps']:
            actual = results['actual'][horizon]
            predicted = results['predictions'][horizon]
            timestamps = results['timestamps'][horizon]

            # タイムスタンプをdatetime型に変換
            datetime_timestamps = pd.to_datetime(timestamps)

            # 一定サイズに制限（最後の2日間分のデータのみ表示）
            max_points = min(576, len(actual))  # 5分間隔で2日間は576ポイント

            if len(actual) > max_points:
                actual = actual[-max_points:]
                predicted = predicted[-max_points:]
                datetime_timestamps = datetime_timestamps[-max_points:]

            # 時系列の折れ線グラフ
            plt.figure(figsize=(15, 7))
            plt.plot(datetime_timestamps, actual, 'b-', linewidth=2, label='実測値')
            plt.plot(datetime_timestamps, predicted, 'r--', linewidth=2, label='予測値')

            # グラフの設定
            plt.title(f'ゾーン {zone} - 予測ホライゾン {horizon}分の実測値と予測値の比較', fontsize=16)
            plt.xlabel('時間', fontsize=14)
            plt.ylabel('温度 (°C)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)

            # X軸の日付フォーマットを設定
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.gcf().autofmt_xdate()

            # RMSEとR²を表示
            rmse = results['rmse'][results['horizon'].index(horizon)]
            r2 = results['r2'][results['horizon'].index(horizon)]
            plt.figtext(0.01, 0.01, f'RMSE: {rmse:.3f}, R²: {r2:.3f}', fontsize=12)

            plt.tight_layout()
            timeseries_path = os.path.join(output_dir, f'zone_{zone}_horizon_{horizon}_timeseries.png')
            plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
            plt.close()

    # 特徴量の重要度変化
    # 各ホライゾンの上位特徴量をまとめる
    top_features_unique = []
    for features in results['top_features']:
        for feature in features:
            if feature not in top_features_unique:
                top_features_unique.append(feature)

    if top_features_unique:
        # 上位10件のみ表示
        top_features_unique = top_features_unique[:10]

        # 各ホライゾンごとに特徴量の順位を記録
        feature_ranks = pd.DataFrame(index=top_features_unique, columns=results['horizon'])
        for horizon_idx, horizon in enumerate(results['horizon']):
            features = results['top_features'][horizon_idx]
            for rank, feature in enumerate(features):
                if feature in top_features_unique:
                    feature_ranks.loc[feature, horizon] = rank + 1

        # 欠損値をNaNで埋める
        feature_ranks = feature_ranks.fillna(np.nan)

        # ヒートマップの作成
        plt.figure(figsize=(12, 8))
        mask = feature_ranks.isnull()
        sns.heatmap(feature_ranks, annot=True, cmap='YlGnBu_r', mask=mask, fmt='.0f',
                   linewidths=0.5, cbar_kws={'label': '順位'})

        plt.title(f'ゾーン {zone} - 異なる予測ホライゾンにおける特徴量重要度の順位', fontsize=16)
        plt.ylabel('特徴量', fontsize=14)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.tight_layout()

        ranks_path = os.path.join(output_dir, f'zone_{zone}_feature_ranks.png')
        plt.savefig(ranks_path, dpi=150, bbox_inches='tight')
        plt.close()

    return metrics_df

def analyze_prediction_horizons():
    """異なる予測ホライゾンでの温度予測モデルの性能を評価するメインプログラム"""

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='異なる予測ホライゾンでの温度予測モデルの性能を評価')
    parser.add_argument('--file_path', type=str, default='./AllDayData.csv', help='入力データファイルのパス')
    parser.add_argument('--output_dir', type=str, default='./output/horizon_analysis', help='出力ディレクトリ')
    parser.add_argument('--zones', type=str, default='all', help='分析するゾーン番号（カンマ区切り。例: 0,2,4 または all で全ゾーン）')
    parser.add_argument('--horizons', type=str, default='5,10,15,20,30', help='評価する予測ホライゾン（分）をカンマ区切りで指定')
    parser.add_argument('--start_date', type=str, default='2024-06-26', help='分析開始日')
    parser.add_argument('--end_date', type=str, default='2024-09-20', help='分析終了日')
    args = parser.parse_args()

    # 出力ディレクトリの作成
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 予測ホライゾンの設定
    horizons = [int(h) for h in args.horizons.split(',') if h.strip()]
    print(f"評価する予測ホライゾン: {horizons} 分")

    # データの読み込み
    print(f"データを読み込んでいます: {args.file_path}")
    df = pd.read_csv(args.file_path)
    print(f"データ読み込み完了: {len(df)}行 × {len(df.columns)}列")

    if 'algo' in df.columns:
        before_rows = len(df)
        df = df.dropna(subset=['algo'])
        print(f"'algo'列のNaN行を削除しました: {before_rows} → {len(df)}行")

    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    filtered_df = df[(df['time_stamp'] >= start) & (df['time_stamp'] <= end)]
    print(f"分析期間: {args.start_date} から {args.end_date} ({len(filtered_df)}行)")

    # サーモ状態の計算
    print("サーモ状態を計算しています...")
    thermo_df = determine_thermo_status(filtered_df)
    wanted_columns = ['time_stamp'] + [f'thermo_{i}' for i in range(12)] + ['thermo_L_or', 'thermo_M_or', 'thermo_R_or']
    for col in wanted_columns:
        if col not in thermo_df.columns and col != 'time_stamp':
            thermo_df[col] = 0
    thermo_df = thermo_df[wanted_columns]
    print("サーモ状態の計算が完了しました")

    # 分析するゾーンの設定
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',
        2: 'M', 3: 'M', 8: 'M', 9: 'M',
        4: 'R', 5: 'R', 10: 'R', 11: 'R'
    }

    if args.zones.lower() == 'all':
        zones_to_analyze = list(range(12))  # 全てのゾーン（0-11）
    else:
        # カンマ区切りでゾーン番号を指定
        zones_to_analyze = [int(z) for z in args.zones.split(',') if z.strip()]

    print(f"分析するゾーン: {zones_to_analyze}")

    all_metrics = []

    for zone in zones_to_analyze:
        print(f"\n===== ゾーン {zone} の異なる予測ホライゾン分析開始 =====")

        if zone not in zone_to_power:
            print(f"ゾーン {zone} はどの室外機にも割り当てられていません。スキップします。")
            continue

        metrics_df = evaluate_prediction_horizons(filtered_df, thermo_df, zone, horizons, output_dir)

        if metrics_df is not None:
            metrics_df['Zone'] = zone
            all_metrics.append(metrics_df)
            print(f"ゾーン {zone} の評価完了")
            print(f"散布図保存場所: {output_dir}/zone_{zone}_horizon_scatter.png")
            # 時系列折れ線グラフに関する情報を表示
            for h in horizons:
                print(f"予測ホライゾン {h}分の時系列折れ線グラフ: {output_dir}/zone_{zone}_horizon_{h}_timeseries.png")

        print(f"===== ゾーン {zone} の分析完了 =====\n")

    # 全ゾーンの結果を比較
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # RMSEの比較
        plt.figure(figsize=(14, 10))
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            plt.plot(zone_data['Horizon'], zone_data['RMSE'], 'o-',
                    linewidth=2, markersize=8, label=f'ゾーン {zone}')

        plt.title('ゾーン別の予測ホライゾンに対するRMSE', fontsize=16)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.ylabel('RMSE (°C)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(horizons)

        plt.tight_layout()
        combined_path = os.path.join(output_dir, 'all_zones_rmse_by_horizon.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()

        # R²の比較
        plt.figure(figsize=(14, 10))
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            plt.plot(zone_data['Horizon'], zone_data['R²'], 'o-',
                    linewidth=2, markersize=8, label=f'ゾーン {zone}')

        plt.title('ゾーン別の予測ホライゾンに対するR²', fontsize=16)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.ylabel('R²', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(horizons)

        plt.tight_layout()
        r2_path = os.path.join(output_dir, 'all_zones_r2_by_horizon.png')
        plt.savefig(r2_path, dpi=150, bbox_inches='tight')
        plt.close()

        # MAEの比較
        plt.figure(figsize=(14, 10))
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            plt.plot(zone_data['Horizon'], zone_data['MAE'], 'o-',
                    linewidth=2, markersize=8, label=f'ゾーン {zone}')

        plt.title('ゾーン別の予測ホライゾンに対するMAE', fontsize=16)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.ylabel('MAE (°C)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(horizons)

        plt.tight_layout()
        mae_path = os.path.join(output_dir, 'all_zones_mae_by_horizon.png')
        plt.savefig(mae_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 結果のCSV保存
        results_path = os.path.join(output_dir, 'horizon_metrics_all_zones.csv')
        combined_metrics.to_csv(results_path, index=False)

        # サマリーテーブルの作成
        summary_data = []
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            best_rmse_idx = zone_data['RMSE'].idxmin()
            best_horizon = zone_data.loc[best_rmse_idx, 'Horizon']
            best_rmse = zone_data.loc[best_rmse_idx, 'RMSE']
            best_mae = zone_data.loc[best_rmse_idx, 'MAE']
            best_r2 = zone_data.loc[best_rmse_idx, 'R²']

            summary_data.append({
                'Zone': int(zone),
                'Best_Horizon': best_horizon,
                'RMSE': best_rmse,
                'MAE': best_mae,
                'R²': best_r2
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Zone')
        summary_path = os.path.join(output_dir, 'best_horizons_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        print("\n最適な予測ホライゾン（RMSEが最小）:")
        print(summary_df.to_string(index=False))

        print(f"\n全ゾーンの予測ホライゾン評価指標をCSVに保存しました: {results_path}")
        print(f"最適な予測ホライゾンのサマリーを保存しました: {summary_path}")
        print(f"全ゾーンの予測ホライゾンRMSE比較図を保存しました: {combined_path}")
        print(f"全ゾーンの予測ホライゾンR²比較図を保存しました: {r2_path}")
        print(f"全ゾーンの予測ホライゾンMAE比較図を保存しました: {mae_path}")
        print("\n時系列折れ線グラフも各ホライゾンごとに生成し、実測値と予測値を時系列で直接比較できるようになりました。")
        print("折れ線グラフは各ゾーンのフォルダ内に 'zone_X_horizon_Y_timeseries.png' として保存されています。")
    else:
        print("評価に成功したゾーンがありません")

    print(f"\nすべての分析が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main()
