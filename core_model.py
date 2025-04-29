#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import time
import logging
from datetime import timedelta
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import argparse

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
    optional_cols = ['outdoor_temp', 'solar_radiation']
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
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
    merged_df['is_night'] = ((merged_df['hour'] >= 19) | (merged_df['hour'] <= 6)).astype(int)

    hour_rad = 2 * np.pi * merged_df['hour'] / 24
    merged_df['hour_sin'] = np.sin(hour_rad)
    merged_df['hour_cos'] = np.cos(hour_rad)

    target_col = f'{sens_temp_col}_future_{prediction_horizon}'
    merged_df[target_col] = merged_df[sens_temp_col].shift(-prediction_horizon)

    lag_cols = {}

    lag_values = [1, 5, 15]
    for lag in lag_values:
        if lag <= look_back:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            merged_df[lag_col] = merged_df[sens_temp_col].shift(lag)
            lag_cols[lag_col] = True

    change_col = f'{sens_temp_col}_change_5'
    merged_df[change_col] = (merged_df[sens_temp_col] - merged_df[f'{sens_temp_col}_lag_5']) / 5
    lag_cols[change_col] = True

    lag_col = f'{power_col}_lag_1'
    merged_df[lag_col] = merged_df[power_col].shift(1)
    lag_cols[lag_col] = True

    window = 15
    roll_temp = f'{sens_temp_col}_roll_{window}'
    temp_past = merged_df[sens_temp_col].shift(1)
    merged_df[roll_temp] = temp_past.rolling(window=window, min_periods=1).mean()

    roll_power = f'{power_col}_roll_{window}'
    power_past = merged_df[power_col].shift(1)
    merged_df[roll_power] = power_past.rolling(window=window, min_periods=1).mean()

    lag_cols[roll_temp] = True
    lag_cols[roll_power] = True

    merged_df['thermo_change'] = merged_df[thermo_col].diff(1).fillna(0)
    reset_points = (merged_df[thermo_col] != merged_df[thermo_col].shift(1)).astype(int)
    reset_points.iloc[0] = 1
    group_id = reset_points.cumsum()
    merged_df['thermo_duration'] = merged_df.groupby(group_id).cumcount()

    if 'outdoor_temp' in merged_df.columns:
        merged_df['outdoor_temp_lag_1'] = merged_df['outdoor_temp'].shift(1)
        merged_df['temp_diff_outdoor'] = merged_df[sens_temp_col] - merged_df['outdoor_temp_lag_1']
        lag_cols['outdoor_temp_lag_1'] = True
        lag_cols['temp_diff_outdoor'] = True

    merged_df['thermo_x_temp'] = merged_df[thermo_col] * merged_df[sens_temp_col]
    merged_df['thermo_on_temp_change'] = merged_df[thermo_col] * merged_df[change_col]

    if 'outdoor_temp' in merged_df.columns and 'solar_radiation' in merged_df.columns:
        merged_df['is_sunny_day'] = ((merged_df['hour'] >= 9) &
                                    (merged_df['hour'] <= 17) &
                                    (merged_df['solar_radiation'] >
                                     merged_df['solar_radiation'].mean())).astype(int)

    merged_df = merged_df.dropna()

    base_feature_columns = [
        'hour', 'day_of_week', 'is_weekend', 'is_night',
        'hour_sin', 'hour_cos',
        valid_col, mode_col, thermo_col, thermo_or_col,
        'thermo_duration', 'thermo_change',
        power_col,
        'thermo_x_temp', f'{sens_temp_col}_lag_1'
    ]

    conditional_features = []
    if 'is_sunny_day' in merged_df.columns:
        conditional_features.append('is_sunny_day')

    lag_feature_columns = [col for col in lag_cols.keys() if col in merged_df.columns]

    all_feature_columns = base_feature_columns + conditional_features + lag_feature_columns
    feature_columns = list(set(all_feature_columns))

    if feature_selection and len(feature_columns) > max_features:
        X_initial = merged_df[feature_columns]
        y_initial = merged_df[target_col]

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

        importance = initial_model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        total_importance = importance_df['Importance'].sum()
        importance_df['Importance_Ratio'] = importance_df['Importance'] / total_importance
        importance_df['Cumulative_Importance'] = importance_df['Importance_Ratio'].cumsum()

        threshold_features = importance_df[
            importance_df['Importance_Ratio'] >= importance_threshold
        ]['Feature'].tolist()

        top_features = importance_df.head(max_features)['Feature'].tolist()
        selected_features = threshold_features

        if len(selected_features) < 5:
            remaining_needed = 5 - len(selected_features)
            for feat in top_features:
                if feat not in selected_features and remaining_needed > 0:
                    selected_features.append(feat)
                    remaining_needed -= 1

        if max_features > 0 and len(selected_features) > max_features:
            selected_features = importance_df[
                importance_df['Feature'].isin(selected_features)
            ].head(max_features)['Feature'].tolist()

        critical_features = [valid_col, mode_col, thermo_col, power_col, f'{sens_temp_col}_lag_1']
        for feat in critical_features:
            if feat in merged_df.columns and feat not in selected_features:
                selected_features.append(feat)

        print(f"Zone {zone}: 特徴量数を {len(feature_columns)} から {len(selected_features)} に削減しました")
        print(f"選択された特徴量: {', '.join(selected_features)}")

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

def main():
    import pandas as pd
    import numpy as np
    import os
    import time
    import logging
    from datetime import timedelta
    import warnings
    import argparse

    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='温度予測モデルの学習と評価')
    parser.add_argument('--file_path', type=str, default='./AllDayData.csv', help='入力データファイルのパス')
    parser.add_argument('--output_dir', type=str, default='./output/sens_temp_predictions', help='出力ディレクトリ')
    parser.add_argument('--no_feature_selection', action='store_true', help='特徴量選択を無効にする（デフォルトは有効）')
    parser.add_argument('--importance_threshold', type=float, default=0.04, help='特徴量選択の重要度閾値')
    parser.add_argument('--max_features', type=int, default=8, help='選択する特徴量の最大数')
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
                # 特徴量作成
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

                # モデル訓練
                train_start = time.time()
                logger.info(f"ゾーン {zone} の LightGBM モデルをトレーニングしています...")
                model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)

                train_time = time.time() - train_start
                logger.info(f"モデルトレーニング完了 ({train_time:.2f}秒)")

                # 評価指標を計算
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = np.mean(np.abs(y_test - y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
                r2 = r2_score(y_test, y_pred)

                # 結果を保存
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

                zone_time = time.time() - zone_start
                logger.info(f"ゾーン {zone} の処理が完了しました (合計: {zone_time:.2f}秒)")

            except Exception as e:
                logger.error(f"ゾーン {zone} の処理中にエラーが発生しました: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # 結果を保存
        if not results_df.empty:
            results_path = os.path.join(output_dir, 'model_results_summary.csv')
            results_df.to_csv(results_path, index=False)
            logger.info(f"モデル結果サマリーを保存しました: {results_path}")
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

if __name__ == "__main__":
    main()
