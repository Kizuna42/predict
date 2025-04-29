"""
特徴量生成ユーティリティ
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from ..config import DEFAULT_CONFIG, get_zone_power_map

def get_zone_power_col(zone):
    """ゾーンに対応する室外機を返す"""
    zone_power_map = get_zone_power_map()
    power_system = zone_power_map.get(zone, None)
    return power_system

def prepare_features_for_sens_temp(df, thermo_df, zone, look_back=None, prediction_horizon=5,
                                feature_selection=None, importance_threshold=None, max_features=None):
    """
    センサー温度予測のための特徴量を作成

    Args:
        df: 入力データフレーム
        thermo_df: サーモ状態データフレーム
        zone: ゾーン番号
        look_back: 過去データの参照時間（分）
        prediction_horizon: 予測ホライゾン（分）
        feature_selection: 特徴量選択を行うかどうか
        importance_threshold: 特徴量重要度閾値
        max_features: 最大特徴量数

    Returns:
        X: 特徴量
        y: 目的変数
        merged_df: 結合されたデータフレーム
    """
    # configからデフォルト値を取得
    if look_back is None:
        look_back = DEFAULT_CONFIG['MAX_LOOK_BACK']
    if feature_selection is None:
        feature_selection = DEFAULT_CONFIG['DEFAULT_FEATURE_SELECTION']
    if importance_threshold is None:
        importance_threshold = DEFAULT_CONFIG['DEFAULT_IMPORTANCE_THRESHOLD']
    if max_features is None:
        max_features = DEFAULT_CONFIG['DEFAULT_MAX_FEATURES']

    # 列名の設定
    valid_col = f'AC_valid_{zone}'
    mode_col = f'AC_mode_{zone}'
    thermo_col = f'thermo_{zone}'
    sens_temp_col = f'sens_temp_{zone}'

    # ゾーンに対応する室外機を特定
    power_col = get_zone_power_col(zone)
    if power_col is None:
        print(f"Zone {zone} not assigned to any outdoor unit")
        return None, None, None

    # power_colの形式を取得 (power_L, power_M, power_R)
    power_system = power_col.split('_')[1] if '_' in power_col else power_col
    thermo_or_col = f'thermo_{power_system}_or'

    # 必要な列を抽出
    required_thermo_cols = ['time_stamp', thermo_col, thermo_or_col]
    thermo_subset = thermo_df[required_thermo_cols].copy()

    required_df_cols = ['time_stamp', valid_col, mode_col, sens_temp_col, power_col]
    optional_cols = ['outdoor_temp', 'solar_radiation']
    for col in optional_cols:
        if col in df.columns:
            required_df_cols.append(col)

    df_subset = df[required_df_cols].copy()

    # データフレームの結合
    merged_df = pd.merge(df_subset, thermo_subset, on='time_stamp', how='left')

    required_cols = [valid_col, mode_col, thermo_col, sens_temp_col, power_col, thermo_or_col]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]
    if missing_cols:
        print(f"Missing required columns for zone {zone}: {missing_cols}")
        return None, None, None

    # 時間特徴量
    merged_df['hour'] = merged_df['time_stamp'].dt.hour
    merged_df['day_of_week'] = merged_df['time_stamp'].dt.dayofweek
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
    merged_df['is_night'] = ((merged_df['hour'] >= 19) | (merged_df['hour'] <= 6)).astype(int)

    # 周期的特徴量（サイン・コサイン変換）
    hour_rad = 2 * np.pi * merged_df['hour'] / 24
    merged_df['hour_sin'] = np.sin(hour_rad)
    merged_df['hour_cos'] = np.cos(hour_rad)

    # 目的変数の設定
    target_col = f'{sens_temp_col}_future_{prediction_horizon}'
    merged_df[target_col] = merged_df[sens_temp_col].shift(-prediction_horizon)

    # ラグ特徴量の辞書
    lag_cols = {}

    # 温度のラグ特徴量
    lag_values = [1, 5, 15]
    for lag in lag_values:
        if lag <= look_back:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            merged_df[lag_col] = merged_df[sens_temp_col].shift(lag)
            lag_cols[lag_col] = True

    # 温度変化率
    change_col = f'{sens_temp_col}_change_5'
    merged_df[change_col] = (merged_df[sens_temp_col] - merged_df[f'{sens_temp_col}_lag_5']) / 5
    lag_cols[change_col] = True

    # 電力消費のラグ特徴量
    lag_col = f'{power_col}_lag_1'
    merged_df[lag_col] = merged_df[power_col].shift(1)
    lag_cols[lag_col] = True

    # 移動平均
    window = 15
    roll_temp = f'{sens_temp_col}_roll_{window}'
    temp_past = merged_df[sens_temp_col].shift(1)
    merged_df[roll_temp] = temp_past.rolling(window=window, min_periods=1).mean()

    roll_power = f'{power_col}_roll_{window}'
    power_past = merged_df[power_col].shift(1)
    merged_df[roll_power] = power_past.rolling(window=window, min_periods=1).mean()

    lag_cols[roll_temp] = True
    lag_cols[roll_power] = True

    # サーモ状態の変化と持続時間
    merged_df['thermo_change'] = merged_df[thermo_col].diff(1).fillna(0)
    reset_points = (merged_df[thermo_col] != merged_df[thermo_col].shift(1)).astype(int)
    reset_points.iloc[0] = 1
    group_id = reset_points.cumsum()
    merged_df['thermo_duration'] = merged_df.groupby(group_id).cumcount()

    # 外気温特徴量
    if 'outdoor_temp' in merged_df.columns:
        merged_df['outdoor_temp_lag_1'] = merged_df['outdoor_temp'].shift(1)
        merged_df['temp_diff_outdoor'] = merged_df[sens_temp_col] - merged_df['outdoor_temp_lag_1']
        lag_cols['outdoor_temp_lag_1'] = True
        lag_cols['temp_diff_outdoor'] = True

    # 交互作用特徴量
    merged_df['thermo_x_temp'] = merged_df[thermo_col] * merged_df[sens_temp_col]

    if f'{sens_temp_col}_change_5' in merged_df.columns:
        merged_df['thermo_on_temp_change'] = merged_df[thermo_col] * merged_df[f'{sens_temp_col}_change_5']
    else:
        merged_df['thermo_on_temp_change'] = 0

    # 天候状態特徴量
    if 'outdoor_temp' in merged_df.columns and 'solar_radiation' in merged_df.columns:
        merged_df['is_sunny_day'] = ((merged_df['hour'] >= 9) &
                                    (merged_df['hour'] <= 17) &
                                    (merged_df['solar_radiation'] >
                                     merged_df['solar_radiation'].mean())).astype(int)

    # NaN値を含む行を削除
    merged_df = merged_df.dropna()

    # 基本特徴量
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

    # 全特徴量の結合と重複除去
    all_feature_columns = base_feature_columns + conditional_features + lag_feature_columns
    feature_columns = list(set(all_feature_columns))

    # 特徴量選択
    if feature_selection and len(feature_columns) > max_features:
        X_initial = merged_df[feature_columns]
        y_initial = merged_df[target_col]

        # 簡易的なモデルを訓練して特徴量重要度を取得
        X_train, X_val, y_train, y_val = train_test_split(
            X_initial, y_initial,
            test_size=DEFAULT_CONFIG['DEFAULT_TEST_SIZE'],
            random_state=DEFAULT_CONFIG['DEFAULT_RANDOM_STATE']
        )
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = DEFAULT_CONFIG['LGBM_PARAMS'].copy()

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

        # 重要度閾値以上の特徴量を選択
        threshold_features = importance_df[
            importance_df['Importance_Ratio'] >= importance_threshold
        ]['Feature'].tolist()

        # 上位max_features個の特徴量を選択
        top_features = importance_df.head(max_features)['Feature'].tolist()

        # 選択特徴量
        selected_features = threshold_features

        # 閾値ベースの選択が十分な特徴量を得られない場合、上位N個を追加
        if len(selected_features) < 5:
            remaining_needed = 5 - len(selected_features)
            for feat in top_features:
                if feat not in selected_features and remaining_needed > 0:
                    selected_features.append(feat)
                    remaining_needed -= 1

        # 特徴量数制限
        if max_features > 0 and len(selected_features) > max_features:
            selected_features = importance_df[
                importance_df['Feature'].isin(selected_features)
            ].head(max_features)['Feature'].tolist()

        # 必須特徴量を追加
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


def prepare_features_for_prediction_without_dropna(df, zone, power_col):
    """
    予測用の特徴量を生成（NaN値を削除せず）

    Args:
        df: 入力データフレーム
        zone: ゾーン番号
        power_col: 電力列名（'power_L', 'power_M', 'power_R'のいずれか）

    Returns:
        特徴量データフレーム
    """
    try:
        valid_col = f'AC_valid_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        sens_temp_col = f'sens_temp_{zone}'

        # power_colから対応するシステム（L、M、R）を抽出
        if '_' in power_col:
            power_system = power_col.split('_')[1]
        else:
            raise ValueError(f"Invalid power column format: {power_col}. Expected format: power_X where X is L, M, or R")

        thermo_or_col = f'thermo_{power_system}_or'

        required_cols = [valid_col, mode_col, thermo_col, sens_temp_col, power_col, thermo_or_col]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns for zone {zone}: {', '.join(missing_cols)}")

        # 特徴量のリスト
        feature_columns = [
            # 時間特徴量
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'hour_sin', 'hour_cos',
            # HVAC操作特徴量
            valid_col, mode_col, thermo_col, thermo_or_col,
            'thermo_duration', 'thermo_change',
            # 電力消費
            power_col,
            # インタラクション特徴量
            'thermo_x_temp'
        ]

        features_df = df.copy()

        # 時間関連特徴量の計算
        hour = features_df['time_stamp'].dt.hour
        day_of_week = features_df['time_stamp'].dt.dayofweek
        features_df['hour'] = hour
        features_df['day_of_week'] = day_of_week
        features_df['is_weekend'] = (day_of_week >= 5).astype(int)
        features_df['is_night'] = ((hour >= 19) | (hour <= 6)).astype(int)

        # 周期的時間特徴量
        hour_rad = 2 * np.pi * hour / 24
        features_df['hour_sin'] = np.sin(hour_rad)
        features_df['hour_cos'] = np.cos(hour_rad)

        # 温度のラグ特徴量
        for lag in [1, 5, 15]:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            features_df[lag_col] = features_df[sens_temp_col].shift(lag)
            feature_columns.append(lag_col)

        # 温度変化率
        change_col = f'{sens_temp_col}_change_5'
        features_df[change_col] = (features_df[sens_temp_col] - features_df[f'{sens_temp_col}_lag_5']) / 5
        feature_columns.append(change_col)

        # 電力消費のラグ特徴量
        lag_col = f'{power_col}_lag_1'
        features_df[lag_col] = features_df[power_col].shift(1)
        feature_columns.append(lag_col)

        # 移動平均
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

        # 外気温関連の特徴量
        if 'outdoor_temp' in features_df.columns:
            outdoor_lag = 'outdoor_temp_lag_1'
            features_df[outdoor_lag] = features_df['outdoor_temp'].shift(1)
            features_df['temp_diff_outdoor'] = features_df[sens_temp_col] - features_df[outdoor_lag]
            feature_columns.extend([outdoor_lag, 'temp_diff_outdoor'])

        # 交互作用特徴量
        features_df['thermo_x_temp'] = features_df[thermo_col] * features_df[sens_temp_col]

        if f'{sens_temp_col}_change_5' in features_df.columns:
            features_df['thermo_on_temp_change'] = features_df[thermo_col] * features_df[f'{sens_temp_col}_change_5']
            feature_columns.append('thermo_on_temp_change')

        # 気象条件の特徴量
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
