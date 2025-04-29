"""
サーモ状態計算ユーティリティ
"""

import pandas as pd
import numpy as np
from ..config import DEFAULT_CONFIG, get_zone_power_map

def determine_thermo_status(df, deadband=None):
    """
    空調機のサーモ状態を判定する

    Args:
        df: 入力データフレーム
        deadband: 不感帯（°C）デフォルト値はconfig.pyから読み込み

    Returns:
        サーモ状態を含むデータフレーム
    """
    # 設定から不感帯の値を取得
    if deadband is None:
        deadband = DEFAULT_CONFIG['THERMO_DEADBAND']

    df = df.reset_index(drop=True)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    min_date = df['time_stamp'].min()
    max_date = df['time_stamp'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='1min')
    result_df = pd.DataFrame({'time_stamp': date_range})
    thermo_cols = [f'thermo_{zone}' for zone in DEFAULT_CONFIG['ALL_ZONES']]
    result_df[thermo_cols] = 0

    for zone in DEFAULT_CONFIG['ALL_ZONES']:
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

                if current_mode == 2:  # 暖房モード
                    if prev_thermo == 0 and df_zone[temp_col].iloc[i] < df_zone[set_col].iloc[i] - deadband:
                        thermo_values[i] = 1
                    elif prev_thermo == 1 and df_zone[temp_col].iloc[i] > df_zone[set_col].iloc[i] + deadband:
                        thermo_values[i] = 0
                    else:
                        thermo_values[i] = prev_thermo
                elif current_mode == 1:  # 冷房モード
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

    # 室外機ごとのOR演算（configから各系統のゾーンリストを取得）
    result_df['thermo_L_or'] = calculate_thermo_or(result_df, DEFAULT_CONFIG['L_ZONES'])
    result_df['thermo_M_or'] = calculate_thermo_or(result_df, DEFAULT_CONFIG['M_ZONES'])
    result_df['thermo_R_or'] = calculate_thermo_or(result_df, DEFAULT_CONFIG['R_ZONES'])

    return result_df


def calculate_thermo_or(df, zones):
    """
    指定されたゾーンのサーモ状態のOR演算を計算する

    Args:
        df: データフレーム
        zones: ゾーン番号のリスト

    Returns:
        OR演算結果の整数値
    """
    # 初期値をFalseにしてOR演算を行う
    result = False
    for zone in zones:
        result = result | df[f'thermo_{zone}'].astype(bool)

    return result.astype(int)
