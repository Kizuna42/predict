"""
サーモ状態計算ユーティリティ
"""

import pandas as pd
import numpy as np

def determine_thermo_status(df, deadband=1.0):
    """
    空調機のサーモ状態を判定する

    Args:
        df: 入力データフレーム
        deadband: 不感帯（°C）

    Returns:
        サーモ状態を含むデータフレーム
    """
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

    # 室外機ごとのOR演算
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
