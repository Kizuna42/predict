#!/usr/bin/env python
# coding: utf-8

"""
データ前処理モジュール
温度データの外れ値処理や平滑化処理など、基本的なデータ前処理関数を提供
"""

import pandas as pd
import numpy as np
from src.config import MIN_TEMP, MAX_TEMP


def filter_temperature_outliers(df, min_temp=MIN_TEMP, max_temp=MAX_TEMP, log=True):
    """
    温度データの外れ値を処理する関数

    Parameters:
    -----------
    df : DataFrame
        処理対象のデータフレーム
    min_temp : float
        最小許容温度（これ未満を外れ値とする）
    max_temp : float
        最大許容温度（これ超過を外れ値とする）
    log : bool
        外れ値処理結果をログ出力するかどうか

    Returns:
    --------
    DataFrame
        外れ値処理後のデータフレーム
    """
    if log:
        print("\n## 温度データの外れ値処理")

    df_filtered = df.copy()

    # センサー温度列を特定
    temp_cols = [col for col in df.columns if 'sens_temp_' in col and 'future' not in col]
    future_temp_cols = [col for col in df.columns if 'sens_temp_' in col and 'future' in col]

    # 現在の温度の外れ値処理
    for col in temp_cols:
        # 外れ値の検出
        outliers = (df[col] < min_temp) | (df[col] > max_temp)
        outlier_count = outliers.sum()

        if outlier_count > 0 and log:
            max_val = df.loc[outliers, col].max() if outlier_count > 0 else np.nan
            min_val = df.loc[outliers, col].min() if outlier_count > 0 else np.nan
            print(f"{col}で{outlier_count}個の外れ値を検出（範囲: {min_val:.2f}～{max_val:.2f}℃）")

        # 外れ値をNaNに置換
        df_filtered.loc[outliers, col] = np.nan

    # 将来温度（目的変数）の外れ値処理
    for col in future_temp_cols:
        # 外れ値の検出
        outliers = (df[col] < min_temp) | (df[col] > max_temp)
        outlier_count = outliers.sum()

        if outlier_count > 0 and log:
            max_val = df.loc[outliers, col].max() if outlier_count > 0 else np.nan
            min_val = df.loc[outliers, col].min() if outlier_count > 0 else np.nan
            print(f"{col}で{outlier_count}個の外れ値を検出（範囲: {min_val:.2f}～{max_val:.2f}℃）")

        # 外れ値をNaNに置換
        df_filtered.loc[outliers, col] = np.nan

    # 処理後の欠損値の数を集計
    if log:
        total_outliers = 0
        for col in temp_cols + future_temp_cols:
            missing_count = df_filtered[col].isna().sum() - df[col].isna().sum()
            if missing_count > 0:
                total_outliers += missing_count

        print(f"全体で{total_outliers}個の外れ値をNaNに置換しました")

    return df_filtered


def apply_smoothing_to_sensors(df, window_sizes):
    """
    センサーデータにスムージングを適用する関数
    データリーク防止のため、過去のデータのみを使用するよう修正
    """
    df_copy = df.copy()
    smoothed_features = []

    # 温度センサーのカラムを取得
    temp_cols = [col for col in df.columns if 'sens_temp' in col and 'future' not in col]

    for col in temp_cols:
        for window in window_sizes:
            # min_periods=windowとし、window分のデータが揃うまでNaNとする
            # center=Falseとし、過去のデータのみを使用
            smoothed_col = f"{col}_smoothed_w{window}"
            df_copy[smoothed_col] = df_copy[col].rolling(
                window=window, min_periods=window, center=False).mean()
            smoothed_features.append(smoothed_col)

    # 前方補間でNaN値を埋める
    for col in smoothed_features:
        df_copy[col] = df_copy[col].fillna(method='ffill')

    return df_copy, smoothed_features


def create_future_targets(df, zone_nums, horizons_minutes, time_diff):
    """
    各ゾーンの将来温度を目的変数として作成

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト（例：[1, 2, 3, ...]）
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff : Timedelta
        データのサンプリング間隔

    Returns:
    --------
    DataFrame
        将来温度特徴量を追加したデータフレーム
    """
    df_copy = df.copy()

    for zone in zone_nums:
        source_col = f'sens_temp_{zone}'
        if source_col not in df.columns:
            print(f"警告: 列 {source_col} が見つかりません")
            continue

        for horizon in horizons_minutes:
            # 時間間隔から必要なシフト数を計算
            shift_periods = int(horizon / time_diff.total_seconds() * 60)

            # 指定分後の温度を取得
            target_col = f'sens_temp_{zone}_future_{horizon}'
            df_copy[target_col] = df_copy[source_col].shift(-shift_periods)

    return df_copy


def prepare_time_features(df):
    """
    時間関連の特徴量を作成する関数
    上司のアドバイスに従い、必要最小限の時間特徴量のみ生成

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム

    Returns:
    --------
    DataFrame
        時間特徴量を追加したデータフレーム
    """
    df_copy = df.copy()

    # 日付が既にインデックスの場合の処理
    if not isinstance(df_copy.index, pd.DatetimeIndex) and 'time_stamp' in df_copy.columns:
        df_copy['time_stamp'] = pd.to_datetime(df_copy['time_stamp'])
        df_copy = df_copy.set_index('time_stamp')
        print(f"時間列 'time_stamp' をインデックスに設定しました")

    # 基本的な時間特徴量：上司のアドバイスに従い、hourのみ追加
    df_copy['hour'] = df_copy.index.hour

    # 周期的時間特徴量：hourの周期的表現のみ追加
    df_copy['hour_sin'] = np.sin(df_copy['hour'] * 2 * np.pi / 24)
    df_copy['hour_cos'] = np.cos(df_copy['hour'] * 2 * np.pi / 24)

    print("時間特徴量を追加しました: hour, hour_sin, hour_cos")

    return df_copy


def get_time_based_train_test_split(df, test_size=0.2):
    """
    時系列データを時間順に分割するための関数

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    test_size : float
        テストデータの割合 (0.0 ~ 1.0)

    Returns:
    --------
    tuple
        (train_df, test_df): トレーニングデータとテストデータのデータフレーム
    """
    # インデックスをソート
    sorted_idx = df.index.sort_values()

    # カットオフポイントの計算
    cutoff_idx = int(len(sorted_idx) * (1 - test_size))
    cutoff_date = sorted_idx[cutoff_idx]

    print(f"時系列分割: カットオフ日時 = {cutoff_date}")
    print(f"トレーニングデータ期間: {sorted_idx[0]} から {cutoff_date}")
    print(f"テストデータ期間: {cutoff_date} から {sorted_idx[-1]}")

    # 実際にデータフレームを分割
    train_df = df[df.index <= cutoff_date].copy()
    test_df = df[df.index > cutoff_date].copy()

    print(f"トレーニングデータ: {train_df.shape[0]}行, テストデータ: {test_df.shape[0]}行")

    return train_df, test_df
