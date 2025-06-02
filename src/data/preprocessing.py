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


def create_temperature_difference_targets(df, zone_nums, horizons_minutes, time_diff):
    """
    各ゾーンの温度差分（将来温度 - 現在温度）を目的変数として作成

    温度そのものではなく変化量を予測することで、ラグの少ない先読み予測を実現する。

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
        温度差分特徴量を追加したデータフレーム
    """
    print("\n## 温度差分予測の目的変数を作成中...")
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
            future_temp = df_copy[source_col].shift(-shift_periods)
            current_temp = df_copy[source_col]

            # 温度差分を計算（将来温度 - 現在温度）
            target_col = f'temp_diff_{zone}_future_{horizon}'
            df_copy[target_col] = future_temp - current_temp

            print(f"温度差分目的変数を作成: {target_col}")

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


def filter_high_value_targets(df, target_cols, threshold=None, percentile=75):
    """
    目的変数の値が高いデータのみをフィルタリングする関数

    Parameters:
    -----------
    df : DataFrame
        フィルタリング対象のデータフレーム
    target_cols : list
        目的変数の列名リスト
    threshold : float, optional
        フィルタリング閾値（指定されない場合はpercentileを使用）
    percentile : float, optional
        パーセンタイル閾値（0-100）

    Returns:
    --------
    DataFrame
        フィルタリング後のデータフレーム
    dict
        フィルタリング情報
    """
    print(f"\n🔍 高値目的変数フィルタリング実行中...")

    df_filtered = df.copy()
    filter_info = {
        'original_count': len(df),
        'filtered_count': 0,
        'removed_count': 0,
        'thresholds': {},
        'target_stats': {}
    }

    # 各目的変数の統計情報を収集
    for target_col in target_cols:
        if target_col in df.columns:
            target_values = df[target_col].dropna()
            if len(target_values) > 0:
                stats = {
                    'mean': target_values.mean(),
                    'std': target_values.std(),
                    'min': target_values.min(),
                    'max': target_values.max(),
                    'q25': target_values.quantile(0.25),
                    'q50': target_values.quantile(0.50),
                    'q75': target_values.quantile(0.75),
                    'q90': target_values.quantile(0.90),
                    'q95': target_values.quantile(0.95)
                }
                filter_info['target_stats'][target_col] = stats

                # 閾値の決定
                if threshold is None:
                    filter_threshold = target_values.quantile(percentile / 100.0)
                else:
                    filter_threshold = threshold

                filter_info['thresholds'][target_col] = filter_threshold

                print(f"📊 {target_col} 統計:")
                print(f"  平均: {stats['mean']:.3f}, 標準偏差: {stats['std']:.3f}")
                print(f"  範囲: {stats['min']:.3f} - {stats['max']:.3f}")
                print(f"  パーセンタイル: 25%={stats['q25']:.3f}, 50%={stats['q50']:.3f}, 75%={stats['q75']:.3f}")
                print(f"  フィルタ閾値: {filter_threshold:.3f} (>{percentile}%ile)")

    # フィルタリング実行
    if filter_info['thresholds']:
        # 全ての目的変数が閾値以上の行のみを保持
        mask = pd.Series(True, index=df.index)

        for target_col, threshold_val in filter_info['thresholds'].items():
            if target_col in df.columns:
                target_mask = df[target_col] >= threshold_val
                mask = mask & target_mask

                before_count = mask.sum()
                print(f"  {target_col} >= {threshold_val:.3f}: {before_count}行が条件を満たす")

        df_filtered = df[mask].copy()
        filter_info['filtered_count'] = len(df_filtered)
        filter_info['removed_count'] = filter_info['original_count'] - filter_info['filtered_count']

        print(f"\n✅ フィルタリング完了:")
        print(f"  元データ: {filter_info['original_count']:,}行")
        print(f"  フィルタ後: {filter_info['filtered_count']:,}行")
        print(f"  除去データ: {filter_info['removed_count']:,}行 ({filter_info['removed_count']/filter_info['original_count']*100:.1f}%)")

        if filter_info['filtered_count'] < 100:
            print(f"⚠️  警告: フィルタ後のデータが少なすぎます ({filter_info['filtered_count']}行)")
    else:
        print("⚠️  警告: 有効な目的変数が見つかりませんでした")

    return df_filtered, filter_info
