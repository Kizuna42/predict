#!/usr/bin/env python
# coding: utf-8

"""
温度予測モデル - LAG特徴量と移動平均特徴量を使用したバージョン
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import os
from datetime import datetime

# 出力ディレクトリの作成
os.makedirs('Output', exist_ok=True)

# グラフ設定
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

print("# 空調システム室内温度予測モデル - LAG特徴量版")

# データ読み込み
print("## データ読み込みと前処理")
try:
    df = pd.read_csv('AllDayData.csv')
    print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
except Exception as e:
    print(f"データ読み込みエラー: {e}")
    import sys
    sys.exit(1)

# 時間列の処理
print("## 時系列データの処理")
df['time_stamp'] = pd.to_datetime(df['time_stamp'])
df = df.set_index('time_stamp')
print("時間列 'time_stamp' をインデックスに設定しました")

# 時間特徴量の追加
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
print("時間特徴量を追加しました: hour, day_of_week, is_weekend")

# 温度センサー列を特定
temp_cols = [col for col in df.columns if 'sens_temp' in col]
existing_zones = sorted(list(set([int(col.split('_')[2]) for col in temp_cols])))
print(f"検出されたゾーン: {existing_zones}")

# ゾーン区分の定義
L_ZONES = [1, 7]
M_ZONES = [2, 3, 8, 9]
R_ZONES = [4, 5, 10, 11]

print("ゾーン区分:")
print(f"L系統ゾーン: {[z for z in existing_zones if z in L_ZONES]}")
print(f"M系統ゾーン: {[z for z in existing_zones if z in M_ZONES]}")
print(f"R系統ゾーン: {[z for z in existing_zones if z in R_ZONES]}")

# 外れ値処理関数
def filter_temperature_outliers(df, min_temp=10, max_temp=40):
    """
    温度データの外れ値をNaNに置換

    Parameters:
    -----------
    df : DataFrame
        外れ値処理するデータフレーム
    min_temp : float
        最小温度しきい値（これ未満は外れ値）
    max_temp : float
        最大温度しきい値（これ超過は外れ値）

    Returns:
    --------
    DataFrame
        外れ値処理後のデータフレーム
    """
    df_filtered = df.copy()
    outlier_count = 0

    # 温度センサー列の外れ値処理
    for col in df.columns:
        if 'sens_temp' in col:
            # 外れ値を特定
            outliers = (df[col] < min_temp) | (df[col] > max_temp)
            outlier_count_col = outliers.sum()

            if outlier_count_col > 0:
                outlier_vals = df.loc[outliers, col]
                print(f"{col}で{outlier_count_col}個の外れ値を検出（範囲: {outlier_vals.min():.2f}～{outlier_vals.max():.2f}℃）")

                # 外れ値をNaNに置換
                df_filtered.loc[outliers, col] = np.nan
                outlier_count += outlier_count_col

    print(f"全体で{outlier_count}個の外れ値をNaNに置換しました")
    return df_filtered

# LAG特徴量を作成する関数
def create_lag_features(df, zone_nums, lag_periods=[1, 3, 6]):
    """
    各ゾーンの過去の温度をLAG特徴量として作成
    """
    print("LAG特徴量を作成中...")
    df_copy = df.copy()

    for zone in zone_nums:
        # 温度のLAG特徴量
        if f'sens_temp_{zone}' in df.columns:
            for lag in lag_periods:
                df_copy[f'sens_temp_{zone}_lag_{lag}'] = df_copy[f'sens_temp_{zone}'].shift(lag)

    return df_copy

# 移動平均特徴量を作成する関数
def create_rolling_features(df, zone_nums, windows=[6, 12]):
    """
    各ゾーンの温度の移動平均を特徴量として作成
    """
    print("移動平均特徴量を作成中...")
    df_copy = df.copy()

    for zone in zone_nums:
        # 温度の移動平均
        if f'sens_temp_{zone}' in df.columns:
            for window in windows:
                df_copy[f'sens_temp_{zone}_rolling_mean_{window}'] = df_copy[f'sens_temp_{zone}'].rolling(window=window).mean()

    return df_copy

# 将来温度の生成関数
def create_future_targets(df, zone_nums, horizon_minutes=15):
    """
    各ゾーンの将来温度を目的変数として作成
    """
    df_copy = df.copy()

    # 時間間隔の検出
    time_diff = pd.Series(df.index).diff().mode()[0]
    print(f"データの時間間隔: {time_diff}")

    for zone in zone_nums:
        source_col = f'sens_temp_{zone}'
        if source_col not in df.columns:
            print(f"警告: 列 {source_col} が見つかりません")
            continue

        # 時間間隔から必要なシフト数を計算
        shift_periods = int(horizon_minutes / time_diff.total_seconds() * 60)

        # 指定分後の温度を取得
        target_col = f'sens_temp_{zone}_future_{horizon_minutes}'
        df_copy[target_col] = df_copy[source_col].shift(-shift_periods)

    return df_copy

# テスト/評価用のゾーンを選択
test_zones = [2, 9]  # M系統からゾーン2と9
print(f"\n## テスト用に選択したゾーン: {test_zones}")

# 目的変数作成（将来温度の予測）
future_horizon = 15  # 15分後を予測
df_with_targets = create_future_targets(df, test_zones, horizon_minutes=future_horizon)

# 外れ値処理
print("\n## 温度データの外れ値処理")
df_filtered = filter_temperature_outliers(df_with_targets, min_temp=10, max_temp=40)

# 時系列特徴量の作成
print("\n## 時系列特徴量の作成")
df_with_features = create_lag_features(df_filtered, test_zones)
df_with_features = create_rolling_features(df_with_features, test_zones)

# 時系列特徴量をリストアップ
lag_cols = [col for col in df_with_features.columns if '_lag_' in col]
rolling_cols = [col for col in df_with_features.columns if '_rolling_' in col]
print(f"LAG特徴量を{len(lag_cols)}個追加しました")
print(f"移動平均特徴量を{len(rolling_cols)}個追加しました")

# 目的変数の例を表示
for zone in test_zones:
    target_col = f'sens_temp_{zone}_future_{future_horizon}'
    print(f"\nゾーン{zone}の{future_horizon}分後温度予測の目的変数サンプル:")
    print(df_with_features[[f'sens_temp_{zone}', target_col]].head(10))

# 各ゾーンごとの特徴量と目的変数を整理
models = {}
results = {}

for zone in test_zones:
    print(f"\n## ゾーン{zone}のモデル構築")
    # 目的変数
    target_col = f'sens_temp_{zone}_future_{future_horizon}'

    # 特徴量リスト
    feature_cols = []

    # 基本特徴量
    feature_cols.append(f'sens_temp_{zone}')  # 現在の温度

    # 時間特徴量
    feature_cols.extend(['hour', 'day_of_week', 'is_weekend'])

    # LAG特徴量
    zone_lag_cols = [col for col in lag_cols if f'sens_temp_{zone}' in col]
    feature_cols.extend(zone_lag_cols)

    # 移動平均特徴量
    zone_rolling_cols = [col for col in rolling_cols if f'sens_temp_{zone}' in col]
    feature_cols.extend(zone_rolling_cols)

    # 全ての欠損値を含む行を削除
    cols_to_use = feature_cols + [target_col]
    df_zone = df_with_features[cols_to_use].dropna()

    print(f"使用する特徴量の数: {len(feature_cols)}")
    print(f"特徴量: {', '.join(feature_cols)}")
    print(f"有効なデータ行数: {len(df_zone)}")

    # トレーニングデータとテストデータに分割
    X = df_zone[feature_cols]
    y = df_zone[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"トレーニングデータ: {X_train.shape}, テストデータ: {X_test.shape}")

    # LightGBMモデルの構築
    print("LightGBMモデルをトレーニング中...")
    model = lgb.LGBMRegressor(
        objective='regression',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )

    # モデルのトレーニング（シンプルなパラメータで）
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    # 予測と評価
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("評価指標:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # 特徴量重要度
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n特徴量重要度 (上位5件):")
    print(feature_importance.head(5))

    # モデルと結果を保存
    models[zone] = model
    results[zone] = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'feature_importance': feature_importance
    }

    # 散布図による予測vs実測の比較
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'ゾーン{zone}の{future_horizon}分後温度予測: 実測値 vs 予測値')
    plt.xlabel('実測値 (℃)')
    plt.ylabel('予測値 (℃)')
    plt.grid(True)
    plt.savefig(f'Output/prediction_scatter_zone_{zone}.png')
    print(f"散布図を保存しました: Output/prediction_scatter_zone_{zone}.png")

    # 特徴量重要度のグラフ
    plt.figure(figsize=(12, 8))
    importance_df = feature_importance.head(10)
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title(f'ゾーン{zone}の{future_horizon}分後温度予測: 特徴量重要度トップ10')
    plt.tight_layout()
    plt.savefig(f'Output/feature_importance_zone_{zone}.png')
    print(f"特徴量重要度グラフを保存しました: Output/feature_importance_zone_{zone}.png")

# 結果のまとめ
print("\n## 分析まとめ")
summary_data = []
for zone in test_zones:
    zone_type = "L系統" if zone in L_ZONES else "M系統" if zone in M_ZONES else "R系統"

    # LAG特徴量と移動平均特徴量の重要度合計
    feature_importance = results[zone]['feature_importance']
    lag_features = [f for f in feature_importance['feature'] if '_lag_' in f]
    rolling_features = [f for f in feature_importance['feature'] if '_rolling_' in f]

    lag_importance = feature_importance[feature_importance['feature'].isin(lag_features)]['importance'].sum()
    rolling_importance = feature_importance[feature_importance['feature'].isin(rolling_features)]['importance'].sum()
    total_importance = feature_importance['importance'].sum()

    lag_percent = lag_importance / total_importance * 100
    rolling_percent = rolling_importance / total_importance * 100
    time_series_percent = lag_percent + rolling_percent

    summary_data.append({
        'ゾーン': zone,
        'ゾーン区分': zone_type,
        'RMSE': results[zone]['rmse'],
        'MAE': results[zone]['mae'],
        'R²': results[zone]['r2'],
        'LAG特徴量重要度(%)': lag_percent,
        '移動平均特徴量重要度(%)': rolling_percent,
        '時系列特徴量合計(%)': time_series_percent,
        '最重要特徴量': feature_importance.iloc[0]['feature']
    })

summary_df = pd.DataFrame(summary_data)
print(summary_df)

# 結果をCSVに保存
summary_df.to_csv('Output/prediction_summary.csv', index=False)
print("分析結果をCSVファイルに保存しました: Output/prediction_summary.csv")

print("\n分析が完了しました。すべての結果はOutputディレクトリに保存されています。")
