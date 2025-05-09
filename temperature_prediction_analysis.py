#!/usr/bin/env python
# coding: utf-8

# 基本ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import os
import math
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Outputディレクトリが存在しない場合は作成
os.makedirs('Output', exist_ok=True)

# グラフ設定
sns.set_theme(style="whitegrid", palette="colorblind")
# フォント設定 - matplotlibに日本語フォントがない場合にも対応
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 温度データの外れ値処理関数
def filter_temperature_outliers(df, min_temp=10, max_temp=40, log=True):
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

print("# 空調システム室内温度予測モデル開発")
print("## データ読み込みと前処理")

# データ読み込み
try:
    df = pd.read_csv('AllDayData.csv')
    print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")
except Exception as e:
    print(f"データ読み込みエラー: {e}")

# メモリ使用状況確認
print("メモリ使用状況:")
mem_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"総メモリ使用量: {mem_usage:.2f} MB")

print("\n## データの基本情報確認")

# 先頭データ確認
print("先頭5行のデータ:")
print(df.head(5).to_string())

# データの基本統計量
print("\nデータの基本統計量:")
desc_stats = df.describe()
print(desc_stats)

# 欠損値の確認
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    '欠損値数': missing_values,
    '欠損率(%)': missing_percent
})

# 欠損のある列のみを表示
print("\n欠損値の状況:")
missing_cols = missing_df[missing_df['欠損値数'] > 0].sort_values('欠損値数', ascending=False)
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("欠損値はありません")

print("\n## 時系列データの処理")

# 時間列の処理
if 'time_stamp' in df.columns:
    print("時間列を処理しています...")
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    df = df.set_index('time_stamp')
    print(f"時間列 'time_stamp' をインデックスに設定しました")

    # 基本的な時間特徴量の追加
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    print("時間特徴量を追加しました: hour, day_of_week, is_weekend")

# 温度センサーデータの概要を視覚化
temp_cols = [col for col in df.columns if 'sens_temp' in col]

print("\n## 目的変数の作成（将来温度の予測）")

# サンプリング間隔の確認
time_diff = df.index.to_series().diff().dropna().value_counts().index[0]
print(f"データの時間間隔: {time_diff}")

# 将来温度の生成関数
def create_future_targets(df, zone_nums, horizons_minutes=[5, 10, 15, 20, 30]):
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

# LAG特徴量を作成する関数
def create_lag_features(df, zone_nums, lag_periods=[1, 3, 6]):
    """
    各ゾーンの過去の温度と湿度をLAG特徴量として作成

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    lag_periods : list
        ラグ期間（データサンプリング単位）のリスト

    Returns:
    --------
    DataFrame
        LAG特徴量を追加したデータフレーム
    """
    print("LAG特徴量を作成中...")
    df_copy = df.copy()

    for zone in zone_nums:
        # 温度のLAG特徴量のみ作成（メモリ節約のため湿度は除外）
        if f'sens_temp_{zone}' in df.columns:
            for lag in lag_periods:
                df_copy[f'sens_temp_{zone}_lag_{lag}'] = df_copy[f'sens_temp_{zone}'].shift(lag)

    return df_copy

# 移動平均特徴量を作成する関数
def create_rolling_features(df, zone_nums, windows=[6, 12]):
    """
    各ゾーンの温度と湿度の移動平均を特徴量として作成

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    windows : list
        移動平均の窓サイズ（データサンプリング単位）のリスト

    Returns:
    --------
    DataFrame
        移動平均特徴量を追加したデータフレーム
    """
    print("移動平均特徴量を作成中...")
    df_copy = df.copy()

    for zone in zone_nums:
        # 温度の移動平均のみ作成（移動標準偏差は除外）
        if f'sens_temp_{zone}' in df.columns:
            for window in windows:
                df_copy[f'sens_temp_{zone}_rolling_mean_{window}'] = df_copy[f'sens_temp_{zone}'].rolling(window=window).mean()

    return df_copy

# 実際のゾーン番号を抽出
existing_zones = sorted([int(col.split('_')[2]) for col in temp_cols])
print(f"検出されたゾーン: {existing_zones}")

# テスト実行用に最初のゾーンだけに制限
test_mode = False  # テストモード用フラグ
if test_mode:
    print(f"テストモード: 最初のゾーンのみを使用します")
    existing_zones = [existing_zones[0]]  # 最初のゾーンだけ使用

# LMRのゾーン区分を定義
L_ZONES = [0, 1, 6, 7]
M_ZONES = [2, 3, 8, 9]
R_ZONES = [4, 5, 10, 11]

# 各ゾーンがどの系統に属するかを表示
print("ゾーン区分:")
print(f"L系統ゾーン: {[z for z in L_ZONES if z in existing_zones]}")
print(f"M系統ゾーン: {[z for z in M_ZONES if z in existing_zones]}")
print(f"R系統ゾーン: {[z for z in R_ZONES if z in existing_zones]}")

# 目的変数の作成
df_with_targets = create_future_targets(df, existing_zones)
print(f"目的変数を追加したデータシェイプ: {df_with_targets.shape}")

# 外れ値処理の実行
df_with_targets = filter_temperature_outliers(df_with_targets, min_temp=10, max_temp=40)

# 時系列特徴量の作成
print("\n## 時系列特徴量の作成")
# LAG特徴量の作成
df_with_targets = create_lag_features(df_with_targets, existing_zones)
# 移動平均特徴量の作成
df_with_targets = create_rolling_features(df_with_targets, existing_zones)

# 新しい特徴量を特定
lag_cols = [col for col in df_with_targets.columns if '_lag_' in col]
rolling_cols = [col for col in df_with_targets.columns if '_rolling_' in col]
print(f"LAG特徴量を{len(lag_cols)}個追加しました")
print(f"移動平均特徴量を{len(rolling_cols)}個追加しました")

# 目的変数の例を表示
target_cols = [col for col in df_with_targets.columns if 'future' in col]
first_zone = existing_zones[0]
print(f"\nゾーン{first_zone}の目的変数サンプル:")
print(df_with_targets[[f'sens_temp_{first_zone}'] + [col for col in target_cols if f'_{first_zone}_future' in col]].head(10))

print("\n## 特徴量エンジニアリング")
print("サーモ状態特徴量と未来特徴量を作成します...")

# 予測ホライゾンの定義
horizons = [5, 10, 15, 20, 30]

# 未来シフト説明変数を作成する関数
def create_future_features(df, feature_name, horizons_minutes=[5, 10, 15, 20, 30]):
    """
    指定された特徴量の未来値を作成する関数

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    feature_name : str
        未来シフトする特徴量の名前
    horizons_minutes : list
        予測ホライゾン（分）のリスト

    Returns:
    --------
    DataFrame
        未来特徴量を追加したデータフレーム
    """
    df_copy = df.copy()
    created_features = []

    if feature_name not in df.columns:
        print(f"警告: 列 {feature_name} が見つかりません")
        return df_copy, created_features

    for horizon in horizons_minutes:
        # 時間間隔から必要なシフト数を計算
        shift_periods = int(horizon / time_diff.total_seconds() * 60)

        # 指定分後の値を取得
        future_col = f'{feature_name}_future_{horizon}'
        df_copy[future_col] = df_copy[feature_name].shift(-shift_periods)
        created_features.append(future_col)

    return df_copy, created_features

# サーモ状態の特徴量を作成（センサー温度とAC設定温度の差）
thermo_features = []
for zone in existing_zones:
    if f'sens_temp_{zone}' in df_with_targets.columns and f'AC_set_{zone}' in df_with_targets.columns:
        # サーモ状態 = センサー温度 - 設定温度
        thermo_col = f'thermo_state_{zone}'
        df_with_targets[thermo_col] = df_with_targets[f'sens_temp_{zone}'] - df_with_targets[f'AC_set_{zone}']
        thermo_features.append(thermo_col)
        print(f"ゾーン{zone}のサーモ状態特徴量を作成しました: {thermo_col}")

# 基本的な特徴量のリスト（修正版）
feature_cols = []

# センサー温度・湿度（未来NG）
feature_cols.extend([col for col in df.columns if ('sens_temp_' in col or 'sens_humid_' in col) and 'future' not in col])

# サーモ状態特徴量を追加
feature_cols.extend(thermo_features)

# 空調システム関連（AC_tempは削除、AC_setも削除予定だがサーモ状態計算のために一時的に保持）
ac_features = [col for col in df.columns if 'AC_' in col and 'AC_temp' not in col]
feature_cols.extend(ac_features)

# 環境データ
env_features = []
if 'atmospheric　temperature' in df.columns:
    feature_cols.append('atmospheric　temperature')
    env_features.append('atmospheric　temperature')
if 'total　solar　radiation' in df.columns:
    feature_cols.append('total　solar　radiation')
    env_features.append('total　solar　radiation')

# 電力データは削除（L, M, R）

# 時間特徴量（hourのみ残す）
feature_cols.append('hour')
# day_of_weekとis_weekendは削除

# LAG特徴量と移動平均特徴量を追加
feature_cols.extend(lag_cols)
feature_cols.extend(rolling_cols)

# LMR系統フラグを特徴量から削除（要件に従い）
# df_with_targets['is_L_zone']などは残すが、特徴量からは除外

# 未来特徴量の追加（予測ホライゾンごとにループ）
future_features_all = {}
for horizon in horizons:
    future_features = []

    # 1. サーモ状態の未来特徴量
    for thermo_col in thermo_features:
        df_with_targets, created_features = create_future_features(df_with_targets, thermo_col, [horizon])
        future_features.extend(created_features)

    # 2. 空調発停(AC_valid)の未来特徴量
    for zone in existing_zones:
        if f'AC_valid_{zone}' in df.columns:
            df_with_targets, created_features = create_future_features(df_with_targets, f'AC_valid_{zone}', [horizon])
            future_features.extend(created_features)

    # 3. 空調モード(AC_mode)の未来特徴量
    for zone in existing_zones:
        if f'AC_mode_{zone}' in df.columns:
            df_with_targets, created_features = create_future_features(df_with_targets, f'AC_mode_{zone}', [horizon])
            future_features.extend(created_features)

    # 4. 環境データの未来特徴量
    for env_feature in env_features:
        df_with_targets, created_features = create_future_features(df_with_targets, env_feature, [horizon])
        future_features.extend(created_features)

    future_features_all[horizon] = future_features
    print(f"{horizon}分後予測用の未来特徴量を{len(future_features)}個作成しました")

# 重複する特徴量を削除
feature_cols = list(dict.fromkeys(feature_cols))

# 多項式特徴量の作成（次数2）- 修正版
print("多項式特徴量を作成中...")
poly_features_all = {}

for horizon in horizons:
    # この予測ホライゾン用の特徴量リスト
    key_features = []

    # 現在のセンサー温度
    for zone in existing_zones:
        if f'sens_temp_{zone}' in df.columns:
            key_features.append(f'sens_temp_{zone}')

    # サーモ状態（現在と未来）
    for zone in existing_zones:
        thermo_col = f'thermo_state_{zone}'
        if thermo_col in df_with_targets.columns:
            key_features.append(thermo_col)
            # 対応する未来シフト特徴量も追加
            future_thermo = f'thermo_state_{zone}_future_{horizon}'
            if future_thermo in df_with_targets.columns:
                key_features.append(future_thermo)

    # 空調発停（現在と未来）
    for zone in existing_zones:
        if f'AC_valid_{zone}' in df.columns:
            key_features.append(f'AC_valid_{zone}')
            # 対応する未来シフト特徴量も追加
            future_valid = f'AC_valid_{zone}_future_{horizon}'
            if future_valid in df_with_targets.columns:
                key_features.append(future_valid)

    # 環境データ（現在と未来）
    for env_feature in env_features:
        if env_feature in df.columns:
            key_features.append(env_feature)
            # 対応する未来シフト特徴量も追加
            future_env = f'{env_feature}_future_{horizon}'
            if future_env in df_with_targets.columns:
                key_features.append(future_env)

    # LMR系統フラグは特徴量から除外

    # 重複を削除
    key_features = list(dict.fromkeys(key_features))

    # 欠損値を含む行を除外
    df_poly = df_with_targets[key_features].dropna()

    if len(df_poly) == 0:
        print(f"警告: {horizon}分後の多項式特徴量作成のためのデータがありません")
        poly_features_all[horizon] = []
        continue

    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df_poly)
    poly_feature_names = [f"poly_{horizon}_{i}" for i in range(poly_features.shape[1])]

    # 元のデータフレームと結合
    df_poly_features = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_poly.index)
    df_with_targets = pd.concat([df_with_targets, df_poly_features], axis=1)

    poly_features_all[horizon] = poly_feature_names
    print(f"{horizon}分後用の多項式特徴量を{len(poly_feature_names)}個追加しました")

# 最終的なデータフレームを作成
df_with_all_features = df_with_targets.copy()
print(f"最終的なデータシェイプ: {df_with_all_features.shape}")

print("\n## モデルトレーニングと評価")

# すべてのゾーンと予測ホライゾンの組み合わせに対する結果を保存するためのディクショナリ
all_results = {}

# LAG依存度分析結果を保存する辞書
lag_dependency = {}

# 各ゾーンのモデルを構築
for zone_to_predict in existing_zones:
    zone_results = {}

    # このゾーンに対応するLMR系統を特定
    if zone_to_predict in L_ZONES:
        zone_system = 'L'
    elif zone_to_predict in M_ZONES:
        zone_system = 'M'
    elif zone_to_predict in R_ZONES:
        zone_system = 'R'
    else:
        zone_system = 'Unknown'

    print(f"\nゾーン{zone_to_predict}({zone_system}系統)のモデル構築を開始します")

    for horizon in horizons:
        target_col = f'sens_temp_{zone_to_predict}_future_{horizon}'

        # 目的変数が存在するか確認
        if target_col not in df_with_all_features.columns:
            print(f"警告: 列 {target_col} が見つかりません。ゾーン{zone_to_predict}の{horizon}分後予測をスキップします。")
            continue

        print(f"\nゾーン{zone_to_predict}({zone_system}系統)の{horizon}分後の温度を予測するモデルを構築します")

        # このホライゾン用の特徴量を準備
        horizon_features = feature_cols.copy()

        # このホライゾン用の未来特徴量を追加
        if horizon in future_features_all:
            horizon_features.extend(future_features_all[horizon])

        # このホライゾン用の多項式特徴量を追加
        if horizon in poly_features_all:
            horizon_features.extend(poly_features_all[horizon])

        # 学習用・評価用データの準備
        X = df_with_all_features[horizon_features].dropna()
        y = df_with_all_features[target_col].loc[X.index]

        # 欠損値がある行を削除
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            print(f"警告: ゾーン{zone_to_predict}の{horizon}分後予測に使用可能なデータがありません。スキップします。")
            continue

        print(f"使用する特徴量の数: {len(horizon_features)}")
        print(f"トレーニングデータのサイズ: {X.shape}")

        # トレーニングデータとテストデータに分割
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"トレーニングデータ: {X_train.shape}, テストデータ: {X_test.shape}")

        # LightGBMモデルのトレーニング
        print("LightGBMモデルをトレーニング中...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=10,
            random_state=42
        )
        lgb_model.fit(X_train, y_train)

        # 予測と評価
        y_pred = lgb_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 結果の保存
        zone_results[horizon] = {
            'model': lgb_model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importance': pd.DataFrame({
                'feature': horizon_features,
                'importance': lgb_model.feature_importances_
            }),
            'system': zone_system  # LMR系統の情報も保存
        }

        # LAG依存度分析(15分後または最初のホライゾン)
        if horizon == 15 or (horizon == list(sorted(horizons))[0] and 15 not in horizons):
            feature_importance = zone_results[horizon]['feature_importance']

            # 基本的な温度センサー特徴量（現在値）
            basic_temp_features = [f for f in feature_importance['feature'] if 'sens_temp' in f and 'future' not in f and '_lag_' not in f and '_rolling_' not in f]

            # 追加のLAG特徴量
            lag_features = [f for f in feature_importance['feature'] if '_lag_' in f]

            # 移動平均特徴量
            rolling_features = [f for f in feature_importance['feature'] if '_rolling_' in f]

            # サーモ状態特徴量
            thermo_features = [f for f in feature_importance['feature'] if 'thermo_state' in f]

            # 未来特徴量
            future_features = [f for f in feature_importance['feature'] if '_future_' in f and 'sens_temp' not in f]

            # 多項式特徴量
            poly_features = [f for f in feature_importance['feature'] if 'poly_' in f]

            # その他の特徴量
            other_features = [f for f in feature_importance['feature']
                             if f not in basic_temp_features
                             and f not in lag_features
                             and f not in rolling_features
                             and f not in thermo_features
                             and f not in future_features
                             and f not in poly_features]

            # 重要度の合計を計算
            basic_temp_importance = feature_importance[feature_importance['feature'].isin(basic_temp_features)]['importance'].sum()
            lag_importance = feature_importance[feature_importance['feature'].isin(lag_features)]['importance'].sum()
            rolling_importance = feature_importance[feature_importance['feature'].isin(rolling_features)]['importance'].sum()
            thermo_importance = feature_importance[feature_importance['feature'].isin(thermo_features)]['importance'].sum()
            future_importance = feature_importance[feature_importance['feature'].isin(future_features)]['importance'].sum()
            poly_importance = feature_importance[feature_importance['feature'].isin(poly_features)]['importance'].sum()
            other_importance = feature_importance[feature_importance['feature'].isin(other_features)]['importance'].sum()

            total_importance = (basic_temp_importance + lag_importance + rolling_importance +
                               thermo_importance + future_importance + poly_importance + other_importance)

            # パーセンテージに変換
            basic_temp_percent = basic_temp_importance/total_importance*100
            lag_percent = lag_importance/total_importance*100
            rolling_percent = rolling_importance/total_importance*100
            thermo_percent = thermo_importance/total_importance*100
            future_percent = future_importance/total_importance*100
            poly_percent = poly_importance/total_importance*100
            other_percent = other_importance/total_importance*100
            time_series_percent = basic_temp_percent + lag_percent + rolling_percent

            lag_dependency[zone_to_predict] = {
                'current_temp_percent': basic_temp_percent,
                'lag_percent': lag_percent,
                'rolling_percent': rolling_percent,
                'time_series_total_percent': time_series_percent,
                'thermo_percent': thermo_percent,
                'future_percent': future_percent,
                'poly_percent': poly_percent,
                'other_percent': other_percent,
                'horizon': horizon,
                'system': zone_system
            }

        print(f"評価指標:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

    all_results[zone_to_predict] = zone_results

# 特徴量重要度を可視化（ゾーンごと）
for zone, results in all_results.items():
    if not results:  # 結果がない場合はスキップ
        continue

    # ゾーンの系統を取得
    zone_system = results[list(results.keys())[0]]['system']

    # 15分後予測(またはある最初のホライゾン)の重要度を使用
    horizon = 15 if 15 in results else list(results.keys())[0]
    feature_importance = results[horizon]['feature_importance']
    top_features = feature_importance.sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'LightGBM Feature Importance (Zone {zone} - {zone_system} System, {horizon}min ahead)')
    plt.tight_layout()
    plt.savefig(f'Output/feature_importance_zone_{zone}.png')
    print(f"ゾーン{zone}({zone_system}系統)の特徴量重要度グラフを保存しました: Output/feature_importance_zone_{zone}.png")

# 予測ホライゾンごとに全ゾーンの時系列と散布図をプロット
for horizon in horizons:
    # 予測ホライゾンに対応するゾーンとデータを収集
    zones_with_data = []

    for zone, results in all_results.items():
        if horizon in results:
            zones_with_data.append(zone)

    if not zones_with_data:
        print(f"警告: {horizon}分後予測のデータがありません。スキップします。")
        continue

    # サブプロットの行数と列数を計算
    n_zones = len(zones_with_data)
    n_cols = min(3, n_zones)
    n_rows = math.ceil(n_zones / n_cols)

    # 散布図（実測値 vs 予測値）
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5), squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(zones_with_data):
        results = all_results[zone][horizon]
        y_test = results['y_test']
        y_pred = results['y_pred']
        r2 = results['r2']
        zone_system = results['system']

        axs[i].scatter(y_test, y_pred, alpha=0.5)
        axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axs[i].set_title(f'Zone {zone} - {zone_system} System (R² = {r2:.4f})')
        axs[i].set_xlabel('Actual')
        axs[i].set_ylabel('Predicted')
        axs[i].grid(True)

    # 使わないサブプロットを非表示
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(f'Output/prediction_vs_actual_horizon_{horizon}.png')
    print(f"{horizon}分後予測の散布図を保存しました: Output/prediction_vs_actual_horizon_{horizon}.png")

    # 時系列プロット（最新100ポイント）
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), sharex=True, squeeze=False)
    axs = axs.flatten()

    for i, zone in enumerate(zones_with_data):
        results = all_results[zone][horizon]
        y_test = results['y_test']
        y_pred = results['y_pred']
        zone_system = results['system']

        test_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        }, index=y_test.index)

        # 時間順にソートして最新の100ポイントを使用
        plot_data = test_df.sort_index().iloc[-100:]

        # プロット
        plot_data.plot(ax=axs[i])

        # 時間軸のフォーマットを改善
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45, ha='right')

        axs[i].set_title(f'Zone {zone} - {zone_system} System')
        axs[i].set_xlabel('Time')
        axs[i].set_ylabel('Temperature (°C)')
        axs[i].grid(True)

    # 使わないサブプロットを非表示
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(f'Output/time_series_horizon_{horizon}.png')
    print(f"{horizon}分後予測の時系列プロットを保存しました: Output/time_series_horizon_{horizon}.png")

# 各ゾーンのホライゾン別性能比較
zone_horizon_results = {}

for zone, results in all_results.items():
    if not results:  # 結果がない場合はスキップ
        continue

    # ゾーンの系統を取得
    zone_system = results[list(results.keys())[0]]['system']

    horizon_data = []
    for h in sorted(results.keys()):
        horizon_data.append({
            '予測ホライゾン(分)': h,
            'RMSE': results[h]['rmse'],
            'MAE': results[h]['mae'],
            'R²': results[h]['r2'],
            'System': zone_system
        })

    # 結果をデータフレームに変換
    zone_horizon_results[zone] = pd.DataFrame(horizon_data)

    # ホライゾン別のRMSEとMAEの折れ線グラフ
    plt.figure(figsize=(10, 6))
    plt.plot(zone_horizon_results[zone]['予測ホライゾン(分)'], zone_horizon_results[zone]['RMSE'], 'o-', label='RMSE')
    plt.plot(zone_horizon_results[zone]['予測ホライゾン(分)'], zone_horizon_results[zone]['MAE'], 's-', label='MAE')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('Error')
    plt.title(f'Zone {zone} - {zone_system} System - Prediction Errors by Horizon')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Output/horizon_comparison_zone_{zone}.png')
    print(f"ゾーン{zone}の予測ホライゾン別の誤差比較グラフを保存しました: Output/horizon_comparison_zone_{zone}.png")

    # ホライゾン別のR²の折れ線グラフ
    plt.figure(figsize=(10, 6))
    plt.plot(zone_horizon_results[zone]['予測ホライゾン(分)'], zone_horizon_results[zone]['R²'], 'o-', color='green')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('R² Score')
    plt.title(f'Zone {zone} - {zone_system} System - R² Score by Prediction Horizon')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Output/horizon_r2_comparison_zone_{zone}.png')
    print(f"ゾーン{zone}の予測ホライゾン別の決定係数比較グラフを保存しました: Output/horizon_r2_comparison_zone_{zone}.png")

# LAG依存度分析結果をCSVにまとめる
lag_dependency_df = pd.DataFrame([
    {
        'ゾーン': zone,
        'ホライゾン(分)': data['horizon'],
        '現在温度依存度(%)': data['current_temp_percent'],
        'LAG特徴量依存度(%)': data['lag_percent'],
        '移動平均依存度(%)': data['rolling_percent'],
        '時系列特徴量合計(%)': data['time_series_total_percent'],
        'サーモ状態依存度(%)': data.get('thermo_percent', 0),
        '未来特徴量依存度(%)': data.get('future_percent', 0),
        '多項式特徴量依存度(%)': data.get('poly_percent', 0),
        'その他特徴量依存度(%)': data['other_percent'],
        '系統': data.get('system', 'Unknown')
    }
    for zone, data in lag_dependency.items()
])
lag_dependency_df.to_csv('Output/lag_dependency.csv', index=False)
print("LAG依存度分析結果をCSVファイルに保存しました: Output/lag_dependency.csv")

# 系統別の平均性能を計算
system_performance = pd.DataFrame([
    {
        'System': 'L',
        'Zones': [z for z in L_ZONES if z in existing_zones],
        'Avg_R2': np.mean([all_results[z][15]['r2'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['r2'] for z in L_ZONES if z in existing_zones and all_results.get(z)]),
        'Avg_RMSE': np.mean([all_results[z][15]['rmse'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['rmse'] for z in L_ZONES if z in existing_zones and all_results.get(z)])
    },
    {
        'System': 'M',
        'Zones': [z for z in M_ZONES if z in existing_zones],
        'Avg_R2': np.mean([all_results[z][15]['r2'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['r2'] for z in M_ZONES if z in existing_zones and all_results.get(z)]),
        'Avg_RMSE': np.mean([all_results[z][15]['rmse'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['rmse'] for z in M_ZONES if z in existing_zones and all_results.get(z)])
    },
    {
        'System': 'R',
        'Zones': [z for z in R_ZONES if z in existing_zones],
        'Avg_R2': np.mean([all_results[z][15]['r2'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['r2'] for z in R_ZONES if z in existing_zones and all_results.get(z)]),
        'Avg_RMSE': np.mean([all_results[z][15]['rmse'] if 15 in all_results[z] else all_results[z][list(all_results[z].keys())[0]]['rmse'] for z in R_ZONES if z in existing_zones and all_results.get(z)])
    }
])
system_performance.to_csv('Output/system_performance.csv', index=False)
print("系統別の予測性能をCSVファイルに保存しました: Output/system_performance.csv")

print("\n## 分析まとめ")
# 各ゾーンの結果をテーブルにまとめる
summary_data = []
for zone, results in all_results.items():
    if 15 in results:  # 15分後予測の結果があれば使用
        h = 15
    elif results:  # なければ最初のホライゾンを使用
        h = list(results.keys())[0]
    else:
        continue

    # ゾーンの系統を取得
    zone_system = results[h]['system']

    summary_data.append({
        'ゾーン': zone,
        '系統': zone_system,
        'ホライゾン(分)': h,
        'RMSE': results[h]['rmse'],
        'MAE': results[h]['mae'],
        'R²': results[h]['r2'],
        '現在温度依存度(%)': lag_dependency[zone]['current_temp_percent'],
        'LAG特徴量依存度(%)': lag_dependency[zone]['lag_percent'],
        '移動平均依存度(%)': lag_dependency[zone]['rolling_percent'],
        '時系列特徴量合計(%)': lag_dependency[zone]['time_series_total_percent'],
        'サーモ状態依存度(%)': lag_dependency[zone].get('thermo_percent', 0),
        '未来特徴量依存度(%)': lag_dependency[zone].get('future_percent', 0),
        '多項式特徴量依存度(%)': lag_dependency[zone].get('poly_percent', 0),
        '重要特徴量': ', '.join(results[h]['feature_importance'].sort_values('importance', ascending=False).head(5)['feature'].tolist())
    })

summary_df = pd.DataFrame(summary_data)
print("各ゾーンの予測性能まとめ:")
print(summary_df)

# CSVファイルとして保存
summary_df.to_csv('Output/prediction_summary.csv', index=False)
print("予測性能まとめをCSVファイルに保存しました: Output/prediction_summary.csv")

print("\n## 系統別分析")
print("系統別の予測性能:")
print(system_performance)

# 系統別のパフォーマンス比較グラフ
plt.figure(figsize=(10, 6))
sns.barplot(x='System', y='Avg_R2', data=system_performance)
plt.title('Average R² Score by System (15min Horizon)')
plt.xlabel('System')
plt.ylabel('Average R² Score')
plt.ylim(0, 1)  # R²は0-1の範囲
plt.tight_layout()
plt.savefig('Output/system_r2_comparison.png')
print("系統別のR²スコア比較グラフを保存しました: Output/system_r2_comparison.png")

plt.figure(figsize=(10, 6))
sns.barplot(x='System', y='Avg_RMSE', data=system_performance)
plt.title('Average RMSE by System (15min Horizon)')
plt.xlabel('System')
plt.ylabel('Average RMSE')
plt.tight_layout()
plt.savefig('Output/system_rmse_comparison.png')
print("系統別のRMSE比較グラフを保存しました: Output/system_rmse_comparison.png")

print("\n分析が完了しました。すべての結果はOutputディレクトリに保存されています。")
