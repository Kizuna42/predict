#!/usr/bin/env python
# coding: utf-8

# ======================================================================
# データリーク防止対策についての注意点:
# 1. 時系列特性を考慮し、未来の値を使った特徴量エンジニアリングを行わない
# 2. 移動平均はmin_periods=1設定で過去のデータのみ使用
# 3. 多項式特徴量の生成はトレーニングデータのみに基づき、テストデータには変換のみ適用
# 4. 時系列データのランダム分割は行わず、時間順に分割して評価
# ======================================================================

# 基本ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# 温度・湿度データのノイズ軽減のための移動平均処理関数
def apply_smoothing_to_sensors(df, window=6):
    """
    センサーの温度・湿度データに移動平均処理を適用してノイズを軽減する関数
    重要: 未来の値を使用せず、現在までのデータのみを使用する

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    window : int
        移動平均の窓サイズ

    Returns:
    --------
    DataFrame
        平滑化された特徴量を追加したデータフレーム
    """
    print(f"\n## センサーデータの平滑化処理（窓サイズ: {window}）")
    df_copy = df.copy()
    smoothed_features = []

    # センサー温度
    temp_cols = [col for col in df.columns if 'sens_temp_' in col and 'future' not in col]
    for col in temp_cols:
        smoothed_col = f'{col}_smoothed'
        # 過去と現在のデータのみを使用した移動平均（min_periods=1を設定）
        df_copy[smoothed_col] = df_copy[col].rolling(window=window, min_periods=1).mean()
        smoothed_features.append(smoothed_col)
        print(f"温度センサー列 '{col}' の平滑化特徴量を作成しました: {smoothed_col}")

    # センサー湿度（あれば）
    humid_cols = [col for col in df.columns if 'sens_humid_' in col and 'future' not in col]
    for col in humid_cols:
        smoothed_col = f'{col}_smoothed'
        # 過去と現在のデータのみを使用した移動平均（min_periods=1を設定）
        df_copy[smoothed_col] = df_copy[col].rolling(window=window, min_periods=1).mean()
        smoothed_features.append(smoothed_col)
        print(f"湿度センサー列 '{col}' の平滑化特徴量を作成しました: {smoothed_col}")

    print(f"合計{len(smoothed_features)}個の平滑化特徴量を作成しました")
    return df_copy, smoothed_features

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

# 移動平均特徴量を作成する関数（デフォルトの中心化された移動平均ではなく、過去のみを使った移動平均に変更）
def create_rolling_features(df, zone_nums, windows=[6, 12]):
    """
    各ゾーンの温度と湿度の移動平均を特徴量として作成
    重要: 未来の値を使わないようにmin_periods=1を設定し、過去のデータのみを使用

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
    print("過去データのみを使った移動平均特徴量を作成中...")
    df_copy = df.copy()

    for zone in zone_nums:
        # 温度の移動平均のみ作成（移動標準偏差は除外）
        if f'sens_temp_{zone}' in df.columns:
            for window in windows:
                # min_periods=1を設定して、過去のデータのみを使用した移動平均
                df_copy[f'sens_temp_{zone}_rolling_mean_{window}'] = df_copy[f'sens_temp_{zone}'].rolling(
                    window=window, min_periods=1).mean()

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

# センサーデータの平滑化処理（ノイズ対策）
df_with_targets, smoothed_features = apply_smoothing_to_sensors(df_with_targets, window=6)

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
print("サーモ状態特徴量を作成します...")

# 予測ホライゾンの定義
horizons = [5, 10, 15, 20, 30]

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

# センサー温度・湿度（平滑化版を優先、未来NG）
feature_cols.extend(smoothed_features)  # 平滑化されたセンサーデータを優先
# 元のセンサーデータは含めない（平滑化版を使用するため）
# feature_cols.extend([col for col in df.columns if ('sens_temp_' in col or 'sens_humid_' in col) and 'future' not in col and col not in smoothed_features])

# サーモ状態特徴量を追加
feature_cols.extend(thermo_features)

# 空調システム関連（AC_tempは削除、AC_setもサーモ状態計算後は除外）
# AC_validやAC_op_modeなど、サーモ状態計算に使用しなかった他のAC関連情報を選択的に追加
ac_control_features = [
    col for col in df.columns
    if ('AC_valid' in col or 'AC_op_mode' in col) and 'future' not in col # 例: ACのON/OFFや運転モードなど
]
feature_cols.extend(ac_control_features)

# 環境データ
env_features = []
if 'atmospheric　temperature' in df.columns:
    feature_cols.append('atmospheric　temperature')
    env_features.append('atmospheric　temperature')
if 'total　solar　radiation' in df.columns:
    feature_cols.append('total　solar　radiation')
    env_features.append('total　solar　radiation')

# 時間特徴量（hourのみ残す）
feature_cols.append('hour')
# day_of_weekとis_weekendは削除

# LAG特徴量と移動平均特徴量を追加
feature_cols.extend(lag_cols)
feature_cols.extend(rolling_cols)

# 未来特徴量関連のコードを削除（データリーク防止）
# future_features_all と rolling_future_features_all は空の辞書として初期化
future_features_all = {h: [] for h in horizons}
rolling_future_features_all = {h: [] for h in horizons}

# 重複する特徴量を削除
feature_cols = list(dict.fromkeys(feature_cols))

# 多項式特徴量の作成（次数2）- データリーク修正版
print("多項式特徴量を作成中...")
poly_features_all = {}

for horizon in horizons:
    # この予測ホライゾン用の特徴量リスト（未来特徴量を使用しない）
    key_features = []

    # 現在のセンサー温度（平滑化版を使用）
    for zone in existing_zones:
        smoothed_temp = f'sens_temp_{zone}_smoothed'
        if smoothed_temp in df_with_targets.columns:
            key_features.append(smoothed_temp)
        elif f'sens_temp_{zone}' in df_with_targets.columns:
            key_features.append(f'sens_temp_{zone}')

    # サーモ状態（現在のみ、未来は除外）
    for zone in existing_zones:
        thermo_col = f'thermo_state_{zone}'
        if thermo_col in df_with_targets.columns:
            key_features.append(thermo_col)

    # 空調発停（現在のみ、未来は除外）
    for zone in existing_zones:
        if f'AC_valid_{zone}' in df.columns:
            key_features.append(f'AC_valid_{zone}')

    # 環境データ（現在のみ、未来は除外）
    for env_feature in env_features:
        if env_feature in df.columns:
            key_features.append(env_feature)

    # 重複を削除
    key_features = list(dict.fromkeys(key_features))

    # 欠損値を含む行を除外
    df_poly = df_with_targets[key_features].dropna()

    if len(df_poly) == 0:
        print(f"警告: {horizon}分後の多項式特徴量作成のためのデータがありません")
        poly_features_all[horizon] = []
        continue

    # 多項式特徴量作成前にインデックスを保存
    original_index = df_poly.index

    # 重要: 各モデルのトレーニング時に行う処理なので、ここでは変換器のみを保存
    # 実際の変換はトレーニングデータのみに基づいて行う必要があります
    poly = PolynomialFeatures(degree=2, include_bias=False)

    # 特徴量名のリストを作成（後でモデルビルディング時に利用）
    # 注: 実際の特徴量生成はモデル構築時に行う
    try:
        # サンプルデータで変換を試して特徴名を取得
        sample_poly_features = poly.fit_transform(df_poly.iloc[:min(100, len(df_poly))])
        poly_feature_names = [f"poly_{horizon}_{i}" for i in range(sample_poly_features.shape[1])]

        # 多項式変換オブジェクトと特徴量名を保存
        poly_features_all[horizon] = {
            'transformer': poly,
            'feature_names': poly_feature_names,
            'source_features': key_features
        }
        print(f"{horizon}分後用の多項式特徴量変換器を準備しました（特徴量数: {len(poly_feature_names)}）")
    except Exception as e:
        print(f"警告: {horizon}分後の多項式特徴量作成中にエラーが発生しました: {e}")
        poly_features_all[horizon] = []
        continue

# 最終的なデータフレームを作成
df_with_all_features = df_with_targets.copy()
print(f"最終的なデータシェイプ: {df_with_all_features.shape}")

print("\n## モデルトレーニングと評価")

# すべてのゾーンと予測ホライゾンの組み合わせに対する結果を保存するためのディクショナリ
all_results = {}

# LAG依存度分析結果を保存する辞書
lag_dependency = {}

# 時系列分割のためのカットオフポイントを計算（データの80%をトレーニングに使用）
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
    cutoff_date : timestamp
        トレーニングデータとテストデータを分ける日時
    """
    # インデックスをソート
    sorted_idx = df.index.sort_values()

    # カットオフポイントの計算
    cutoff_idx = int(len(sorted_idx) * (1 - test_size))
    cutoff_date = sorted_idx[cutoff_idx]

    print(f"時系列分割: カットオフ日時 = {cutoff_date}")
    print(f"トレーニングデータ期間: {sorted_idx[0]} から {cutoff_date}")
    print(f"テストデータ期間: {cutoff_date} から {sorted_idx[-1]}")

    return cutoff_date

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

        # このホライゾン用の特徴量を準備（未来特徴量は使用しない）
        horizon_features = feature_cols.copy()

        # 多項式特徴量の特徴名をリストに追加
        poly_feature_names = []
        if horizon in poly_features_all and isinstance(poly_features_all[horizon], dict):
            poly_feature_names = poly_features_all[horizon]['feature_names']
            horizon_features.extend(poly_feature_names)

        # 基本的な特徴量に関する学習用・評価用データの準備
        X_base = df_with_all_features[feature_cols].dropna()
        y = df_with_all_features[target_col].loc[X_base.index]

        # 欠損値がある行を削除
        valid_idx = y.notna()
        X_base = X_base[valid_idx]
        y = y[valid_idx]

        if len(X_base) == 0:
            print(f"警告: ゾーン{zone_to_predict}の{horizon}分後予測に使用可能なデータがありません。スキップします。")
            continue

        print(f"使用する特徴量の数: {len(horizon_features)}")
        print(f"基本データセットのサイズ: {X_base.shape}")

        # 時系列に基づいてデータを分割
        cutoff_date = get_time_based_train_test_split(X_base, test_size=0.2)
        X_base_train = X_base[X_base.index <= cutoff_date]
        X_base_test = X_base[X_base.index > cutoff_date]
        y_train = y[y.index <= cutoff_date]
        y_test = y[y.index > cutoff_date]

        # 多項式特徴量を生成（トレーニングデータのみに基づく）
        X_train = X_base_train.copy()
        X_test = X_base_test.copy()

        if horizon in poly_features_all and isinstance(poly_features_all[horizon], dict):
            try:
                poly_transformer = poly_features_all[horizon]['transformer']
                source_features = poly_features_all[horizon]['source_features']

                # トレーニングデータから多項式特徴量を生成
                X_poly_train = poly_transformer.fit_transform(X_base_train[source_features])
                poly_df_train = pd.DataFrame(
                    X_poly_train,
                    columns=poly_feature_names,
                    index=X_base_train.index
                )

                # テストデータに同じ変換を適用
                X_poly_test = poly_transformer.transform(X_base_test[source_features])
                poly_df_test = pd.DataFrame(
                    X_poly_test,
                    columns=poly_feature_names,
                    index=X_base_test.index
                )

                # 特徴量を結合
                X_train = pd.concat([X_train, poly_df_train], axis=1)
                X_test = pd.concat([X_test, poly_df_test], axis=1)

                print(f"多項式特徴量を追加しました（トレーニング: {X_train.shape}, テスト: {X_test.shape}）")
            except Exception as e:
                print(f"警告: 多項式特徴量の生成中にエラーが発生しました: {e}")
                # エラーが発生した場合は多項式特徴量を使わずに続行
                horizon_features = feature_cols

        print(f"最終的なデータセット: トレーニングデータ: {X_train.shape}, テストデータ: {X_test.shape}")

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

        # インデックスの検証と修正
        valid_index = X_test.index
        try:
            # 日付インデックスの有効性確認（matplotlib制限：0001～9999年）
            invalid_dates = valid_index.map(lambda x: not isinstance(x, pd.Timestamp) or x.year < 1 or x.year > 9999)
            if invalid_dates.any():
                print(f"警告: インデックスに無効な日付があります。これらを除外します。")
                valid_mask = ~invalid_dates
                X_test = X_test[valid_mask]
                y_test = y_test[valid_mask]
                y_pred = y_pred[valid_mask]
                # 性能指標を再計算
                if len(y_test) > 0:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
        except Exception as e:
            print(f"インデックス検証中にエラーが発生しました: {e}")

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
                'feature': X_train.columns,  # 修正：実際にモデルが使用した特徴量名
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
    try:
        # sharexをFalseに設定して各プロットが独自のx軸を持つようにする
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), squeeze=False)
        axs = axs.flatten()

        for i, zone in enumerate(zones_with_data):
            try:
                results = all_results[zone][horizon]
                y_test = results['y_test']
                y_pred = results['y_pred']
                zone_system = results['system']

                # インデックスの確認
                if isinstance(y_test.index, pd.DatetimeIndex):
                    # 日付インデックスを確認
                    has_valid_dates = True
                    try:
                        # 有効な日付範囲かチェック
                        for idx in y_test.index[-100:]:
                            if idx.year < 1 or idx.year > 9999:
                                has_valid_dates = False
                                break
                    except Exception:
                        has_valid_dates = False

                    if has_valid_dates:
                        # 有効な日付インデックスを使用
                        test_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred
                        }, index=y_test.index)
                        # 時間順にソート
                        test_df = test_df.sort_index()
                    else:
                        # 連番インデックスを使用
                        range_index = pd.RangeIndex(start=0, stop=len(y_test))
                        test_df = pd.DataFrame({
                            'Actual': y_test.values,
                            'Predicted': y_pred
                        }, index=range_index)
                else:
                    # 日付インデックスでない場合は連番を使用
                    range_index = pd.RangeIndex(start=0, stop=len(y_test))
                    test_df = pd.DataFrame({
                        'Actual': y_test.values,
                        'Predicted': y_pred
                    }, index=range_index)

                # 最新の100ポイントを使用
                plot_data = test_df.iloc[-100:]

                # データがない場合、メッセージを表示して次へ
                if len(plot_data) == 0:
                    axs[i].text(0.5, 0.5, "データがありません",
                              horizontalalignment='center', verticalalignment='center',
                              transform=axs[i].transAxes)
                    continue

                # 実測値と予測値をカラー分けして明確に表示
                axs[i].plot(plot_data.index, plot_data['Actual'], 'b-', label='Actual', linewidth=2)
                axs[i].plot(plot_data.index, plot_data['Predicted'], 'r--', label='Predicted', linewidth=2)

                # 判読しやすいように凡例を表示
                axs[i].legend(loc='best')

                # インデックスがDatetimeIndexの場合のみ日付フォーマットを設定
                if isinstance(plot_data.index, pd.DatetimeIndex):
                    try:
                        # 時間軸のフォーマットを改善
                        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                        plt.setp(axs[i].xaxis.get_majorticklabels(), rotation=45, ha='right')
                        axs[i].set_xlabel('Time')
                    except Exception:
                        # 日付フォーマットが失敗したら通常のインデックス表示
                        axs[i].set_xlabel('Sample')
                else:
                    # データポイント数を表示
                    axs[i].set_xlabel('Sample')
                    # 読みやすいようにx軸のティック数を制限
                    if len(plot_data) > 10:
                        step = max(1, len(plot_data) // 10)
                        ticks = plot_data.index[::step].tolist()
                        axs[i].set_xticks(ticks)

                axs[i].set_title(f'Zone {zone} - {zone_system} System ({horizon}min Ahead Prediction)')
                axs[i].set_ylabel('Temperature (°C)')
                axs[i].grid(True)

                # エラーの可視化 (RMSE)
                rmse = np.sqrt(mean_squared_error(plot_data['Actual'], plot_data['Predicted']))
                axs[i].annotate(f'RMSE: {rmse:.3f}°C',
                             xy=(0.02, 0.95),
                             xycoords='axes fraction',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            except Exception as e:
                print(f"警告: ゾーン{zone}の時系列プロット作成中にエラーが発生しました: {e}")
                axs[i].text(0.5, 0.5, f"プロットエラー",
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[i].transAxes)

        # 使わないサブプロットを非表示
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f'{horizon}min Ahead Temperature Prediction - Actual vs Predicted', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # suptitleのスペースを確保
        plt.savefig(f'Output/time_series_horizon_{horizon}.png')
        print(f"{horizon}分後予測の時系列プロットを保存しました: Output/time_series_horizon_{horizon}.png")
    except Exception as e:
        print(f"エラー: {horizon}分後予測の時系列プロット作成中にエラーが発生しました: {e}")

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

print("\n## 未来特徴量を使用しない検証結果")
print("データリーク修正後は未来の情報を特徴量として使用しないため、")
print("実際の予測状況に近い性能評価となります。")
print("特に多項式特徴量の生成においては、トレーニングデータのみに基づいて変換を行い、")
print("その変換をテストデータに適用することで、未来のデータリークを防止しています。")

# テスト用に最初のゾーンと15分後予測のみを使用
if existing_zones and 15 in horizons:
    test_zone = existing_zones[0]
    test_horizon = 15

    print(f"ゾーン{test_zone}の{test_horizon}分後予測の評価結果:")

    # このゾーンの結果を取得
    if test_zone in all_results and test_horizon in all_results[test_zone]:
        result = all_results[test_zone][test_horizon]
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"MAE: {result['mae']:.4f}")
        print(f"R²: {result['r2']:.4f}")

        # 重要な特徴量を表示
        top_features = result['feature_importance'].sort_values('importance', ascending=False).head(10)
        print("\n上位10個の重要な特徴量:")
        for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
            print(f"{i}. {feature}: {importance:.4f}")
    else:
        print(f"ゾーン{test_zone}の{test_horizon}分後予測の結果が見つかりません")

print("\n分析が完了しました。すべての結果はOutputディレクトリに保存されています。")

# 将来のセンサー以外の特徴量（説明変数）を生成する関数
def create_future_explanatory_features(df, base_features_config, horizons_minutes, time_diff_seconds):
    """
    指定されたベース特徴量の将来値を生成する

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    base_features_config : list of dicts
        各ベース特徴量の設定（例: [{'name': 'atmospheric　temperature', 'type': 'common'},
                                    {'name': 'thermo_state_0', 'type': 'zone_specific', 'zone': 0}])
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : float
        データのサンプリング間隔（秒）

    Returns:
    --------
    DataFrame
        将来の特徴量を追加したデータフレーム
    list
        生成された未来特徴量のカラム名リスト
    """
    df_copy = df.copy()
    created_future_features = []

    for config in base_features_config:
        base_col_name = config['name']

        if base_col_name not in df_copy.columns:
            print(f"警告: ベース列 {base_col_name} がデータフレームに存在しません。スキップします。")
            continue

        for horizon in horizons_minutes:
            shift_periods = int(horizon * 60 / time_diff_seconds) # 分を秒に変換してからシフト数を計算
            future_col_name = f"{base_col_name}_future_{horizon}"
            df_copy[future_col_name] = df_copy[base_col_name].shift(-shift_periods)
            created_future_features.append(future_col_name)
            # print(f"未来特徴量を作成: {future_col_name} (元: {base_col_name}, {horizon}分後)")

    return df_copy, list(set(created_future_features)) # 重複除去して返す
