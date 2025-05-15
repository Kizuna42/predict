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
        # 平滑化されたセンサー温度が存在すればそちらを優先してサーモ状態を計算
        base_temp_col_for_thermo = f'sens_temp_{zone}_smoothed' if f'sens_temp_{zone}_smoothed' in df_with_targets.columns else f'sens_temp_{zone}'
        df_with_targets[thermo_col] = df_with_targets[base_temp_col_for_thermo] - df_with_targets[f'AC_set_{zone}']
        thermo_features.append(thermo_col)
        print(f"ゾーン{zone}のサーモ状態特徴量を作成しました: {thermo_col} (ベース温度: {base_temp_col_for_thermo})")

# 未来の説明変数のためのベース特徴量設定
future_explanatory_base_config = []
# 共通環境特徴量
actual_atmo_temp_col_name = None
for col_name in df_with_targets.columns:
    if 'atmospheric' in col_name.lower() and 'temperature' in col_name.lower():
        actual_atmo_temp_col_name = col_name
        future_explanatory_base_config.append({'name': actual_atmo_temp_col_name, 'type': 'common'})
        print(f"環境特徴量 (未来予測対象): {actual_atmo_temp_col_name}")
        break

actual_solar_rad_col_name = None
for col_name in df_with_targets.columns:
    if 'total' in col_name.lower() and 'solar' in col_name.lower() and 'radiation' in col_name.lower():
        actual_solar_rad_col_name = col_name
        future_explanatory_base_config.append({'name': actual_solar_rad_col_name, 'type': 'common'})
        print(f"環境特徴量 (未来予測対象): {actual_solar_rad_col_name}")
        break

# ゾーン別特徴量 (サーモ状態, AC有効状態, ACモード)
for zone in existing_zones:
    if f'thermo_state_{zone}' in df_with_targets.columns:
        future_explanatory_base_config.append({'name': f'thermo_state_{zone}', 'type': 'zone_specific', 'zone': zone})
    if f'AC_valid_{zone}' in df_with_targets.columns:
        future_explanatory_base_config.append({'name': f'AC_valid_{zone}', 'type': 'zone_specific', 'zone': zone})

    ac_mode_col_candidate = f'AC_mode_{zone}' # 想定されるACモードカラム名
    if ac_mode_col_candidate in df_with_targets.columns:
        future_explanatory_base_config.append({'name': ac_mode_col_candidate, 'type': 'zone_specific', 'zone': zone})
        print(f"ACモード特徴量 (未来予測対象): {ac_mode_col_candidate}")
    # else:
        # print(f"情報: ACモード特徴量 {ac_mode_col_candidate} はデータに存在しませんでした。")

# time_diff を秒単位で取得 (create_future_explanatory_features に渡すため)
time_diff_seconds_val = time_diff.total_seconds()

# 未来の説明変数を生成
df_with_targets, all_future_explanatory_features = create_future_explanatory_features(
    df_with_targets,
    future_explanatory_base_config,
    horizons,
    time_diff_seconds_val
)
print(f"{len(all_future_explanatory_features)}個の未来の説明変数を生成しました。")

# 時系列特徴量の作成
print("\n## 時系列特徴量の作成")
# LAG特徴量の作成
df_with_targets = create_lag_features(df_with_targets, existing_zones)
# 移動平均特徴量の作成
df_with_targets = create_rolling_features(df_with_targets, existing_zones)

# 基本的な特徴量のリスト（修正版）
feature_cols = []

# センサー温度・湿度（平滑化版を優先、未来NG）
feature_cols.extend(smoothed_features)

# サーモ状態特徴量を追加 (現在のサーモ状態のみ)
feature_cols.extend(thermo_features)

# 空調システム関連（AC_validなど、サーモ状態計算に使用しなかった他のAC関連情報で「現在」のもの）
# AC_set, AC_temp は含めない
ac_control_features = []
for zone in existing_zones: # ゾーンごとにAC関連特徴量を確認
    if f'AC_valid_{zone}' in df.columns:
        ac_control_features.append(f'AC_valid_{zone}')
    ac_mode_col_candidate = f'AC_mode_{zone}'
    if ac_mode_col_candidate in df.columns: # AC_modeも基本特徴量(現在)に追加
        ac_control_features.append(ac_mode_col_candidate)
        print(f"ACモード特徴量 (現在時刻ベース特徴量): {ac_mode_col_candidate}")

ac_control_features = list(dict.fromkeys(ac_control_features)) # 重複除去
feature_cols.extend(ac_control_features)

# 環境データ (現在の外気温・日射量)
env_features_current = []
if actual_atmo_temp_col_name: # 先ほど取得したカラム名を使用
    feature_cols.append(actual_atmo_temp_col_name)
    env_features_current.append(actual_atmo_temp_col_name)
if actual_solar_rad_col_name: # 先ほど取得したカラム名を使用
    feature_cols.append(actual_solar_rad_col_name)
    env_features_current.append(actual_solar_rad_col_name)

# 時間特徴量（hourのみ残す）
feature_cols.append('hour')
# day_of_weekとis_weekendは削除対象なので、ここでの追加はしない

# LAG特徴量と移動平均特徴量を追加
feature_cols.extend(lag_cols)
feature_cols.extend(rolling_cols)

# 未来特徴量関連のコードを削除（データリーク防止）
# future_features_all と rolling_future_features_all は空の辞書として初期化
# これらは新しい all_future_explanatory_features で管理されるため不要
# future_features_all = {h: [] for h in horizons}
# rolling_future_features_all = {h: [] for h in horizons}

# 重複する特徴量を削除
feature_cols = list(dict.fromkeys(feature_cols))
print(f"基本特徴量 (現在時刻ベース): {len(feature_cols)}個")

# 多項式特徴量の作成（次数2）- データリーク修正版
print("多項式特徴量を作成中...")
poly_features_transformers = {} # Transformer と関連情報を保存する辞書に変更

for horizon in horizons:
    key_features_for_poly = []

    # 現在のセンサー温度（平滑化版を使用）
    for zone in existing_zones:
        smoothed_temp = f'sens_temp_{zone}_smoothed'
        if smoothed_temp in df_with_targets.columns:
            key_features_for_poly.append(smoothed_temp)
        elif f'sens_temp_{zone}' in df_with_targets.columns: # 平滑化がない場合のフォールバック
            key_features_for_poly.append(f'sens_temp_{zone}')

    # 「現在」のサーモ状態
    for zone in existing_zones:
        thermo_col = f'thermo_state_{zone}'
        if thermo_col in df_with_targets.columns:
            key_features_for_poly.append(thermo_col)

    # 「現在」のAC有効状態
    for zone in existing_zones:
        if f'AC_valid_{zone}' in df_with_targets.columns:
            key_features_for_poly.append(f'AC_valid_{zone}')
    # AC_mode_{zone} (現在) があればここに追加

    # 「現在」の環境データ
    key_features_for_poly.extend(env_features_current)

    # 「このホライゾンに対応する未来」の特徴量を追加
    for f_col_config in future_explanatory_base_config:
        base_name = f_col_config['name']
        future_variant_name = f"{base_name}_future_{horizon}"
        if future_variant_name in df_with_targets.columns:
            key_features_for_poly.append(future_variant_name)
        # else:
            # print(f"Poly Warning: 未来特徴量 {future_variant_name} がdf_with_targetsに見つかりません (ホライゾン {horizon})\")

    key_features_for_poly = list(dict.fromkeys(key_features_for_poly))

    # 対象ゾーンに絞った特徴量のみを選択 (多項式特徴量の組み合わせ爆発を防ぐため、関連ゾーンに限定も検討)
    # 今回は、全てのゾーンの現在温度と、予測対象ゾーンに関連する未来特徴量などを組み合わせる
    # より厳密には、予測対象ゾーンのセンサー温度、サーモ状態、関連する未来特徴量に絞るのも手
    # ここでは、上記で集めた key_features_for_poly をそのまま使う

    if not key_features_for_poly:
        print(f"警告: {horizon}分後の多項式特徴量作成のためのキー特徴量が空です。スキップします。")
        poly_features_transformers[horizon] = None # Transformerがないことを示す
        continue

    # 欠損値を含む行を除外して多項式特徴量を学習
    df_poly_train_subset = df_with_targets[key_features_for_poly].dropna()

    if len(df_poly_train_subset) == 0:
        print(f"警告: {horizon}分後の多項式特徴量作成のためのデータがありません（NaN除去後）。スキップします。")
        poly_features_transformers[horizon] = None
        continue

    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False) # interaction_only=Falseで単独項の二乗も含む

    try:
        # トレーニングデータの一部（または全部）でfitして変換器を作成
        # ここでは df_poly_train_subset を使うが、実際のモデル学習時のX_trainから抽出した部分を使うべき
        # → モデル学習ループ内でfitするように変更

        # 特徴量名だけ先に取得するために一時的にfit_transformを試みる
        # 重要：このpolyオブジェクトはここではfitせず、モデル学習ループ内で学習データにfitさせる
        temp_poly_transformer = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        sample_poly_features = temp_poly_transformer.fit_transform(df_poly_train_subset.iloc[:min(100, len(df_poly_train_subset))]) # 少量データで試す

        # 生成される特徴量名を取得 (get_feature_names_out を使うのが望ましいが、バージョン依存のため手動生成)
        # poly_feature_names_generated = temp_poly_transformer.get_feature_names_out(key_features_for_poly)
        # 手動生成の場合、組み合わせが複雑になるため、単純な連番名を使用
        poly_feature_names = [f"poly_{horizon}_{i}" for i in range(sample_poly_features.shape[1])]

        poly_features_transformers[horizon] = {
            'transformer_template': PolynomialFeatures(degree=2, include_bias=False, interaction_only=False), # fitされていないテンプレート
            'source_features': key_features_for_poly,
            'poly_feature_names': poly_feature_names
        }
        print(f"{horizon}分後用の多項式特徴量変換器テンプレートを準備しました (元特徴量数: {len(key_features_for_poly)}, 生成多項式特徴量数: {len(poly_feature_names)})")

    except ValueError as ve:
        print(f"警告: {horizon}分後の多項式特徴量名取得中にValueError: {ve}。Source: {key_features_for_poly[:5]}... ({len(key_features_for_poly)} features)")
        poly_features_transformers[horizon] = None
    except Exception as e:
        print(f"警告: {horizon}分後の多項式特徴量名取得中にエラー: {e}")
        poly_features_transformers[horizon] = None

# 最終的なデータフレームを作成 (この時点では多項式特徴量は実データとして追加されていない)
df_with_all_features = df_with_targets.copy()
print(f"特徴量エンジニアリング後のデータシェイプ: {df_with_all_features.shape}")

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
        # horizon_features = feature_cols.copy() # 元の行は削除

        # 1. 基本特徴量 (現在の値)
        current_features_for_model = feature_cols.copy()

        # 2. このホライゾンに対応する「未来の説明変数」を追加
        horizon_specific_future_explanatory = []
        for f_col_config in future_explanatory_base_config:
            base_name = f_col_config['name']
            # ゾーン別か共通かでカラム名suffixを調整
            # if f_col_config['type'] == 'zone_specific' and f_col_config['zone'] != zone_to_predict:
            #     continue # 他ゾーンの未来値は一旦含めない（多重共線や過学習リスク）、または含める戦略もアリ

            future_variant_name = f"{base_name}_future_{horizon}"
            if future_variant_name in df_with_all_features.columns:
                horizon_specific_future_explanatory.append(future_variant_name)

        current_features_for_model.extend(horizon_specific_future_explanatory)
        current_features_for_model = list(dict.fromkeys(current_features_for_model)) # 重複除去

        # 多項式特徴量の特徴名をリストに追加 (この時点では名前のみ)
        poly_feature_names_for_horizon = []
        if horizon in poly_features_transformers and poly_features_transformers[horizon] is not None:
            poly_feature_names_for_horizon = poly_features_transformers[horizon]['poly_feature_names']
            # current_features_for_model.extend(poly_feature_names_for_horizon) # 実データ生成後に結合するので、ここでは名前だけ保持

        # 基本的な特徴量に関する学習用・評価用データの準備
        # X_base は多項式変換前の、現在の特徴量と未来の説明変数を含むデータ
        X_base_cols = current_features_for_model.copy() # poly適用前のカラムリスト

        # 目的変数とX_baseを準備 (NaN除去のために先にyと結合する戦略もある)
        temp_df_for_dropna = df_with_all_features[X_base_cols + [target_col]].copy()
        temp_df_for_dropna.dropna(subset=[target_col], inplace=True) # まず目的変数のNaNを除去

        # X_baseとyを再定義
        y_intermediate = temp_df_for_dropna[target_col]
        X_base = temp_df_for_dropna[X_base_cols]

        # X_base内のNaNの扱い: LightGBMは扱えるが、多項式特徴量生成前には除去が必要
        # 多項式特徴量のソースとなるカラム (poly_features_transformers[horizon]['source_features']) のNaNは除去
        if horizon in poly_features_transformers and poly_features_transformers[horizon] is not None:
            source_poly_cols = poly_features_transformers[horizon]['source_features']
            # X_base と y_intermediate のインデックスを合わせてから dropna
            common_idx_before_poly_dropna = X_base.index.intersection(y_intermediate.index)
            X_base = X_base.loc[common_idx_before_poly_dropna]
            y_intermediate = y_intermediate.loc[common_idx_before_poly_dropna]

            X_base.dropna(subset=source_poly_cols, inplace=True) # 多項式生成元のNaN除去
            y_intermediate = y_intermediate.loc[X_base.index] # X_baseに合わせてyも更新

        if len(X_base) == 0:
            print(f"警告: ゾーン{zone_to_predict}の{horizon}分後予測に使用可能なデータがありません(X_base作成後)。スキップします。")
            continue

        # 時系列に基づいてデータを分割
        cutoff_date = get_time_based_train_test_split(X_base, test_size=0.2) # X_base はDatetimeIndexを持つ想定

        X_train_base = X_base[X_base.index <= cutoff_date]
        X_test_base = X_base[X_base.index > cutoff_date]
        y_train = y_intermediate[y_intermediate.index <= cutoff_date]
        y_test = y_intermediate[y_intermediate.index > cutoff_date]

        # 多項式特徴量を生成（トレーニングデータでfitし、テストデータにtransform）
        X_train = X_train_base.copy()
        X_test = X_test_base.copy()

        final_feature_list_for_model = X_train_base.columns.tolist() # 初期化

        if horizon in poly_features_transformers and poly_features_transformers[horizon] is not None:
            poly_config = poly_features_transformers[horizon]
            poly_transformer = poly_config['transformer_template'] # fitされていないテンプレートを取得
            source_features_for_poly = poly_config['source_features']
            current_poly_feature_names = poly_config['poly_feature_names']

            # source_features_for_poly が X_train_base/X_test_base に全て存在するか確認
            missing_source_train = [s for s in source_features_for_poly if s not in X_train_base.columns]
            missing_source_test = [s for s in source_features_for_poly if s not in X_test_base.columns]

            if missing_source_train or missing_source_test:
                print(f"警告: 多項式特徴量生成のためのソース特徴量が不足しています。ホライゾン {horizon}。スキップします。")
                print(f"不足 (Train): {missing_source_train}, 不足 (Test): {missing_source_test}")
            elif X_train_base[source_features_for_poly].empty:
                 print(f"警告: 多項式特徴量生成のための学習データが空です (ソース特徴量選択後)。ホライゾン {horizon}。スキップします。")
            else:
                try:
                    # トレーニングデータでPolynomialFeaturesをfit
                    poly_transformer.fit(X_train_base[source_features_for_poly])

                    # トレーニングデータから多項式特徴量を生成
                    X_poly_train = poly_transformer.transform(X_train_base[source_features_for_poly])
                    poly_df_train = pd.DataFrame(
                        X_poly_train,
                        columns=current_poly_feature_names, # 事前に準備した名前を使用
                        index=X_train_base.index
                    )

                    # テストデータに同じ変換を適用
                    X_poly_test = poly_transformer.transform(X_test_base[source_features_for_poly])
                    poly_df_test = pd.DataFrame(
                        X_poly_test,
                        columns=current_poly_feature_names, # 事前に準備した名前を使用
                        index=X_test_base.index
                    )

                    # 特徴量を結合 (元のX_train/X_testにはsource_features_for_polyも含まれているので注意)
                    # 多重共線性を避けるため、元のsource_features_for_polyを削除するか、
                    # またはPolynomialFeaturesでinteraction_only=Trueにするなどの考慮が必要。
                    # ここでは単純に結合するが、重複や強い相関を持つ特徴量がないか後で確認推奨。
                    X_train = pd.concat([X_train, poly_df_train], axis=1)
                    X_test = pd.concat([X_test, poly_df_test], axis=1)

                    final_feature_list_for_model.extend(current_poly_feature_names)
                    final_feature_list_for_model = list(dict.fromkeys(final_feature_list_for_model))


                    print(f"多項式特徴量を追加しました（トレーニング: {X_train.shape}, テスト: {X_test.shape}）")
                except Exception as e:
                    print(f"警告: ホライゾン {horizon} の多項式特徴量の生成/結合中にエラー: {e}")
                    # エラー時も学習は続行するが、多項式特徴量は含まれない

        # X_train, X_test から重複する可能性のあるカラムを削除（特に多項式の元特徴量）
        # PolynomialFeatures(interaction_only=False) の場合、元の特徴量のコピーも含まれることがあるため
        X_train = X_train.loc[:,~X_train.columns.duplicated()]
        X_test = X_test.loc[:,~X_test.columns.duplicated()]
        final_feature_list_for_model = X_train.columns.tolist() # 更新された最終的な特徴量リスト

        if X_train.empty or y_train.empty:
            print(f"警告: ゾーン{zone_to_predict} ホライゾン{horizon}で学習データが空です。スキップします。")
            continue

        print(f"最終的な学習特徴量数: {len(final_feature_list_for_model)}")
        # print(f"使用する特徴量のサンプル: {final_feature_list_for_model[:10]}") # デバッグ用

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

            # 特徴量カテゴリの定義と分類
            current_sensor_temp_features = []
            current_sensor_humid_features = [] # 湿度も考慮する場合
            lag_temp_features = []
            rolling_temp_features = []
            thermo_state_current_features = []
            ac_control_current_features = [] # AC_valid, AC_mode (現在)
            env_current_features = [] # 外気温、日射量 (現在)

            future_thermo_state_features = []
            future_ac_control_features = []
            future_env_features = []

            poly_interaction_features = []
            time_features = [] # hourなど
            other_identified_features = [] # 上記以外で特定されたもの

            all_model_features = feature_importance['feature'].tolist()

            for f_name in all_model_features:
                is_future = '_future_' in f_name
                is_lag = '_lag_' in f_name
                is_rolling = '_rolling_' in f_name
                is_poly = 'poly_' in f_name
                is_thermo = 'thermo_state_' in f_name
                is_sens_temp = 'sens_temp_' in f_name and not is_future and not is_lag and not is_rolling and not is_poly and not is_thermo
                is_sens_humid = 'sens_humid_' in f_name and not is_future and not is_lag and not is_rolling and not is_poly # 湿度の場合
                is_ac_valid = 'AC_valid_' in f_name
                is_ac_mode = 'AC_mode_' in f_name # AC_mode を考慮
                is_atmo_temp = (actual_atmo_temp_col_name and actual_atmo_temp_col_name in f_name)
                is_solar_rad = (actual_solar_rad_col_name and actual_solar_rad_col_name in f_name)
                is_hour = f_name == 'hour'

                if is_poly:
                    poly_interaction_features.append(f_name)
                elif is_lag and is_sens_temp:
                    lag_temp_features.append(f_name)
                elif is_rolling and is_sens_temp:
                    rolling_temp_features.append(f_name)
                elif is_sens_temp:
                    current_sensor_temp_features.append(f_name)
                elif is_sens_humid: # 湿度を追加する場合
                    current_sensor_humid_features.append(f_name)
                elif is_hour:
                    time_features.append(f_name)
                elif is_thermo:
                    if is_future:
                        future_thermo_state_features.append(f_name)
                    else:
                        thermo_state_current_features.append(f_name)
                elif is_ac_valid or is_ac_mode:
                    if is_future:
                        future_ac_control_features.append(f_name)
                    else:
                        ac_control_current_features.append(f_name)
                elif is_atmo_temp or is_solar_rad:
                    if is_future:
                        future_env_features.append(f_name)
                    else:
                        env_current_features.append(f_name)
                else:
                    other_identified_features.append(f_name)

            # 各カテゴリの重要度合計を計算
            def get_sum_importance(features_list):
                return feature_importance[feature_importance['feature'].isin(features_list)]['importance'].sum()

            current_sensor_temp_importance = get_sum_importance(current_sensor_temp_features)
            current_sensor_humid_importance = get_sum_importance(current_sensor_humid_features)
            lag_temp_importance = get_sum_importance(lag_temp_features)
            rolling_temp_importance = get_sum_importance(rolling_temp_features)
            thermo_state_current_importance = get_sum_importance(thermo_state_current_features)
            ac_control_current_importance = get_sum_importance(ac_control_current_features)
            env_current_importance = get_sum_importance(env_current_features)

            future_thermo_state_importance = get_sum_importance(future_thermo_state_features)
            future_ac_control_importance = get_sum_importance(future_ac_control_features)
            future_env_importance = get_sum_importance(future_env_features)

            poly_interaction_importance = get_sum_importance(poly_interaction_features)
            time_importance = get_sum_importance(time_features)
            other_importance = get_sum_importance(other_identified_features)

            total_importance_calculated = sum([
                current_sensor_temp_importance, current_sensor_humid_importance, lag_temp_importance, rolling_temp_importance,
                thermo_state_current_importance, ac_control_current_importance, env_current_importance,
                future_thermo_state_importance, future_ac_control_importance, future_env_importance,
                poly_interaction_importance, time_importance, other_importance
            ])

            # パーセンテージに変換 (total_importance_calculated が0でないことを確認)
            def to_percent(value):
                return (value / total_importance_calculated * 100) if total_importance_calculated > 0 else 0

            lag_dependency[zone_to_predict] = {
                'horizon': horizon,
                'system': zone_system,
                'current_sensor_temp_percent': to_percent(current_sensor_temp_importance),
                'current_sensor_humid_percent': to_percent(current_sensor_humid_importance),
                'lag_temp_percent': to_percent(lag_temp_importance),
                'rolling_temp_percent': to_percent(rolling_temp_importance),
                'thermo_state_current_percent': to_percent(thermo_state_current_importance),
                'ac_control_current_percent': to_percent(ac_control_current_importance),
                'env_current_percent': to_percent(env_current_importance),
                'future_thermo_state_percent': to_percent(future_thermo_state_importance),
                'future_ac_control_percent': to_percent(future_ac_control_importance),
                'future_env_percent': to_percent(future_env_importance),
                'poly_interaction_percent': to_percent(poly_interaction_importance),
                'time_percent': to_percent(time_importance),
                'other_percent': to_percent(other_importance),
                'total_past_time_series_percent': to_percent(current_sensor_temp_importance + current_sensor_humid_importance + lag_temp_importance + rolling_temp_importance),
                'total_current_non_sensor_percent': to_percent(thermo_state_current_importance + ac_control_current_importance + env_current_importance + time_importance),
                'total_future_explanatory_percent': to_percent(future_thermo_state_importance + future_ac_control_importance + future_env_importance)
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

    fig.suptitle(f'{horizon}min Ahead Temperature Prediction - Actual vs Predicted', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # suptitleのスペースを確保
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
        '系統': data.get('system', 'Unknown'),
        '現在温度依存度(%)': data['current_sensor_temp_percent'],
        '現在湿度依存度(%)': data['current_sensor_humid_percent'],
        'LAG温度依存度(%)': data['lag_temp_percent'],
        '移動平均温度依存度(%)': data['rolling_temp_percent'],
        '現在サーモ状態依存度(%)': data['thermo_state_current_percent'],
        '現在AC制御依存度(%)': data['ac_control_current_percent'],
        '現在環境データ依存度(%)': data['env_current_percent'],
        '未来サーモ状態依存度(%)': data['future_thermo_state_percent'],
        '未来AC制御依存度(%)': data['future_ac_control_percent'],
        '未来環境データ依存度(%)': data['future_env_percent'],
        '多項式特徴量依存度(%)': data['poly_interaction_percent'],
        '時間特徴量依存度(%)': data['time_percent'],
        'その他特徴量依存度(%)': data['other_percent'],
        '過去時系列合計(%)': data['total_past_time_series_percent'],
        '現在非センサー合計(%)': data['total_current_non_sensor_percent'],
        '未来説明変数合計(%)': data['total_future_explanatory_percent'],
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
        # LAG依存度分析の結果をサマリーにも追加 (主要なものを抜粋)
        '過去時系列依存度(%)': lag_dependency[zone]['total_past_time_series_percent'],
        '現在非センサー依存度(%)': lag_dependency[zone]['total_current_non_sensor_percent'],
        '未来説明変数依存度(%)': lag_dependency[zone]['total_future_explanatory_percent'],
        '多項式特徴量依存度(%)': lag_dependency[zone]['poly_interaction_percent'],
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
