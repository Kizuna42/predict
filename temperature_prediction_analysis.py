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

# 実際のゾーン番号を抽出
existing_zones = sorted([int(col.split('_')[2]) for col in temp_cols])
print(f"検出されたゾーン: {existing_zones}")

# 目的変数の作成
df_with_targets = create_future_targets(df, existing_zones)
print(f"目的変数を追加したデータシェイプ: {df_with_targets.shape}")

# 目的変数の例を表示
target_cols = [col for col in df_with_targets.columns if 'future' in col]
first_zone = existing_zones[0]
print(f"\nゾーン{first_zone}の目的変数サンプル:")
print(df_with_targets[[f'sens_temp_{first_zone}'] + [col for col in target_cols if f'_{first_zone}_future' in col]].head(10))

print("\n## 特徴量エンジニアリング")

# 基本的な特徴量のリスト
feature_cols = []

# センサー温度・湿度
feature_cols.extend([col for col in df.columns if ('sens_temp_' in col or 'sens_humid_' in col) and 'future' not in col])

# 空調システム関連
feature_cols.extend([col for col in df.columns if 'AC_' in col])

# 環境データ
if 'atmospheric　temperature' in df.columns:
    feature_cols.append('atmospheric　temperature')
if 'total　solar　radiation' in df.columns:
    feature_cols.append('total　solar　radiation')

# 電力データ
if 'L' in df.columns:
    feature_cols.append('L')
if 'M' in df.columns:
    feature_cols.append('M')
if 'R' in df.columns:
    feature_cols.append('R')

# 時間特徴量
feature_cols.extend(['hour', 'day_of_week', 'is_weekend'])

# 重複する特徴量を削除
feature_cols = list(dict.fromkeys(feature_cols))

# 多項式特徴量の作成（次数2）
print("多項式特徴量を作成中...")
key_features = []
for zone in existing_zones:
    if f'sens_temp_{zone}' in df.columns and f'AC_set_{zone}' in df.columns:
        key_features.extend([f'sens_temp_{zone}', f'AC_set_{zone}'])
    if 'atmospheric　temperature' in df.columns:
        key_features.append('atmospheric　temperature')

# 重複を削除
key_features = list(dict.fromkeys(key_features))

# 欠損値を含む行を除外
df_poly = df_with_targets[key_features].dropna()

poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_poly)
poly_feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]

# 元のデータフレームと結合
df_poly_features = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_poly.index)
df_with_all_features = pd.concat([df_with_targets, df_poly_features], axis=1)

print(f"多項式特徴量を追加しました: {len(poly_feature_names)}個の特徴量")
print(f"最終的なデータシェイプ: {df_with_all_features.shape}")

print("\n## モデルトレーニングと評価")

# すべてのゾーンと予測ホライゾンの組み合わせに対する結果を保存するためのディクショナリ
all_results = {}
horizons = [5, 10, 15, 20, 30]

# LAG依存度分析結果を保存する辞書
lag_dependency = {}

# 各ゾーンのモデルを構築
for zone_to_predict in existing_zones:
    zone_results = {}

    for horizon in horizons:
        target_col = f'sens_temp_{zone_to_predict}_future_{horizon}'

        # 目的変数が存在するか確認
        if target_col not in df_with_all_features.columns:
            print(f"警告: 列 {target_col} が見つかりません。ゾーン{zone_to_predict}の{horizon}分後予測をスキップします。")
            continue

        print(f"\nゾーン{zone_to_predict}の{horizon}分後の温度を予測するモデルを構築します")

        # 学習用・評価用データの準備
        features = feature_cols + poly_feature_names
        X = df_with_all_features[features].dropna()
        y = df_with_all_features[target_col].loc[X.index]

        # 欠損値がある行を削除
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]

        if len(X) == 0:
            print(f"警告: ゾーン{zone_to_predict}の{horizon}分後予測に使用可能なデータがありません。スキップします。")
            continue

        print(f"使用する特徴量の数: {len(features)}")
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
                'feature': features,
                'importance': lgb_model.feature_importances_
            })
        }

        # LAG依存度分析(15分後または最初のホライゾン)
        if horizon == 15 or (horizon == list(sorted(horizons))[0] and 15 not in horizons):
            feature_importance = zone_results[horizon]['feature_importance']
            lag_features = [f for f in feature_importance['feature'] if 'sens_temp' in f and 'future' not in f]
            non_lag_features = [f for f in feature_importance['feature'] if f not in lag_features]

            lag_importance = feature_importance[feature_importance['feature'].isin(lag_features)]['importance'].sum()
            non_lag_importance = feature_importance[feature_importance['feature'].isin(non_lag_features)]['importance'].sum()
            total_importance = lag_importance + non_lag_importance

            lag_percent = lag_importance/total_importance*100
            non_lag_percent = non_lag_importance/total_importance*100

            lag_dependency[zone_to_predict] = {
                'lag_percent': lag_percent,
                'non_lag_percent': non_lag_percent,
                'horizon': horizon
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

    # 15分後予測(またはある最初のホライゾン)の重要度を使用
    horizon = 15 if 15 in results else list(results.keys())[0]
    feature_importance = results[horizon]['feature_importance']
    top_features = feature_importance.sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'LightGBM Feature Importance (Zone {zone}, {horizon}min ahead)')
    plt.tight_layout()
    plt.savefig(f'Output/feature_importance_zone_{zone}.png')
    print(f"ゾーン{zone}の特徴量重要度グラフを保存しました: Output/feature_importance_zone_{zone}.png")

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

        axs[i].scatter(y_test, y_pred, alpha=0.5)
        axs[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axs[i].set_title(f'Zone {zone} (R² = {r2:.4f})')
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

        axs[i].set_title(f'Zone {zone}')
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

    horizon_data = []
    for h in sorted(results.keys()):
        horizon_data.append({
            '予測ホライゾン(分)': h,
            'RMSE': results[h]['rmse'],
            'MAE': results[h]['mae'],
            'R²': results[h]['r2']
        })

    # 結果をデータフレームに変換
    zone_horizon_results[zone] = pd.DataFrame(horizon_data)

    # ホライゾン別のRMSEとMAEの折れ線グラフ
    plt.figure(figsize=(10, 6))
    plt.plot(zone_horizon_results[zone]['予測ホライゾン(分)'], zone_horizon_results[zone]['RMSE'], 'o-', label='RMSE')
    plt.plot(zone_horizon_results[zone]['予測ホライゾン(分)'], zone_horizon_results[zone]['MAE'], 's-', label='MAE')
    plt.xlabel('Prediction Horizon (minutes)')
    plt.ylabel('Error')
    plt.title(f'Zone {zone} - Prediction Errors by Horizon')
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
    plt.title(f'Zone {zone} - R² Score by Prediction Horizon')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Output/horizon_r2_comparison_zone_{zone}.png')
    print(f"ゾーン{zone}の予測ホライゾン別の決定係数比較グラフを保存しました: Output/horizon_r2_comparison_zone_{zone}.png")

# LAG依存度分析結果をCSVにまとめる
lag_dependency_df = pd.DataFrame([
    {
        'ゾーン': zone,
        'ホライゾン(分)': data['horizon'],
        'LAG依存度(%)': data['lag_percent'],
        'その他特徴量依存度(%)': data['non_lag_percent']
    }
    for zone, data in lag_dependency.items()
])
lag_dependency_df.to_csv('Output/lag_dependency.csv', index=False)
print("LAG依存度分析結果をCSVファイルに保存しました: Output/lag_dependency.csv")

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

    summary_data.append({
        'ゾーン': zone,
        'ホライゾン(分)': h,
        'RMSE': results[h]['rmse'],
        'MAE': results[h]['mae'],
        'R²': results[h]['r2'],
        'LAG依存度(%)': lag_dependency[zone]['lag_percent'],
        '重要特徴量': ', '.join(results[h]['feature_importance'].sort_values('importance', ascending=False).head(3)['feature'].tolist())
    })

summary_df = pd.DataFrame(summary_data)
print("各ゾーンの予測性能まとめ:")
print(summary_df)

# CSVファイルとして保存
summary_df.to_csv('Output/prediction_summary.csv', index=False)
print("予測性能まとめをCSVファイルに保存しました: Output/prediction_summary.csv")

print("\n分析が完了しました。すべての結果はOutputディレクトリに保存されています。")
