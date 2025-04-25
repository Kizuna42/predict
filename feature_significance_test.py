#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import logging
from datetime import timedelta
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from improve import prepare_features_for_sens_temp, determine_thermo_status, train_lgbm_model
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='./AllDayData.csv', start_date='2024-06-26', end_date='2024-09-20'):
    """データを読み込み、指定期間でフィルタリングする"""
    print(f"データを読み込んでいます: {file_path}")
    df = pd.read_csv(file_path)
    print(f"データ読み込み完了: {len(df)}行 × {len(df.columns)}列")

    if 'algo' in df.columns:
        before_rows = len(df)
        df = df.dropna(subset=['algo'])
        print(f"'algo'列のNaN行を削除しました: {before_rows} → {len(df)}行")

    df['time_stamp'] = pd.to_datetime(df['time_stamp'])
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    filtered_df = df[(df['time_stamp'] >= start) & (df['time_stamp'] <= end)]
    print(f"分析期間: {start_date} から {end_date} ({len(filtered_df)}行)")

    print("サーモ状態を計算しています...")
    thermo_df = determine_thermo_status(filtered_df)
    wanted_columns = ['time_stamp'] + [f'thermo_{i}' for i in range(12)] + ['thermo_L_or', 'thermo_M_or', 'thermo_R_or']
    for col in wanted_columns:
        if col not in thermo_df.columns and col != 'time_stamp':
            thermo_df[col] = 0
    thermo_df = thermo_df[wanted_columns]
    print("サーモ状態の計算が完了しました")

    return filtered_df, thermo_df

def analyze_lag_feature_significance(df, thermo_df, zone, output_dir='./output/feature_significance'):
    """Lag特徴量の意義を詳細に分析"""
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',
        2: 'M', 3: 'M', 8: 'M', 9: 'M',
        4: 'R', 5: 'R', 10: 'R', 11: 'R'
    }

    if zone not in zone_to_power:
        print(f"ゾーン {zone} はどの室外機にも割り当てられていません")
        return None

    power_col = zone_to_power[zone]
    sens_temp_col = f'sens_temp_{zone}'

    # 必要なカラムが存在するか確認
    required_cols = [f'AC_valid_{zone}', f'AC_mode_{zone}', sens_temp_col, power_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ゾーン {zone} に必要なカラムがありません: {', '.join(missing_cols)}")
        return None

    print(f"\n===== ゾーン {zone} のLag特徴量分析 =====")

    # 基本モデル（標準の予測ホライゾン5分）
    prediction_horizon = 5
    print(f"基本モデル（予測ホライゾン: {prediction_horizon}分）の特徴量を作成中...")
    X, y, features_df = prepare_features_for_sens_temp(df, thermo_df, zone, prediction_horizon=prediction_horizon)
    if X is None or y is None or features_df is None:
        print(f"ゾーン {zone} の特徴量を作成できませんでした")
        return None

    # 予測対象の温度列名
    target_col = f'{sens_temp_col}_future_{prediction_horizon}'

    # 特徴量の相関分析
    # ラグ特徴量と予測対象変数の相関を計算
    lag_cols = [col for col in X.columns if '_lag_' in col or '_change_' in col]

    # ラグ特徴量と現在の温度データを結合
    corr_data = pd.concat([X[lag_cols], features_df[sens_temp_col].reset_index(drop=True),
                          features_df[target_col].reset_index(drop=True)], axis=1)
    corr_data.columns = list(corr_data.columns)

    # 相関行列を計算
    print("相関行列を計算しています...")
    corr_matrix = corr_data.corr()

    # 予測対象変数とラグ特徴量の相関
    target_corr = corr_matrix[target_col].loc[lag_cols].sort_values(ascending=False)
    print(f"\n予測対象（{target_col}）と最も相関の高いラグ特徴量:")
    print(target_corr.head(10))

    # 現在温度とラグ特徴量の相関
    current_corr = corr_matrix[sens_temp_col].loc[lag_cols].sort_values(ascending=False)
    print(f"\n現在温度（{sens_temp_col}）と最も相関の高いラグ特徴量:")
    print(current_corr.head(10))

    # 相関ヒートマップを作成
    plt.figure(figsize=(16, 14))

    # ヒートマップに表示する変数を選択（すべての変数だと多すぎるため）
    selected_vars = [sens_temp_col, target_col] + target_corr.index[:10].tolist()
    selected_corr = corr_matrix.loc[selected_vars, selected_vars]

    sns.heatmap(selected_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'ゾーン {zone} - 温度とラグ特徴量の相関ヒートマップ', fontsize=16)
    plt.tight_layout()

    # 結果ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    corr_path = os.path.join(output_dir, f'zone_{zone}_lag_correlation.png')
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()

    # ラグ特徴量と予測ホライゾンの関係分析
    # モデルを学習
    print("モデルをトレーニングしています...")
    model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)
    if model is None:
        print("モデルのトレーニングに失敗しました")
        return None

    # LightGBMの重要度を使用（permutation importanceの代わりに）
    print("LightGBMの特徴量重要度を分析しています...")
    if importance_df is None or importance_df.empty:
        print("特徴量重要度が取得できませんでした")
        return None

    # ラグ特徴量のみを抽出
    lag_importance = importance_df[importance_df['Feature'].str.contains('_lag_|_change_')]

    # 重要度が高いラグ特徴量のプロット
    plt.figure(figsize=(14, 8))

    top_n = min(15, len(lag_importance))
    top_lag_features = lag_importance.head(top_n)

    bars = plt.bar(
        top_lag_features['Feature'],
        top_lag_features['Importance'],
        color=plt.cm.viridis(np.linspace(0, 0.8, top_n))
    )
    plt.xticks(rotation=45, ha='right')
    plt.title(f'ゾーン {zone} - 上位ラグ特徴量の重要度', fontsize=16)
    plt.ylabel('重要度', fontsize=14)
    plt.xlabel('特徴量', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    imp_path = os.path.join(output_dir, f'zone_{zone}_feature_importance.png')
    plt.savefig(imp_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Lag特徴量と予測ホライゾンの関係を分析
    # 異なるラグ値を持つ特徴量がどのように重要度に影響するかを確認
    lag_values = [1, 2, 5, 15, 30, 60]
    lag_importance_by_value = {}

    for lag in lag_values:
        # 特定のラグ値を持つ特徴量のみを抽出
        lag_features = [f for f in lag_importance['Feature'] if f'_lag_{lag}' in f or f'_change_{lag}' in f]

        if lag_features:
            # ラグ値ごとの重要度を合計
            total_importance = lag_importance[lag_importance['Feature'].isin(lag_features)]['Importance'].sum()
            lag_importance_by_value[lag] = total_importance

    # ラグ値と重要度の関係をプロット
    if lag_importance_by_value:
        lags = list(lag_importance_by_value.keys())
        importances = list(lag_importance_by_value.values())

        plt.figure(figsize=(12, 7))
        plt.bar(lags, importances, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(lags))))
        plt.axvline(x=prediction_horizon, color='red', linestyle='--',
                   label=f'予測ホライゾン ({prediction_horizon}分)')
        plt.title(f'ゾーン {zone} - ラグ値別の特徴量重要度', fontsize=16)
        plt.xlabel('ラグ値（分）', fontsize=14)
        plt.ylabel('合計重要度', fontsize=14)
        plt.xticks(lags)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        lag_imp_path = os.path.join(output_dir, f'zone_{zone}_lag_value_importance.png')
        plt.savefig(lag_imp_path, dpi=150, bbox_inches='tight')
        plt.close()

    # Lag5が重要であることを視覚的に確認
    # 時系列での温度変化とLag5の関係をプロット
    sample_size = min(1000, len(features_df))
    sample_data = features_df.iloc[-sample_size:].reset_index(drop=True).copy()

    plt.figure(figsize=(16, 8))

    # 現在温度と5分後温度をプロット
    plt.plot(sample_data.index, sample_data[sens_temp_col], 'b-', linewidth=2, label='現在温度')
    plt.plot(sample_data.index, sample_data[target_col], 'r-', linewidth=2, label='5分後温度')

    # 5分前温度（Lag5）をプロット
    lag5_col = f'{sens_temp_col}_lag_5'
    if lag5_col in sample_data.columns:
        plt.plot(sample_data.index, sample_data[lag5_col], 'g--', linewidth=1.5, label='5分前温度 (Lag5)')

    plt.title(f'ゾーン {zone} - 現在温度・5分後温度・5分前温度の比較', fontsize=16)
    plt.xlabel('時間ステップ', fontsize=14)
    plt.ylabel('温度 (°C)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    timeseries_path = os.path.join(output_dir, f'zone_{zone}_temperature_lag_comparison.png')
    plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 温度の自己相関分析
    # 現在温度と過去のラグ温度の関係を調べる
    temperature_series = features_df[sens_temp_col].dropna()

    # 自己相関関数を計算
    max_lag = 60  # 60分まで
    autocorr = [temperature_series.autocorr(lag=i) for i in range(1, max_lag+1)]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(1, max_lag+1), autocorr,
                  color=plt.cm.viridis(np.linspace(0, 0.8, max_lag)))

    # 予測ホライゾンを示す縦線
    plt.axvline(x=prediction_horizon, color='red', linestyle='--',
               label=f'予測ホライゾン ({prediction_horizon}分)')

    # 自己相関が高いラグを強調
    high_autocorr_lags = sorted([(i+1, autocorr[i]) for i in range(len(autocorr))],
                               key=lambda x: abs(x[1]), reverse=True)[:5]

    for lag, corr in high_autocorr_lags:
        plt.plot(lag, corr, 'ro', markersize=8)
        plt.annotate(f'Lag {lag}: {corr:.3f}',
                    xy=(lag, corr),
                    xytext=(lag, corr + 0.05 if corr > 0 else corr - 0.05),
                    ha='center',
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->'))

    plt.title(f'ゾーン {zone} - 温度の自己相関関数', fontsize=16)
    plt.xlabel('ラグ（分）', fontsize=14)
    plt.ylabel('自己相関', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()

    autocorr_path = os.path.join(output_dir, f'zone_{zone}_temperature_autocorrelation.png')
    plt.savefig(autocorr_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 温度変化率と予測精度の関係分析
    # 温度変化率を計算
    sample_data['temp_change_rate'] = sample_data[sens_temp_col].diff(1)

    # 変化率の大きさによってデータをビン分け
    sample_data['change_magnitude'] = pd.cut(
        sample_data['temp_change_rate'].abs(),
        bins=[0, 0.01, 0.05, 0.1, float('inf')],
        labels=['微小変化', '小変化', '中変化', '大変化']
    )

    # モデル予測の追加（最後のサンプルデータ部分）
    if len(y_test) >= sample_size:
        sample_data['predicted'] = y_pred[-sample_size:]
        sample_data['error'] = np.abs(sample_data[target_col] - sample_data['predicted'])

        # 変化率の大きさごとの予測誤差
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='change_magnitude', y='error', data=sample_data.dropna(subset=['change_magnitude', 'error']))
        plt.title(f'ゾーン {zone} - 温度変化率の大きさと予測誤差の関係', fontsize=16)
        plt.xlabel('温度変化の大きさ', fontsize=14)
        plt.ylabel('絶対誤差 (°C)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        error_path = os.path.join(output_dir, f'zone_{zone}_change_rate_vs_error.png')
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        plt.close()

    # 最後に結論をまとめた図を作成
    # 予測ホライゾンとラグ5の関係を示す図
    plt.figure(figsize=(12, 7))

    # サンプルデータからランダムに抽出して時系列を表示
    sample_indices = np.random.choice(len(sample_data) - prediction_horizon, 100, replace=False)
    sample_indices = np.sort(sample_indices)

    for idx in sample_indices[:5]:  # 5つのケースのみ表示
        time_range = range(idx, idx + prediction_horizon + 6)  # lag5, 現在, 予測まで

        temp_series = sample_data.iloc[time_range][sens_temp_col].values
        lag5_point = temp_series[0]  # Lag5
        current_point = temp_series[5]  # 現在値
        future_point = temp_series[-1]  # 5分後

        xs = range(len(temp_series))
        plt.plot(xs, temp_series, '-', alpha=0.6, linewidth=1)
        plt.plot([0], [lag5_point], 'go', label='Lag5' if idx == sample_indices[0] else "")
        plt.plot([5], [current_point], 'bo', label='現在値' if idx == sample_indices[0] else "")
        plt.plot([10], [future_point], 'ro', label='5分後 (予測対象)' if idx == sample_indices[0] else "")

    plt.axvline(x=0, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=5, color='blue', linestyle='--', alpha=0.5)
    plt.axvline(x=10, color='red', linestyle='--', alpha=0.5)

    plt.title(f'ゾーン {zone} - Lag5・現在値・予測対象の関係', fontsize=16)
    plt.xlabel('時間ステップ', fontsize=14)
    plt.ylabel('温度 (°C)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    relationship_path = os.path.join(output_dir, f'zone_{zone}_lag5_prediction_relationship.png')
    plt.savefig(relationship_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"ゾーン {zone} のLag特徴量分析が完了しました。結果は {output_dir} に保存されています。")

    return {
        'zone': zone,
        'target_corr': target_corr,
        'current_corr': current_corr,
        'lag_importance': lag_importance,
        'lag_importance_by_value': lag_importance_by_value,
        'autocorr': autocorr
    }

def main():
    output_dir = './output/feature_significance'
    os.makedirs(output_dir, exist_ok=True)

    # データの読み込み
    df, thermo_df = load_data()

    # 分析するゾーン
    zones_to_analyze = [2, 4]  # M, R システムの代表ゾーン

    results = []
    for zone in zones_to_analyze:
        result = analyze_lag_feature_significance(df, thermo_df, zone, output_dir)
        if result:
            results.append(result)

    # 全ゾーンの結果を比較
    if len(results) > 1:
        # 自己相関の比較
        plt.figure(figsize=(14, 8))
        for result in results:
            zone = result['zone']
            autocorr = result['autocorr']
            plt.plot(range(1, len(autocorr)+1), autocorr, '-', linewidth=2, label=f'ゾーン {zone}')

        # 予測ホライゾンを示す縦線
        plt.axvline(x=5, color='red', linestyle='--', label=f'予測ホライゾン (5分)')

        plt.title('ゾーン間の温度自己相関比較', fontsize=16)
        plt.xlabel('ラグ（分）', fontsize=14)
        plt.ylabel('自己相関', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()

        compare_path = os.path.join(output_dir, 'all_zones_autocorrelation.png')
        plt.savefig(compare_path, dpi=150, bbox_inches='tight')
        plt.close()

        # ラグ値別重要度の比較
        if all('lag_importance_by_value' in result and result['lag_importance_by_value'] for result in results):
            # 結果を一つのデータフレームにまとめる
            lag_imp_df = pd.DataFrame()

            for result in results:
                zone = result['zone']
                lag_imp = result['lag_importance_by_value']

                zone_df = pd.DataFrame({
                    'Lag': lag_imp.keys(),
                    'Importance': lag_imp.values(),
                    'Zone': zone
                })

                lag_imp_df = pd.concat([lag_imp_df, zone_df], ignore_index=True)

            # ラグ値と重要度のヒートマップ
            pivot_df = lag_imp_df.pivot(index='Zone', columns='Lag', values='Importance')

            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=0.5)
            plt.title('ラグ値別特徴量重要度のゾーン間比較', fontsize=16)
            plt.ylabel('ゾーン', fontsize=14)
            plt.xlabel('ラグ値（分）', fontsize=14)
            plt.tight_layout()

            heatmap_path = os.path.join(output_dir, 'all_zones_lag_importance_heatmap.png')
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"すべての分析が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main()
