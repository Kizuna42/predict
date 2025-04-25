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

def evaluate_prediction_horizons(df, thermo_df, zone, horizons=[1, 5, 10, 30], output_dir='./output/horizon_analysis'):
    """異なる予測ホライゾンでのモデル性能を評価"""
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',
        2: 'M', 3: 'M', 8: 'M', 9: 'M',
        4: 'R', 5: 'R', 10: 'R', 11: 'R'
    }

    if zone not in zone_to_power:
        print(f"ゾーン {zone} はどの室外機にも割り当てられていません")
        return None

    power_col = zone_to_power[zone]
    valid_col = f'AC_valid_{zone}'
    mode_col = f'AC_mode_{zone}'
    sens_temp_col = f'sens_temp_{zone}'

    required_cols = [valid_col, mode_col, sens_temp_col, power_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ゾーン {zone} に必要なカラムがありません: {', '.join(missing_cols)}")
        return None

    # 結果格納用
    results = {
        'horizon': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'top_features': [],
        'predictions': {},
        'actual': {}
    }

    # 各予測ホライゾンでモデルを訓練・評価
    for horizon in horizons:
        print(f"\nゾーン {zone} の予測ホライゾン {horizon} 分のモデル評価を開始")

        print(f"特徴量を作成中 (予測ホライゾン: {horizon}分)...")
        X, y, features_df = prepare_features_for_sens_temp(df, thermo_df, zone, prediction_horizon=horizon)
        if X is None or y is None or features_df is None:
            print(f"ゾーン {zone} の特徴量を作成できませんでした")
            continue

        print(f"モデルをトレーニングしています...")
        model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)
        if model is None:
            print(f"モデルのトレーニングに失敗しました")
            continue

        # 評価指標を計算
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"予測ホライゾン {horizon}分の結果: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

        # 上位特徴量
        top_features = importance_df.head(5)['Feature'].tolist() if importance_df is not None else []

        # 結果を保存
        results['horizon'].append(horizon)
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['r2'].append(r2)
        results['top_features'].append(top_features)

        # テストデータの最後の部分を保存（可視化用）
        sample_size = min(1000, len(y_test))
        results['predictions'][horizon] = y_pred[-sample_size:]
        results['actual'][horizon] = y_test.iloc[-sample_size:].values

    if not results['horizon']:
        print(f"ゾーン {zone} の評価結果がありません")
        return None

    # 結果ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 評価指標の可視化
    metrics_df = pd.DataFrame({
        'Horizon': results['horizon'],
        'RMSE': results['rmse'],
        'MAE': results['mae'],
        'R²': results['r2']
    })

    # RMSEとMAEのグラフ
    fig, ax1 = plt.subplots(figsize=(12, 7))

    line1 = ax1.plot(metrics_df['Horizon'], metrics_df['RMSE'], 'bo-', linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('予測ホライゾン（分）', fontsize=14)
    ax1.set_ylabel('RMSE (°C)', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    line2 = ax2.plot(metrics_df['Horizon'], metrics_df['MAE'], 'ro-', linewidth=2, markersize=8, label='MAE')
    ax2.set_ylabel('MAE (°C)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')

    # R²のグラフ（第二Y軸）
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # 右軸を少し外側に
    line3 = ax3.plot(metrics_df['Horizon'], metrics_df['R²'], 'go-', linewidth=2, markersize=8, label='R²')
    ax3.set_ylabel('R²', color='green', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='green')

    # 凡例を作成
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=12)

    plt.title(f'ゾーン {zone} - 予測ホライゾンに対する評価指標の変化', fontsize=16)
    plt.grid(True, alpha=0.3)

    # X軸の設定（予測ホライゾンの値を直接表示）
    plt.xticks(metrics_df['Horizon'])

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f'zone_{zone}_horizon_metrics.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 異なるホライゾンでの予測結果の時系列比較
    # 複数のホライゾンの予測結果を同じタイムステップで比較するのは難しいため、
    # 各ホライゾンごとに実測値と予測値の関係を可視化

    # 各ホライゾンについて散布図
    ncols = min(2, len(horizons))
    nrows = (len(horizons) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, horizon in enumerate(horizons):
        if i < len(axes) and horizon in results['predictions']:
            ax = axes[i]
            actual = results['actual'][horizon]
            predicted = results['predictions'][horizon]

            ax.scatter(actual, predicted, alpha=0.5, color=plt.cm.viridis(i/len(horizons)))

            # 完全一致線
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

            # 回帰線
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val, max_val, 100)
            ax.plot(x_trend, p(x_trend), 'r-', linewidth=1.5, alpha=0.6,
                    label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

            rmse = results['rmse'][results['horizon'].index(horizon)]
            r2 = results['r2'][results['horizon'].index(horizon)]

            ax.set_title(f'予測ホライゾン: {horizon}分\nRMSE: {rmse:.3f}, R²: {r2:.3f}', fontsize=12)
            ax.set_xlabel('実測値 (°C)', fontsize=10)
            ax.set_ylabel('予測値 (°C)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=9)

    # 使用されていない軸を非表示
    for i in range(len(horizons), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f'zone_{zone}_horizon_scatter.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 特徴量の重要度変化
    # 各ホライゾンの上位特徴量をまとめる
    top_features_unique = []
    for features in results['top_features']:
        for feature in features:
            if feature not in top_features_unique:
                top_features_unique.append(feature)

    if top_features_unique:
        # 上位10件のみ表示
        top_features_unique = top_features_unique[:10]

        # 各ホライゾンごとに特徴量の順位を記録
        feature_ranks = pd.DataFrame(index=top_features_unique, columns=results['horizon'])
        for horizon_idx, horizon in enumerate(results['horizon']):
            features = results['top_features'][horizon_idx]
            for rank, feature in enumerate(features):
                if feature in top_features_unique:
                    feature_ranks.loc[feature, horizon] = rank + 1

        # 欠損値をNaNで埋める
        feature_ranks = feature_ranks.fillna(np.nan)

        # ヒートマップの作成
        plt.figure(figsize=(12, 8))
        mask = feature_ranks.isnull()
        sns.heatmap(feature_ranks, annot=True, cmap='YlGnBu_r', mask=mask, fmt='.0f',
                   linewidths=0.5, cbar_kws={'label': '順位'})

        plt.title(f'ゾーン {zone} - 異なる予測ホライゾンにおける特徴量重要度の順位', fontsize=16)
        plt.ylabel('特徴量', fontsize=14)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.tight_layout()

        ranks_path = os.path.join(output_dir, f'zone_{zone}_feature_ranks.png')
        plt.savefig(ranks_path, dpi=150, bbox_inches='tight')
        plt.close()

    return metrics_df

def main():
    output_dir = './output/horizon_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # データの読み込み
    df, thermo_df = load_data()

    # 分析するゾーン
    zones_to_analyze = [0, 2, 4]  # L, M, R 各システムの代表ゾーン

    # 評価する予測ホライゾン（分）
    horizons = [1, 5, 10, 30, 60]

    all_metrics = []

    for zone in zones_to_analyze:
        print(f"\n===== ゾーン {zone} の異なる予測ホライゾン分析開始 =====")

        metrics_df = evaluate_prediction_horizons(df, thermo_df, zone, horizons, output_dir)

        if metrics_df is not None:
            metrics_df['Zone'] = zone
            all_metrics.append(metrics_df)

        print(f"===== ゾーン {zone} の分析完了 =====\n")

    # 全ゾーンの結果を比較
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # RMSEの比較
        plt.figure(figsize=(14, 8))
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            plt.plot(zone_data['Horizon'], zone_data['RMSE'], 'o-',
                    linewidth=2, markersize=8, label=f'ゾーン {zone}')

        plt.title('ゾーン別の予測ホライゾンに対するRMSE', fontsize=16)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.ylabel('RMSE (°C)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(horizons)

        plt.tight_layout()
        combined_path = os.path.join(output_dir, 'all_zones_rmse_by_horizon.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()

        # R²の比較
        plt.figure(figsize=(14, 8))
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            plt.plot(zone_data['Horizon'], zone_data['R²'], 'o-',
                    linewidth=2, markersize=8, label=f'ゾーン {zone}')

        plt.title('ゾーン別の予測ホライゾンに対するR²', fontsize=16)
        plt.xlabel('予測ホライゾン（分）', fontsize=14)
        plt.ylabel('R²', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xticks(horizons)

        plt.tight_layout()
        r2_path = os.path.join(output_dir, 'all_zones_r2_by_horizon.png')
        plt.savefig(r2_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 結果のCSV保存
        combined_metrics.to_csv(os.path.join(output_dir, 'horizon_metrics_all_zones.csv'), index=False)

    print(f"すべての分析が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main()
