#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

def train_and_test_with_feature_ablation(df, thermo_df, zone, output_dir='./output/lag_analysis'):
    """特定のゾーンについて、特徴量を段階的に除外しながらモデルを訓練・評価する"""
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

    # 標準の特徴量とモデル
    print(f"ゾーン {zone} の標準特徴量を作成中...")
    X, y, features_df = prepare_features_for_sens_temp(df, thermo_df, zone, prediction_horizon=5)
    if X is None or y is None or features_df is None:
        print(f"ゾーン {zone} の特徴量を作成できませんでした")
        return None

    print(f"標準モデルをトレーニングしています...")
    model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)

    # Lag5特徴量を除外したモデル
    print(f"Lag5特徴量を除外したモデルを作成中...")
    lag5_cols = [col for col in X.columns if '_lag_5' in col or '_change_5' in col]
    X_no_lag5 = X.drop(columns=lag5_cols)

    print(f"Lag5除外モデルをトレーニングしています...")
    model_no_lag5, X_test_no_lag5, y_test_no_lag5, y_pred_no_lag5, _ = train_lgbm_model(X_no_lag5, y)

    # すべてのラグ特徴量を除外したモデル
    print(f"すべてのラグ特徴量を除外したモデルを作成中...")
    all_lag_cols = [col for col in X.columns if '_lag_' in col or '_change_' in col]
    X_no_lags = X.drop(columns=all_lag_cols)

    print(f"すべてのラグ除外モデルをトレーニングしています...")
    model_no_lags, X_test_no_lags, y_test_no_lags, y_pred_no_lags, _ = train_lgbm_model(X_no_lags, y)

    # ナイーブ予測（現在値をそのまま未来の予測とする）
    print(f"ナイーブ予測を計算しています...")
    # 特徴量にsens_temp_X列があることを前提
    current_temp_idx = features_df.index[-len(y_test):]
    naive_pred = features_df.loc[current_temp_idx, sens_temp_col].values

    # 予測値を5分後の実際の値と比較するためのデータを作成
    prediction_comparison = pd.DataFrame({
        'actual': y_test.values,
        'predicted': y_pred,
        'predicted_no_lag5': y_pred_no_lag5,
        'predicted_no_lags': y_pred_no_lags,
        'naive_pred': naive_pred
    })

    # 結果の保存ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 結果の評価
    metrics = {}
    for model_name, predictions in [
        ('標準モデル', y_pred),
        ('Lag5除外', y_pred_no_lag5),
        ('全ラグ除外', y_pred_no_lags),
        ('ナイーブ予測', naive_pred)
    ]:
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = np.mean(np.abs(y_test - predictions))
        r2 = r2_score(y_test, predictions)
        metrics[model_name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
        print(f"{model_name}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # メトリクスの可視化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_names = ['RMSE', 'MAE', 'R²']
    model_names = list(metrics.keys())

    for i, metric in enumerate(metric_names):
        values = [metrics[model][metric] for model in model_names]
        axes[i].bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 0.8, len(model_names))))
        axes[i].set_title(f'{metric} 比較', fontsize=14)
        axes[i].set_ylabel(metric, fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f'zone_{zone}_metrics_comparison.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 時系列での予測比較（最後の1000ポイント）
    sample_size = min(1000, len(prediction_comparison))
    sample_data = prediction_comparison.iloc[-sample_size:].reset_index(drop=True)

    plt.figure(figsize=(15, 8))
    plt.plot(sample_data.index, sample_data['actual'], 'k-', linewidth=2, label='実測値')
    plt.plot(sample_data.index, sample_data['predicted'], 'b-', linewidth=1.5, label='標準モデル予測')
    plt.plot(sample_data.index, sample_data['predicted_no_lag5'], 'g--', linewidth=1.5, label='Lag5除外予測')
    plt.plot(sample_data.index, sample_data['predicted_no_lags'], 'r:', linewidth=1.5, label='全ラグ除外予測')
    plt.plot(sample_data.index, sample_data['naive_pred'], 'm-.', linewidth=1.5, label='ナイーブ予測(現在値)')

    plt.title(f'ゾーン {zone} - 異なるモデルによる予測比較', fontsize=16)
    plt.xlabel('時間ステップ', fontsize=14)
    plt.ylabel('温度 (°C)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()

    time_series_path = os.path.join(output_dir, f'zone_{zone}_time_series_comparison.png')
    plt.savefig(time_series_path, dpi=150, bbox_inches='tight')
    plt.close()

    # 散布図による比較
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    model_data = [
        ('標準モデル', sample_data['predicted'], 'blue'),
        ('Lag5除外', sample_data['predicted_no_lag5'], 'green'),
        ('全ラグ除外', sample_data['predicted_no_lags'], 'red'),
        ('ナイーブ予測', sample_data['naive_pred'], 'magenta')
    ]

    for i, (model_name, pred_data, color) in enumerate(model_data):
        row, col = i // 2, i % 2
        ax = axes[row, col]

        ax.scatter(sample_data['actual'], pred_data, alpha=0.6, c=color)

        # 完全一致線
        min_val = min(sample_data['actual'].min(), pred_data.min())
        max_val = max(sample_data['actual'].max(), pred_data.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)

        # 回帰線
        z = np.polyfit(sample_data['actual'], pred_data, 1)
        p = np.poly1d(z)
        ax.plot(np.array([min_val, max_val]), p(np.array([min_val, max_val])),
                color=color, linestyle='-', linewidth=2,
                label=f'y = {z[0]:.2f}x + {z[1]:.2f}')

        # 評価メトリクス
        rmse = np.sqrt(mean_squared_error(sample_data['actual'], pred_data))
        r2 = r2_score(sample_data['actual'], pred_data)

        ax.set_title(f'{model_name}\nRMSE: {rmse:.3f}, R²: {r2:.3f}', fontsize=14)
        ax.set_xlabel('実測値 (°C)', fontsize=12)
        ax.set_ylabel('予測値 (°C)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')

    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f'zone_{zone}_scatter_comparison.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Lag5特徴量と他のラグ特徴量の重要度比較
    if importance_df is not None:
        lag5_importance = importance_df[importance_df['Feature'].str.contains('_lag_5|_change_5')]
        other_lags_importance = importance_df[
            importance_df['Feature'].str.contains('_lag_|_change_') &
            ~importance_df['Feature'].str.contains('_lag_5|_change_5')
        ]

        lag5_sum = lag5_importance['Importance'].sum()
        other_lags_sum = other_lags_importance['Importance'].sum()
        non_lag_sum = importance_df[
            ~importance_df['Feature'].str.contains('_lag_|_change_')
        ]['Importance'].sum()

        importance_summary = pd.DataFrame({
            'Feature Type': ['Lag5 特徴量', 'その他のラグ特徴量', '非ラグ特徴量'],
            'Importance': [lag5_sum, other_lags_sum, non_lag_sum]
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Feature Type', y='Importance', data=importance_summary,
                   palette=plt.cm.viridis(np.linspace(0.2, 0.8, 3)))

        # 割合を追加
        total = importance_summary['Importance'].sum()
        for i, val in enumerate(importance_summary['Importance']):
            plt.text(i, val + 0.01 * total, f'{val/total*100:.1f}%',
                    ha='center', va='bottom', fontsize=12)

        plt.title(f'ゾーン {zone} - 特徴量タイプ別の重要度', fontsize=16)
        plt.ylabel('重要度', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        importance_path = os.path.join(output_dir, f'zone_{zone}_feature_type_importance.png')
        plt.savefig(importance_path, dpi=150, bbox_inches='tight')
        plt.close()

    return {
        'zone': zone,
        'metrics': metrics,
        'importance': {
            'lag5': lag5_sum,
            'other_lags': other_lags_sum,
            'non_lag': non_lag_sum
        } if importance_df is not None else None
    }

def main():
    output_dir = './output/lag_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # データの読み込み
    df, thermo_df = load_data()

    # 分析するゾーン
    zones_to_analyze = [0, 2, 4]  # L, M, R 各システムの代表ゾーン

    results = []
    for zone in zones_to_analyze:
        print(f"\n===== ゾーン {zone} の分析開始 =====")
        result = train_and_test_with_feature_ablation(df, thermo_df, zone, output_dir)
        if result:
            results.append(result)
        print(f"===== ゾーン {zone} の分析完了 =====\n")

    # 全ゾーンの結果を比較
    if results:
        # メトリクス比較グラフ
        metrics_df = pd.DataFrame(columns=['Zone', 'Model', 'RMSE', 'MAE', 'R²'])

        for result in results:
            zone = result['zone']
            for model_name, model_metrics in result['metrics'].items():
                metrics_df = pd.concat([
                    metrics_df,
                    pd.DataFrame({
                        'Zone': [zone],
                        'Model': [model_name],
                        'RMSE': [model_metrics['RMSE']],
                        'MAE': [model_metrics['MAE']],
                        'R²': [model_metrics['R²']]
                    })
                ], ignore_index=True)

        # RMSEの比較グラフ
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Zone', y='RMSE', hue='Model', data=metrics_df,
                   palette=plt.cm.viridis(np.linspace(0, 0.8, len(metrics_df['Model'].unique()))))
        plt.title('各ゾーンにおけるモデル別のRMSE比較', fontsize=16)
        plt.ylabel('RMSE (°C)', fontsize=14)
        plt.xlabel('ゾーン', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='モデル', fontsize=12)
        plt.tight_layout()

        rmse_path = os.path.join(output_dir, 'all_zones_rmse_comparison.png')
        plt.savefig(rmse_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 特徴量タイプの重要度比較
        importance_results = [r for r in results if r['importance'] is not None]
        if importance_results:
            importance_df = pd.DataFrame(columns=['Zone', 'Feature Type', 'Importance'])

            for result in importance_results:
                zone = result['zone']
                for feature_type, importance in result['importance'].items():
                    feature_name = {
                        'lag5': 'Lag5 特徴量',
                        'other_lags': 'その他のラグ特徴量',
                        'non_lag': '非ラグ特徴量'
                    }.get(feature_type, feature_type)

                    importance_df = pd.concat([
                        importance_df,
                        pd.DataFrame({
                            'Zone': [zone],
                            'Feature Type': [feature_name],
                            'Importance': [importance]
                        })
                    ], ignore_index=True)

            # 各ゾーンごとに正規化
            for zone in importance_df['Zone'].unique():
                zone_mask = importance_df['Zone'] == zone
                total = importance_df.loc[zone_mask, 'Importance'].sum()
                importance_df.loc[zone_mask, 'Importance Pct'] = importance_df.loc[zone_mask, 'Importance'] / total * 100

            plt.figure(figsize=(12, 6))
            sns.barplot(x='Zone', y='Importance Pct', hue='Feature Type', data=importance_df,
                       palette=plt.cm.viridis(np.linspace(0.2, 0.8, 3)))
            plt.title('各ゾーンにおける特徴量タイプの重要度比較 (%)', fontsize=16)
            plt.ylabel('重要度 (%)', fontsize=14)
            plt.xlabel('ゾーン', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.legend(title='特徴量タイプ', fontsize=12)
            plt.tight_layout()

            importance_path = os.path.join(output_dir, 'all_zones_feature_importance_comparison.png')
            plt.savefig(importance_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"すべての分析が完了しました。結果は {output_dir} に保存されています。")

if __name__ == "__main__":
    main()
