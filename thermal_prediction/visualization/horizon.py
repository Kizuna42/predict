"""
予測ホライゾン分析の可視化機能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def visualize_horizon_metrics(metrics_df, zone, output_dir):
    """
    予測ホライゾンごとの評価指標を可視化する

    Args:
        metrics_df: 評価指標のデータフレーム
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        metrics_path: 保存したグラフのパス
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # RMSEのプロット
    line1 = ax1.plot(metrics_df['Horizon'], metrics_df['RMSE'], 'bo-',
                    linewidth=2, markersize=8, label='RMSE')
    ax1.set_xlabel('予測ホライゾン（分）', fontsize=14)
    ax1.set_ylabel('RMSE (°C)', color='blue', fontsize=14)
    ax1.tick_params(axis='y', labelcolor='blue')

    # MAEのプロット（同じY軸）
    ax2 = ax1.twinx()
    line2 = ax2.plot(metrics_df['Horizon'], metrics_df['MAE'], 'ro-',
                    linewidth=2, markersize=8, label='MAE')
    ax2.set_ylabel('MAE (°C)', color='red', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='red')

    # R²のプロット（別のY軸）
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # 右軸を少し外側に
    line3 = ax3.plot(metrics_df['Horizon'], metrics_df['R²'], 'go-',
                    linewidth=2, markersize=8, label='R²')
    ax3.set_ylabel('R²', color='green', fontsize=14)
    ax3.tick_params(axis='y', labelcolor='green')

    # 凡例の設定
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=12)

    plt.title(f'ゾーン {zone} - 予測ホライゾンに対する評価指標の変化', fontsize=16)
    plt.grid(True, alpha=0.3)

    # X軸の設定
    plt.xticks(metrics_df['Horizon'])

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, f'zone_{zone}_horizon_metrics.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    plt.close()

    return metrics_path

def visualize_horizon_scatter(results, horizons, zone, output_dir):
    """
    異なる予測ホライゾンでの散布図を作成

    Args:
        results: 結果辞書
        horizons: 予測ホライゾンのリスト
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        scatter_path: 保存したグラフのパス
    """
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

            # 評価指標
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

    return scatter_path

def visualize_horizon_timeseries(results, horizons, zone, output_dir):
    """
    異なる予測ホライゾンでの時系列グラフを作成

    Args:
        results: 結果辞書
        horizons: 予測ホライゾンのリスト
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        各ホライゾンの時系列グラフパスのリスト
    """
    timeseries_paths = []

    # 個別の時系列グラフ
    for horizon in horizons:
        if horizon in results['predictions'] and horizon in results['timestamps']:
            actual = results['actual'][horizon]
            predicted = results['predictions'][horizon]
            timestamps = results['timestamps'][horizon]

            # タイムスタンプをdatetime型に変換
            datetime_timestamps = pd.to_datetime(timestamps)

            # 一定サイズに制限（最後の2日間分のデータのみ表示）
            max_points = min(576, len(actual))  # 5分間隔で2日間は576ポイント

            if len(actual) > max_points:
                actual = actual[-max_points:]
                predicted = predicted[-max_points:]
                datetime_timestamps = datetime_timestamps[-max_points:]

            # 時系列の折れ線グラフ
            plt.figure(figsize=(15, 7))
            plt.plot(datetime_timestamps, actual, 'b-', linewidth=2, label='実測値')
            plt.plot(datetime_timestamps, predicted, 'r--', linewidth=2, label='予測値')

            # グラフの設定
            plt.title(f'ゾーン {zone} - 予測ホライゾン {horizon}分の実測値と予測値の比較', fontsize=16)
            plt.xlabel('時間', fontsize=14)
            plt.ylabel('温度 (°C)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best', fontsize=12)

            # X軸の日付フォーマットを設定
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.gcf().autofmt_xdate()

            # RMSEとR²を表示
            rmse = results['rmse'][results['horizon'].index(horizon)]
            r2 = results['r2'][results['horizon'].index(horizon)]
            plt.figtext(0.01, 0.01, f'RMSE: {rmse:.3f}, R²: {r2:.3f}', fontsize=12)

            plt.tight_layout()
            timeseries_path = os.path.join(output_dir, f'zone_{zone}_horizon_{horizon}_timeseries.png')
            plt.savefig(timeseries_path, dpi=150, bbox_inches='tight')
            plt.close()

            timeseries_paths.append(timeseries_path)

    return timeseries_paths

def visualize_all_horizons_timeseries(results, horizons, zone, output_dir):
    """
    全ての予測ホライゾンを1つのグラフにまとめる

    Args:
        results: 結果辞書
        horizons: 予測ホライゾンのリスト
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        combined_path: 保存したグラフのパス
    """
    if not horizons or not any(h in results['predictions'] for h in horizons):
        return None

    # サブプロットのレイアウトを設定
    ncols = min(2, len(horizons))
    nrows = (len(horizons) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 各ホライゾンの時系列データをプロット
    for i, horizon in enumerate(horizons):
        if i < len(axes) and horizon in results['predictions'] and horizon in results['timestamps']:
            ax = axes[i]
            actual = results['actual'][horizon]
            predicted = results['predictions'][horizon]
            timestamps = results['timestamps'][horizon]

            # タイムスタンプをdatetime型に変換
            datetime_timestamps = pd.to_datetime(timestamps)

            # 一定サイズに制限（最後の12時間分のデータのみ表示）
            max_points = min(144, len(actual))  # 5分間隔で12時間は144ポイント

            if len(actual) > max_points:
                actual = actual[-max_points:]
                predicted = predicted[-max_points:]
                datetime_timestamps = datetime_timestamps[-max_points:]

            # 時系列プロット
            ax.plot(datetime_timestamps, actual, 'b-', linewidth=1.5, label='実測値')
            ax.plot(datetime_timestamps, predicted, 'r--', linewidth=1.5, label='予測値')

            # グラフの設定
            ax.set_title(
                f'予測ホライゾン: {horizon}分\nRMSE: {results["rmse"][results["horizon"].index(horizon)]:.3f}, R²: {results["r2"][results["horizon"].index(horizon)]:.3f}',
                fontsize=12
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)

            # X軸の日付フォーマットを設定
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)

            # Y軸ラベルは左端の列のみに表示
            if i % ncols == 0:
                ax.set_ylabel('温度 (°C)', fontsize=10)

            # X軸ラベルは最下段のみに表示
            if i >= len(horizons) - ncols:
                ax.set_xlabel('時間', fontsize=10)

    # 使用されていない軸を非表示
    for i in range(len(horizons), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'zone_{zone}_horizon_all_timeseries.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()

    return combined_path

def visualize_feature_ranks(results, zone, output_dir):
    """
    各ホライゾンにおける特徴量の重要度順位を可視化

    Args:
        results: 結果辞書
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        ranks_path: 保存したグラフのパス
    """
    # 特徴量の重要度変化
    # 各ホライゾンの上位特徴量をまとめる
    top_features_unique = []
    for features in results['top_features']:
        for feature in features:
            if feature not in top_features_unique:
                top_features_unique.append(feature)

    if not top_features_unique:
        return None

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

    return ranks_path

def visualize_zones_comparison(combined_metrics, horizons, output_dir):
    """
    複数ゾーンのRMSE/R²/MAEを比較する

    Args:
        combined_metrics: 結合されたメトリクスデータフレーム
        horizons: 予測ホライゾンのリスト
        output_dir: 出力ディレクトリ

    Returns:
        保存したグラフのパスのタプル
    """
    # RMSEの比較
    plt.figure(figsize=(14, 10))
    for zone in combined_metrics['Zone'].unique():
        zone_data = combined_metrics[combined_metrics['Zone'] == zone]
        plt.plot(zone_data['Horizon'], zone_data['RMSE'], 'o-',
                linewidth=2, markersize=8, label=f'ゾーン {zone}')

    plt.title('ゾーン別の予測ホライゾンに対するRMSE', fontsize=16)
    plt.xlabel('予測ホライゾン（分）', fontsize=14)
    plt.ylabel('RMSE (°C)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(horizons)

    plt.tight_layout()
    rmse_path = os.path.join(output_dir, 'all_zones_rmse_by_horizon.png')
    plt.savefig(rmse_path, dpi=150, bbox_inches='tight')
    plt.close()

    # R²の比較
    plt.figure(figsize=(14, 10))
    for zone in combined_metrics['Zone'].unique():
        zone_data = combined_metrics[combined_metrics['Zone'] == zone]
        plt.plot(zone_data['Horizon'], zone_data['R²'], 'o-',
                linewidth=2, markersize=8, label=f'ゾーン {zone}')

    plt.title('ゾーン別の予測ホライゾンに対するR²', fontsize=16)
    plt.xlabel('予測ホライゾン（分）', fontsize=14)
    plt.ylabel('R²', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(horizons)

    plt.tight_layout()
    r2_path = os.path.join(output_dir, 'all_zones_r2_by_horizon.png')
    plt.savefig(r2_path, dpi=150, bbox_inches='tight')
    plt.close()

    # MAEの比較
    plt.figure(figsize=(14, 10))
    for zone in combined_metrics['Zone'].unique():
        zone_data = combined_metrics[combined_metrics['Zone'] == zone]
        plt.plot(zone_data['Horizon'], zone_data['MAE'], 'o-',
                linewidth=2, markersize=8, label=f'ゾーン {zone}')

    plt.title('ゾーン別の予測ホライゾンに対するMAE', fontsize=16)
    plt.xlabel('予測ホライゾン（分）', fontsize=14)
    plt.ylabel('MAE (°C)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(horizons)

    plt.tight_layout()
    mae_path = os.path.join(output_dir, 'all_zones_mae_by_horizon.png')
    plt.savefig(mae_path, dpi=150, bbox_inches='tight')
    plt.close()

    return rmse_path, r2_path, mae_path
