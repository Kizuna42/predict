#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
異なる予測ホライゾンでの温度予測モデルの性能を評価するモジュール
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import sys
import pathlib
current_dir = pathlib.Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from thermal_prediction.utils import determine_thermo_status, prepare_features_for_sens_temp, get_zone_power_col
from thermal_prediction.models import train_lgbm_model
from thermal_prediction.visualization import (
    visualize_horizon_metrics,
    visualize_horizon_scatter,
    visualize_horizon_timeseries,
    visualize_all_horizons_timeseries,
    visualize_feature_ranks,
    visualize_zones_comparison
)

def evaluate_prediction_horizons(df, thermo_df, zone, horizons=[5, 10, 15, 20, 30], output_dir='./output/horizon_analysis'):
    """
    異なる予測ホライゾンでのモデル性能を評価

    Args:
        df: 入力データフレーム
        thermo_df: サーモ状態データフレーム
        zone: ゾーン番号
        horizons: 評価する予測ホライゾン（分）のリスト
        output_dir: 出力ディレクトリ

    Returns:
        metrics_df: 評価指標のデータフレーム
    """
    # ゾーンに対応する室外機を特定
    power_col = get_zone_power_col(zone)
    if power_col is None:
        print(f"ゾーン {zone} はどの室外機にも割り当てられていません")
        return None

    valid_col = f'AC_valid_{zone}'
    mode_col = f'AC_mode_{zone}'
    sens_temp_col = f'sens_temp_{zone}'

    # 必要な列を確認
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
        'actual': {},
        'timestamps': {},
        'features_df': {}
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

        # 時系列可視化用にタイムスタンプと特徴量データフレームを保存
        if not y_test.empty:
            test_timestamps = pd.Series(X_test.index).iloc[-sample_size:].values
            results['timestamps'][horizon] = test_timestamps
            results['features_df'][horizon] = features_df

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
    metrics_path = visualize_horizon_metrics(metrics_df, zone, output_dir)
    print(f"評価指標グラフを保存しました: {metrics_path}")

    # 異なるホライゾンでの予測結果の散布図
    scatter_path = visualize_horizon_scatter(results, horizons, zone, output_dir)
    print(f"散布図を保存しました: {scatter_path}")

    # 個別の時系列折れ線グラフ
    timeseries_paths = visualize_horizon_timeseries(results, horizons, zone, output_dir)
    for horizon, path in zip(horizons, timeseries_paths):
        print(f"予測ホライゾン {horizon}分の時系列グラフを保存しました: {path}")

    # 全ホライゾンの時系列比較
    combined_path = visualize_all_horizons_timeseries(results, horizons, zone, output_dir)
    print(f"全予測ホライゾンの時系列比較グラフを保存しました: {combined_path}")

    # 特徴量の重要度変化
    ranks_path = visualize_feature_ranks(results, zone, output_dir)
    if ranks_path:
        print(f"特徴量重要度順位の変化を保存しました: {ranks_path}")

    return metrics_df

def analyze_prediction_horizons(file_path, output_dir, zones='all', horizons=[5, 10, 15, 20, 30],
                              start_date='2024-06-26', end_date='2024-09-20'):
    """
    異なる予測ホライゾンでの温度予測モデルの性能を評価するメイン関数

    Args:
        file_path: 入力データファイルのパス
        output_dir: 出力ディレクトリ
        zones: 分析するゾーン番号（'all'または数値のリスト）
        horizons: 評価する予測ホライゾン（分）のリスト
        start_date: 分析開始日
        end_date: 分析終了日

    Returns:
        combined_metrics: 結合された評価指標のデータフレーム
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)

    # 予測ホライゾンの設定
    if isinstance(horizons, str):
        horizons = [int(h) for h in horizons.split(',') if h.strip()]
    print(f"評価する予測ホライゾン: {horizons} 分")

    # データの読み込み
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

    # サーモ状態の計算
    print("サーモ状態を計算しています...")
    thermo_df = determine_thermo_status(filtered_df)
    wanted_columns = ['time_stamp'] + [f'thermo_{i}' for i in range(12)] + ['thermo_L_or', 'thermo_M_or', 'thermo_R_or']
    for col in wanted_columns:
        if col not in thermo_df.columns and col != 'time_stamp':
            thermo_df[col] = 0
    thermo_df = thermo_df[wanted_columns]
    print("サーモ状態の計算が完了しました")

    # 分析するゾーンの設定
    if zones == 'all':
        zones_to_analyze = list(range(12))  # 全てのゾーン（0-11）
    elif isinstance(zones, str) and zones != 'all':
        # カンマ区切りでゾーン番号を指定
        zones_to_analyze = [int(z) for z in zones.split(',') if z.strip()]
    else:
        zones_to_analyze = zones

    print(f"分析するゾーン: {zones_to_analyze}")

    all_metrics = []

    for zone in zones_to_analyze:
        print(f"\n===== ゾーン {zone} の異なる予測ホライゾン分析開始 =====")

        # ゾーンに対応する室外機を特定
        power_col = get_zone_power_col(zone)
        if power_col is None:
            print(f"ゾーン {zone} はどの室外機にも割り当てられていません。スキップします。")
            continue

        metrics_df = evaluate_prediction_horizons(filtered_df, thermo_df, zone, horizons, output_dir)

        if metrics_df is not None:
            metrics_df['Zone'] = zone
            all_metrics.append(metrics_df)
            print(f"ゾーン {zone} の評価完了")
            print(f"散布図保存場所: {output_dir}/zone_{zone}_horizon_scatter.png")
            print(f"全予測ホライゾンを統合した時系列グラフ: {output_dir}/zone_{zone}_horizon_all_timeseries.png")
            # 各ホライゾンの時系列折れ線グラフに関する情報を表示
            for h in horizons:
                print(f"予測ホライゾン {h}分の個別時系列折れ線グラフ: {output_dir}/zone_{zone}_horizon_{h}_timeseries.png")

        print(f"===== ゾーン {zone} の分析完了 =====\n")

    # 全ゾーンの結果を比較
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)

        # ゾーン間比較グラフの作成
        rmse_path, r2_path, mae_path = visualize_zones_comparison(combined_metrics, horizons, output_dir)

        # 結果のCSV保存
        results_path = os.path.join(output_dir, 'horizon_metrics_all_zones.csv')
        combined_metrics.to_csv(results_path, index=False)

        # サマリーテーブルの作成
        summary_data = []
        for zone in combined_metrics['Zone'].unique():
            zone_data = combined_metrics[combined_metrics['Zone'] == zone]
            best_rmse_idx = zone_data['RMSE'].idxmin()
            best_horizon = zone_data.loc[best_rmse_idx, 'Horizon']
            best_rmse = zone_data.loc[best_rmse_idx, 'RMSE']
            best_mae = zone_data.loc[best_rmse_idx, 'MAE']
            best_r2 = zone_data.loc[best_rmse_idx, 'R²']

            summary_data.append({
                'Zone': int(zone),
                'Best_Horizon': best_horizon,
                'RMSE': best_rmse,
                'MAE': best_mae,
                'R²': best_r2
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Zone')
        summary_path = os.path.join(output_dir, 'best_horizons_summary.csv')
        summary_df.to_csv(summary_path, index=False)

        print("\n最適な予測ホライゾン（RMSEが最小）:")
        print(summary_df.to_string(index=False))

        print(f"\n全ゾーンの予測ホライゾン評価指標をCSVに保存しました: {results_path}")
        print(f"最適な予測ホライゾンのサマリーを保存しました: {summary_path}")
        print(f"全ゾーンの予測ホライゾンRMSE比較図を保存しました: {rmse_path}")
        print(f"全ゾーンの予測ホライゾンR²比較図を保存しました: {r2_path}")
        print(f"全ゾーンの予測ホライゾンMAE比較図を保存しました: {mae_path}")

        return combined_metrics
    else:
        print("評価に成功したゾーンがありません")
        return None
