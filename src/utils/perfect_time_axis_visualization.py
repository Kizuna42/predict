#!/usr/bin/env python
# coding: utf-8

"""
完璧な時間軸修正可視化システム
予測値と同じ時刻（予測対象時刻）の実測値を正しく比較
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional, Tuple, List
import os
import math
from .font_config import setup_japanese_font


def get_future_actual_values(original_data: pd.Series,
                           input_timestamps: pd.DatetimeIndex,
                           horizon: int) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    予測対象時刻の実測値を取得

    Parameters:
    -----------
    original_data : pd.Series
        元の時系列データ
    input_timestamps : pd.DatetimeIndex
        入力時刻のタイムスタンプ
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    tuple
        (予測対象時刻, 予測対象時刻の実測値)
    """
    # 予測対象時刻の計算
    future_timestamps = input_timestamps + pd.Timedelta(minutes=horizon)

    # 予測対象時刻の実測値を取得
    future_actual_values = []
    for ts in future_timestamps:
        if ts in original_data.index:
            future_actual_values.append(original_data.loc[ts])
        else:
            future_actual_values.append(np.nan)

    return future_timestamps, np.array(future_actual_values)


def plot_perfect_time_axis_comparison(input_timestamps: pd.DatetimeIndex,
                                    input_actual_values: np.ndarray,
                                    predicted_values: np.ndarray,
                                    future_actual_values: np.ndarray,
                                    horizon: int,
                                    zone: int = None,
                                    title: str = None,
                                    save_path: str = None) -> plt.Figure:
    """
    完璧な時間軸比較プロット

    Parameters:
    -----------
    input_timestamps : pd.DatetimeIndex
        入力時刻
    input_actual_values : np.ndarray
        入力時刻の実測値
    predicted_values : np.ndarray
        予測値
    future_actual_values : np.ndarray
        予測対象時刻の実測値
    horizon : int
        予測ホライゾン（分）
    zone : int, optional
        ゾーン番号
    title : str, optional
        プロットタイトル
    save_path : str, optional
        保存パス

    Returns:
    --------
    plt.Figure
        作成されたフィギュア
    """

    # フォント設定（安定化）
    try:
        setup_japanese_font()
    except Exception:
        # フォント設定に失敗した場合はデフォルトを使用
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    # 絵文字や特殊文字の文字化け対策
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14

    # 予測対象時刻の計算
    future_timestamps = input_timestamps + pd.Timedelta(minutes=horizon)

    # 有効なデータのマスク
    valid_future_mask = ~np.isnan(future_actual_values)
    valid_input_mask = ~np.isnan(input_actual_values)
    valid_pred_mask = ~np.isnan(predicted_values)

    # 3つのサブプロット作成
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))

    # 1. 従来の間違った方法
    axes[0].plot(input_timestamps[valid_input_mask],
                input_actual_values[valid_input_mask],
                'b-', linewidth=2, label='実測値（入力時刻）', alpha=0.8)
    axes[0].plot(input_timestamps[valid_pred_mask],
                predicted_values[valid_pred_mask],
                'r--', linewidth=2, label='予測値（間違った時間軸）', alpha=0.8)

    axes[0].set_title('[間違った方法] 従来の方法: 予測値が入力時刻に表示',
                     fontsize=14, color='red', fontweight='bold')
    axes[0].set_ylabel('温度 (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 修正された方法（予測値の時間軸のみ修正）
    axes[1].plot(input_timestamps[valid_input_mask],
                input_actual_values[valid_input_mask],
                'b-', linewidth=2, label='実測値（入力時刻）', alpha=0.8)
    axes[1].plot(future_timestamps[valid_pred_mask],
                predicted_values[valid_pred_mask],
                'r--', linewidth=2, label=f'予測値（+{horizon}分後）', alpha=0.8)

    axes[1].set_title('[部分修正] 予測値の時間軸は修正されたが、比較対象が不適切',
                     fontsize=14, color='orange', fontweight='bold')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 完璧な方法（予測値と同じ時刻の実測値で比較）
    if np.sum(valid_future_mask) > 0:
        # 予測対象時刻の実測値をプロット
        axes[2].plot(future_timestamps[valid_future_mask],
                    future_actual_values[valid_future_mask],
                    'g-', linewidth=2, label=f'実測値（+{horizon}分後）', alpha=0.8)

        # 予測値を同じ時刻でプロット
        axes[2].plot(future_timestamps[valid_pred_mask],
                    predicted_values[valid_pred_mask],
                    'r--', linewidth=2, label=f'予測値（+{horizon}分後）', alpha=0.8)

        # 性能指標の計算
        common_indices = valid_future_mask & valid_pred_mask
        if np.sum(common_indices) > 0:
            mae = np.mean(np.abs(future_actual_values[common_indices] - predicted_values[common_indices]))
            rmse = np.sqrt(np.mean((future_actual_values[common_indices] - predicted_values[common_indices]) ** 2))
            corr = np.corrcoef(future_actual_values[common_indices], predicted_values[common_indices])[0, 1]

            # 性能指標を表示
            axes[2].text(0.02, 0.98,
                        f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C\n相関: {corr:.3f}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)

        axes[2].set_title('[完璧な方法] 予測値と同じ時刻の実測値で比較',
                         fontsize=14, color='green', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, f'{horizon}分後の実測値データが不足',
                    transform=axes[2].transAxes, ha='center', va='center', fontsize=12)
        axes[2].set_title('[データ不足] 予測対象時刻の実測値なし',
                         fontsize=14, color='red', fontweight='bold')

    axes[2].set_ylabel('温度 (°C)')
    axes[2].set_xlabel('日時')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # X軸の書式設定
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.tick_params(axis='x', rotation=45)

    # 全体タイトル
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    else:
        zone_str = f'ゾーン {zone} ' if zone is not None else ''
        fig.suptitle(f'{zone_str}完璧な時間軸修正比較（{horizon}分予測）',
                    fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"完璧な時間軸比較プロット保存: {save_path}")

    return fig


def create_perfect_visualization_for_zone(results_dict: Dict,
                                        original_df: pd.DataFrame,
                                        zone: int,
                                        horizon: int,
                                        save_dir: str = None,
                                        sample_size: int = 200) -> Dict[str, Any]:
    """
    特定ゾーンの完璧な可視化を作成

    Parameters:
    -----------
    results_dict : dict
        モデル結果辞書
    original_df : pd.DataFrame
        元のデータフレーム
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        保存ディレクトリ
    sample_size : int
        サンプルサイズ

    Returns:
    --------
    dict
        可視化結果
    """

    result = {
        'zone': zone,
        'horizon': horizon,
        'success': False,
        'error_message': None,
        'file_paths': [],
        'metrics': {}
    }

    try:
        # データの取得
        zone_results = results_dict.get(zone, {})
        horizon_results = zone_results.get(horizon, {})

        if not horizon_results:
            result['error_message'] = f"ゾーン {zone}, ホライゾン {horizon} のデータが見つかりません"
            return result

        # 必要なデータの確認
        required_keys = ['test_data', 'test_y', 'test_predictions']
        if not all(k in horizon_results for k in required_keys):
            result['error_message'] = f"必要なデータが不足: {[k for k in required_keys if k not in horizon_results]}"
            return result

        test_data = horizon_results['test_data']
        test_y = horizon_results['test_y']
        test_predictions = horizon_results['test_predictions']

        # 有効なデータのインデックス
        valid_indices = test_y.dropna().index
        if len(valid_indices) < 10:
            result['error_message'] = f"有効データが不足: {len(valid_indices)}ポイント"
            return result

        # サンプルデータの選択
        actual_sample_size = min(sample_size, len(valid_indices))
        sample_indices = valid_indices[-actual_sample_size:]

        # 入力時刻の実測値を取得
        temp_col = f'sens_temp_{zone}'
        if temp_col not in test_data.columns:
            result['error_message'] = f"温度データ列が見つかりません: {temp_col}"
            return result

        if temp_col not in original_df.columns:
            result['error_message'] = f"元データに温度列が見つかりません: {temp_col}"
            return result

        # データの抽出
        input_actual_values = test_data.loc[sample_indices, temp_col].values
        sample_predictions = test_predictions[-actual_sample_size:] if len(test_predictions) >= actual_sample_size else test_predictions

        # 予測対象時刻の実測値を取得
        future_timestamps, future_actual_values = get_future_actual_values(
            original_df[temp_col], sample_indices, horizon
        )

        # 完璧な時間軸比較プロットの作成
        if save_dir:
            save_path = os.path.join(save_dir, f'perfect_time_axis_zone_{zone}_horizon_{horizon}.png')
        else:
            save_path = None

        fig = plot_perfect_time_axis_comparison(
            input_timestamps=sample_indices,
            input_actual_values=input_actual_values,
            predicted_values=sample_predictions,
            future_actual_values=future_actual_values,
            horizon=horizon,
            zone=zone,
            save_path=save_path
        )

        if save_path:
            result['file_paths'].append(save_path)

        # 性能指標の計算
        valid_future_mask = ~np.isnan(future_actual_values)
        valid_pred_mask = ~np.isnan(sample_predictions)
        common_mask = valid_future_mask & valid_pred_mask

        if np.sum(common_mask) > 0:
            mae = np.mean(np.abs(future_actual_values[common_mask] - sample_predictions[common_mask]))
            rmse = np.sqrt(np.mean((future_actual_values[common_mask] - sample_predictions[common_mask]) ** 2))
            corr = np.corrcoef(future_actual_values[common_mask], sample_predictions[common_mask])[0, 1]

            result['metrics'] = {
                'mae': mae,
                'rmse': rmse,
                'correlation': corr,
                'valid_points': np.sum(common_mask),
                'total_points': len(sample_predictions)
            }

        plt.close(fig)
        result['success'] = True

    except Exception as e:
        result['error_message'] = f"可視化作成エラー: {str(e)}"

    return result


def create_perfect_visualization_for_all_zones(results_dict: Dict,
                                             original_df: pd.DataFrame,
                                             horizon: int,
                                             save_dir: str = None,
                                             sample_size: int = 200) -> Dict[str, Any]:
    """
    全ゾーンの完璧な可視化を作成

    Parameters:
    -----------
    results_dict : dict
        モデル結果辞書
    original_df : pd.DataFrame
        元のデータフレーム
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        保存ディレクトリ
    sample_size : int
        サンプルサイズ

    Returns:
    --------
    dict
        全体の可視化結果
    """

    print(f"\n{'='*80}")
    print(f"【{horizon}分予測の完璧な時間軸修正可視化】")
    print(f"{'='*80}")

    overall_result = {
        'horizon': horizon,
        'total_zones': 0,
        'successful_zones': 0,
        'failed_zones': 0,
        'zone_results': {},
        'summary_metrics': {},
        'file_paths': []
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 各ゾーンの処理
    for zone in sorted(results_dict.keys()):
        print(f"\n--- ゾーン {zone} の完璧な可視化作成 ---")

        zone_result = create_perfect_visualization_for_zone(
            results_dict=results_dict,
            original_df=original_df,
            zone=zone,
            horizon=horizon,
            save_dir=save_dir,
            sample_size=sample_size
        )

        overall_result['zone_results'][zone] = zone_result
        overall_result['total_zones'] += 1

        if zone_result['success']:
            overall_result['successful_zones'] += 1
            overall_result['file_paths'].extend(zone_result['file_paths'])

            # メトリクスの表示
            if zone_result['metrics']:
                metrics = zone_result['metrics']
                print(f"[成功] MAE: {metrics['mae']:.3f}°C, "
                      f"RMSE: {metrics['rmse']:.3f}°C, "
                      f"相関: {metrics['correlation']:.3f}")
        else:
            overall_result['failed_zones'] += 1
            print(f"[失敗] {zone_result['error_message']}")

    # サマリーメトリクスの計算
    successful_metrics = [r['metrics'] for r in overall_result['zone_results'].values()
                         if r['success'] and r['metrics']]

    if successful_metrics:
        overall_result['summary_metrics'] = {
            'average_mae': np.mean([m['mae'] for m in successful_metrics]),
            'average_rmse': np.mean([m['rmse'] for m in successful_metrics]),
            'average_correlation': np.mean([m['correlation'] for m in successful_metrics]),
            'total_valid_points': sum([m['valid_points'] for m in successful_metrics])
        }

    # 結果サマリーの表示
    print(f"\n【{horizon}分予測の完璧な可視化結果サマリー】")
    print(f"  総ゾーン数: {overall_result['total_zones']}")
    print(f"  成功ゾーン: {overall_result['successful_zones']}")
    print(f"  失敗ゾーン: {overall_result['failed_zones']}")

    if overall_result['summary_metrics']:
        metrics = overall_result['summary_metrics']
        print(f"  平均MAE: {metrics['average_mae']:.3f}°C")
        print(f"  平均RMSE: {metrics['average_rmse']:.3f}°C")
        print(f"  平均相関: {metrics['average_correlation']:.3f}")

    if save_dir:
        print(f"  保存ディレクトリ: {save_dir}")
        print(f"  生成ファイル数: {len(overall_result['file_paths'])}")

    return overall_result


def create_comprehensive_perfect_visualization(results_dict: Dict,
                                             original_df: pd.DataFrame,
                                             horizons: List[int],
                                             save_dir: str = None) -> Dict[str, Any]:
    """
    全ホライゾンの包括的な完璧可視化

    Parameters:
    -----------
    results_dict : dict
        モデル結果辞書
    original_df : pd.DataFrame
        元のデータフレーム
    horizons : list
        予測ホライゾンのリスト
    save_dir : str, optional
        保存ディレクトリ

    Returns:
    --------
    dict
        包括的可視化結果
    """

    print(f"\n{'='*100}")
    print(f"【包括的完璧時間軸修正可視化システム】")
    print(f"{'='*100}")

    comprehensive_result = {
        'horizons_processed': [],
        'total_visualizations': 0,
        'successful_visualizations': 0,
        'horizon_results': {},
        'overall_summary': {}
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # 各ホライゾンの処理
    for horizon in horizons:
        horizon_save_dir = os.path.join(save_dir, f'horizon_{horizon}') if save_dir else None

        horizon_result = create_perfect_visualization_for_all_zones(
            results_dict=results_dict,
            original_df=original_df,
            horizon=horizon,
            save_dir=horizon_save_dir
        )

        comprehensive_result['horizon_results'][horizon] = horizon_result
        comprehensive_result['horizons_processed'].append(horizon)
        comprehensive_result['total_visualizations'] += horizon_result['total_zones']
        comprehensive_result['successful_visualizations'] += horizon_result['successful_zones']

    # 全体サマリーの計算
    all_metrics = []
    for horizon_result in comprehensive_result['horizon_results'].values():
        if horizon_result['summary_metrics']:
            all_metrics.append(horizon_result['summary_metrics'])

    if all_metrics:
        comprehensive_result['overall_summary'] = {
            'overall_average_mae': np.mean([m['average_mae'] for m in all_metrics]),
            'overall_average_rmse': np.mean([m['average_rmse'] for m in all_metrics]),
            'overall_average_correlation': np.mean([m['average_correlation'] for m in all_metrics]),
            'total_data_points': sum([m['total_valid_points'] for m in all_metrics])
        }

    # 最終結果の表示
    print(f"\n【包括的完璧可視化完了】")
    print(f"  処理ホライゾン: {comprehensive_result['horizons_processed']}")
    print(f"  総可視化数: {comprehensive_result['total_visualizations']}")
    print(f"  成功可視化数: {comprehensive_result['successful_visualizations']}")
    print(f"  成功率: {comprehensive_result['successful_visualizations']/comprehensive_result['total_visualizations']*100:.1f}%")

    if comprehensive_result['overall_summary']:
        summary = comprehensive_result['overall_summary']
        print(f"  全体平均MAE: {summary['overall_average_mae']:.3f}°C")
        print(f"  全体平均RMSE: {summary['overall_average_rmse']:.3f}°C")
        print(f"  全体平均相関: {summary['overall_average_correlation']:.3f}")

    return comprehensive_result


def create_simple_demo(zone: int = 1, horizon: int = 15, save_dir: str = None) -> Dict[str, Any]:
    """
    簡単な完璧時間軸修正デモ（スタンドアロン版）

    Parameters:
    -----------
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        保存ディレクトリ

    Returns:
    --------
    dict
        デモ結果
    """
    print(f"Perfect Time Axis Demo: Zone {zone}, {horizon}min prediction")

    try:
        # データ読み込み
        df = pd.read_csv('AllDayData.csv')

        # 時間インデックス設定
        if 'time_stamp' in df.columns:
            df['time_stamp'] = pd.to_datetime(df['time_stamp'])
            df = df.set_index('time_stamp')

        temp_col = f'sens_temp_{zone}'
        if temp_col not in df.columns:
            return {'success': False, 'error': f"Temperature column {temp_col} not found"}

        # 最新データを使用
        recent_data = df.iloc[-1000:].copy()
        input_timestamps = recent_data.index[-500:]  # 最新500ポイント
        input_actual_values = recent_data.loc[input_timestamps, temp_col].values

        # 予測対象時刻の実測値を取得
        future_timestamps, future_actual_values = get_future_actual_values(
            df[temp_col], input_timestamps, horizon
        )

        # 有効な未来値のチェック
        valid_future_mask = ~np.isnan(future_actual_values)
        if np.sum(valid_future_mask) < 10:
            return {'success': False, 'error': f"Insufficient future data: {np.sum(valid_future_mask)} points"}

        # 予測値の作成（実測値にノイズを加えてシミュレート）
        np.random.seed(42)
        noise_std = np.nanstd(future_actual_values) * 0.05  # 5%のノイズ
        predicted_values = future_actual_values + np.random.normal(0, noise_std, len(future_actual_values))

        # プロット作成
        save_path = None
        if save_dir:
            save_path = os.path.join(save_dir, f'perfect_demo_zone_{zone}_horizon_{horizon}.png')

        fig = plot_perfect_time_axis_comparison(
            input_timestamps=input_timestamps,
            input_actual_values=input_actual_values,
            predicted_values=predicted_values,
            future_actual_values=future_actual_values,
            horizon=horizon,
            zone=zone,
            save_path=save_path
        )

        # 性能指標の計算
        valid_mask = ~np.isnan(future_actual_values) & ~np.isnan(predicted_values)
        if np.sum(valid_mask) > 0:
            mae = np.mean(np.abs(future_actual_values[valid_mask] - predicted_values[valid_mask]))
            rmse = np.sqrt(np.mean((future_actual_values[valid_mask] - predicted_values[valid_mask]) ** 2))
            corr = np.corrcoef(future_actual_values[valid_mask], predicted_values[valid_mask])[0, 1]

            print(f"Performance Metrics:")
            print(f"  MAE: {mae:.3f}°C")
            print(f"  RMSE: {rmse:.3f}°C")
            print(f"  Correlation: {corr:.3f}")
            print(f"  Valid points: {np.sum(valid_mask)}")

        plt.close(fig)

        return {
            'success': True,
            'mae': mae if 'mae' in locals() else None,
            'rmse': rmse if 'rmse' in locals() else None,
            'correlation': corr if 'corr' in locals() else None,
            'save_path': save_path,
            'valid_points': np.sum(valid_mask) if 'valid_mask' in locals() else 0
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}
