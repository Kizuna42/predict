#!/usr/bin/env python
# coding: utf-8

"""
簡単な完璧時間軸修正可視化（フォント設定なし）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional, Tuple, List
import os


def get_future_actual_values_simple(original_data: pd.Series,
                                  input_timestamps: pd.DatetimeIndex,
                                  horizon: int) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    予測対象時刻の実測値を取得（簡単版）
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


def plot_simple_perfect_comparison(input_timestamps: pd.DatetimeIndex,
                                 input_actual_values: np.ndarray,
                                 predicted_values: np.ndarray,
                                 future_actual_values: np.ndarray,
                                 horizon: int,
                                 zone: int = None,
                                 save_path: str = None) -> plt.Figure:
    """
    簡単な完璧時間軸比較プロット
    """

    # 基本設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

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
                'b-', linewidth=2, label='Actual (Input Time)', alpha=0.8)
    axes[0].plot(input_timestamps[valid_pred_mask],
                predicted_values[valid_pred_mask],
                'r--', linewidth=2, label='Prediction (Wrong Time Axis)', alpha=0.8)

    axes[0].set_title(f'Wrong Method: Prediction at Input Time',
                     fontsize=14, color='red', fontweight='bold')
    axes[0].set_ylabel('Temperature (C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 修正された方法（予測値の時間軸のみ修正）
    axes[1].plot(input_timestamps[valid_input_mask],
                input_actual_values[valid_input_mask],
                'b-', linewidth=2, label='Actual (Input Time)', alpha=0.8)
    axes[1].plot(future_timestamps[valid_pred_mask],
                predicted_values[valid_pred_mask],
                'r--', linewidth=2, label=f'Prediction (+{horizon}min)', alpha=0.8)

    axes[1].set_title(f'Partial Fix: Prediction Time Axis Corrected',
                     fontsize=14, color='orange', fontweight='bold')
    axes[1].set_ylabel('Temperature (C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 完璧な方法（予測値と同じ時刻の実測値で比較）
    if np.sum(valid_future_mask) > 0:
        # 予測対象時刻の実測値をプロット
        axes[2].plot(future_timestamps[valid_future_mask],
                    future_actual_values[valid_future_mask],
                    'g-', linewidth=2, label=f'Actual (+{horizon}min)', alpha=0.8)

        # 予測値を同じ時刻でプロット
        axes[2].plot(future_timestamps[valid_pred_mask],
                    predicted_values[valid_pred_mask],
                    'r--', linewidth=2, label=f'Prediction (+{horizon}min)', alpha=0.8)

        # 性能指標の計算
        common_indices = valid_future_mask & valid_pred_mask
        if np.sum(common_indices) > 0:
            mae = np.mean(np.abs(future_actual_values[common_indices] - predicted_values[common_indices]))
            rmse = np.sqrt(np.mean((future_actual_values[common_indices] - predicted_values[common_indices]) ** 2))
            corr = np.corrcoef(future_actual_values[common_indices], predicted_values[common_indices])[0, 1]

            # 性能指標を表示
            axes[2].text(0.02, 0.98,
                        f'MAE: {mae:.3f}C\nRMSE: {rmse:.3f}C\nCorr: {corr:.3f}',
                        transform=axes[2].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)

        axes[2].set_title(f'Perfect Method: Same Time Comparison',
                         fontsize=14, color='green', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, f'No future actual data for +{horizon}min',
                    transform=axes[2].transAxes, ha='center', va='center', fontsize=12)
        axes[2].set_title(f'Data Shortage: No Future Actual Values',
                         fontsize=14, color='red', fontweight='bold')

    axes[2].set_ylabel('Temperature (C)')
    axes[2].set_xlabel('DateTime')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # X軸の書式設定
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.tick_params(axis='x', rotation=45)

    # 全体タイトル
    zone_str = f'Zone {zone} ' if zone is not None else ''
    fig.suptitle(f'{zone_str}Perfect Time Axis Correction ({horizon}min Prediction)',
                fontsize=16, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Perfect time axis plot saved: {save_path}")

    return fig


def create_simple_perfect_demo(zone=1, horizon=15, save_dir=None):
    """
    簡単な完璧時間軸修正デモ
    """
    print(f"Simple Perfect Time Axis Demo: Zone {zone}, {horizon}min prediction")

    try:
        # データ読み込み
        df = pd.read_csv('AllDayData.csv')

        # 時間インデックス設定
        if 'time_stamp' in df.columns:
            df['time_stamp'] = pd.to_datetime(df['time_stamp'])
            df = df.set_index('time_stamp')

        temp_col = f'sens_temp_{zone}'
        if temp_col not in df.columns:
            print(f"Error: {temp_col} not found")
            return None

        # 最新1000ポイントを使用
        recent_data = df.iloc[-1000:].copy()

        # 入力時刻とデータ
        input_timestamps = recent_data.index[-500:]  # 最新500ポイント
        input_actual_values = recent_data.loc[input_timestamps, temp_col].values

        # 予測値の作成（実際の未来値にノイズを加える）
        future_timestamps, future_actual_values = get_future_actual_values_simple(
            df[temp_col], input_timestamps, horizon
        )

        # 有効な未来値のマスク
        valid_future_mask = ~np.isnan(future_actual_values)

        if np.sum(valid_future_mask) < 10:
            print(f"Error: Not enough future data points: {np.sum(valid_future_mask)}")
            return None

        # 予測値の作成（未来の実測値にノイズを加える）
        np.random.seed(42)
        noise_std = np.nanstd(future_actual_values) * 0.05  # 5%のノイズ
        predicted_values = future_actual_values + np.random.normal(0, noise_std, len(future_actual_values))

        # プロット作成
        if save_dir:
            save_path = os.path.join(save_dir, f'simple_perfect_zone_{zone}_horizon_{horizon}.png')
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_path = None

        fig = plot_simple_perfect_comparison(
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
            print(f"  MAE: {mae:.3f}C")
            print(f"  RMSE: {rmse:.3f}C")
            print(f"  Correlation: {corr:.3f}")
            print(f"  Valid points: {np.sum(valid_mask)}")

        plt.close(fig)

        return {
            'success': True,
            'mae': mae if 'mae' in locals() else None,
            'rmse': rmse if 'rmse' in locals() else None,
            'correlation': corr if 'corr' in locals() else None,
            'save_path': save_path
        }

    except Exception as e:
        print(f"Error in simple demo: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    # テスト実行
    result = create_simple_perfect_demo(zone=1, horizon=15, save_dir="Output/simple_demo")

    if result and result['success']:
        print("Simple demo completed successfully!")
        if result.get('save_path'):
            print(f"Plot saved to: {result['save_path']}")
    else:
        print("Simple demo failed!")
        if result:
            print(f"Error: {result.get('error', 'Unknown error')}")
