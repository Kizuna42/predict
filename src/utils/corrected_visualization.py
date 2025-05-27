#!/usr/bin/env python
# coding: utf-8

"""
時間軸修正済み可視化モジュール
予測値を正しい時間軸でプロットする機能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any, Optional, Tuple
import os
from .font_config import setup_japanese_font


def plot_corrected_time_series(timestamps: pd.DatetimeIndex,
                              actual_values: np.ndarray,
                              predicted_values: np.ndarray,
                              horizon: int,
                              zone: int = None,
                              title: str = None,
                              save_path: str = None,
                              show_comparison: bool = True) -> plt.Figure:
    """
    時間軸を修正した時系列プロット

    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        入力時刻のタイムスタンプ
    actual_values : np.ndarray
        実測値（入力時刻の値）
    predicted_values : np.ndarray
        予測値（未来の値）
    horizon : int
        予測ホライゾン（分）
    zone : int, optional
        ゾーン番号
    title : str, optional
        プロットタイトル
    save_path : str, optional
        保存パス
    show_comparison : bool
        比較プロットを表示するかどうか

    Returns:
    --------
    plt.Figure
        作成されたフィギュア
    """

    # フォント設定
    setup_japanese_font()

    # 正しい予測タイムスタンプの計算
    prediction_timestamps = timestamps + pd.Timedelta(minutes=horizon)

    if show_comparison:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 1. 間違った表示方法
        axes[0].plot(timestamps, actual_values, 'b-', linewidth=2,
                    label='実測値', alpha=0.8)
        axes[0].plot(timestamps, predicted_values, 'r--', linewidth=2,
                    label='予測値（間違った時間軸）', alpha=0.8)
        axes[0].set_title('間違った表示方法: 予測値が入力と同じ時刻に表示',
                         fontsize=14, color='red', fontweight='bold')
        axes[0].set_ylabel('温度 (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 正しい表示方法
        axes[1].plot(timestamps, actual_values, 'b-', linewidth=2,
                    label='実測値（入力時刻）', alpha=0.8)
        axes[1].plot(prediction_timestamps, predicted_values, 'r--', linewidth=2,
                    label=f'予測値（正しい時間軸: +{horizon}分）', alpha=0.8)
        axes[1].set_title('正しい表示方法: 予測値が未来の時刻に表示',
                         fontsize=14, color='green', fontweight='bold')
        axes[1].set_ylabel('温度 (°C)')
        axes[1].set_xlabel('日時')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # X軸の書式設定
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
            ax.tick_params(axis='x', rotation=45)

    else:
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(timestamps, actual_values, 'b-', linewidth=2,
               label='実測値（入力時刻）', alpha=0.8)
        ax.plot(prediction_timestamps, predicted_values, 'r--', linewidth=2,
               label=f'予測値（+{horizon}分後）', alpha=0.8)

        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            zone_str = f'ゾーン {zone} ' if zone is not None else ''
            ax.set_title(f'{zone_str}時間軸修正済み予測プロット（{horizon}分予測）',
                        fontsize=14, fontweight='bold')

        ax.set_ylabel('温度 (°C)')
        ax.set_xlabel('日時')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # X軸の書式設定
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"時間軸修正済みプロット保存: {save_path}")

    return fig


def plot_prediction_vs_future_actual(timestamps: pd.DatetimeIndex,
                                    predicted_values: np.ndarray,
                                    original_data: pd.Series,
                                    horizon: int,
                                    zone: int = None,
                                    save_path: str = None) -> plt.Figure:
    """
    予測値と実際の未来値の比較プロット

    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        入力時刻のタイムスタンプ
    predicted_values : np.ndarray
        予測値
    original_data : pd.Series
        元の時系列データ（未来値取得用）
    horizon : int
        予測ホライゾン（分）
    zone : int, optional
        ゾーン番号
    save_path : str, optional
        保存パス

    Returns:
    --------
    plt.Figure
        作成されたフィギュア
    """

    # フォント設定
    setup_japanese_font()

    # 予測対象時刻の計算
    prediction_timestamps = timestamps + pd.Timedelta(minutes=horizon)

    # 実際の未来値を取得
    future_actual_values = []
    for ts in prediction_timestamps:
        if ts in original_data.index:
            future_actual_values.append(original_data.loc[ts])
        else:
            future_actual_values.append(np.nan)

    future_actual_values = np.array(future_actual_values)
    valid_indices = ~np.isnan(future_actual_values)

    fig, ax = plt.subplots(figsize=(12, 6))

    if np.sum(valid_indices) > 0:
        # 有効なデータのみプロット
        valid_timestamps = prediction_timestamps[valid_indices]
        valid_actual = future_actual_values[valid_indices]
        valid_predicted = predicted_values[valid_indices]

        ax.plot(valid_timestamps, valid_actual, 'g-', linewidth=2,
               label=f'実測値（+{horizon}分後）', alpha=0.8)
        ax.plot(valid_timestamps, valid_predicted, 'r--', linewidth=2,
               label=f'予測値（+{horizon}分後）', alpha=0.8)

        # 誤差の計算と表示
        mae = np.mean(np.abs(valid_actual - valid_predicted))
        rmse = np.sqrt(np.mean((valid_actual - valid_predicted) ** 2))

        ax.text(0.02, 0.98, f'MAE: {mae:.3f}°C\nRMSE: {rmse:.3f}°C',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    else:
        ax.text(0.5, 0.5, f'{horizon}分後の実測値データが不足',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)

    zone_str = f'ゾーン {zone} ' if zone is not None else ''
    ax.set_title(f'{zone_str}予測値 vs 実際の{horizon}分後の実測値',
                fontsize=14, fontweight='bold')
    ax.set_ylabel('温度 (°C)')
    ax.set_xlabel('日時')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # X軸の書式設定
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"予測vs実測比較プロット保存: {save_path}")

    return fig


def create_comprehensive_time_axis_report(results_dict: Dict,
                                        original_df: pd.DataFrame,
                                        save_dir: str) -> Dict[str, Any]:
    """
    包括的な時間軸修正レポートの作成

    Parameters:
    -----------
    results_dict : dict
        モデル結果辞書
    original_df : pd.DataFrame
        元のデータフレーム
    save_dir : str
        保存ディレクトリ

    Returns:
    --------
    dict
        レポート結果
    """

    print("\n" + "="*80)
    print("📈 包括的時間軸修正レポート作成")
    print("="*80)

    report = {
        'total_plots_created': 0,
        'zones_processed': [],
        'horizons_processed': [],
        'file_paths': []
    }

    os.makedirs(save_dir, exist_ok=True)

    for zone, zone_results in results_dict.items():
        for horizon, horizon_results in zone_results.items():
            # 必要なデータの確認
            required_keys = ['test_data', 'test_y', 'test_predictions']
            if not all(k in horizon_results for k in required_keys):
                continue

            print(f"\n--- ゾーン {zone} - {horizon}分予測の修正プロット作成 ---")

            # データの取得
            test_data = horizon_results['test_data']
            test_y = horizon_results['test_y']
            test_predictions = horizon_results['test_predictions']

            # 有効なデータのインデックス
            valid_indices = test_y.dropna().index
            if len(valid_indices) < 50:
                print(f"データ不足のためスキップ: {len(valid_indices)}ポイント")
                continue

            # サンプルデータの選択
            sample_size = min(200, len(valid_indices))
            sample_indices = valid_indices[-sample_size:]

            # 入力時刻の実測値を取得
            temp_col = f'sens_temp_{zone}'
            if temp_col in test_data.columns:
                input_actual_values = test_data.loc[sample_indices, temp_col].values
            else:
                print(f"温度データ列が見つかりません: {temp_col}")
                continue

            # 予測値の取得
            sample_predictions = test_predictions[-sample_size:] if len(test_predictions) >= sample_size else test_predictions

            # 1. 時間軸修正比較プロット
            comparison_path = os.path.join(save_dir, f'corrected_comparison_zone_{zone}_horizon_{horizon}.png')
            fig1 = plot_corrected_time_series(
                timestamps=sample_indices,
                actual_values=input_actual_values,
                predicted_values=sample_predictions,
                horizon=horizon,
                zone=zone,
                save_path=comparison_path,
                show_comparison=True
            )
            plt.close(fig1)

            # 2. 予測vs未来実測値プロット
            if temp_col in original_df.columns:
                future_comparison_path = os.path.join(save_dir, f'prediction_vs_future_zone_{zone}_horizon_{horizon}.png')
                fig2 = plot_prediction_vs_future_actual(
                    timestamps=sample_indices,
                    predicted_values=sample_predictions,
                    original_data=original_df[temp_col],
                    horizon=horizon,
                    zone=zone,
                    save_path=future_comparison_path
                )
                plt.close(fig2)

                report['file_paths'].extend([comparison_path, future_comparison_path])
            else:
                report['file_paths'].append(comparison_path)

            report['total_plots_created'] += 2

            if zone not in report['zones_processed']:
                report['zones_processed'].append(zone)
            if horizon not in report['horizons_processed']:
                report['horizons_processed'].append(horizon)

            print(f"✅ プロット作成完了")

    # サマリーの表示
    print(f"\n📊 レポート作成完了:")
    print(f"  作成プロット数: {report['total_plots_created']}")
    print(f"  処理ゾーン: {sorted(report['zones_processed'])}")
    print(f"  処理ホライゾン: {sorted(report['horizons_processed'])}")
    print(f"  保存ディレクトリ: {save_dir}")

    return report


def validate_time_axis_alignment(timestamps: pd.DatetimeIndex,
                               predictions: np.ndarray,
                               horizon: int) -> Dict[str, Any]:
    """
    時間軸整合性の自動検証

    Parameters:
    -----------
    timestamps : pd.DatetimeIndex
        入力タイムスタンプ
    predictions : np.ndarray
        予測値
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        検証結果
    """

    validation_result = {
        'is_correct_length': len(timestamps) == len(predictions),
        'expected_prediction_timestamps': timestamps + pd.Timedelta(minutes=horizon),
        'input_timestamps': timestamps,
        'horizon_minutes': horizon,
        'recommendations': []
    }

    if not validation_result['is_correct_length']:
        validation_result['recommendations'].append(
            f"データ長が不一致: timestamps={len(timestamps)}, predictions={len(predictions)}"
        )

    # 基本的な推奨事項
    validation_result['recommendations'].extend([
        f"予測値は入力時刻 + {horizon}分でプロットしてください",
        "実測値との比較は同じ時刻の値同士で行ってください",
        "プロットのラベルに時間軸の説明を含めてください"
    ])

    return validation_result
