#!/usr/bin/env python
# coding: utf-8

"""
時間軸整合性の簡単なテストスクリプト
モデルなしで時間軸の問題を検証
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.data.preprocessing import prepare_time_features, create_future_targets
from src.utils.font_config import setup_japanese_font
import os


def analyze_time_axis_structure():
    """
    時間軸構造の詳細分析
    """
    print("=" * 80)
    print("🕐 時間軸構造の詳細分析")
    print("=" * 80)

    # データの読み込み
    print("\n📊 データ読み込み...")
    df = pd.read_csv('AllDayData.csv')
    print(f"データ形状: {df.shape}")

    # 時間特徴量の準備
    df = prepare_time_features(df)

    # 時間間隔の確認
    time_diff = df.index.to_series().diff().dropna().value_counts().index[0]
    print(f"⏱️ データの時間間隔: {time_diff}")

    # テスト用のゾーンとホライゾン
    test_zone = 1
    test_horizon = 15

    print(f"\n🎯 テスト対象: ゾーン {test_zone}, ホライゾン {test_horizon}分")

    # 元の温度データの確認
    temp_col = f'sens_temp_{test_zone}'
    if temp_col not in df.columns:
        print(f"❌ 温度データ列が見つかりません: {temp_col}")
        return

    print(f"✅ 元の温度データ列: {temp_col}")

    # 目的変数の作成
    df_with_targets = create_future_targets(df, [test_zone], [test_horizon], time_diff)
    target_col = f'sens_temp_{test_zone}_future_{test_horizon}'

    print(f"✅ 目的変数列: {target_col}")

    # 最新1000ポイントでテスト
    test_data = df_with_targets.iloc[-1000:].copy()

    # データの詳細分析
    print(f"\n🔍 データ詳細分析:")
    print(f"  テストデータ期間: {test_data.index.min()} ～ {test_data.index.max()}")
    print(f"  データ長: {len(test_data)}")

    # 元の温度と目的変数の関係を確認
    original_temp = test_data[temp_col].dropna()
    target_temp = test_data[target_col].dropna()

    print(f"  元の温度データ: {len(original_temp)}ポイント")
    print(f"  目的変数データ: {len(target_temp)}ポイント")

    # 共通インデックスでの比較
    common_indices = original_temp.index.intersection(target_temp.index)
    print(f"  共通インデックス: {len(common_indices)}ポイント")

    if len(common_indices) > 50:
        # 時間軸マッピングの検証
        print(f"\n🕐 時間軸マッピングの検証:")

        # サンプルとして最初の5つを表示
        sample_indices = common_indices[:5]

        for i, timestamp in enumerate(sample_indices):
            input_time = timestamp
            expected_prediction_time = timestamp + pd.Timedelta(minutes=test_horizon)

            original_value = original_temp.loc[timestamp]
            target_value = target_temp.loc[timestamp]

            print(f"  {i+1}. 入力時刻: {input_time}")
            print(f"     → 予測対象時刻: {expected_prediction_time}")
            print(f"     → 元の値: {original_value:.2f}°C")
            print(f"     → 目的変数値: {target_value:.2f}°C")

            # 実際の未来値を確認
            if expected_prediction_time in original_temp.index:
                actual_future_value = original_temp.loc[expected_prediction_time]
                print(f"     → 実際の{test_horizon}分後の値: {actual_future_value:.2f}°C")
                print(f"     → 目的変数との差: {abs(target_value - actual_future_value):.3f}°C")
            else:
                print(f"     → 実際の{test_horizon}分後の値: データなし")
            print()

        # シフトの正確性を検証
        print(f"🔄 シフト正確性の検証:")
        verify_shift_accuracy(original_temp, target_temp, test_horizon)

        # 可視化デモンストレーション
        print(f"\n📈 可視化デモンストレーション作成中...")
        create_visualization_demo(original_temp, target_temp, test_zone, test_horizon)


def verify_shift_accuracy(original_temp, target_temp, horizon):
    """
    シフトの正確性を相関分析で検証
    """
    correlations = {}

    # 0分から horizon+20分まで5分刻みで検証
    for shift_min in range(0, horizon + 25, 5):
        try:
            # 元データをshift_min分後にシフト（5分間隔と仮定）
            shift_periods = shift_min // 5
            shifted_original = original_temp.shift(-shift_periods)

            # 共通インデックスで相関を計算
            common_idx = shifted_original.index.intersection(target_temp.index)
            if len(common_idx) > 10:
                corr = np.corrcoef(
                    shifted_original.loc[common_idx].values,
                    target_temp.loc[common_idx].values
                )[0, 1]

                if not np.isnan(corr):
                    correlations[shift_min] = corr

        except Exception as e:
            continue

    if correlations:
        # 最高相関のシフト量を特定
        best_shift = max(correlations.keys(), key=lambda k: abs(correlations[k]))
        best_corr = correlations[best_shift]

        print(f"  期待シフト: {horizon}分")
        print(f"  検出シフト: {best_shift}分")
        print(f"  最高相関: {best_corr:.3f}")
        print(f"  シフト正確性: {'✅ 正確' if abs(best_shift - horizon) <= 5 else '❌ 不正確'}")

        # 相関の詳細表示
        print(f"  相関詳細:")
        for shift, corr in sorted(correlations.items()):
            marker = "★" if shift == best_shift else " "
            print(f"    {marker} {shift:2d}分シフト: 相関 {corr:.3f}")


def create_visualization_demo(original_temp, target_temp, zone, horizon):
    """
    時間軸表示の正誤比較デモンストレーション
    """
    # フォント設定
    setup_japanese_font()

    # サンプルデータの選択（最新200ポイント）
    sample_size = min(200, len(original_temp))
    sample_indices = original_temp.index[-sample_size:]

    # データの抽出
    input_timestamps = sample_indices
    original_values = original_temp.loc[sample_indices].values

    # 目的変数の値（シフト済み）
    target_values = []
    for ts in input_timestamps:
        if ts in target_temp.index:
            target_values.append(target_temp.loc[ts])
        else:
            target_values.append(np.nan)

    target_values = np.array(target_values)

    # 予測値のシミュレーション（目的変数にノイズを加える）
    predicted_values = target_values + np.random.normal(0, 0.3, len(target_values))

    # 正しい予測タイムスタンプの計算
    correct_prediction_timestamps = input_timestamps + pd.Timedelta(minutes=horizon)

    # プロット作成
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. 間違った表示方法
    axes[0].plot(input_timestamps, original_values, 'b-', linewidth=2, label='実測値', alpha=0.8)
    axes[0].plot(input_timestamps, predicted_values, 'r--', linewidth=2, label='予測値（間違った時間軸）', alpha=0.8)
    axes[0].set_title(f'❌ 間違った表示方法: 予測値が入力と同じ時刻に表示',
                     fontsize=14, color='red', fontweight='bold')
    axes[0].set_ylabel('温度 (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. 正しい表示方法
    axes[1].plot(input_timestamps, original_values, 'b-', linewidth=2, label='実測値（入力時刻）', alpha=0.8)
    axes[1].plot(correct_prediction_timestamps, predicted_values, 'r--', linewidth=2,
                label=f'予測値（正しい時間軸: +{horizon}分）', alpha=0.8)
    axes[1].set_title(f'✅ 正しい表示方法: 予測値が未来の時刻（+{horizon}分）に表示',
                     fontsize=14, color='green', fontweight='bold')
    axes[1].set_ylabel('温度 (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. 比較用：実測値の未来値との比較
    future_actual_timestamps = input_timestamps + pd.Timedelta(minutes=horizon)

    # 未来の実測値を取得
    future_actual_values = []
    for ts in future_actual_timestamps:
        if ts in original_temp.index:
            future_actual_values.append(original_temp.loc[ts])
        else:
            future_actual_values.append(np.nan)

    future_actual_values = np.array(future_actual_values)
    valid_future = ~np.isnan(future_actual_values)

    if np.sum(valid_future) > 0:
        axes[2].plot(future_actual_timestamps[valid_future], future_actual_values[valid_future],
                    'g-', linewidth=2, label=f'実測値（+{horizon}分後）', alpha=0.8)
        axes[2].plot(correct_prediction_timestamps[valid_future], predicted_values[valid_future],
                    'r--', linewidth=2, label=f'予測値（+{horizon}分後）', alpha=0.8)
        axes[2].set_title(f'📊 比較検証: 予測値 vs 実際の{horizon}分後の実測値',
                         fontsize=14, color='blue', fontweight='bold')
    else:
        axes[2].text(0.5, 0.5, f'{horizon}分後の実測値データが不足',
                    transform=axes[2].transAxes, ha='center', va='center', fontsize=12)
        axes[2].set_title(f'📊 比較検証: データ不足のため表示不可',
                         fontsize=14, color='orange')

    axes[2].set_xlabel('日時')
    axes[2].set_ylabel('温度 (°C)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # X軸の書式設定
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 保存
    os.makedirs('output', exist_ok=True)
    save_path = f'output/time_axis_demo_zone_{zone}_horizon_{horizon}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📁 時間軸デモンストレーション保存: {save_path}")

    plt.show()


def analyze_current_visualization_approach():
    """
    現在の可視化アプローチの分析
    """
    print(f"\n📈 現在の可視化アプローチの分析:")

    print(f"\n🔍 現在のコードベースでの時間軸処理:")
    print(f"  1. 目的変数作成: df[target_col] = df[source_col].shift(-shift_periods)")
    print(f"     → これにより、入力時刻のインデックスに未来の値が格納される")

    print(f"\n  2. test_yの作成: test_y = test_df[target_col]")
    print(f"     → test_yは入力時刻をインデックスに持つが、値は未来の温度")

    print(f"\n  3. 予測値の生成: test_predictions = model.predict(test_X)")
    print(f"     → 予測値は配列形式で、test_yと同じ長さ")

    print(f"\n❓ 問題の所在:")
    print(f"  - test_yのインデックス: 入力時刻（例：13:00）")
    print(f"  - test_yの値: 未来の温度（例：13:15の温度）")
    print(f"  - 予測値: 同じく未来の温度（例：13:15の温度）")

    print(f"\n  ✅ 正しい可視化方法:")
    print(f"     - 実測値: 入力時刻（13:00）にプロット")
    print(f"     - 予測値: 予測対象時刻（13:15）にプロット")

    print(f"\n  ❌ 間違った可視化方法:")
    print(f"     - 実測値: 入力時刻（13:00）にプロット")
    print(f"     - 予測値: 入力時刻（13:00）にプロット ← これが後追いに見える原因")


if __name__ == "__main__":
    analyze_time_axis_structure()
    analyze_current_visualization_approach()
