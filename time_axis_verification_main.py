#!/usr/bin/env python
# coding: utf-8

"""
時間軸整合性検証メインスクリプト
予測値と実測値の時間軸対応関係を詳細に検証
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import HORIZONS, OUTPUT_DIR

# 時間軸検証機能のインポート
from src.diagnostics.time_axis_verification import (
    verify_time_axis_alignment,
    run_comprehensive_time_axis_verification
)

# データ前処理のインポート
from src.data.preprocessing import prepare_time_features


def load_original_data():
    """
    元のデータフレームを読み込む
    """
    print("元のデータフレームを読み込み中...")

    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")

        # 時間特徴量の準備
        df = prepare_time_features(df)

        return df

    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None


def load_model_results():
    """
    保存されたモデル結果を読み込む
    """
    print("保存されたモデル結果を読み込み中...")

    # 結果ディクショナリの初期化
    results = {}

    # モデルディレクトリの確認
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"エラー: モデルディレクトリ '{models_dir}' が見つかりません")
        return None

    # 保存されたモデルファイルを検索
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    if not model_files:
        print("エラー: 保存されたモデルファイルが見つかりません")
        return None

    # ゾーンとホライゾンの組み合わせを抽出
    zone_horizon_combinations = set()
    for file in model_files:
        if 'zone_' in file and 'horizon_' in file:
            parts = file.replace('.pkl', '').split('_')
            zone_idx = parts.index('zone') + 1
            horizon_idx = parts.index('horizon') + 1

            if zone_idx < len(parts) and horizon_idx < len(parts):
                try:
                    zone = int(parts[zone_idx])
                    horizon = int(parts[horizon_idx])
                    zone_horizon_combinations.add((zone, horizon))
                except ValueError:
                    continue

    print(f"検出されたゾーン・ホライゾン組み合わせ: {len(zone_horizon_combinations)}個")

    # 各組み合わせについてデータを読み込み
    for zone, horizon in zone_horizon_combinations:
        if zone not in results:
            results[zone] = {}

        # モデルファイルの読み込み
        model_file = f"model_zone_{zone}_horizon_{horizon}.pkl"
        features_file = f"features_zone_{zone}_horizon_{horizon}.pkl"

        model_path = os.path.join(models_dir, model_file)
        features_path = os.path.join(models_dir, features_file)

        if os.path.exists(model_path) and os.path.exists(features_path):
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                with open(features_path, 'rb') as f:
                    features = pickle.load(f)

                # 特徴量重要度の取得
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                else:
                    # 特徴量重要度が取得できない場合はダミーデータ
                    feature_importance = pd.DataFrame({
                        'feature': features,
                        'importance': [1.0] * len(features)
                    })

                results[zone][horizon] = {
                    'model': model,
                    'selected_features': features,
                    'feature_importance': feature_importance
                }

                print(f"ゾーン {zone}, ホライゾン {horizon}分: モデル読み込み完了")

            except Exception as e:
                print(f"ゾーン {zone}, ホライゾン {horizon}分: 読み込みエラー - {e}")

    return results


def create_mock_test_data(original_df, zone, horizon):
    """
    テスト用のモックデータを作成
    実際の実装では、保存されたテストデータを使用
    """
    # 最新1000データポイントを使用
    test_data = original_df.iloc[-1000:].copy()

    # 目的変数の作成（シフト処理）
    temp_col = f'sens_temp_{zone}'
    if temp_col in test_data.columns:
        # 時間間隔の推定
        time_diff = test_data.index.to_series().diff().dropna().value_counts().index[0]
        shift_periods = int(horizon / time_diff.total_seconds() * 60)

        # 目的変数の作成
        target_col = f'sens_temp_{zone}_future_{horizon}'
        test_data[target_col] = test_data[temp_col].shift(-shift_periods)

        # test_yの作成
        test_y = test_data[target_col].dropna()

        # 予測値の作成（実際の実装では保存された予測結果を使用）
        # ここでは元の値に小さなノイズを加えたものを予測値とする
        test_predictions = test_y.values + np.random.normal(0, 0.5, len(test_y))

        return test_data, test_y, test_predictions

    return None, None, None


def run_time_axis_verification(target_zones=None, target_horizons=None):
    """
    時間軸整合性検証の実行

    Parameters:
    -----------
    target_zones : list, optional
        検証対象のゾーン番号のリスト
    target_horizons : list, optional
        検証対象の予測ホライゾン（分）のリスト
    """
    print("=" * 80)
    print("🕐 時間軸整合性検証")
    print("=" * 80)

    # 対象ホライゾンの設定
    if target_horizons is None:
        target_horizons = HORIZONS

    print(f"検証対象ホライゾン: {target_horizons}")

    # 元データの読み込み
    original_df = load_original_data()
    if original_df is None:
        print("エラー: 元データの読み込みに失敗しました")
        return

    # モデル結果の読み込み
    model_results = load_model_results()
    if model_results is None:
        print("エラー: モデル結果の読み込みに失敗しました")
        return

    # 出力ディレクトリの作成
    verification_dir = os.path.join(OUTPUT_DIR, "time_axis_verification")
    os.makedirs(verification_dir, exist_ok=True)

    # 検証用のresults_dictを作成
    verification_results_dict = {}

    # 利用可能なゾーンの確認
    available_zones = list(model_results.keys())
    if target_zones:
        available_zones = [z for z in target_zones if z in available_zones]

    print(f"検証対象ゾーン: {available_zones}")

    # 各ゾーン・ホライゾンについて検証を実行
    for zone in available_zones:
        verification_results_dict[zone] = {}

        for horizon in target_horizons:
            if horizon not in model_results[zone]:
                continue

            print(f"\n{'='*50}")
            print(f"ゾーン {zone} - {horizon}分予測の検証")
            print(f"{'='*50}")

            # テストデータの作成（実際の実装では保存されたデータを使用）
            test_data, test_y, test_predictions = create_mock_test_data(
                original_df, zone, horizon
            )

            if test_data is None or test_y is None or test_predictions is None:
                print(f"警告: ゾーン {zone} のデータが不足しています")
                continue

            # 検証用データの準備
            verification_results_dict[zone][horizon] = {
                'test_data': test_data,
                'test_y': test_y,
                'test_predictions': test_predictions,
                'feature_importance': model_results[zone][horizon]['feature_importance']
            }

            # 個別検証の実行
            try:
                verification_result = verify_time_axis_alignment(
                    df=original_df,
                    zone=zone,
                    horizon=horizon,
                    test_predictions=test_predictions,
                    test_y=test_y,
                    test_data=test_data,
                    save_dir=verification_dir
                )

                # 結果の表示
                print(f"\n📊 検証結果サマリー:")
                print(f"  データ構造: {'✅' if verification_result['data_structure_analysis']['original_data_available'] else '❌'}")
                print(f"  シフト正確性: {'✅' if verification_result['time_axis_mapping'].get('shift_verification', {}).get('is_correct_shift', False) else '❌'}")
                print(f"  データ長整合性: {'✅' if verification_result['alignment_verification']['data_length_match'] else '❌'}")
                print(f"  可視化デモ: {'✅' if verification_result['visualization_correctness']['demonstration_created'] else '❌'}")

                if verification_result['recommendations']:
                    print(f"\n💡 推奨事項:")
                    for rec in verification_result['recommendations'][:3]:  # 上位3つのみ表示
                        print(f"  - {rec}")

            except Exception as e:
                print(f"検証エラー: {e}")
                continue

    # 包括的検証の実行
    if verification_results_dict:
        print(f"\n{'='*80}")
        print("🔍 包括的時間軸整合性検証")
        print(f"{'='*80}")

        try:
            comprehensive_verification = run_comprehensive_time_axis_verification(
                results_dict=verification_results_dict,
                original_df=original_df,
                save_dir=verification_dir
            )

            # 結果の保存
            import json
            verification_report_path = os.path.join(verification_dir, 'time_axis_verification_report.json')
            with open(verification_report_path, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_verification, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n📁 検証レポート保存: {verification_report_path}")

        except Exception as e:
            print(f"包括的検証エラー: {e}")

    print(f"\n{'='*80}")
    print("時間軸整合性検証完了")
    print(f"結果は {verification_dir} に保存されました")
    print(f"{'='*80}")


def print_verification_summary():
    """
    検証項目のサマリーを表示
    """
    print("\n" + "="*80)
    print("🕐 時間軸整合性検証 - 検証項目")
    print("="*80)

    print("\n🔍 検証内容:")
    print("1. データ構造の分析")
    print("   - 元の温度データの利用可能性")
    print("   - 目的変数（シフト済み）の確認")
    print("   - test_yの出所の特定")

    print("\n2. 時間軸マッピングの分析")
    print("   - 入力タイムスタンプと予測対象タイムスタンプの対応")
    print("   - シフト処理の正確性検証")
    print("   - 相関分析による検証")

    print("\n3. 予測値と実測値の整合性検証")
    print("   - データ長の一致確認")
    print("   - タイムスタンプの整合性")
    print("   - 値の範囲の一貫性")

    print("\n4. 可視化の正確性検証")
    print("   - 正しい時間軸表示方法のデモンストレーション")
    print("   - 間違った表示方法との比較")
    print("   - 実際の未来値との比較検証")

    print("\n📊 出力される検証結果:")
    print("- 各ゾーン・ホライゾン別の詳細検証結果")
    print("- 時間軸表示の正誤比較プロット")
    print("- 包括的検証レポート（JSON形式）")
    print("- 具体的な修正推奨事項")

    print("\n❓ 解決される疑問:")
    print("- 予測値をそのままプロットしてよいか？")
    print("- 実測値とは何を指すのか？")
    print("- 予測値と実測値が同じ時間軸でプロットされているか？")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='時間軸整合性検証')
    parser.add_argument('--zones', type=int, nargs='+',
                       help='検証対象のゾーン番号')
    parser.add_argument('--horizons', type=int, nargs='+',
                       help='検証対象の予測ホライゾン（分）')
    parser.add_argument('--summary', action='store_true',
                       help='検証項目のサマリーのみを表示')

    args = parser.parse_args()

    if args.summary:
        print_verification_summary()
    else:
        run_time_axis_verification(
            target_zones=args.zones,
            target_horizons=args.horizons
        )
