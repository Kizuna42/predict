#!/usr/bin/env python
# coding: utf-8

"""
LAG特徴量による後追い問題の包括的診断スクリプト
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import HORIZONS, OUTPUT_DIR

# 新しい包括的診断機能のインポート
from src.diagnostics.comprehensive_lag_analysis import (
    analyze_lag_following_comprehensive,
    generate_lag_analysis_report
)

# 既存の診断機能のインポート
from src.diagnostics.time_validation import (
    validate_prediction_timing,
    analyze_time_axis_consistency
)

# 可視化機能のインポート
from src.utils.advanced_visualization import (
    plot_corrected_time_series_by_horizon
)


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


def load_test_data_and_predictions():
    """
    テストデータと予測結果を読み込む
    注意: この関数は実際のデータ構造に合わせて調整が必要
    """
    print("テストデータと予測結果を読み込み中...")

    # データファイルの読み込み
    try:
        df = pd.read_csv('AllDayData.csv')
        print(f"データ読み込み成功: {df.shape[0]}行 x {df.shape[1]}列")

        # 時間インデックスの設定（実際のデータ構造に合わせて調整）
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif df.index.name != 'timestamp':
            # 最初の列がタイムスタンプの場合
            df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        print(f"データ読み込みエラー: {e}")
        return None


def run_comprehensive_lag_diagnosis(target_horizons=None):
    """
    包括的LAG診断の実行

    Parameters:
    -----------
    target_horizons : list, optional
        診断対象のホライゾン（分）のリスト
    """
    print("=" * 60)
    print("LAG特徴量による後追い問題の包括的診断")
    print("=" * 60)

    # 対象ホライゾンの設定
    if target_horizons is None:
        target_horizons = HORIZONS

    print(f"診断対象ホライゾン: {target_horizons}")

    # モデル結果の読み込み
    model_results = load_model_results()
    if model_results is None:
        print("エラー: モデル結果の読み込みに失敗しました")
        return

    # テストデータの読み込み
    test_data = load_test_data_and_predictions()
    if test_data is None:
        print("エラー: テストデータの読み込みに失敗しました")
        return

    # 出力ディレクトリの作成
    diagnosis_dir = os.path.join(OUTPUT_DIR, "lag_diagnosis")
    os.makedirs(diagnosis_dir, exist_ok=True)

    # 各ホライゾンについて診断を実行
    for horizon in target_horizons:
        print(f"\n{'='*40}")
        print(f"ホライゾン {horizon}分の診断開始")
        print(f"{'='*40}")

        # 該当するゾーンの確認
        zones_for_horizon = [zone for zone in model_results.keys()
                           if horizon in model_results[zone]]

        if not zones_for_horizon:
            print(f"警告: ホライゾン {horizon}分のデータが見つかりません")
            continue

        print(f"診断対象ゾーン: {zones_for_horizon}")

        # 各ゾーンについて詳細診断を実行
        detailed_results = {}

        for zone in zones_for_horizon:
            print(f"\n--- ゾーン {zone} の詳細診断 ---")

            # モデル情報の取得
            zone_model_info = model_results[zone][horizon]
            feature_importance = zone_model_info['feature_importance']

            # テストデータの準備（実際のデータ構造に合わせて調整が必要）
            # ここでは仮のデータを使用
            timestamps = test_data.index[-1000:]  # 最新1000データポイント

            # 実測値の取得（実際の列名に合わせて調整）
            temp_col = f'sens_temp_{zone}'
            if temp_col in test_data.columns:
                actual_values = test_data[temp_col].iloc[-1000:].values
            else:
                print(f"警告: ゾーン {zone} の温度データが見つかりません")
                continue

            # 予測値の生成（実際の予測結果がない場合の仮実装）
            # 実際の実装では保存された予測結果を使用
            predicted_values = actual_values + np.random.normal(0, 0.5, len(actual_values))

            # 包括的分析の実行
            try:
                analysis_result = analyze_lag_following_comprehensive(
                    timestamps=timestamps,
                    actual_values=actual_values,
                    predicted_values=predicted_values,
                    feature_importance=feature_importance,
                    zone=zone,
                    horizon=horizon
                )

                detailed_results[zone] = analysis_result

                # 結果の表示
                print(f"  重要度: {analysis_result['severity']}")
                print(f"  時間軸整合性: {'OK' if analysis_result['timestamp_analysis']['is_correct_alignment'] else 'NG'}")
                print(f"  パターン後追い: {'検出' if analysis_result['pattern_analysis']['valley_following'] or analysis_result['pattern_analysis']['peak_following'] else '未検出'}")
                print(f"  LAG依存度リスク: {analysis_result['lag_dependency']['risk_level']}")

                # 推奨事項の表示
                if analysis_result['recommendations']:
                    print("  推奨事項:")
                    for rec in analysis_result['recommendations'][:3]:  # 上位3つのみ表示
                        print(f"    - {rec['description']}")

            except Exception as e:
                print(f"  エラー: 分析中にエラーが発生しました - {e}")
                continue

        # ホライゾン全体のレポート生成
        if detailed_results:
            print(f"\n--- ホライゾン {horizon}分の総合レポート生成 ---")

            # 仮のresults_dictを作成（実際の実装では実際のデータを使用）
            results_dict_for_report = {}
            for zone in detailed_results.keys():
                results_dict_for_report[zone] = {
                    horizon: {
                        'test_data': test_data.iloc[-1000:],
                        'test_y': test_data[f'sens_temp_{zone}'].iloc[-1000:] if f'sens_temp_{zone}' in test_data.columns else pd.Series([0]*1000),
                        'test_predictions': np.random.normal(25, 2, 1000),  # 仮の予測値
                        'feature_importance': model_results[zone][horizon]['feature_importance']
                    }
                }

            try:
                report = generate_lag_analysis_report(
                    results_dict=results_dict_for_report,
                    horizon=horizon,
                    save_dir=diagnosis_dir
                )

                print(f"  分析対象ゾーン数: {report['zones_analyzed']}")
                print(f"  高リスクゾーン: {len(report['high_risk_zones'])}個")
                print(f"  中リスクゾーン: {len(report['medium_risk_zones'])}個")
                print(f"  低リスクゾーン: {len(report['low_risk_zones'])}個")

                if report['overall_recommendations']:
                    print("  全体的な推奨事項:")
                    for rec in report['overall_recommendations']:
                        print(f"    - {rec}")

            except Exception as e:
                print(f"  レポート生成エラー: {e}")

        # 時間軸修正済み可視化の生成
        print(f"\n--- ホライゾン {horizon}分の可視化生成 ---")
        try:
            if detailed_results:
                # 可視化の実行（実際のデータを使用する場合は調整が必要）
                fig = plot_corrected_time_series_by_horizon(
                    results_dict=results_dict_for_report,
                    horizon=horizon,
                    save_dir=diagnosis_dir,
                    points=200,
                    save=True,
                    validate_timing=True
                )

                if fig:
                    print(f"  時間軸修正済み可視化を保存しました")
                else:
                    print(f"  可視化の生成に失敗しました")

        except Exception as e:
            print(f"  可視化生成エラー: {e}")

    print(f"\n{'='*60}")
    print("LAG診断完了")
    print(f"結果は {diagnosis_dir} に保存されました")
    print(f"{'='*60}")


def print_diagnosis_summary():
    """
    診断結果のサマリーを表示
    """
    print("\n" + "="*60)
    print("LAG特徴量による後追い問題 - 診断項目")
    print("="*60)

    print("\n🔍 診断内容:")
    print("1. 予測プロットの時間軸表示確認")
    print("   - 15分先予測が正しく未来の時刻に表示されているか")
    print("   - 予測値が過去のタイムスタンプに表示されていないか")

    print("\n2. LAG特徴量による単純な過去値スライド検証")
    print("   - モデルが過去の実測値を単純にコピーしていないか")
    print("   - 谷や山のパターンが遅れて出現していないか")

    print("\n3. LAG特徴量への過度な依存度確認")
    print("   - 明示的なLAG特徴量の使用状況")
    print("   - 暗黙的なLAG効果（平滑化、差分など）の依存度")

    print("\n4. 未来情報の参照漏れ確認")
    print("   - データリークの検出")
    print("   - 特徴量エンジニアリングの妥当性検証")

    print("\n📊 出力される診断結果:")
    print("- 各ゾーン・ホライゾン別の詳細分析")
    print("- 時間軸修正済み可視化")
    print("- 包括的診断レポート（JSON形式）")
    print("- 具体的な改善推奨事項")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='LAG特徴量による後追い問題の包括的診断')
    parser.add_argument('--horizons', type=int, nargs='+',
                       help='診断対象の予測ホライゾン（分）')
    parser.add_argument('--summary', action='store_true',
                       help='診断項目のサマリーのみを表示')

    args = parser.parse_args()

    if args.summary:
        print_diagnosis_summary()
    else:
        run_comprehensive_lag_diagnosis(target_horizons=args.horizons)
