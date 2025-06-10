#!/usr/bin/env python
# coding: utf-8

"""
予測実行ランナーモジュール
直接予測、差分予測、比較分析の実行機能を統合
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 設定のインポート
from src.config import (
    HORIZONS, SMOOTHING_WINDOWS, L_ZONES, M_ZONES, R_ZONES,
    MODELS_DIR, OUTPUT_DIR, TEST_SIZE, FEATURE_SELECTION_THRESHOLD
)

# データ前処理関数のインポート
from src.data.preprocessing import (
    filter_temperature_outliers,
    apply_smoothing_to_sensors,
    create_future_targets,
    create_temperature_difference_targets,
    prepare_time_features,
    get_time_based_train_test_split,
    filter_high_value_targets
)

# 特徴量エンジニアリング関数のインポート
from src.data.feature_engineering import (
    create_optimized_features_pipeline,
    create_difference_prediction_pipeline
)

# モデル訓練関数のインポート
from src.models.training import (
    train_physics_guided_model,
    save_model_and_features,
    train_temperature_difference_model,
    save_difference_model_and_features
)

# 評価関数のインポート
from src.models.evaluation import (
    calculate_metrics,
    print_metrics,
    evaluate_temperature_difference_model,
    print_difference_metrics,
    restore_temperature_from_difference,
    compare_difference_vs_direct_prediction,
    print_prediction_comparison,
    test_physical_validity,
    test_difference_prediction_behavior
)

# 可視化関数のインポート
from src.utils.basic_plots import (
    plot_feature_importance,
    plot_time_series_comparison,
    plot_scatter_analysis,
    plot_comparison_analysis,
    create_comprehensive_visualization_report,
    create_comprehensive_minute_analysis_report
)

# データ検証ユーティリティ
from src.utils.data_validation import safe_data_preparation, print_data_preparation_summary


class PredictionRunner:
    """統合予測実行クラス"""
    
    def __init__(self, save_models: bool = True, create_visualizations: bool = True):
        """
        初期化
        
        Parameters:
        -----------
        save_models : bool
            モデルを保存するか
        create_visualizations : bool
            可視化を作成するか
        """
        self.save_models = save_models
        self.create_visualizations = create_visualizations
        self.models_dir, self.viz_dir = self._setup_output_directories()
        
    def _setup_output_directories(self) -> Tuple[Path, Path]:
        """出力ディレクトリの構造を設定"""
        base_output = Path("Output")
        models_dir = base_output / "models"
        viz_dir = base_output / "visualizations"

        models_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)

        return models_dir, viz_dir
    
    def run_direct_prediction(self, df: pd.DataFrame, zones: List[int], horizons: List[int]) -> Dict:
        """
        直接温度予測を実行

        Parameters:
        -----------
        df : DataFrame
            前処理済みデータ
        zones : list
            対象ゾーン
        horizons : list
            予測ホライゾン

        Returns:
        --------
        dict
            実行結果
        """
        print("\n🎯 直接温度予測システム実行中...")
        results = {'models': [], 'metrics': {}}

        # 時間差の計算
        time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

        # 直接予測用目的変数の作成
        df_with_targets = create_future_targets(df, zones, horizons,
                                              pd.Timedelta(seconds=time_diff_seconds))

        # 特徴量エンジニアリング
        df_processed, selected_features, feature_info = create_optimized_features_pipeline(
            df=df_with_targets,
            zone_nums=zones,
            horizons_minutes=horizons,
            time_diff_seconds=time_diff_seconds,
            smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
            feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
        )

        for zone in zones:
            print(f"\n🎯 ゾーン {zone} 処理中...")

            for horizon in horizons:
                print(f"\n--- {horizon}分後予測 ---")

                target_col = f'sens_temp_{zone}_future_{horizon}'
                if target_col not in df_processed.columns:
                    print(f"警告: 目的変数 {target_col} が見つかりません")
                    continue

                # 安全なデータ準備
                feature_cols = [col for col in selected_features if col in df_processed.columns]
                clean_data, prep_info = safe_data_preparation(df_processed, feature_cols, target_col)
                print_data_preparation_summary(prep_info)

                if len(clean_data) < 100:
                    print(f"警告: 有効データが不足 ({len(clean_data)}行)")
                    continue

                # 高値目的変数フィルタリング（75パーセンタイル以上）
                filtered_data, filter_info = filter_high_value_targets(
                    clean_data, [target_col], percentile=75
                )

                if len(filtered_data) < 50:
                    print(f"警告: フィルタ後のデータが不足 ({len(filtered_data)}行) - フィルタリングをスキップ")
                    filtered_data = clean_data

                # 時系列分割
                train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

                X_train = train_df[feature_cols]
                y_train = train_df[target_col]
                X_test = test_df[feature_cols]
                y_test = test_df[target_col]

                print(f"訓練データ: {X_train.shape[0]}行, テストデータ: {X_test.shape[0]}行")

                # モデル訓練
                model = train_physics_guided_model(X_train, y_train)

                # 予測
                y_pred = model.predict(X_test)

                # 評価
                metrics = calculate_metrics(y_test, y_pred)
                print_metrics(metrics, zone, horizon)

                results['metrics'][f'direct_zone_{zone}_horizon_{horizon}'] = metrics

                # モデル保存
                if self.save_models:
                    model_info = save_model_and_features(
                        model, feature_cols, zone, horizon
                    )
                    if model_info[0]:  # model_pathが存在する場合
                        results['models'].append({
                            'type': 'direct',
                            'zone': zone,
                            'horizon': horizon,
                            'model_path': model_info[0],
                            'features_path': model_info[1]
                        })

                # 可視化作成
                if self.create_visualizations:
                    self._create_direct_prediction_visualizations(
                        model, X_test, y_test, y_pred, zone, horizon, feature_cols
                    )

        return results

    def run_difference_prediction(self, df: pd.DataFrame, zones: List[int], horizons: List[int]) -> Dict:
        """
        差分予測を実行

        Parameters:
        -----------
        df : DataFrame
            前処理済みデータ
        zones : list
            対象ゾーン
        horizons : list
            予測ホライゾン

        Returns:
        --------
        dict
            実行結果
        """
        print("\n🔥 差分予測システム実行中...")
        results = {'models': [], 'metrics': {}}

        # 時間差の計算
        time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

        # 差分予測用目的変数の作成
        df_with_diff_targets = create_temperature_difference_targets(df, zones, horizons,
                                                                   pd.Timedelta(seconds=time_diff_seconds))

        # 差分予測専用特徴量エンジニアリング
        df_processed, selected_features, feature_info = create_difference_prediction_pipeline(
            df=df_with_diff_targets,
            zone_nums=zones,
            horizons_minutes=horizons,
            time_diff_seconds=time_diff_seconds,
            smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
            feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
        )

        for zone in zones:
            print(f"\n🎯 ゾーン {zone} 処理中...")

            for horizon in horizons:
                print(f"\n--- {horizon}分後差分予測 ---")

                diff_target_col = f'temp_diff_{zone}_future_{horizon}'
                if diff_target_col not in df_processed.columns:
                    print(f"警告: 差分目的変数 {diff_target_col} が見つかりません")
                    continue

                # 安全なデータ準備
                feature_cols = [col for col in selected_features if col in df_processed.columns]
                clean_data, prep_info = safe_data_preparation(df_processed, feature_cols, diff_target_col)
                print_data_preparation_summary(prep_info)

                if len(clean_data) < 100:
                    print(f"警告: 有効データが不足 ({len(clean_data)}行)")
                    continue

                # 高値目的変数フィルタリング（差分の絶対値で5パーセンタイル以上）
                abs_diff_col = f'abs_temp_diff_{zone}_future_{horizon}'
                clean_data[abs_diff_col] = clean_data[diff_target_col].abs()

                filtered_data, filter_info = filter_high_value_targets(
                    clean_data, [abs_diff_col], percentile=5
                )

                # フィルタ後データが少なすぎる場合は段階的にフィルタを緩める
                if len(filtered_data) < 100:
                    print(f"⚠️  5%ileフィルタ後のデータが不足 ({len(filtered_data)}行) - 20%ileで再試行")
                    filtered_data, filter_info = filter_high_value_targets(
                        clean_data, [abs_diff_col], percentile=20
                    )

                    if len(filtered_data) < 30:
                        print(f"⚠️  20%ileフィルタ後のデータが不足 ({len(filtered_data)}行) - フィルタリングをスキップ")
                        filtered_data = clean_data

                # 時系列分割
                train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

                X_train = train_df[feature_cols]
                y_train = train_df[diff_target_col]
                X_test = test_df[feature_cols]
                y_test = test_df[diff_target_col]

                print(f"訓練データ: {X_train.shape[0]}行, テストデータ: {X_test.shape[0]}行")

                # モデル訓練（PhysicsConstrainedLGBMを使用）
                model = train_temperature_difference_model(X_train, y_train)

                # 予測と評価
                y_pred_diff = model.predict(X_test)
                current_temps_test = test_df[f'sens_temp_{zone}']
                metrics = evaluate_temperature_difference_model(
                    y_test, y_pred_diff, current_temps_test
                )
                print_difference_metrics(metrics, zone, horizon)

                results['metrics'][f'difference_zone_{zone}_horizon_{horizon}'] = metrics

                # 物理的妥当性テスト
                print(f"\n🔬 物理的妥当性テスト (Zone {zone}, {horizon}分後)")
                validity_score = test_physical_validity(
                    model, feature_cols, test_df, zone, horizon,
                    is_difference_model=True, current_temp_col=f'sens_temp_{zone}'
                )
                if validity_score and 'validity_score' in validity_score:
                    print(f"物理的妥当性スコア: {validity_score['validity_score']:.1%}")
                else:
                    print("物理的妥当性テストが実行できませんでした")

                # モデル保存
                if self.save_models:
                    model_info = save_difference_model_and_features(
                        model, feature_cols, zone, horizon
                    )
                    if model_info[0]:  # model_pathが存在する場合
                        results['models'].append({
                            'type': 'difference',
                            'zone': zone,
                            'horizon': horizon,
                            'model_path': model_info[0],
                            'features_path': model_info[1]
                        })

                # 可視化作成
                if self.create_visualizations:
                    self._create_difference_prediction_visualizations(
                        model, X_test, y_test, current_temps_test, zone, horizon, feature_cols
                    )

        return results

    def _create_direct_prediction_visualizations(self, model, X_test, y_test, y_pred, zone, horizon, feature_cols):
        """直接予測の可視化を作成"""
        try:
            # 特徴量重要度プロット
            plot_feature_importance(
                model, feature_cols, zone, horizon,
                self.viz_dir, prefix="direct"
            )

            # 時系列比較プロット
            plot_time_series_comparison(
                y_test, y_pred, zone, horizon,
                self.viz_dir, prefix="direct"
            )

            # 散布図分析
            plot_scatter_analysis(
                y_test, y_pred, zone, horizon,
                self.viz_dir, prefix="direct"
            )

        except Exception as e:
            print(f"可視化作成エラー: {e}")

    def _create_difference_prediction_visualizations(self, model, X_test, y_test, current_temps, zone, horizon, feature_cols):
        """差分予測の可視化を作成"""
        try:
            # 特徴量重要度プロット
            plot_feature_importance(
                model, feature_cols, zone, horizon,
                self.viz_dir, prefix="difference"
            )

            # 差分予測特有の可視化
            y_pred = model.predict(X_test)
            plot_time_series_comparison(
                y_test, y_pred, zone, horizon,
                self.viz_dir, prefix="difference"
            )

            # 復元温度での散布図分析
            restored_temps = restore_temperature_from_difference(y_pred, current_temps)
            actual_future_temps = current_temps + y_test

            plot_scatter_analysis(
                actual_future_temps, restored_temps, zone, horizon,
                self.viz_dir, prefix="difference_restored"
            )

        except Exception as e:
            print(f"差分予測可視化作成エラー: {e}") 