"""
特徴量選択モジュールのテスト
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import sys
import pathlib

# プロジェクトのルートディレクトリをパスに追加
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from thermal_prediction.utils.feature_selection import (
    select_features_with_multiple_methods,
    visualize_feature_selection_results,
    evaluate_feature_sets,
    visualize_feature_evaluation
)

class TestFeatureSelection(unittest.TestCase):
    """特徴量選択モジュールのテストクラス"""

    def setUp(self):
        """テストデータの作成"""
        # テスト用の特徴量データフレームを作成
        np.random.seed(42)
        n_samples = 1000
        n_features = 20

        # 特徴量行列の作成
        X_data = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X_data, columns=feature_names)

        # 目的変数を作成（一部の特徴量だけに依存するようにする）
        important_features = [0, 2, 5, 10]
        y_data = 3 * X_data[:, 0] + 2 * X_data[:, 2] + 5 * X_data[:, 5] + 1 * X_data[:, 10] + np.random.randn(n_samples) * 0.5
        self.y = pd.Series(y_data, name='target')

        # 一時ディレクトリを作成（可視化結果の保存用）
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """テスト後のクリーンアップ"""
        # 一時ディレクトリを削除
        shutil.rmtree(self.test_dir)

    def test_select_features_with_multiple_methods(self):
        """複数の特徴量選択手法のテスト"""
        # 特徴量選択を実行
        selected_features_dict, feature_importance_df = select_features_with_multiple_methods(
            self.X, self.y, output_dir=None
        )

        # 返り値の検証
        self.assertIsInstance(selected_features_dict, dict)
        self.assertIsInstance(feature_importance_df, pd.DataFrame)

        # 各手法が実行されていることを確認
        expected_methods = ['LightGBM', 'RandomForest', 'MutualInfo', 'RFE']
        for method in expected_methods:
            self.assertIn(method, selected_features_dict)
            self.assertIn(f'{method}_importance', feature_importance_df.columns)

        # 各手法で特徴量が選択されていることを確認
        for method, features in selected_features_dict.items():
            self.assertGreater(len(features), 0)
            for feature in features:
                self.assertIn(feature, self.X.columns)

    def test_select_features_with_visualization(self):
        """可視化付きの特徴量選択のテスト"""
        # 可視化付きで特徴量選択を実行
        selected_features_dict, feature_importance_df = select_features_with_multiple_methods(
            self.X, self.y, output_dir=self.test_dir, zone=0, prediction_horizon=5
        )

        # 出力ファイルの存在確認
        expected_files = [
            'feature_importance_lightgbm_zone_0_horizon_5.png',
            'feature_importance_randomforest_zone_0_horizon_5.png',
            'feature_importance_mutualinfo_zone_0_horizon_5.png',
            'feature_importance_rfe_zone_0_horizon_5.png',
            'feature_selection_venn_zone_0_horizon_5.png',
            'common_selected_features_zone_0_horizon_5.csv'
        ]

        for file_name in expected_files:
            file_path = os.path.join(self.test_dir, file_name)
            self.assertTrue(os.path.exists(file_path), f"{file_name}が存在しません")

    def test_visualize_feature_selection_results(self):
        """特徴量選択結果の可視化のテスト"""
        # 特徴量選択を実行
        selected_features_dict, feature_importance_df = select_features_with_multiple_methods(
            self.X, self.y, output_dir=None
        )

        # 特徴量重要度辞書の作成
        feature_importance_dict = {}
        for method in selected_features_dict.keys():
            col_name = f'{method}_importance'
            if col_name in feature_importance_df.columns:
                importance_df = pd.DataFrame({
                    'Feature': feature_importance_df['Feature'],
                    'Importance': feature_importance_df[col_name]
                }).sort_values('Importance', ascending=False)
                feature_importance_dict[method] = importance_df

        # 可視化を実行
        paths = visualize_feature_selection_results(
            feature_importance_dict, selected_features_dict,
            output_dir=self.test_dir, zone=1, prediction_horizon=10
        )

        # 返り値の検証
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

        # 出力ファイルの存在確認
        for path in paths:
            self.assertTrue(os.path.exists(path))

    def test_evaluate_feature_sets(self):
        """特徴量セット評価のテスト"""
        # 特徴量選択を実行
        selected_features_dict, _ = select_features_with_multiple_methods(
            self.X, self.y, output_dir=None
        )

        # 特徴量セットを評価
        results_df = evaluate_feature_sets(
            self.X, self.y, selected_features_dict, test_size=0.2, random_state=42
        )

        # 返り値の検証
        self.assertIsInstance(results_df, pd.DataFrame)

        # 結果の列を確認
        expected_columns = ['Method', 'Feature Count', 'RMSE', 'Feature List']
        for column in expected_columns:
            self.assertIn(column, results_df.columns)

        # 各手法の結果が含まれていることを確認
        methods = list(selected_features_dict.keys()) + ['All Features']
        for method in methods:
            self.assertIn(method, results_df['Method'].values)

    def test_visualize_feature_evaluation(self):
        """特徴量評価結果の可視化のテスト"""
        # 特徴量選択を実行
        selected_features_dict, _ = select_features_with_multiple_methods(
            self.X, self.y, output_dir=None
        )

        # 特徴量セットを評価
        results_df = evaluate_feature_sets(
            self.X, self.y, selected_features_dict, test_size=0.2, random_state=42
        )

        # 評価結果を可視化
        path = visualize_feature_evaluation(
            results_df, output_dir=self.test_dir, zone=2, prediction_horizon=15
        )

        # 返り値の検証
        self.assertIsInstance(path, str)

        # 出力ファイルの存在確認
        self.assertTrue(os.path.exists(path))
        self.assertEqual(path, os.path.join(self.test_dir, 'feature_selection_comparison_zone_2_horizon_15.png'))
