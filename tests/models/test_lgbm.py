"""
LightGBMモデルのテスト
"""

import unittest
import pandas as pd
import numpy as np
import sys
import pathlib

# プロジェクトのルートディレクトリをパスに追加
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from thermal_prediction.models.lgbm import train_lgbm_model

class TestLGBM(unittest.TestCase):
    """LightGBMモデルのテストクラス"""

    def setUp(self):
        """テストデータの作成"""
        # テスト用のデータフレームを作成
        np.random.seed(42)
        n_samples = 1000
        n_features = 10

        # 特徴量行列の作成
        X_data = np.random.randn(n_samples, n_features)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X_data, columns=feature_names)

        # 目的変数を作成
        y_data = 3 * X_data[:, 0] + 2 * X_data[:, 2] + np.random.randn(n_samples) * 0.5
        self.y = pd.Series(y_data, name='target')

    def test_train_lgbm_model(self):
        """LightGBMモデルのトレーニングのテスト"""
        # モデルを訓練
        model, X_test, y_test, y_pred, importance_df = train_lgbm_model(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # 返り値の検証
        self.assertIsNotNone(model)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)
        self.assertIsInstance(y_pred, np.ndarray)
        self.assertIsInstance(importance_df, pd.DataFrame)

        # データ分割の検証
        self.assertEqual(len(X_test), int(len(self.X) * 0.2))
        self.assertEqual(len(y_test), len(X_test))
        self.assertEqual(len(y_pred), len(X_test))

        # 特徴量重要度の検証
        self.assertEqual(len(importance_df), len(self.X.columns))
        self.assertIn('Feature', importance_df.columns)
        self.assertIn('Importance', importance_df.columns)

    def test_train_lgbm_model_prediction(self):
        """LightGBMモデルの予測精度のテスト"""
        # モデルを訓練
        model, X_test, y_test, y_pred, importance_df = train_lgbm_model(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # 予測が適切な範囲内にあることを確認
        self.assertEqual(len(y_pred), len(y_test))

        # MSEが適切な値（ランダムよりも良い）であることを確認
        mse = np.mean((y_test - y_pred) ** 2)
        y_mean = y_test.mean()
        baseline_mse = np.mean((y_test - y_mean) ** 2)

        self.assertLess(mse, baseline_mse)

    def test_train_lgbm_model_feature_importance(self):
        """LightGBMモデルの特徴量重要度のテスト"""
        # モデルを訓練
        model, X_test, y_test, y_pred, importance_df = train_lgbm_model(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # 特徴量重要度の検証
        self.assertEqual(len(importance_df), len(self.X.columns))

        # 特徴量の重要度の合計が0より大きいことを確認
        self.assertGreater(importance_df['Importance'].sum(), 0)

        # 上位の特徴量に、目的変数の生成に使用した特徴量が含まれていることを確認
        top_features = importance_df.head(3)['Feature'].tolist()
        self.assertTrue('feature_0' in top_features or 'feature_2' in top_features)

    def test_train_lgbm_model_with_empty_data(self):
        """空のデータでのLightGBMモデルのトレーニングのテスト"""
        # 空のデータフレームを作成
        X_empty = pd.DataFrame(columns=self.X.columns)
        y_empty = pd.Series(name='target')

        # モデルを訓練
        model, X_test, y_test, y_pred, importance_df = train_lgbm_model(
            X_empty, y_empty, test_size=0.2, random_state=42
        )

        # エラーハンドリングの検証
        self.assertIsNone(model)
        self.assertIsNone(X_test)
        self.assertIsNone(y_test)
        self.assertIsNone(y_pred)
        self.assertIsNone(importance_df)
