"""
特徴量生成機能のテスト
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import pathlib

# プロジェクトのルートディレクトリをパスに追加
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from thermal_prediction.utils.features import (
    prepare_features_for_sens_temp,
    get_zone_power_col
)
from thermal_prediction.utils.thermo import determine_thermo_status

class TestFeatures(unittest.TestCase):
    """特徴量生成機能のテストクラス"""

    def setUp(self):
        """テストデータの作成"""
        # テスト用のデータフレームを作成
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')

        # 基本データフレーム
        self.df = pd.DataFrame({
            'time_stamp': dates,
            'AC_valid_0': 1,
            'AC_mode_0': 1,  # 冷房モード
            'AC_temp_0': 26.0,
            'AC_set_0': 24.0,
            'sens_temp_0': np.sin(np.arange(1000) * 0.01) * 2 + 25,  # 温度の波形
            'power_L': np.abs(np.sin(np.arange(1000) * 0.05)) * 1000,  # 電力消費の波形
            'outdoor_temp': np.sin(np.arange(1000) * 0.005) * 5 + 30,  # 外気温
            'solar_radiation': np.maximum(0, np.sin(np.arange(1000) * 0.02)) * 1000  # 日射量
        })

        # サーモ状態を計算
        self.thermo_df = determine_thermo_status(self.df)

    def test_get_zone_power_col(self):
        """ゾーンに対応する室外機のマッピングのテスト"""
        # L系統のゾーン
        self.assertEqual(get_zone_power_col(0), 'L')
        self.assertEqual(get_zone_power_col(1), 'L')
        self.assertEqual(get_zone_power_col(6), 'L')
        self.assertEqual(get_zone_power_col(7), 'L')

        # M系統のゾーン
        self.assertEqual(get_zone_power_col(2), 'M')
        self.assertEqual(get_zone_power_col(3), 'M')
        self.assertEqual(get_zone_power_col(8), 'M')
        self.assertEqual(get_zone_power_col(9), 'M')

        # R系統のゾーン
        self.assertEqual(get_zone_power_col(4), 'R')
        self.assertEqual(get_zone_power_col(5), 'R')
        self.assertEqual(get_zone_power_col(10), 'R')
        self.assertEqual(get_zone_power_col(11), 'R')

        # 範囲外のゾーン
        self.assertIsNone(get_zone_power_col(12))
        self.assertIsNone(get_zone_power_col(-1))

    def test_prepare_features_for_sens_temp(self):
        """特徴量生成機能のテスト"""
        zone = 0
        prediction_horizon = 5

        # 特徴量生成
        X, y, features_df = prepare_features_for_sens_temp(
            self.df, self.thermo_df, zone,
            look_back=15,
            prediction_horizon=prediction_horizon,
            feature_selection=False
        )

        # 基本検証
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(features_df)

        # 行数の検証
        # NaN値を除去しているので、元のデータより少ない行数になる
        self.assertLess(len(X), len(self.df))
        self.assertEqual(len(X), len(y))

        # 目的変数の検証
        target_col = f'sens_temp_0_future_{prediction_horizon}'
        self.assertEqual(y.name, target_col)

        # 必須特徴量の存在検証
        required_features = [
            'hour', 'day_of_week', 'is_weekend', 'thermo_0',
            'sens_temp_0_lag_1', 'power_L'
        ]
        for feature in required_features:
            self.assertIn(feature, X.columns)

        # 値の範囲検証
        self.assertTrue(all(X['hour'] >= 0) and all(X['hour'] < 24))
        self.assertTrue(all(X['day_of_week'] >= 0) and all(X['day_of_week'] < 7))
        self.assertTrue(all(X['is_weekend'].isin([0, 1])))
        self.assertTrue(all(X['thermo_0'].isin([0, 1])))

    def test_prepare_features_different_horizons(self):
        """異なる予測ホライゾンでの特徴量生成のテスト"""
        zone = 0
        horizons = [5, 15, 30]

        for horizon in horizons:
            X, y, features_df = prepare_features_for_sens_temp(
                self.df, self.thermo_df, zone,
                prediction_horizon=horizon,
                feature_selection=False
            )

            # 目的変数の検証
            target_col = f'sens_temp_0_future_{horizon}'
            self.assertEqual(y.name, target_col)

            # データの長さ検証
            # ホライゾンが長いほど末尾のNaN値が増えるため行数が減少
            if horizon > 5:
                self.assertLessEqual(len(X), self.df.shape[0] - horizon)

    def test_missing_data_handling(self):
        """欠損値がある場合の処理のテスト"""
        zone = 0
        prediction_horizon = 5

        # 一部のデータをNaNに設定
        df_with_nans = self.df.copy()
        df_with_nans.loc[10:20, 'sens_temp_0'] = np.nan

        # 特徴量生成
        X, y, features_df = prepare_features_for_sens_temp(
            df_with_nans, self.thermo_df, zone,
            prediction_horizon=prediction_horizon,
            feature_selection=False
        )

        # NaN値を含む行が除去されていることを確認
        self.assertLess(len(X), len(df_with_nans) - 11)  # 11行のNaNが除去されている

    def test_feature_selection(self):
        """特徴量選択のテスト"""
        zone = 0
        prediction_horizon = 5

        # 特徴量選択ありで特徴量生成
        X, y, features_df = prepare_features_for_sens_temp(
            self.df, self.thermo_df, zone,
            prediction_horizon=prediction_horizon,
            feature_selection=True,
            max_features=6  # 最大6特徴量を選択
        )

        # 特徴量数の検証
        # 特徴量選択が有効なので、特徴量数は減少または同等
        all_features = features_df.columns.drop(f'sens_temp_0_future_{prediction_horizon}')
        self.assertLessEqual(len(X.columns), len(all_features))

    def test_prepare_features_mock(self):
        """特徴量生成のモックテスト - get_zone_power_colの結果を直接使わずにテスト"""
        # このテストでは電力列名を一時的に上書き
        import thermal_prediction.utils.features as features_module
        original_get_zone_power_col = features_module.get_zone_power_col

        # get_zone_power_colをモックに置き換え
        features_module.get_zone_power_col = lambda zone: 'power_L' if zone == 0 else None

        try:
            zone = 0
            prediction_horizon = 5

            # 特徴量生成
            X, y, features_df = prepare_features_for_sens_temp(
                self.df, self.thermo_df, zone,
                look_back=15,
                prediction_horizon=prediction_horizon,
                feature_selection=False
            )

            # 基本検証
            self.assertIsNotNone(X)
            self.assertIsNotNone(y)
            self.assertIsNotNone(features_df)

            # 行数の検証
            # NaN値を除去しているので、元のデータより少ない行数になる
            self.assertLess(len(X), len(self.df))
            self.assertEqual(len(X), len(y))

            # 目的変数の検証
            target_col = f'sens_temp_0_future_{prediction_horizon}'
            self.assertEqual(y.name, target_col)

            # 必須特徴量の存在検証
            required_features = [
                'hour', 'day_of_week', 'is_weekend', 'thermo_0',
                'sens_temp_0_lag_1', 'power_L'
            ]
            for feature in required_features:
                self.assertIn(feature, X.columns)

            # 値の範囲検証
            self.assertTrue(all(X['hour'] >= 0) and all(X['hour'] < 24))
            self.assertTrue(all(X['day_of_week'] >= 0) and all(X['day_of_week'] < 7))
            self.assertTrue(all(X['is_weekend'].isin([0, 1])))
            self.assertTrue(all(X['thermo_0'].isin([0, 1])))

        finally:
            # 元の関数に戻す
            features_module.get_zone_power_col = original_get_zone_power_col

    def test_prepare_features_different_horizons_mock(self):
        """異なる予測ホライゾンでの特徴量生成のモックテスト"""
        import thermal_prediction.utils.features as features_module
        original_get_zone_power_col = features_module.get_zone_power_col

        # get_zone_power_colをモックに置き換え
        features_module.get_zone_power_col = lambda zone: 'power_L' if zone == 0 else None

        try:
            zone = 0
            horizons = [5, 15, 30]

            for horizon in horizons:
                X, y, features_df = prepare_features_for_sens_temp(
                    self.df, self.thermo_df, zone,
                    prediction_horizon=horizon,
                    feature_selection=False
                )

                # 目的変数の検証
                target_col = f'sens_temp_0_future_{horizon}'
                self.assertEqual(y.name, target_col)

                # データの長さ検証
                # ホライゾンが長いほど末尾のNaN値が増えるため行数が減少
                if horizon > 5:
                    self.assertLessEqual(len(X), self.df.shape[0] - horizon)

        finally:
            # 元の関数に戻す
            features_module.get_zone_power_col = original_get_zone_power_col

    def test_missing_data_handling_mock(self):
        """欠損値がある場合の処理のモックテスト"""
        import thermal_prediction.utils.features as features_module
        original_get_zone_power_col = features_module.get_zone_power_col

        # get_zone_power_colをモックに置き換え
        features_module.get_zone_power_col = lambda zone: 'power_L' if zone == 0 else None

        try:
            zone = 0
            prediction_horizon = 5

            # 一部のデータをNaNに設定
            df_with_nans = self.df.copy()
            df_with_nans.loc[10:20, 'sens_temp_0'] = np.nan

            # 特徴量生成
            X, y, features_df = prepare_features_for_sens_temp(
                df_with_nans, self.thermo_df, zone,
                prediction_horizon=prediction_horizon,
                feature_selection=False
            )

            # NaN値を含む行が除去されていることを確認
            self.assertLess(len(X), len(df_with_nans) - 11)  # 11行のNaNが除去されている

        finally:
            # 元の関数に戻す
            features_module.get_zone_power_col = original_get_zone_power_col

    def test_feature_selection_mock(self):
        """特徴量選択のモックテスト"""
        import thermal_prediction.utils.features as features_module
        original_get_zone_power_col = features_module.get_zone_power_col

        # get_zone_power_colをモックに置き換え
        features_module.get_zone_power_col = lambda zone: 'power_L' if zone == 0 else None

        try:
            zone = 0
            prediction_horizon = 5

            # 特徴量選択ありで特徴量生成
            X, y, features_df = prepare_features_for_sens_temp(
                self.df, self.thermo_df, zone,
                prediction_horizon=prediction_horizon,
                feature_selection=True,
                max_features=6  # 最大6特徴量を選択
            )

            # 特徴量数の検証
            # 特徴量選択が有効なので、特徴量数は減少または同等
            all_features = features_df.columns.drop(f'sens_temp_0_future_{prediction_horizon}')
            self.assertLessEqual(len(X.columns), len(all_features))

        finally:
            # 元の関数に戻す
            features_module.get_zone_power_col = original_get_zone_power_col
