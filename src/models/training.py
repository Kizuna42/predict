#!/usr/bin/env python
# coding: utf-8

"""
モデルトレーニングモジュール
物理法則を考慮したLightGBMモデルのトレーニング関数を提供
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from src.config import LGBM_PARAMS, MODELS_DIR


def train_physics_guided_model(X_train, y_train, params=None):
    """
    物理法則を考慮したモデルのトレーニング
    修正: 特徴量の重複をチェックし、重複を排除

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数
    params : dict
        LightGBMのパラメータ（Noneの場合はデフォルト値を使用）

    Returns:
    --------
    LGBMRegressor
        トレーニング済みモデル
    """
    print("物理法則ガイド付きモデルをトレーニング中...")

    # 列名の重複チェック
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: トレーニングデータの列名に重複があります。重複を排除します。")
        # 重複を排除したデータフレームを作成
        unique_cols = []
        seen_cols = set()
        for col in X_train.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)

        # 重複を排除した特徴量のみを使用
        duplicate_count = len(X_train.columns) - len(unique_cols)
        X_train = X_train[unique_cols]
        print(f"{duplicate_count}個の重複特徴量を排除しました。残り特徴量数: {len(unique_cols)}")

    # パラメータが指定されていない場合は、デフォルト値を使用
    if params is None:
        params = LGBM_PARAMS.copy()  # コピーを作成して元の設定を変更しないようにする

    # 警告メッセージを抑制するため、verboseを-1に設定
    params['verbose'] = -1

    # 物理モデルに適したパラメータを使用
    lgb_model = lgb.LGBMRegressor(**params)

    try:
        # Pythonの標準警告を一時的に抑制
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # クラス重み付け：急激な温度変化を重視
            if 'weight' in y_train.index.names:
                print("サンプル重み付けを使用します")
                lgb_model.fit(X_train, y_train)
            else:
                # 急激な温度変化に対する重み付け
                temp_changes = y_train.diff().abs().fillna(0)
                weights = 1 + temp_changes / temp_changes.mean()

                # スパイクの影響を制限
                max_weight = 3.0  # 最大ウェイト値を制限
                weights = weights.clip(upper=max_weight)

                lgb_model.fit(X_train, y_train, sample_weight=weights)

        return lgb_model

    except Exception as e:
        print(f"モデルトレーニング中にエラーが発生しました: {e}")
        print("緊急対処モードでトレーニングを試みます...")

        # 最小限の特徴量だけを使用して再トレーニング
        # 基本的な特徴量のみを選択（温度、設定値、制御状態など）
        basic_features = [col for col in X_train.columns if any(key in col for key in [
            'sens_temp', 'thermo_state', 'AC_valid', 'AC_mode', 'atmospheric', 'solar'
        ])]

        if len(basic_features) > 0:
            print(f"基本特徴量{len(basic_features)}個のみを使用してトレーニングします")
            simple_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbose=-1  # 警告を抑制
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simple_model.fit(X_train[basic_features], y_train)
            return simple_model
        else:
            # すべての対処が失敗した場合はダミーモデルを返す
            print("ダミーモデルを作成します（平均値予測）")
            from sklearn.dummy import DummyRegressor
            dummy_model = DummyRegressor(strategy='mean')
            dummy_model.fit(X_train.iloc[:, 0].values.reshape(-1, 1), y_train)
            return dummy_model


def save_model_and_features(model, feature_list, zone, horizon, poly_config=None):
    """
    モデルと特徴量情報を保存する関数

    Parameters:
    -----------
    model : 学習済みモデル
        保存するモデル
    feature_list : list
        特徴量のリスト
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    poly_config : dict, optional
        多項式特徴量の設定情報

    Returns:
    --------
    tuple
        (モデルのファイルパス, 特徴量情報のファイルパス)
    """
    # モデルと特徴量情報のファイルパス
    model_filename = os.path.join(MODELS_DIR, f"lgbm_model_zone_{zone}_horizon_{horizon}.pkl")
    features_filename = os.path.join(MODELS_DIR, f"features_zone_{zone}_horizon_{horizon}.pkl")

    try:
        # モデルの保存
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"モデルを保存しました: {model_filename}")

        # 特徴量情報の保存
        feature_info = {
            'feature_cols': feature_list,
            'poly_config': poly_config if poly_config else {
                'use_poly': False,
                'poly_features': []
            }
        }
        with open(features_filename, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"特徴量情報を保存しました: {features_filename}")

        return model_filename, features_filename
    except Exception as e:
        print(f"モデルまたは特徴量情報の保存中にエラーが発生しました: {e}")
        return None, None


def load_model_and_features(zone, horizon):
    """
    保存されたモデルと特徴量情報を読み込む関数

    Parameters:
    -----------
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    tuple
        (モデル, 特徴量情報)
    """
    model_filename = os.path.join(MODELS_DIR, f"lgbm_model_zone_{zone}_horizon_{horizon}.pkl")
    features_filename = os.path.join(MODELS_DIR, f"features_zone_{zone}_horizon_{horizon}.pkl")

    try:
        # モデルの読み込み
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        # 特徴量情報の読み込み
        with open(features_filename, 'rb') as f:
            feature_info = pickle.load(f)

        print(f"モデルと特徴量情報を読み込みました: ゾーン{zone}, {horizon}分後")
        return model, feature_info
    except Exception as e:
        print(f"モデルまたは特徴量情報の読み込み中にエラーが発生しました: {e}")
        return None, None


def train_temperature_difference_model(X_train, y_train, params=None):
    """
    温度差分予測専用のモデルトレーニング関数

    温度の変化量を予測するため、従来の温度予測とは異なるパラメータ調整を行う：
    - より高い学習率で変化パターンを捉える
    - 小さな差分値に対する感度を向上
    - 変化の激しい期間への重み付け強化

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数（温度差分）
    params : dict
        LightGBMのパラメータ（Noneの場合は差分予測用デフォルト値を使用）

    Returns:
    --------
    LGBMRegressor
        トレーニング済み差分予測モデル
    """
    print("温度差分予測専用モデルをトレーニング中...")

    # 列名の重複チェック
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: トレーニングデータの列名に重複があります。重複を排除します。")
        unique_cols = []
        seen_cols = set()
        for col in X_train.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)

        duplicate_count = len(X_train.columns) - len(unique_cols)
        X_train = X_train[unique_cols]
        print(f"{duplicate_count}個の重複特徴量を排除しました。残り特徴量数: {len(unique_cols)}")

    # 差分予測に最適化されたパラメータ
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',  # 温度差分では平均絶対誤差が適切
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.08,  # やや高めの学習率で変化パターンを捉える
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 8,
            'min_data_in_leaf': 10,  # 小さな差分値を捉えるため小さめに設定
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.02,
            'random_state': 42,
            'n_estimators': 1000,
            'verbose': -1
        }

    lgb_model = lgb.LGBMRegressor(**params)

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 差分予測専用の重み付け戦略
            # 1. 大きな変化量への重み付け
            abs_diff = y_train.abs()
            change_weights = 1 + (abs_diff / abs_diff.std()) * 0.5

            # 2. 0付近の小さな変化も重要視（ラグ対策）
            small_change_bonus = np.where(abs_diff < abs_diff.quantile(0.3), 1.2, 1.0)

            # 3. 変化方向の多様性を重視
            direction_changes = np.abs(np.sign(y_train).diff()).fillna(0)
            direction_weights = 1 + direction_changes * 0.3

            # 最終的な重み
            final_weights = change_weights * small_change_bonus * direction_weights
            final_weights = final_weights.clip(upper=3.0)  # 最大重みを制限

            print(f"差分予測用重み付け適用 - 平均重み: {final_weights.mean():.3f}")

            lgb_model.fit(X_train, y_train, sample_weight=final_weights)

        return lgb_model

    except Exception as e:
        print(f"差分予測モデルトレーニング中にエラーが発生しました: {e}")
        print("シンプルな差分予測モデルで再試行...")

        # フォールバック: シンプルなパラメータ
        simple_model = lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simple_model.fit(X_train, y_train)

        return simple_model


def save_difference_model_and_features(model, feature_list, zone, horizon, poly_config=None):
    """
    差分予測モデルと特徴量情報を保存する関数

    Parameters:
    -----------
    model : 学習済みモデル
        保存する差分予測モデル
    feature_list : list
        特徴量のリスト
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    poly_config : dict, optional
        多項式特徴量の設定情報

    Returns:
    --------
    tuple
        (モデルのファイルパス, 特徴量情報のファイルパス)
    """
    # 差分予測モデル専用のファイル名
    model_filename = os.path.join(MODELS_DIR, f"diff_model_zone_{zone}_horizon_{horizon}.pkl")
    features_filename = os.path.join(MODELS_DIR, f"diff_features_zone_{zone}_horizon_{horizon}.pkl")

    try:
        # モデルの保存
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"差分予測モデルを保存しました: {model_filename}")

        # 特徴量情報の保存
        feature_info = {
            'feature_cols': feature_list,
            'model_type': 'temperature_difference',
            'poly_config': poly_config if poly_config else {
                'use_poly': False,
                'poly_features': []
            }
        }
        with open(features_filename, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"差分予測特徴量情報を保存しました: {features_filename}")

        return model_filename, features_filename
    except Exception as e:
        print(f"差分予測モデルまたは特徴量情報の保存中にエラーが発生しました: {e}")
        return None, None
