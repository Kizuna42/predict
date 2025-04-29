"""
LightGBM モデル関連機能
"""

import numpy as np
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from ..config import DEFAULT_CONFIG

def train_lgbm_model(X, y, test_size=None, random_state=None):
    """
    LightGBM モデルを訓練する

    Args:
        X: 特徴量
        y: 目的変数
        test_size: テストデータの割合
        random_state: 乱数シード

    Returns:
        model: 訓練されたモデル
        X_test: テストデータの特徴量
        y_test: テストデータの目的変数
        y_pred: 予測値
        importance_df: 特徴量重要度のデータフレーム
    """
    try:
        # configからデフォルト値を取得
        if test_size is None:
            test_size = DEFAULT_CONFIG['DEFAULT_TEST_SIZE']
        if random_state is None:
            random_state = DEFAULT_CONFIG['DEFAULT_RANDOM_STATE']

        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # configからパラメータを取得
        params = DEFAULT_CONFIG['LGBM_PARAMS'].copy()

        callbacks = [
            lgb.early_stopping(DEFAULT_CONFIG['LGBM_EARLY_STOPPING_ROUNDS'], verbose=False),
            lgb.log_evaluation(period=100, show_stdv=False)
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=DEFAULT_CONFIG['LGBM_NUM_BOOST_ROUND'],
            valid_sets=[train_data, test_data],
            callbacks=callbacks
        )

        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        # 評価指標の計算
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")

        # 特徴量重要度の計算
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        print("\nTop 10 Feature Importance:")
        print(importance_df.head(10))

        return model, X_test, y_test, y_pred, importance_df

    except Exception as e:
        print(f"Error in train_lgbm_model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None
