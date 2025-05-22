import numpy as np
import pandas as pd

def prepare_prediction_features(current_data, future_settings, feature_list, horizon):
    """
    予測用の特徴量を準備

    Parameters:
    -----------
    current_data : DataFrame
        現在のセンサーデータなど
    future_settings : dict
        予測時の設定値（AC_valid, AC_mode, サーモ状態など）
    feature_list : list
        モデルが使用する特徴量リスト
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    DataFrame
        予測用の特徴量データフレーム
    """
    # 現在データのコピー
    df = current_data.copy()

    # 未来設定値を追加
    for key, value in future_settings.items():
        future_col = f"{key}_future_{horizon}"
        if future_col in feature_list:
            df[future_col] = value

    # 不要な列を削除し、必要な特徴量のみを抽出
    prediction_features = df[feature_list].copy()

    # 欠損値処理
    prediction_features = prediction_features.fillna(method='ffill').fillna(method='bfill')

    return prediction_features

def predict_temperature(model, current_data, future_settings, feature_list, horizon):
    """
    温度予測を実行

    Parameters:
    -----------
    model : 学習済みモデル
    current_data : DataFrame
        現在のセンサーデータなど
    future_settings : dict
        予測時の設定値（AC_valid, AC_mode, サーモ状態など）
    feature_list : list
        モデルが使用する特徴量リスト
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    float
        予測温度
    """
    # 予測用特徴量の準備
    prediction_features = prepare_prediction_features(
        current_data, future_settings, feature_list, horizon
    )

    # 予測実行
    prediction = model.predict(prediction_features)

    return prediction[0]  # 単一の予測値を返す
