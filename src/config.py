#!/usr/bin/env python
# coding: utf-8

"""
設定値モジュール
温度予測モデルに関する各種パラメータと設定値を定義
"""

import os

# パス設定
OUTPUT_DIR = 'Output'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

# ディレクトリが存在しない場合は作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# データの前処理設定
MIN_TEMP = 10  # 最小許容温度
MAX_TEMP = 40  # 最大許容温度

# スムージング窓サイズ
SMOOTHING_WINDOWS = [3, 6, 12]

# 予測ホライゾン（分）
HORIZONS = [5, 10, 15, 20, 30]

# LMRのゾーン区分
L_ZONES = [0, 1, 6, 7]
M_ZONES = [2, 3, 8, 9]
R_ZONES = [4, 5, 10, 11]

# モデルハイパーパラメータ
LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.005,
    'max_depth': 4,
    'min_child_samples': 40,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    'random_state': 42,
    'importance_type': 'gain',
    'boosting_type': 'gbdt',
    'objective': 'regression'
}

# 特徴量選択設定
FEATURE_SELECTION_THRESHOLD = 'median'

# トレーニング/テスト分割比率
TEST_SIZE = 0.2
