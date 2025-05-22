#!/usr/bin/env python
# coding: utf-8

import os

OUTPUT_DIR = 'Output'
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# データの前処理設定
MIN_TEMP = 10  # 最小許容温度
MAX_TEMP = 40  # 最大許容温度

# スムージング窓サイズ
SMOOTHING_WINDOWS = [3, 6, 12, 18, 24]

# 予測ホライゾン（分）
HORIZONS = [5, 10, 15, 20, 30]

L_ZONES = [0, 1, 6, 7]
M_ZONES = [2, 3, 8, 9]
R_ZONES = [4, 5, 10, 11]

# モデルハイパーパラメータ
LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.005,
    'max_depth': 5,       # 少し深くして複雑な関係を捉える
    'min_child_samples': 30, # サンプル数を少し減らして柔軟性を高める
    'subsample': 0.8,
    'colsample_bytree': 0.8, # 列のサンプリング率を上げて特徴量の利用率を高める
    'reg_alpha': 0.05,
    'reg_lambda': 0.1,
    'random_state': 42,
    'importance_type': 'gain',
    'boosting_type': 'gbdt',
    'objective': 'regression'
}

# 特徴量選択設定 - 上司のアドバイスに従い少し緩和
FEATURE_SELECTION_THRESHOLD = '30%' # サーモ状態や環境データなどの重要な特徴量をより多く含める

# トレーニング/テスト分割比率
TEST_SIZE = 0.2
