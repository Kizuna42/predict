"""
温度予測モジュールの設定を一元管理するモジュール
"""

import os

# デフォルト設定
DEFAULT_CONFIG = {
    # データ関連
    'DEFAULT_FILE_PATH': './AllDayData.csv',
    'DEFAULT_OUTPUT_DIR': './output/sens_temp_predictions',
    'DEFAULT_START_DATE': '2024-07-01',
    'DEFAULT_END_DATE': '2024-07-03',

    # 予測関連
    'DEFAULT_PREDICTION_HORIZONS': [5, 10, 15, 20, 30],
    'DEFAULT_TEST_SIZE': 0.2,
    'DEFAULT_RANDOM_STATE': 42,

    # 特徴量関連
    'MAX_LOOK_BACK': 60,  # 過去データの参照最大時間（分）
    'DEFAULT_FEATURE_SELECTION': True,
    'DEFAULT_IMPORTANCE_THRESHOLD': 0.05,
    'DEFAULT_MAX_FEATURES': 6,

    # サーモ状態関連
    'THERMO_DEADBAND': 1.0,  # サーモ状態の不感帯

    # ゾーン関連
    'ALL_ZONES': list(range(12)),  # 全てのゾーン（0-11）
    'L_ZONES': [0, 1, 6, 7],  # L系統のゾーン
    'M_ZONES': [2, 3, 8, 9],  # M系統のゾーン
    'R_ZONES': [4, 5, 10, 11],  # R系統のゾーン

    # モデル関連
    'LGBM_PARAMS': {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbosity': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'force_col_wise': True,
    },
    'LGBM_EARLY_STOPPING_ROUNDS': 50,
    'LGBM_NUM_BOOST_ROUND': 1000,

    # 可視化関連
    'FIGURE_DPI': 100,
    'FIGURE_SIZE': (12, 8),
    'COLOR_PALETTE': 'viridis',

    # 実行設定
    'RUN_ANALYZE_HORIZONS': True,  # 予測ホライゾン分析を実行するかどうか
    'ZONES_TO_ANALYZE': [1],  # テスト用に1ゾーンのみ分析
    'HORIZONS_TO_ANALYZE': [5, 10, 15]  # テスト用に3つのホライゾンのみ分析
}


def get_zone_power_map():
    """ゾーンと対応する室外機のマッピングを返す"""
    zone_to_power = {}
    for zone in DEFAULT_CONFIG['L_ZONES']:
        zone_to_power[zone] = 'L'
    for zone in DEFAULT_CONFIG['M_ZONES']:
        zone_to_power[zone] = 'M'
    for zone in DEFAULT_CONFIG['R_ZONES']:
        zone_to_power[zone] = 'R'
    return zone_to_power


def get_thermo_or_map():
    """室外機とOR関連のサーモ状態列のマッピングを返す"""
    return {
        'L': 'thermo_L_or',
        'M': 'thermo_M_or',
        'R': 'thermo_R_or'
    }
