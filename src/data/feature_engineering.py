#!/usr/bin/env python
# coding: utf-8

"""
特徴量エンジニアリングモジュール
LAG特徴量、移動平均特徴量、物理モデルベースの特徴量など、各種特徴量生成関数を提供
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb


def create_lag_features(df, zone_nums, lag_periods=[12, 24]):
    """
    各ゾーンの過去の温度と湿度をLAG特徴量として作成
    修正: LAGに過度に依存しないように、直近の短期LAGを削除し、より長期のLAGのみを使用
    上司のアドバイスに従い、生データLAG特徴量を減らし、変化率や物理量の特徴量を強化

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    lag_periods : list
        ラグ期間（データサンプリング単位）のリスト（長期LAGのみに変更）

    Returns:
    --------
    DataFrame
        LAG特徴量を追加したデータフレーム
    list
        作成した特徴量のリスト
    """
    print("物理モデルベースのLAG特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    for zone in zone_nums:
        # 温度のLAG特徴量ではなく、温度変化率に焦点を当てる
        if f'sens_temp_{zone}' in df.columns:
            # 温度変化量（一定期間での温度差）- 物理的意味を持つ特徴量
            for lag in lag_periods:
                # 長期的な変化率（直前からの変化ではなく、より長期の変化を捉える）
                df_copy[f'temp_change_rate_{zone}_{lag}'] = (df_copy[f'sens_temp_{zone}'] - df_copy[f'sens_temp_{zone}'].shift(lag)) / lag
                created_features.append(f'temp_change_rate_{zone}_{lag}')

                # 長期的な変化率の加速度（変化の変化）
                df_copy[f'temp_change_accel_{zone}_{lag}'] = df_copy[f'temp_change_rate_{zone}_{lag}'].diff()
                created_features.append(f'temp_change_accel_{zone}_{lag}')

            # 温度の加速度（変化率の変化）- 物理的なパラメータ
            df_copy[f'temp_acceleration_{zone}'] = df_copy[f'sens_temp_{zone}'].diff().diff()
            created_features.append(f'temp_acceleration_{zone}')

            # 温度変化のトレンド（上昇/下降の持続性）- 長期的なトレンドを捉える
            df_copy[f'temp_trend_{zone}_long'] = np.sign(df_copy[f'sens_temp_{zone}'].diff(12))
            created_features.append(f'temp_trend_{zone}_long')

            # 温度の振動性（変化方向の反転頻度）- 制御系の安定性指標
            temp_changes = df_copy[f'sens_temp_{zone}'].diff()
            df_copy[f'temp_oscillation_{zone}'] = ((temp_changes.shift(1) * temp_changes) < 0).astype(int).rolling(window=24, min_periods=1).mean()
            created_features.append(f'temp_oscillation_{zone}')

    print(f"作成した物理ベースの特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_rolling_features(df, zone_nums, windows=[6, 12, 24]):
    """
    各ゾーンの温度と湿度の移動平均を特徴量として作成
    重要: 未来の値を使わないようにmin_periods=1を設定し、過去のデータのみを使用
    修正: 変化率を中心とした特徴量に変更し、より長い窓幅を追加

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    windows : list
        移動平均の窓サイズ（データサンプリング単位）のリスト

    Returns:
    --------
    DataFrame
        移動平均特徴量を追加したデータフレーム
    list
        作成した特徴量のリスト
    """
    print("物理的意味を持つ移動平均特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    for zone in zone_nums:
        # 温度の移動平均ではなく、物理的な意味を持つ特徴量を作成
        if f'sens_temp_{zone}' in df.columns:
            for window in windows:
                # 温度変化率の移動平均（温度の一次微分の平滑化）
                df_copy[f'temp_rate_mean_{zone}_{window}'] = df_copy[f'sens_temp_{zone}'].diff().rolling(
                    window=window, min_periods=1).mean()
                created_features.append(f'temp_rate_mean_{zone}_{window}')

                # 温度加速度の移動平均（温度の二次微分の平滑化）
                df_copy[f'temp_accel_mean_{zone}_{window}'] = df_copy[f'sens_temp_{zone}'].diff().diff().rolling(
                    window=window, min_periods=1).mean()
                created_features.append(f'temp_accel_mean_{zone}_{window}')

                # 温度変動の振幅（温度の安定性/変動性を物理的に表現）
                df_copy[f'temp_amplitude_{zone}_{window}'] = df_copy[f'sens_temp_{zone}'].rolling(
                    window=window, min_periods=1).max() - df_copy[f'sens_temp_{zone}'].rolling(
                    window=window, min_periods=1).min()
                created_features.append(f'temp_amplitude_{zone}_{window}')

    print(f"作成した物理的特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_physics_based_features(df, zone_nums):
    """
    物理法則に基づいた特徴量を作成する関数
    改善: より詳細な熱力学モデルに基づいた特徴量を追加

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト

    Returns:
    --------
    DataFrame
        物理モデルベースの特徴量を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    print("拡張された物理モデルベースの特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    # 1. 熱力学の法則に基づく特徴量
    for zone in zone_nums:
        if f'sens_temp_{zone}' in df.columns:
            # 温度変化率（一次微分）
            df_copy[f'temp_rate_{zone}'] = df_copy[f'sens_temp_{zone}'].diff()
            created_features.append(f'temp_rate_{zone}')

            # 温度加速度（二次微分 - 温度変化の変化率）
            df_copy[f'temp_accel_{zone}'] = df_copy[f'temp_rate_{zone}'].diff()
            created_features.append(f'temp_accel_{zone}')

            # 正規化した温度変化率（過去に対する相対変化）
            df_copy[f'temp_rel_change_{zone}'] = df_copy[f'temp_rate_{zone}'] / df_copy[f'sens_temp_{zone}'].shift(1).replace(0, np.nan)
            df_copy[f'temp_rel_change_{zone}'].fillna(0, inplace=True)
            created_features.append(f'temp_rel_change_{zone}')

            # 温度の慣性（過去の変化の持続性 - より長期的な視点）
            df_copy[f'temp_momentum_{zone}'] = df_copy[f'sens_temp_{zone}'].diff(3) * df_copy[f'sens_temp_{zone}'].diff()
            created_features.append(f'temp_momentum_{zone}')

            # 1.1 設定温度との関係
            if f'AC_set_{zone}' in df.columns:
                # 設定温度との差（熱力学における駆動力）
                df_copy[f'temp_diff_to_setpoint_{zone}'] = df_copy[f'sens_temp_{zone}'] - df_copy[f'AC_set_{zone}']
                created_features.append(f'temp_diff_to_setpoint_{zone}')

                # 熱伝達の効率性指標（温度差に対する変化率の比）
                df_copy[f'heat_transfer_efficiency_{zone}'] = df_copy[f'sens_temp_{zone}'].diff() / df_copy[f'temp_diff_to_setpoint_{zone}'].abs()
                # 分母が0になる可能性があるため、無限大や無効な値を処理
                df_copy[f'heat_transfer_efficiency_{zone}'].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_copy[f'heat_transfer_efficiency_{zone}'].fillna(0, inplace=True)  # NaNを0に置換
                created_features.append(f'heat_transfer_efficiency_{zone}')

                # 設定との差の変化率（制御応答の指標）
                df_copy[f'setpoint_response_{zone}'] = df_copy[f'temp_diff_to_setpoint_{zone}'].diff()
                created_features.append(f'setpoint_response_{zone}')

                # 設定温度への収束速度の推定
                df_copy[f'convergence_rate_{zone}'] = df_copy[f'temp_diff_to_setpoint_{zone}'] / df_copy[f'temp_diff_to_setpoint_{zone}'].shift(1)
                df_copy[f'convergence_rate_{zone}'].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_copy[f'convergence_rate_{zone}'].fillna(1, inplace=True)  # NaNを1に置換（変化なし）
                created_features.append(f'convergence_rate_{zone}')

            # 1.2 空調制御状態に基づく特徴量
            if f'AC_valid_{zone}' in df.columns:
                # AC有効状態と温度変化の交互作用
                df_copy[f'ac_temp_interaction_{zone}'] = df_copy[f'AC_valid_{zone}'] * df_copy[f'temp_rate_{zone}']
                created_features.append(f'ac_temp_interaction_{zone}')

                # AC状態変化の検出（ON/OFFが切り替わった時点）
                df_copy[f'ac_state_change_{zone}'] = df_copy[f'AC_valid_{zone}'].diff().abs()
                created_features.append(f'ac_state_change_{zone}')

                # AC OFF後の温度ドリフト検出（自然温度変化）
                ac_off_mask = (df_copy[f'AC_valid_{zone}'] == 0)
                df_copy[f'natural_temp_drift_{zone}'] = 0.0
                if ac_off_mask.sum() > 0:
                    df_copy.loc[ac_off_mask, f'natural_temp_drift_{zone}'] = df_copy.loc[ac_off_mask, f'temp_rate_{zone}']
                created_features.append(f'natural_temp_drift_{zone}')

            # 1.3 外気温との関係
            # atmospheric temperatureのカラム名を部分一致で検索
            atmos_cols = [col for col in df.columns if 'atmospheric' in col.lower() and 'temperature' in col.lower()]
            if atmos_cols:
                atmos_col = atmos_cols[0]
                # 外気温との温度差（熱伝導の駆動力）
                df_copy[f'temp_diff_to_outside_{zone}'] = df_copy[f'sens_temp_{zone}'] - df_copy[atmos_col]
                created_features.append(f'temp_diff_to_outside_{zone}')

                # 外気温との温度差の変化率
                df_copy[f'temp_diff_to_outside_rate_{zone}'] = df_copy[f'temp_diff_to_outside_{zone}'].diff()
                created_features.append(f'temp_diff_to_outside_rate_{zone}')

                # 外気温の変化率
                df_copy[f'atmos_temp_rate'] = df_copy[atmos_col].diff()
                created_features.append('atmos_temp_rate')

            # 1.4 日射量との関係
            # solar radiationのカラム名を部分一致で検索
            solar_cols = [col for col in df.columns if 'solar' in col.lower() and 'radiation' in col.lower()]
            if solar_cols:
                solar_col = solar_cols[0]
                # 日射量の変化率
                df_copy[f'solar_radiation_rate'] = df_copy[solar_col].diff()
                created_features.append('solar_radiation_rate')

                # 日射量と温度変化の交互作用
                df_copy[f'solar_temp_interaction_{zone}'] = df_copy[solar_col] * df_copy[f'temp_rate_{zone}']
                created_features.append(f'solar_temp_interaction_{zone}')

    print(f"作成した物理的特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_future_explanatory_features(df, base_features_config, horizons_minutes, time_diff_seconds, is_prediction_mode=False):
    """
    未来の説明変数を作成する関数
    is_prediction_mode=Trueの場合は、プレースホルダーを作成（予測時用）
    修正: カラム名の特殊文字に対応

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    base_features_config : list of dict
        基本特徴量の設定（例: [{'name': 'thermo_state_1', 'type': 'zone_specific', 'zone': 1}]）
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : int または float
        データのサンプリング間隔（秒）
    is_prediction_mode : bool
        予測モード時はTrueを指定（未来値はプレースホルダーとなる）

    Returns:
    --------
    DataFrame
        未来の説明変数を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    print("制御可能なパラメータと環境データの未来値を特徴量として作成中...")
    df_copy = df.copy()
    created_features = []

    # 制御可能なパラメータと環境データの前缀
    controllable_params_prefixes = ['AC_', 'thermo_state']
    environmental_prefixes = ['atmospheric', 'solar', 'radiation']

    # 利用可能なカラム名を取得
    available_columns = df_copy.columns.tolist()

    for horizon in horizons_minutes:
        # シフト量を計算
        shift_steps = int(horizon * 60 / time_diff_seconds)

        for config in base_features_config:
            base_col_name = config['name']

            # カラムが存在するか確認
            if base_col_name not in available_columns:
                # 部分一致で検索を試みる
                matching_cols = [col for col in available_columns
                               if all(keyword.lower() in col.lower() for keyword in base_col_name.split())]
                if matching_cols:
                    base_col_name = matching_cols[0]
                    print(f"カラム名を置換しました: '{config['name']}' → '{base_col_name}'")
                else:
                    print(f"警告: カラム '{config['name']}' が見つかりません。このカラムの未来値はスキップします。")
                    continue

            is_controllable = any(base_col_name.startswith(prefix) for prefix in controllable_params_prefixes)
            is_environmental = any(prefix in base_col_name.lower() for prefix in environmental_prefixes)

            if is_controllable or is_environmental:
                future_col = f"{base_col_name}_future_{horizon}"

                if not is_prediction_mode:
                    # 学習時: 実際の未来値を使用
                    df_copy[future_col] = df_copy[base_col_name].shift(-shift_steps)
                else:
                    # 予測時: プレースホルダーを作成（NaNまたは0で初期化）
                    df_copy[future_col] = np.nan

                created_features.append(future_col)

    print(f"作成した未来の説明変数: {len(created_features)}個")
    return df_copy, created_features


def create_thermo_state_features(df, zone_nums):
    """
    サーモ状態の特徴量を作成
    センサー温度とAC設定温度の差を計算
    サーモの状態を導入し、AC_setの代わりにもなるようにする

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト

    Returns:
    --------
    DataFrame
        サーモ状態特徴量を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    df_copy = df.copy()
    thermo_features = []

    for zone in zone_nums:
        if f'sens_temp_{zone}' in df_copy.columns and f'AC_set_{zone}' in df_copy.columns:
            # 平滑化されたセンサー温度が存在すればそちらを優先してサーモ状態を計算
            base_temp_col_for_thermo = f'sens_temp_{zone}_smoothed' if f'sens_temp_{zone}_smoothed' in df_copy.columns else f'sens_temp_{zone}'

            # サーモ状態 = センサー温度 - 設定温度
            thermo_col = f'thermo_state_{zone}'
            df_copy[thermo_col] = df_copy[base_temp_col_for_thermo] - df_copy[f'AC_set_{zone}']
            thermo_features.append(thermo_col)

            # 上司のアドバイスに基づき、サーモの状態を基準とした特徴量を追加
            # 冷房モードと暖房モードで異なるサーモ状態の特性を表現
            if f'AC_mode_{zone}' in df_copy.columns:
                # AC_modeに基づいてサーモ状態の意味を調整（冷房/暖房）
                ac_mode_col = f'AC_mode_{zone}'
                # 冷房モード時は正のサーモ状態が「暑い」、暖房モード時は負のサーモ状態が「寒い」
                df_copy[f'thermo_state_adjusted_{zone}'] = df_copy[thermo_col] * df_copy[ac_mode_col].map({0: 1, 1: -1})
                thermo_features.append(f'thermo_state_adjusted_{zone}')

            print(f"ゾーン{zone}のサーモ状態特徴量を作成しました: {thermo_col} (ベース温度: {base_temp_col_for_thermo})")

    return df_copy, thermo_features


def select_important_features(X_train, y_train, X_test, feature_names, threshold='25%'):
    """
    LightGBMを使った特徴量選択により、重要な特徴量のみを選択
    閾値を変更：物理特徴量を優先的に選択するため
    修正: 特徴量名の重複を排除する処理を追加
    さらに修正: 警告メッセージを抑制

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数
    X_test : DataFrame
        テスト用特徴量
    feature_names : list
        特徴量名のリスト
    threshold : str or float
        特徴量選択の閾値

    Returns:
    --------
    X_train_selected, X_test_selected : 選択された特徴量のみを含むデータフレーム
    selected_features : 選択された特徴量名のリスト
    """
    # 特徴量名に重複がないか確認し、重複があれば警告して排除
    unique_feature_names = []
    seen_names = set()
    duplicates = []

    for feature in feature_names:
        if feature in seen_names:
            duplicates.append(feature)
        else:
            unique_feature_names.append(feature)
            seen_names.add(feature)

    if duplicates:
        print(f"警告: 重複する特徴量名を検出し排除しました: {len(duplicates)}個")
        # 必要に応じて最初の数個だけ表示
        if len(duplicates) > 5:
            print(f"例: {duplicates[:5]} など...")
        else:
            print(f"重複特徴量: {duplicates}")
        feature_names = unique_feature_names

    # 列名に重複がないか確認（データフレームの列名も確認）
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: データフレームの列名に重複があります。重複を排除します。")
        # 重複のない新しいデータフレームを作成
        unique_cols = []
        seen_cols = set()
        for col in X_train.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)

        # 重複のない列だけを使用
        X_train = X_train[unique_cols]
        X_test = X_test[unique_cols]

        # feature_namesも更新
        feature_names = [f for f in feature_names if f in unique_cols]

    # 特徴量選択用の軽量なモデル
    selection_model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.01,
        max_depth=3,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.05,
        random_state=42,
        importance_type='gain',  # 利得ベースの重要度を使用
        verbose=-1  # 警告メッセージを抑制
    )

    # SelectFromModelで重要な特徴量を選択
    # パーセント表記の閾値を処理（'30%'のような文字列）
    selector_threshold = threshold
    if isinstance(threshold, str) and '%' in threshold:
        # パーセント表記の閾値をfloatに変換（'30%' → 0.3）
        try:
            selector_threshold = float(threshold.strip('%')) / 100.0
            print(f"閾値を変換しました: {threshold} → {selector_threshold}")
        except ValueError:
            # 変換できない場合は'mean'を使用
            selector_threshold = 'mean'
            print(f"閾値の解析に失敗したため、'mean'を使用します: {threshold}")

    selector = SelectFromModel(selection_model, threshold=selector_threshold)

    try:
        # 警告を抑制
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 物理特徴量を優先して選択するために重みを付ける
            # 温度変化の傾向と関連する特徴量
            temp_changes = y_train.diff().abs().fillna(0)
            weights = 1 + temp_changes / temp_changes.mean()

            # スパイクの影響を制限
            max_weight = 3.0  # 最大ウェイト値を制限
            weights = weights.clip(upper=max_weight)

            # モデルをトレーニング
            selector.fit(X_train, y_train, sample_weight=weights)

        # 選択された特徴量のマスクを取得
        feature_mask = selector.get_support()
        selected_feature_indices = [i for i, selected in enumerate(feature_mask) if selected]
        selected_features = [feature_names[i] for i in selected_feature_indices if i < len(feature_names)]

        # LAG特徴量の依存度を減らす追加処理
        # すべての選択された特徴量の中でLAG特徴量の割合を確認
        lag_features = [f for f in selected_features if 'lag' in f.lower()]
        if len(lag_features) > len(selected_features) * 0.25:  # LAG特徴量が25%を超える場合
            # LAG特徴量を減らし、物理ベースの特徴量を優先
            physical_features = [f for f in selected_features if any(
                key in f.lower() for key in ['diff', 'rate', 'accel', 'momentum', 'exchange', 'efficiency', 'amplitude']
            )]
            # 環境特徴量
            env_features = [f for f in selected_features if any(
                key in f.lower() for key in ['atmospheric', 'solar', 'radiation', 'outside']
            )]
            # 制御特徴量
            control_features = [f for f in selected_features if any(
                key in f.lower() for key in ['ac_', 'thermo_state', 'valid']
            )]
            # 時間特徴量
            time_features = [f for f in selected_features if any(
                key in f.lower() for key in ['hour', 'day', 'sin', 'cos']
            )]

            # LAG特徴量を減らして他の特徴量を優先
            max_lag_features = max(1, int(len(selected_features) * 0.15))  # 最大でも15%までに制限
            if len(lag_features) > max_lag_features:
                # 長期のLAGを優先して選択（短期LAGを除外）
                lag_periods = [int(f.split('_')[-1]) for f in lag_features if f.split('_')[-1].isdigit()]
                sorted_lag_features = [f for _, f in sorted(zip(lag_periods, lag_features), key=lambda x: x[0], reverse=True)]
                selected_lag_features = sorted_lag_features[:max_lag_features]

                # 最終的な特徴量リストを作成
                final_selected_features = physical_features + env_features + control_features + time_features + selected_lag_features
                final_selected_features = list(dict.fromkeys(final_selected_features))  # 重複を除去

                # 特徴量マスクを更新
                feature_mask = [f in final_selected_features for f in feature_names]
                selected_features = final_selected_features

                print(f"LAG特徴量の依存度を下げるため、特徴量を調整しました: {len(selected_features)}個")

        # 選択された特徴量でデータセットをフィルタリング
        # 選択された特徴量名が実際にデータフレームに存在することを確認
        existing_features = [f for f in selected_features if f in X_train.columns]
        if len(existing_features) < len(selected_features):
            print(f"警告: {len(selected_features) - len(existing_features)}個の選択特徴量がデータフレームに存在しません")
            selected_features = existing_features

        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        print(f"特徴量選択により{len(selected_features)}/{len(feature_names)}個の特徴量を選択しました")

        return X_train_selected, X_test_selected, selected_features

    except Exception as e:
        print(f"特徴量選択中にエラーが発生しました: {e}")
        # エラーの場合は重複のない特徴量のみを返す
        unique_cols = list(dict.fromkeys(X_train.columns))
        print(f"エラーのため特徴量選択をスキップし、{len(unique_cols)}個の重複のない特徴量を使用します")
        return X_train[unique_cols], X_test[unique_cols], unique_cols


def create_polynomial_features(X_train, X_test, base_features, degree=2):
    """
    多項式特徴量を生成する関数
    修正: 特徴量の存在確認を追加し、カラム名の問題を解決
    さらに修正: 重複特徴量名を防止する処理を追加

    Parameters:
    -----------
    X_train : DataFrame
        トレーニングデータ
    X_test : DataFrame
        テストデータ
    base_features : list
        多項式特徴量の基となる特徴量リスト
    degree : int
        多項式の次数

    Returns:
    --------
    X_train_poly, X_test_poly : 多項式特徴量を追加したデータフレーム
    poly_feature_names : 生成された多項式特徴量の名前リスト
    """
    # 警告を抑制
    import warnings

    # 実際に存在する特徴量のみをフィルタリング
    available_columns = X_train.columns.tolist()
    filtered_base_features = []

    for feature in base_features:
        # 特徴量が存在するか確認
        if feature in available_columns:
            filtered_base_features.append(feature)
        else:
            # 全角スペースなどの問題がある可能性を考慮
            # 部分一致でも確認
            matching_columns = [col for col in available_columns if feature.replace('\\u3000', '　').strip() in col.strip()]
            if matching_columns:
                filtered_base_features.append(matching_columns[0])
            else:
                print(f"警告: 特徴量 '{feature}' はデータセットに存在しないため、多項式特徴量生成から除外します")

    # 特徴量が存在しない場合は処理をスキップ
    if len(filtered_base_features) == 0:
        print("多項式特徴量の生成をスキップします: 有効な特徴量がありません")
        return X_train, X_test, []

    # 特徴量名が重複しないように、基底特徴量の一意性を確保
    filtered_base_features = list(dict.fromkeys(filtered_base_features))
    print(f"多項式特徴量生成のため、{len(filtered_base_features)}/{len(base_features)}個の有効な特徴量を使用します")

    # ベース特徴量のみを抽出
    X_train_base = X_train[filtered_base_features].copy()
    X_test_base = X_test[filtered_base_features].copy()

    # 多項式特徴量の生成
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)

    try:
        # 警告を抑制しながら多項式特徴量を生成
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # トレーニングデータに基づいて変換
            X_train_poly_array = poly.fit_transform(X_train_base)

            # テストデータには同じ変換を適用
            X_test_poly_array = poly.transform(X_test_base)

        # 特徴量名の生成
        feature_names = poly.get_feature_names_out(filtered_base_features)

        # 元の特徴量名を除外（1次の項は元データに含まれる）
        poly_feature_names = [name for name in feature_names if ' ' in name]

        # 重複する特徴量名をチェックして一意にする
        unique_poly_feature_names = []
        seen_names = set()

        for i, name in enumerate(poly_feature_names):
            if name in seen_names:
                # 重複する名前には連番を付ける
                base_name = name
                counter = 1
                new_name = f"{base_name}_{counter}"
                while new_name in seen_names:
                    counter += 1
                    new_name = f"{base_name}_{counter}"

                print(f"重複する特徴量名を検出: '{name}' → '{new_name}'に変更しました")
                unique_poly_feature_names.append(new_name)
                seen_names.add(new_name)
            else:
                unique_poly_feature_names.append(name)
                seen_names.add(name)

        # データフレームに変換して結合
        X_train_poly_df = pd.DataFrame(
            X_train_poly_array[:, len(filtered_base_features):],
            columns=unique_poly_feature_names,
            index=X_train.index
        )

        X_test_poly_df = pd.DataFrame(
            X_test_poly_array[:, len(filtered_base_features):],
            columns=unique_poly_feature_names,
            index=X_test.index
        )

        # 元のデータフレームと結合
        X_train_with_poly = pd.concat([X_train, X_train_poly_df], axis=1)
        X_test_with_poly = pd.concat([X_test, X_test_poly_df], axis=1)

        print(f"多項式特徴量を{len(unique_poly_feature_names)}個生成しました")

        return X_train_with_poly, X_test_with_poly, unique_poly_feature_names

    except Exception as e:
        print(f"多項式特徴量生成中にエラーが発生しました: {e}")
        # エラーの場合は元のデータフレームをそのまま返す
        return X_train, X_test, []
