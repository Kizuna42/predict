#!/usr/bin/env python
# coding: utf-8

"""
特徴量エンジニアリングモジュール
物理モデルベースの特徴量、サーモ状態、多項式特徴量を提供
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from scipy import ndimage


def create_physics_based_features(df, zone_nums):
    """
    物理法則に基づいた必要最小限の特徴量を作成する関数

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
    print("物理モデルベース特徴量を作成中...")
    df_copy = df.copy()
    created_features = []

    # 各ゾーンの基本的な物理特徴量
    for zone in zone_nums:
        if f'sens_temp_{zone}' in df.columns:
            # 温度変化率（一次微分）
            df_copy[f'temp_rate_{zone}'] = df_copy[f'sens_temp_{zone}'].diff()
            created_features.append(f'temp_rate_{zone}')

            # 設定温度との差（サーモ状態）
            if f'AC_set_{zone}' in df.columns:
                df_copy[f'temp_diff_to_setpoint_{zone}'] = df_copy[f'sens_temp_{zone}'] - df_copy[f'AC_set_{zone}']
                created_features.append(f'temp_diff_to_setpoint_{zone}')

                # 設定温度差の変化率（制御応答の指標）
                df_copy[f'setpoint_response_{zone}'] = df_copy[f'temp_diff_to_setpoint_{zone}'].diff()
                created_features.append(f'setpoint_response_{zone}')

            # 空調制御状態との交互作用
            if f'AC_valid_{zone}' in df.columns:
                df_copy[f'ac_temp_interaction_{zone}'] = df_copy[f'AC_valid_{zone}'] * df_copy[f'temp_rate_{zone}']
                created_features.append(f'ac_temp_interaction_{zone}')

    # 外気温関連特徴量
    atmos_cols = [col for col in df.columns if 'atmospheric' in col.lower() and 'temperature' in col.lower()]
    if atmos_cols:
        atmos_col = atmos_cols[0]
        # 外気温の変化率
        df_copy['atmos_temp_rate'] = df_copy[atmos_col].diff()
        created_features.append('atmos_temp_rate')

        # ゾーンごとの外気温との温度差
        for zone in zone_nums:
            if f'sens_temp_{zone}' in df.columns:
                df_copy[f'temp_diff_to_outside_{zone}'] = df_copy[f'sens_temp_{zone}'] - df_copy[atmos_col]
                created_features.append(f'temp_diff_to_outside_{zone}')

    # 日射量関連特徴量
    solar_cols = [col for col in df.columns if 'solar' in col.lower() and 'radiation' in col.lower()]
    if solar_cols:
        solar_col = solar_cols[0]
        # 日射量の変化率
        df_copy['solar_radiation_rate'] = df_copy[solar_col].diff()
        created_features.append('solar_radiation_rate')

    print(f"作成した物理特徴量: {len(created_features)}個")
    return df_copy, created_features


def create_future_explanatory_features(df, base_features_config, horizons_minutes, time_diff_seconds, is_prediction_mode=False):
    """
    未来の説明変数を作成する関数
    is_prediction_mode=Trueの場合は、プレースホルダーを作成（予測時用）

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    base_features_config : list of dict
        基本特徴量の設定
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
    print("制御可能パラメータと環境データの未来値を特徴量として作成中...")
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
                    # 予測時: プレースホルダーを作成（NaNで初期化）
                    df_copy[future_col] = np.nan

                created_features.append(future_col)

    print(f"作成した未来の説明変数: {len(created_features)}個")
    return df_copy, created_features


def create_thermo_state_features(df, zone_nums):
    """
    サーモ状態の特徴量を作成
    センサー温度とAC設定温度の差を計算

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

            print(f"ゾーン{zone}のサーモ状態特徴量を作成しました: {thermo_col} (ベース温度: {base_temp_col_for_thermo})")

    return df_copy, thermo_features


def select_important_features(X_train, y_train, X_test, feature_names, threshold='25%'):
    """
    LightGBMを使った特徴量選択により、重要な特徴量のみを選択（オリジナル版）

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
        feature_names = unique_feature_names

    # 列名に重複がないか確認（データフレームの列名も確認）
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: データフレームの列名に重複があります。重複を排除します。")
        unique_cols = list(dict.fromkeys(X_train.columns))
        X_train = X_train[unique_cols]
        X_test = X_test[unique_cols]
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
        importance_type='gain',
        verbose=-1
    )

    # SelectFromModelで重要な特徴量を選択
    selector_threshold = threshold
    if isinstance(threshold, str) and '%' in threshold:
        try:
            selector_threshold = float(threshold.strip('%')) / 100.0
            print(f"閾値を変換しました: {threshold} → {selector_threshold}")
        except ValueError:
            selector_threshold = 'mean'
            print(f"閾値の解析に失敗したため、'mean'を使用します: {threshold}")

    selector = SelectFromModel(selection_model, threshold=selector_threshold)

    try:
        # 警告を抑制
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # 物理特徴量を優先して選択するために重みを付ける
            temp_changes = y_train.diff().abs().fillna(0)
            weights = 1 + temp_changes / temp_changes.mean()

            # スパイクの影響を制限
            max_weight = 3.0
            weights = weights.clip(upper=max_weight)

            # モデルをトレーニング
            selector.fit(X_train, y_train, sample_weight=weights)

        # 選択された特徴量のマスクを取得
        feature_mask = selector.get_support()
        selected_feature_indices = [i for i, selected in enumerate(feature_mask) if selected]
        selected_features = [feature_names[i] for i in selected_feature_indices if i < len(feature_names)]

        # 選択された特徴量でデータセットをフィルタリング
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
    import warnings

    # 実際に存在する特徴量のみをフィルタリング
    available_columns = X_train.columns.tolist()
    filtered_base_features = []

    for feature in base_features:
        if feature in available_columns:
            filtered_base_features.append(feature)
        else:
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


def apply_smoothing_to_sensors(df, zone_nums, window_size=5):
    """
    温度・湿度センサーデータにノイズ対策の移動平均を適用
    未来の情報を含まないように注意して平滑化を行う

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    window_size : int
        移動平均のウィンドウサイズ（デフォルト: 5）

    Returns:
    --------
    DataFrame
        平滑化された特徴量を追加したデータフレーム
    list
        作成された平滑化特徴量のリスト
    """
    print(f"温度・湿度センサーデータの平滑化処理中（ウィンドウサイズ: {window_size}）...")
    df_copy = df.copy()
    smoothed_features = []

    # 各ゾーンの温度・湿度データを平滑化
    for zone in zone_nums:
        # 温度データの平滑化
        temp_col = f'sens_temp_{zone}'
        if temp_col in df_copy.columns:
            # 未来の情報を含まない右寄りの移動平均（過去のデータのみ使用）
            smoothed_col = f'{temp_col}_smoothed'
            df_copy[smoothed_col] = df_copy[temp_col].rolling(
                window=window_size,
                min_periods=1,
                center=False  # 右寄りの移動平均（未来の情報を含まない）
            ).mean()
            smoothed_features.append(smoothed_col)
            print(f"温度データを平滑化しました: {temp_col} → {smoothed_col}")

        # 湿度データの平滑化
        humid_col = f'sens_humid_{zone}'
        if humid_col in df_copy.columns:
            smoothed_col = f'{humid_col}_smoothed'
            df_copy[smoothed_col] = df_copy[humid_col].rolling(
                window=window_size,
                min_periods=1,
                center=False
            ).mean()
            smoothed_features.append(smoothed_col)
            print(f"湿度データを平滑化しました: {humid_col} → {smoothed_col}")

    # 外気温データの平滑化
    atmos_temp_cols = [col for col in df_copy.columns if 'atmospheric' in col.lower() and 'temperature' in col.lower()]
    for col in atmos_temp_cols:
        smoothed_col = f'{col}_smoothed'
        df_copy[smoothed_col] = df_copy[col].rolling(
            window=window_size,
            min_periods=1,
            center=False
        ).mean()
        smoothed_features.append(smoothed_col)
        print(f"外気温データを平滑化しました: {col} → {smoothed_col}")

    # 外気湿度データの平滑化
    atmos_humid_cols = [col for col in df_copy.columns if 'atmospheric' in col.lower() and 'humidity' in col.lower()]
    for col in atmos_humid_cols:
        smoothed_col = f'{col}_smoothed'
        df_copy[smoothed_col] = df_copy[col].rolling(
            window=window_size,
            min_periods=1,
            center=False
        ).mean()
        smoothed_features.append(smoothed_col)
        print(f"外気湿度データを平滑化しました: {col} → {smoothed_col}")

    print(f"平滑化特徴量を{len(smoothed_features)}個作成しました")
    return df_copy, smoothed_features


def create_important_features(df, zone_nums, horizons_minutes, time_diff_seconds, is_prediction_mode=False):
    """
    重要な特徴量を統合して作成する関数
    - 外気温・日射量
    - サーモ状態
    - 発停・モード
    - 平滑化された温度・湿度
    - 未来の制御パラメータ

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : int または float
        データのサンプリング間隔（秒）
    is_prediction_mode : bool
        予測モード時はTrueを指定

    Returns:
    --------
    DataFrame
        重要特徴量を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    print("重要特徴量の統合作成中...")
    df_copy = df.copy()
    all_features = []

    # 1. ノイズ対策：温度・湿度の平滑化
    df_copy, smoothed_features = apply_smoothing_to_sensors(df_copy, zone_nums)
    all_features.extend(smoothed_features)

    # 2. サーモ状態の作成
    df_copy, thermo_features = create_thermo_state_features(df_copy, zone_nums)
    all_features.extend(thermo_features)

    # 3. 基本的な環境・制御特徴量の収集
    important_feature_patterns = [
        # 外気温・日射量
        'atmospheric.*temperature',
        'solar.*radiation',

        # 発停・モード（AC_validやAC_modeなど）
        'AC_valid',
        'AC_mode',
        'AC_on',
        'AC_off',

        # 平滑化された温度・湿度（既に追加済み）
        # 'sens_temp.*_smoothed',
        # 'sens_humid.*_smoothed',

        # サーモ状態（既に追加済み）
        # 'thermo_state',
    ]

    base_features = []
    for pattern in important_feature_patterns:
        import re
        matching_cols = [col for col in df_copy.columns if re.search(pattern, col, re.IGNORECASE)]
        base_features.extend(matching_cols)

    # 重複排除
    base_features = list(set(base_features))
    all_features.extend(base_features)

    # 4. 未来の制御パラメータと環境データの作成
    future_features = []
    for horizon in horizons_minutes:
        shift_steps = int(horizon * 60 / time_diff_seconds)

        # 制御可能なパラメータの未来値
        controllable_prefixes = ['AC_valid', 'AC_mode', 'AC_set', 'thermo_state']
        for prefix in controllable_prefixes:
            matching_cols = [col for col in df_copy.columns if col.startswith(prefix)]
            for col in matching_cols:
                future_col = f"{col}_future_{horizon}min"
                if not is_prediction_mode:
                    df_copy[future_col] = df_copy[col].shift(-shift_steps)
                else:
                    df_copy[future_col] = np.nan
                future_features.append(future_col)

        # 環境データの未来値
        environmental_patterns = ['atmospheric.*temperature', 'solar.*radiation']
        for pattern in environmental_patterns:
            matching_cols = [col for col in df_copy.columns if re.search(pattern, col, re.IGNORECASE)]
            for col in matching_cols:
                future_col = f"{col}_future_{horizon}min"
                if not is_prediction_mode:
                    df_copy[future_col] = df_copy[col].shift(-shift_steps)
                else:
                    df_copy[future_col] = np.nan
                future_features.append(future_col)

    all_features.extend(future_features)

    # 5. AC_setやAC_tempの除外（サーモ状態に一本化したため）
    excluded_patterns = ['AC_set_[0-9]+$', 'AC_temp_[0-9]+$']
    for pattern in excluded_patterns:
        excluded_cols = [col for col in all_features if re.search(pattern, col)]
        for col in excluded_cols:
            if col in all_features:
                all_features.remove(col)
                print(f"除外した特徴量: {col} (サーモ状態に統合済み)")

    # 重複排除
    all_features = list(set(all_features))

    # 実際に存在する特徴量のみをフィルタリング
    existing_features = [f for f in all_features if f in df_copy.columns]

    print(f"重要特徴量を{len(existing_features)}個作成しました")
    print(f"  - 平滑化特徴量: {len(smoothed_features)}個")
    print(f"  - サーモ状態特徴量: {len(thermo_features)}個")
    print(f"  - 基本特徴量: {len(base_features)}個")
    print(f"  - 未来特徴量: {len(future_features)}個")

    return df_copy, existing_features


def select_important_features_enhanced(X_train, y_train, X_test, feature_names, threshold='25%', priority_patterns=None):
    """
    重要特徴量パターンを優先した特徴量選択関数

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
    priority_patterns : list of str
        優先する特徴量のパターンリスト

    Returns:
    --------
    X_train_selected, X_test_selected : 選択された特徴量のみを含むデータフレーム
    selected_features : 選択された特徴量名のリスト
    """
    import re

    if priority_patterns is None:
        priority_patterns = [
            'thermo_state',
            'atmospheric.*temperature',
            'solar.*radiation',
            'AC_valid',
            'AC_mode',
            'smoothed',
            'future.*min'
        ]

    print("重要特徴量パターンを優先した特徴量選択を開始...")

    # 重複排除
    unique_feature_names = list(dict.fromkeys(feature_names))
    if len(unique_feature_names) != len(feature_names):
        print(f"警告: 重複する特徴量名を{len(feature_names) - len(unique_feature_names)}個排除しました")
        feature_names = unique_feature_names

    # データフレームの列名重複チェック
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: データフレームの列名に重複があります。重複を排除します。")
        unique_cols = list(dict.fromkeys(X_train.columns))
        X_train = X_train[unique_cols]
        X_test = X_test[unique_cols]
        feature_names = [f for f in feature_names if f in unique_cols]

    # 1. 優先パターンにマッチする特徴量を強制選択
    priority_features = []
    for pattern in priority_patterns:
        matching_features = [f for f in feature_names if re.search(pattern, f, re.IGNORECASE)]
        priority_features.extend(matching_features)

    priority_features = list(set(priority_features))  # 重複排除
    existing_priority_features = [f for f in priority_features if f in X_train.columns]

    print(f"優先特徴量を{len(existing_priority_features)}個特定しました")

    # 2. 残りの特徴量から重要度ベースで選択
    remaining_features = [f for f in feature_names if f not in existing_priority_features and f in X_train.columns]

    if len(remaining_features) > 0:
        # 特徴量選択用モデル
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
            importance_type='gain',
            verbose=-1
        )

        # 閾値の処理
        selector_threshold = threshold
        if isinstance(threshold, str) and '%' in threshold:
            try:
                selector_threshold = float(threshold.strip('%')) / 100.0
            except ValueError:
                selector_threshold = 'mean'

        selector = SelectFromModel(selection_model, threshold=selector_threshold)

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 残りの特徴量でモデルを訓練
                X_remaining = X_train[remaining_features]

                # 重みの計算（温度変化の大きさに基づく）
                temp_changes = y_train.diff().abs().fillna(0)
                weights = 1 + temp_changes / (temp_changes.mean() + 1e-8)
                weights = weights.clip(upper=3.0)  # スパイクの影響を制限

                # 特徴量選択の実行
                selector.fit(X_remaining, y_train, sample_weight=weights)

            # 選択された特徴量
            feature_mask = selector.get_support()
            selected_remaining = [remaining_features[i] for i, selected in enumerate(feature_mask) if selected]

            print(f"重要度ベースで{len(selected_remaining)}個の特徴量を追加選択しました")

        except Exception as e:
            print(f"重要度ベース選択中にエラー: {e}")
            # エラーの場合は上位の特徴量を適当に選択
            max_remaining = min(len(remaining_features), 20)
            selected_remaining = remaining_features[:max_remaining]
            print(f"エラーのため上位{len(selected_remaining)}個の特徴量を使用します")
    else:
        selected_remaining = []

    # 3. 最終的な特徴量リストの作成
    final_features = existing_priority_features + selected_remaining
    final_features = [f for f in final_features if f in X_train.columns]  # 存在確認

    # 4. 結果の返却
    X_train_selected = X_train[final_features]
    X_test_selected = X_test[final_features]

    print(f"最終選択特徴量: {len(final_features)}個")
    print(f"  - 優先特徴量: {len(existing_priority_features)}個")
    print(f"  - 重要度ベース: {len(selected_remaining)}個")

    return X_train_selected, X_test_selected, final_features


def create_optimized_features_pipeline(df, zone_nums, horizons_minutes, time_diff_seconds,
                                      is_prediction_mode=False, use_enhanced_selection=True,
                                      smoothing_window=5, feature_selection_threshold='25%'):
    """
    最適化された特徴量作成パイプライン
    要求された処理を統合して実行

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : int または float
        データのサンプリング間隔（秒）
    is_prediction_mode : bool
        予測モード時はTrueを指定
    use_enhanced_selection : bool
        改良された特徴量選択を使用するかどうか
    smoothing_window : int
        移動平均のウィンドウサイズ
    feature_selection_threshold : str or float
        特徴量選択の閾値

    Returns:
    --------
    df_processed : DataFrame
        処理済みデータフレーム
    selected_features : list
        選択された特徴量名のリスト
    feature_info : dict
        作成された特徴量の詳細情報
    """
    print("=== 最適化された特徴量エンジニアリングパイプライン開始 ===")

    # 統合特徴量作成
    df_processed, all_created_features = create_important_features(
        df, zone_nums, horizons_minutes, time_diff_seconds, is_prediction_mode
    )

    # 物理ベース特徴量も追加
    df_processed, physics_features = create_physics_based_features(df_processed, zone_nums)
    all_created_features.extend(physics_features)

    # 重複排除
    all_created_features = list(set(all_created_features))

    # 実際に存在する特徴量のみをフィルタリング
    existing_features = [f for f in all_created_features if f in df_processed.columns]

    print(f"作成された特徴量総数: {len(existing_features)}個")

    # 特徴量の詳細情報
    feature_info = {
        'total_features': len(existing_features),
        'smoothed_features': [f for f in existing_features if 'smoothed' in f],
        'thermo_features': [f for f in existing_features if 'thermo_state' in f],
        'future_features': [f for f in existing_features if 'future' in f],
        'physics_features': [f for f in existing_features if any(p in f for p in ['rate', 'diff', 'interaction'])],
        'environmental_features': [f for f in existing_features if any(p in f for p in ['atmospheric', 'solar', 'radiation'])]
    }

    print("特徴量カテゴリ別の詳細:")
    for category, features in feature_info.items():
        if category != 'total_features':
            print(f"  - {category}: {len(features)}個")

    return df_processed, existing_features, feature_info


def create_difference_prediction_features(df, zone_nums, horizons_minutes, time_diff_seconds):
    """
    差分予測に特化した特徴量を作成する関数

    温度変化量の予測に有効な特徴量を生成：
    - 温度変化率の履歴
    - 空調制御の変化パターン
    - 外部環境の変化率
    - 時系列パターン特徴量

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : int または float
        データのサンプリング間隔（秒）

    Returns:
    --------
    DataFrame
        差分予測特化特徴量を追加したデータフレーム
    list
        作成された特徴量のリスト
    """
    print("🔥 差分予測特化特徴量を作成中...")
    df_copy = df.copy()
    diff_features = []

    # 各ゾーンの差分予測特化特徴量
    for zone in zone_nums:
        if f'sens_temp_{zone}' in df.columns:
            # 1. 温度変化率の履歴（複数時間窓）
            temp_col = f'sens_temp_{zone}'

            # 短期変化率（5分、10分、15分前との差）
            for minutes_back in [5, 10, 15]:
                shift_steps = int(minutes_back * 60 / time_diff_seconds)
                change_col = f'temp_change_{zone}_{minutes_back}min'
                df_copy[change_col] = df_copy[temp_col] - df_copy[temp_col].shift(shift_steps)
                diff_features.append(change_col)

            # 変化率の変化率（加速度的変化）
            temp_rate = df_copy[temp_col].diff()
            df_copy[f'temp_acceleration_{zone}'] = temp_rate.diff()
            diff_features.append(f'temp_acceleration_{zone}')

            # 変化方向の持続性（連続する変化方向の回数）
            temp_direction = np.sign(temp_rate)
            direction_persistence = temp_direction.groupby((temp_direction != temp_direction.shift()).cumsum()).cumcount() + 1
            df_copy[f'temp_direction_persistence_{zone}'] = direction_persistence
            diff_features.append(f'temp_direction_persistence_{zone}')

            # 2. 空調制御の変化パターン
            if f'AC_valid_{zone}' in df.columns:
                # 空調状態の変化
                ac_change = df_copy[f'AC_valid_{zone}'].diff()
                df_copy[f'ac_state_change_{zone}'] = ac_change
                diff_features.append(f'ac_state_change_{zone}')

                # 空調ON/OFF後の経過時間
                ac_on_periods = (df_copy[f'AC_valid_{zone}'] == 1).astype(int)
                ac_off_periods = (df_copy[f'AC_valid_{zone}'] == 0).astype(int)

                # ON状態の継続時間
                ac_on_duration = ac_on_periods.groupby((ac_on_periods != ac_on_periods.shift()).cumsum()).cumcount() + 1
                ac_on_duration = ac_on_duration * ac_on_periods  # OFF時は0にリセット
                df_copy[f'ac_on_duration_{zone}'] = ac_on_duration
                diff_features.append(f'ac_on_duration_{zone}')

                # OFF状態の継続時間
                ac_off_duration = ac_off_periods.groupby((ac_off_periods != ac_off_periods.shift()).cumsum()).cumcount() + 1
                ac_off_duration = ac_off_duration * ac_off_periods  # ON時は0にリセット
                df_copy[f'ac_off_duration_{zone}'] = ac_off_duration
                diff_features.append(f'ac_off_duration_{zone}')

            if f'AC_set_{zone}' in df.columns:
                # 設定温度の変化
                setpoint_change = df_copy[f'AC_set_{zone}'].diff()
                df_copy[f'setpoint_change_{zone}'] = setpoint_change
                diff_features.append(f'setpoint_change_{zone}')

                # 設定温度変化後の経過時間
                setpoint_changed = (setpoint_change != 0).astype(int)
                time_since_setpoint_change = setpoint_changed.groupby((setpoint_changed == 1).cumsum()).cumcount()
                df_copy[f'time_since_setpoint_change_{zone}'] = time_since_setpoint_change
                diff_features.append(f'time_since_setpoint_change_{zone}')

    # 3. 外部環境の変化率特徴量
    # 外気温の変化率
    atmos_cols = [col for col in df.columns if 'atmospheric' in col.lower() and 'temperature' in col.lower()]
    if atmos_cols:
        atmos_col = atmos_cols[0]

        # 外気温の変化率（複数時間窓）
        for minutes_back in [5, 10, 15, 30]:
            shift_steps = int(minutes_back * 60 / time_diff_seconds)
            change_col = f'atmos_temp_change_{minutes_back}min'
            df_copy[change_col] = df_copy[atmos_col] - df_copy[atmos_col].shift(shift_steps)
            diff_features.append(change_col)

        # 外気温変化の加速度
        atmos_rate = df_copy[atmos_col].diff()
        df_copy['atmos_temp_acceleration'] = atmos_rate.diff()
        diff_features.append('atmos_temp_acceleration')

    # 日射量の変化率
    solar_cols = [col for col in df.columns if 'solar' in col.lower() and 'radiation' in col.lower()]
    if solar_cols:
        solar_col = solar_cols[0]

        # 日射量の変化率
        for minutes_back in [5, 10, 15]:
            shift_steps = int(minutes_back * 60 / time_diff_seconds)
            change_col = f'solar_change_{minutes_back}min'
            df_copy[change_col] = df_copy[solar_col] - df_copy[solar_col].shift(shift_steps)
            diff_features.append(change_col)

    # 4. 時系列パターン特徴量
    # 時間帯別の変化パターン
    hour = df_copy.index.hour

    # 朝の昇温期（6-10時）、昼の安定期（10-16時）、夕方の降温期（16-20時）、夜の安定期（20-6時）
    df_copy['is_morning_heating'] = ((hour >= 6) & (hour < 10)).astype(int)
    df_copy['is_daytime_stable'] = ((hour >= 10) & (hour < 16)).astype(int)
    df_copy['is_evening_cooling'] = ((hour >= 16) & (hour < 20)).astype(int)
    df_copy['is_night_stable'] = ((hour >= 20) | (hour < 6)).astype(int)

    diff_features.extend(['is_morning_heating', 'is_daytime_stable', 'is_evening_cooling', 'is_night_stable'])

    # 5. ゾーン間の相互作用特徴量（複数ゾーンがある場合）
    if len(zone_nums) > 1:
        for i, zone1 in enumerate(zone_nums):
            for zone2 in zone_nums[i+1:]:
                if f'sens_temp_{zone1}' in df.columns and f'sens_temp_{zone2}' in df.columns:
                    # ゾーン間温度差
                    inter_zone_diff = f'temp_diff_zone_{zone1}_to_{zone2}'
                    df_copy[inter_zone_diff] = df_copy[f'sens_temp_{zone1}'] - df_copy[f'sens_temp_{zone2}']
                    diff_features.append(inter_zone_diff)

                    # ゾーン間温度差の変化率
                    inter_zone_diff_rate = f'temp_diff_rate_zone_{zone1}_to_{zone2}'
                    df_copy[inter_zone_diff_rate] = df_copy[inter_zone_diff].diff()
                    diff_features.append(inter_zone_diff_rate)

    print(f"✅ 差分予測特化特徴量を{len(diff_features)}個作成しました")

    return df_copy, diff_features


def create_difference_prediction_pipeline(df, zone_nums, horizons_minutes, time_diff_seconds,
                                        smoothing_window=5, feature_selection_threshold='25%'):
    """
    差分予測専用の最適化された特徴量エンジニアリングパイプライン

    差分予測に特化した特徴量を重点的に作成し、高精度な温度変化量予測を実現する。

    Parameters:
    -----------
    df : DataFrame
        時系列インデックスを持つデータフレーム
    zone_nums : list
        ゾーン番号のリスト
    horizons_minutes : list
        予測ホライゾン（分）のリスト
    time_diff_seconds : int または float
        データのサンプリング間隔（秒）
    smoothing_window : int
        平滑化ウィンドウサイズ
    feature_selection_threshold : str
        特徴量選択の閾値

    Returns:
    --------
    DataFrame
        処理済みデータフレーム
    list
        選択された特徴量のリスト
    dict
        特徴量情報
    """
    print("🔥 差分予測専用パイプライン開始...")

    df_processed = df.copy()
    all_features = []
    feature_info = {
        'difference_specific_features': [],
        'smoothed_features': [],
        'physics_features': [],
        'future_features': [],
        'selected_features': [],
        'total_features_created': 0,
        'total_features_selected': 0
    }

    # 1. 差分予測特化特徴量の作成（最優先）
    df_processed, diff_features = create_difference_prediction_features(
        df_processed, zone_nums, horizons_minutes, time_diff_seconds
    )
    all_features.extend(diff_features)
    feature_info['difference_specific_features'] = diff_features
    print(f"差分予測特化特徴量: {len(diff_features)}個")

    # 2. 平滑化特徴量（差分予測でも重要）
    df_processed, smoothed_features = apply_smoothing_to_sensors(
        df_processed, zone_nums, window_size=smoothing_window
    )
    all_features.extend(smoothed_features)
    feature_info['smoothed_features'] = smoothed_features
    print(f"平滑化特徴量: {len(smoothed_features)}個")

    # 3. サーモ状態特徴量（制御応答の理解に重要）
    df_processed, thermo_features = create_thermo_state_features(df_processed, zone_nums)
    all_features.extend(thermo_features)
    feature_info['thermo_features'] = thermo_features
    print(f"サーモ状態特徴量: {len(thermo_features)}個")

    # 4. 物理モデルベース特徴量（基本的な物理法則）
    df_processed, physics_features = create_physics_based_features(df_processed, zone_nums)
    all_features.extend(physics_features)
    feature_info['physics_features'] = physics_features
    print(f"物理モデル特徴量: {len(physics_features)}個")

    # 5. 重要な基本特徴量の選択
    important_features_config = [
        {'name': 'atmospheric　temperature', 'type': 'environmental'},
        {'name': 'solar radiation', 'type': 'environmental'},
        {'name': 'hour', 'type': 'temporal'},
        {'name': 'hour_sin', 'type': 'temporal'},
        {'name': 'hour_cos', 'type': 'temporal'}
    ]

    # ゾーン関連の基本特徴量
    for zone in zone_nums:
        zone_features = [
            {'name': f'sens_temp_{zone}', 'type': 'sensor'},
            {'name': f'sens_humid_{zone}', 'type': 'sensor'},
            {'name': f'AC_valid_{zone}', 'type': 'control'},
            {'name': f'AC_set_{zone}', 'type': 'control'},
            {'name': f'AC_mode_{zone}', 'type': 'control'}
        ]
        important_features_config.extend(zone_features)

    # 基本特徴量の追加
    basic_features = []
    for config in important_features_config:
        if config['name'] in df_processed.columns:
            basic_features.append(config['name'])

    all_features.extend(basic_features)
    feature_info['basic_features'] = basic_features
    print(f"基本特徴量: {len(basic_features)}個")

    # 6. 未来の制御・環境特徴量（差分予測でも有効）
    df_processed, future_features = create_future_explanatory_features(
        df_processed, important_features_config, horizons_minutes, time_diff_seconds
    )
    all_features.extend(future_features)
    feature_info['future_features'] = future_features
    print(f"未来特徴量: {len(future_features)}個")

    # 7. 特徴量の重複除去
    unique_features = list(dict.fromkeys(all_features))  # 順序を保持しつつ重複除去
    available_features = [f for f in unique_features if f in df_processed.columns]

    feature_info['total_features_created'] = len(available_features)
    print(f"作成された特徴量総数: {len(available_features)}個")

    # 8. 差分予測に特化した特徴量選択
    # 差分予測特化特徴量は優先的に保持
    priority_patterns = [
        'temp_change_', 'temp_acceleration_', 'temp_direction_persistence_',
        'ac_state_change_', 'ac_on_duration_', 'ac_off_duration_',
        'setpoint_change_', 'time_since_setpoint_change_',
        'atmos_temp_change_', 'atmos_temp_acceleration',
        'solar_change_', 'is_morning_heating', 'is_daytime_stable',
        'is_evening_cooling', 'is_night_stable',
        'temp_diff_zone_', 'temp_diff_rate_zone_'
    ]

    # 差分予測用の目的変数を仮作成（特徴量選択用）
    temp_target_cols = []
    for zone in zone_nums:
        for horizon in horizons_minutes:
            target_col = f'temp_diff_{zone}_future_{horizon}'
            if target_col in df_processed.columns:
                temp_target_cols.append(target_col)
                break  # 最初の有効な目的変数のみ使用
        if temp_target_cols:
            break

    if temp_target_cols and len(available_features) > 50:
        print(f"🎯 差分予測特化特徴量選択を実行中...")

        # 有効なデータのみで特徴量選択
        target_col = temp_target_cols[0]
        valid_data = df_processed.dropna(subset=[target_col] + available_features[:50])  # 最初の50特徴量で試行

        if len(valid_data) > 100:
            X_temp = valid_data[available_features[:50]]
            y_temp = valid_data[target_col]

            try:
                # 差分予測特化の特徴量選択
                selected_features = select_important_features_enhanced(
                    X_temp, X_temp, y_temp, available_features[:50],
                    threshold=feature_selection_threshold,
                    priority_patterns=priority_patterns
                )

                # 差分予測特化特徴量を追加で保持
                for pattern in priority_patterns:
                    pattern_features = [f for f in available_features if pattern in f and f not in selected_features]
                    selected_features.extend(pattern_features[:3])  # 各パターンから最大3個

                # 重複除去
                selected_features = list(dict.fromkeys(selected_features))

            except Exception as e:
                print(f"特徴量選択エラー: {e}")
                selected_features = available_features[:100]  # フォールバック
        else:
            selected_features = available_features
    else:
        selected_features = available_features

    feature_info['selected_features'] = selected_features
    feature_info['total_features_selected'] = len(selected_features)

    print(f"✅ 差分予測専用パイプライン完了:")
    print(f"  - 差分特化特徴量: {len(diff_features)}個")
    print(f"  - 平滑化特徴量: {len(smoothed_features)}個")
    print(f"  - 物理特徴量: {len(physics_features)}個")
    print(f"  - 基本特徴量: {len(basic_features)}個")
    print(f"  - 未来特徴量: {len(future_features)}個")
    print(f"  - 最終選択特徴量: {len(selected_features)}個")

    return df_processed, selected_features, feature_info


"""
使用例:

# 基本的な使用法
df_processed, selected_features, feature_info = create_optimized_features_pipeline(
    df=data_df,
    zone_nums=[1, 2, 3],
    horizons_minutes=[10, 15],
    time_diff_seconds=300,  # 5分間隔
    is_prediction_mode=False,  # 学習時
    smoothing_window=5,
    feature_selection_threshold='25%'
)

# 特徴量選択の実行（改良版を使用）
X_train_selected, X_test_selected, final_features = select_important_features_enhanced(
    X_train=X_train[selected_features],
    y_train=y_train,
    X_test=X_test[selected_features],
    feature_names=selected_features,
    threshold='25%'
)

# 処理内容:
# 1. 温度・湿度のノイズ対策（移動平均）
# 2. サーモ状態の作成（AC_setとAC_tempを統合）
# 3. 未来の制御パラメータ・環境データの作成
# 4. 重要特徴量の優先選択
# 5. AC_setやAC_tempの除外（サーモ状態に一本化）

重要特徴量パターン:
- 外気温: atmospheric.*temperature
- 日射量: solar.*radiation
- サーモ状態: thermo_state
- 発停・モード: AC_valid, AC_mode
- 平滑化データ: *_smoothed
- 未来情報: *_future_*min
"""
