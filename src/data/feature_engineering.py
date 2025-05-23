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
    LightGBMを使った特徴量選択により、重要な特徴量のみを選択

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
