"""
特徴量選択と視覚化のための機能モジュール
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os
from ..config import DEFAULT_CONFIG

def select_features_with_multiple_methods(X, y, output_dir=None, zone=None, prediction_horizon=None):
    """
    複数の特徴量選択手法を適用し、結果を比較・視覚化する

    Args:
        X: 特徴量データフレーム
        y: 目的変数
        output_dir: 出力ディレクトリ（グラフ保存用）
        zone: ゾーン番号（グラフタイトル用）
        prediction_horizon: 予測ホライゾン（グラフタイトル用）

    Returns:
        selected_features_dict: 各手法で選択された特徴量の辞書
        feature_importance_df: 各手法での特徴量重要度を統合したデータフレーム
    """
    feature_names = X.columns.tolist()
    selected_features_dict = {}
    feature_importance_dict = {}

    # 1. LightGBM特徴量重要度
    print("LightGBMによる特徴量選択を実行中...")
    lgb_model = LGBMRegressor(**DEFAULT_CONFIG['LGBM_PARAMS'])
    lgb_model.fit(X, y)
    lgb_importance = lgb_model.feature_importances_

    lgb_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': lgb_importance
    }).sort_values('Importance', ascending=False)

    threshold = lgb_importance_df['Importance'].max() * DEFAULT_CONFIG['DEFAULT_IMPORTANCE_THRESHOLD']
    selected_features_lgb = lgb_importance_df[lgb_importance_df['Importance'] > threshold]['Feature'].tolist()

    selected_features_dict['LightGBM'] = selected_features_lgb
    feature_importance_dict['LightGBM'] = lgb_importance_df

    # 2. Random Forest特徴量重要度
    print("Random Forestによる特徴量選択を実行中...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=DEFAULT_CONFIG['DEFAULT_RANDOM_STATE'])
    rf_model.fit(X, y)
    rf_importance = rf_model.feature_importances_

    rf_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_importance
    }).sort_values('Importance', ascending=False)

    threshold = rf_importance_df['Importance'].max() * DEFAULT_CONFIG['DEFAULT_IMPORTANCE_THRESHOLD']
    selected_features_rf = rf_importance_df[rf_importance_df['Importance'] > threshold]['Feature'].tolist()

    selected_features_dict['RandomForest'] = selected_features_rf
    feature_importance_dict['RandomForest'] = rf_importance_df

    # 3. 相互情報量
    print("相互情報量による特徴量選択を実行中...")
    mi_scores = mutual_info_regression(X, y, random_state=DEFAULT_CONFIG['DEFAULT_RANDOM_STATE'])

    mi_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mi_scores
    }).sort_values('Importance', ascending=False)

    threshold = mi_importance_df['Importance'].max() * DEFAULT_CONFIG['DEFAULT_IMPORTANCE_THRESHOLD']
    selected_features_mi = mi_importance_df[mi_importance_df['Importance'] > threshold]['Feature'].tolist()

    selected_features_dict['MutualInfo'] = selected_features_mi
    feature_importance_dict['MutualInfo'] = mi_importance_df

    # 4. 再帰的特徴量除去（RFE）
    print("再帰的特徴量除去（RFE）を実行中...")
    n_features_to_select = min(DEFAULT_CONFIG['DEFAULT_MAX_FEATURES'], X.shape[1])
    rfe = RFE(
        estimator=RandomForestRegressor(n_estimators=100, random_state=DEFAULT_CONFIG['DEFAULT_RANDOM_STATE']),
        n_features_to_select=n_features_to_select,
        step=1
    )
    rfe.fit(X, y)

    rfe_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rfe.ranking_
    }).sort_values('Importance')  # ランキングなので昇順

    selected_features_rfe = rfe_importance_df[rfe_importance_df['Importance'] == 1]['Feature'].tolist()

    selected_features_dict['RFE'] = selected_features_rfe
    feature_importance_dict['RFE'] = rfe_importance_df

    # 特徴量選択結果をデータフレームに統合
    feature_importance_df = pd.DataFrame({'Feature': feature_names})

    for method, imp_df in feature_importance_dict.items():
        # RFEの場合はランキングを反転して重要度に変換
        if method == 'RFE':
            max_rank = imp_df['Importance'].max()
            feature_importance_df[f'{method}_importance'] = imp_df.set_index('Feature')['Importance'].map(lambda x: max_rank - x + 1)
        else:
            feature_importance_df[f'{method}_importance'] = imp_df.set_index('Feature')['Importance']

    # 出力ディレクトリが指定されている場合は可視化を行う
    if output_dir:
        visualize_feature_selection_results(feature_importance_dict, selected_features_dict, output_dir, zone, prediction_horizon)

    return selected_features_dict, feature_importance_df

def visualize_feature_selection_results(feature_importance_dict, selected_features_dict, output_dir, zone=None, prediction_horizon=None):
    """
    特徴量選択結果を視覚化する

    Args:
        feature_importance_dict: 各手法の特徴量重要度データフレーム
        selected_features_dict: 各手法で選択された特徴量
        output_dir: 出力ディレクトリ
        zone: ゾーン番号
        prediction_horizon: 予測ホライゾン

    Returns:
        paths: 保存された画像のパスリスト
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    # 1. 各手法の特徴量重要度のバープロット
    for method, imp_df in feature_importance_dict.items():
        plt.figure(figsize=(12, 8))

        # 上位10個の特徴量のみ表示
        top_features = imp_df.head(10)

        # RFEの場合はランキングを反転して表示
        if method == 'RFE':
            max_rank = imp_df['Importance'].max()
            importance_values = max_rank - top_features['Importance'] + 1
            plt.barh(top_features['Feature'], importance_values)
            plt.xlabel('Importance (Ranking Converted)')
        else:
            plt.barh(top_features['Feature'], top_features['Importance'])
            plt.xlabel('Importance')

        title = f'Top Features by {method}'
        if zone is not None and prediction_horizon is not None:
            title += f' (Zone {zone}, Horizon {prediction_horizon}min)'

        plt.title(title)
        plt.ylabel('Feature')
        plt.tight_layout()

        file_name = f'feature_importance_{method.lower()}'
        if zone is not None:
            file_name += f'_zone_{zone}'
        if prediction_horizon is not None:
            file_name += f'_horizon_{prediction_horizon}'
        file_name += '.png'

        path = os.path.join(output_dir, file_name)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        paths.append(path)

    # 2. 全手法の特徴量選択の一致度を視覚化（ベン図）
    plt.figure(figsize=(10, 8))

    from matplotlib_venn import venn3, venn2

    # 手法が3つ以下の場合はVenn図を描画
    methods = list(selected_features_dict.keys())
    if len(methods) == 2:
        set1 = set(selected_features_dict[methods[0]])
        set2 = set(selected_features_dict[methods[1]])
        venn2([set1, set2], [methods[0], methods[1]])
    elif len(methods) >= 3:
        # 上位3つの手法のみ使用
        set1 = set(selected_features_dict[methods[0]])
        set2 = set(selected_features_dict[methods[1]])
        set3 = set(selected_features_dict[methods[2]])
        venn3([set1, set2, set3], [methods[0], methods[1], methods[2]])

    title = 'Feature Selection Agreement Between Methods'
    if zone is not None and prediction_horizon is not None:
        title += f' (Zone {zone}, Horizon {prediction_horizon}min)'
    plt.title(title)

    file_name = 'feature_selection_venn'
    if zone is not None:
        file_name += f'_zone_{zone}'
    if prediction_horizon is not None:
        file_name += f'_horizon_{prediction_horizon}'
    file_name += '.png'

    path = os.path.join(output_dir, file_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    paths.append(path)

    # 3. 特徴量の相関ヒートマップ
    all_selected_features = set()
    for features in selected_features_dict.values():
        all_selected_features.update(features)

    # 全ての手法で選択された特徴量をデータフレームとして作成
    common_selected_features = []
    for feature in all_selected_features:
        methods_count = sum(1 for method_features in selected_features_dict.values() if feature in method_features)
        common_selected_features.append({
            'Feature': feature,
            'Selected_by_count': methods_count,
            'Selected_by': ', '.join(method for method in methods if feature in selected_features_dict[method])
        })

    common_df = pd.DataFrame(common_selected_features).sort_values('Selected_by_count', ascending=False)

    file_name = 'common_selected_features'
    if zone is not None:
        file_name += f'_zone_{zone}'
    if prediction_horizon is not None:
        file_name += f'_horizon_{prediction_horizon}'
    file_name += '.csv'

    path = os.path.join(output_dir, file_name)
    common_df.to_csv(path, index=False)

    return paths

def evaluate_feature_sets(X, y, selected_features_dict, test_size=0.2, random_state=42):
    """
    各特徴量セットの性能を評価

    Args:
        X: 特徴量データフレーム
        y: 目的変数
        selected_features_dict: 各手法で選択された特徴量
        test_size: テストデータの割合
        random_state: 乱数シード

    Returns:
        results_df: 各特徴量セットの評価結果
    """
    # テストデータとトレーニングデータの分割
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    results = []

    # ベースラインとして全特徴量を使った場合の評価
    lgb_model = LGBMRegressor(**DEFAULT_CONFIG['LGBM_PARAMS'])
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results.append({
        'Method': 'All Features',
        'Feature Count': X.shape[1],
        'RMSE': rmse,
        'Feature List': ', '.join(X.columns)
    })

    # 各手法で選択された特徴量セットの評価
    for method, selected_features in selected_features_dict.items():
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

        lgb_model = LGBMRegressor(**DEFAULT_CONFIG['LGBM_PARAMS'])
        lgb_model.fit(X_train_selected, y_train)
        y_pred = lgb_model.predict(X_test_selected)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'Method': method,
            'Feature Count': len(selected_features),
            'RMSE': rmse,
            'Feature List': ', '.join(selected_features)
        })

    # すべての手法で共通して選択された特徴量の評価
    common_features = set.intersection(*map(set, selected_features_dict.values()))
    if common_features:
        common_features = list(common_features)
        X_train_common = X_train[common_features]
        X_test_common = X_test[common_features]

        lgb_model = LGBMRegressor(**DEFAULT_CONFIG['LGBM_PARAMS'])
        lgb_model.fit(X_train_common, y_train)
        y_pred = lgb_model.predict(X_test_common)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            'Method': 'Common Features',
            'Feature Count': len(common_features),
            'RMSE': rmse,
            'Feature List': ', '.join(common_features)
        })

    results_df = pd.DataFrame(results)
    return results_df

def visualize_feature_evaluation(results_df, output_dir, zone=None, prediction_horizon=None):
    """
    特徴量セット評価結果を視覚化

    Args:
        results_df: 評価結果のデータフレーム
        output_dir: 出力ディレクトリ
        zone: ゾーン番号
        prediction_horizon: 予測ホライゾン

    Returns:
        path: 保存された画像のパス
    """
    plt.figure(figsize=(12, 8))

    # RMSE値のバープロット
    ax = sns.barplot(x='Method', y='RMSE', data=results_df)

    for i, row in enumerate(results_df.itertuples()):
        feature_count_idx = list(results_df.columns).index('Feature Count') + 1  # +1はインデックスのオフセット
        feature_count = getattr(row, f'_{feature_count_idx}')
        ax.text(i, row.RMSE + 0.01, f'{feature_count} features',
                ha='center', va='bottom', rotation=0, fontsize=10)

    title = 'Feature Selection Methods Comparison'
    if zone is not None and prediction_horizon is not None:
        title += f' (Zone {zone}, Horizon {prediction_horizon}min)'

    plt.title(title)
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    file_name = 'feature_selection_comparison'
    if zone is not None:
        file_name += f'_zone_{zone}'
    if prediction_horizon is not None:
        file_name += f'_horizon_{prediction_horizon}'
    file_name += '.png'

    path = os.path.join(output_dir, file_name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()

    return path
