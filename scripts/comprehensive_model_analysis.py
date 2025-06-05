#!/usr/bin/env python
# coding: utf-8

"""
包括的モデル分析スクリプト
- データリークチェック
- 時間軸整合性確認
- モデル性能の詳細分析
- 改善方針の提案
"""

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import os
from datetime import datetime, timedelta

# プロジェクトのルートディレクトリをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore')

# 設定のインポート
from src.config import SMOOTHING_WINDOWS, FEATURE_SELECTION_THRESHOLD, TEST_SIZE

# データ前処理関数のインポート
from src.data.preprocessing import (
    filter_temperature_outliers,
    create_temperature_difference_targets,
    prepare_time_features,
    get_time_based_train_test_split,
    filter_high_value_targets
)

# 特徴量エンジニアリング関数のインポート
from src.data.feature_engineering import create_difference_prediction_pipeline

# モデル訓練関数のインポート
from src.models.training import train_temperature_difference_model

# 評価関数のインポート
from src.models.evaluation import evaluate_temperature_difference_model


def check_data_leakage(df, zone, horizon, time_diff_seconds):
    """
    データリークの可能性をチェック

    Parameters:
    -----------
    df : DataFrame
        前処理済みデータ
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン
    time_diff_seconds : float
        データのサンプリング間隔

    Returns:
    --------
    dict
        リークチェック結果
    """
    print("\\n🔍 データリークチェック開始...")

    results = {
        'potential_leaks': [],
        'future_features': [],
        'temporal_consistency': {},
        'feature_timing': {}
    }

    # 1. 未来情報を含む可能性のある特徴量の検出
    future_keywords = ['future', 'lag', 'shift', 'lead']

    for col in df.columns:
        col_lower = col.lower()
        for keyword in future_keywords:
            if keyword in col_lower:
                if 'future' in col_lower:
                    results['future_features'].append(col)
                else:
                    results['potential_leaks'].append(col)

    # 2. 目的変数との時間的関係チェック
    target_col = f'temp_diff_{zone}_future_{horizon}'
    if target_col in df.columns:
        shift_steps = int(horizon * 60 / time_diff_seconds)

        # 基本温度カラムとの相関チェック
        base_temp_col = f'sens_temp_{zone}'
        if base_temp_col in df.columns:
            # 現在時点の温度と未来の差分の相関
            correlation = df[base_temp_col].corr(df[target_col])
            results['temporal_consistency']['current_temp_vs_future_diff'] = correlation

            # 未来の温度を逆算してチェック
            df_temp = df.copy()
            df_temp['future_temp_calculated'] = df_temp[base_temp_col] + df_temp[target_col]
            df_temp['future_temp_actual'] = df_temp[base_temp_col].shift(-shift_steps)

            # 計算された未来温度と実際の未来温度の一致度
            valid_mask = df_temp[['future_temp_calculated', 'future_temp_actual']].notna().all(axis=1)
            if valid_mask.sum() > 0:
                calc_vs_actual_corr = df_temp.loc[valid_mask, 'future_temp_calculated'].corr(
                    df_temp.loc[valid_mask, 'future_temp_actual']
                )
                results['temporal_consistency']['calculated_vs_actual_future'] = calc_vs_actual_corr

                # RMSE計算
                rmse = np.sqrt(np.mean((
                    df_temp.loc[valid_mask, 'future_temp_calculated'] -
                    df_temp.loc[valid_mask, 'future_temp_actual']
                )**2))
                results['temporal_consistency']['future_temp_rmse'] = rmse

    # 3. 特徴量のタイムスタンプ整合性チェック
    horizon_minutes = horizon
    expected_shift = pd.Timedelta(minutes=horizon_minutes)

    for col in results['future_features']:
        if f'future_{horizon}' in col:
            base_col = col.replace(f'_future_{horizon}', '')
            if base_col in df.columns:
                # 非ナン値のインデックス差分をチェック
                future_valid = df[col].notna()
                base_valid = df[base_col].notna()

                if future_valid.sum() > 0 and base_valid.sum() > 0:
                    # 最初と最後の有効データのタイムスタンプ差
                    future_first = df.index[future_valid][0] if future_valid.sum() > 0 else None
                    future_last = df.index[future_valid][-1] if future_valid.sum() > 0 else None
                    base_first = df.index[base_valid][0] if base_valid.sum() > 0 else None
                    base_last = df.index[base_valid][-1] if base_valid.sum() > 0 else None

                    results['feature_timing'][col] = {
                        'future_first': future_first,
                        'future_last': future_last,
                        'base_first': base_first,
                        'base_last': base_last,
                        'expected_shift_minutes': horizon_minutes
                    }

    print(f"✅ データリークチェック完了")
    print(f"   - 未来特徴量数: {len(results['future_features'])}")
    print(f"   - 潜在的リスク特徴量数: {len(results['potential_leaks'])}")

    return results


def analyze_model_performance(model, X_test, y_test, feature_names, current_temps=None):
    """
    モデル性能の詳細分析

    Parameters:
    -----------
    model : trained model
        学習済みモデル
    X_test : DataFrame
        テスト特徴量
    y_test : Series
        テスト目的変数
    feature_names : list
        特徴量名
    current_temps : Series, optional
        現在温度データ

    Returns:
    --------
    dict
        詳細分析結果
    """
    print("\\n📊 モデル性能詳細分析開始...")

    # 予測実行
    y_pred = model.predict(X_test)

    # 基本性能指標
    if current_temps is not None:
        metrics = evaluate_temperature_difference_model(y_test, y_pred, current_temps)
    else:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        metrics = {
            'diff_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'diff_mae': mean_absolute_error(y_test, y_pred),
            'diff_r2': r2_score(y_test, y_pred)
        }

    # 詳細分析
    analysis = {
        'basic_metrics': metrics,
        'feature_importance': {},
        'residual_analysis': {},
        'prediction_distribution': {},
        'temporal_patterns': {},
        'outlier_analysis': {}
    }

    # 1. 特徴量重要度分析
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        analysis['feature_importance'] = {
            'top_10': feature_importance_df.head(10).to_dict('records'),
            'total_features': len(feature_names),
            'importance_concentration': importances.max() / importances.sum(),
            'effective_features': (importances > 0.01).sum()
        }

    # 2. 残差分析
    residuals = y_test - y_pred
    analysis['residual_analysis'] = {
        'residual_mean': float(residuals.mean()),
        'residual_std': float(residuals.std()),
        'residual_skewness': float(residuals.skew()),
        'residual_kurtosis': float(residuals.kurtosis()),
        'residual_autocorr': float(residuals.autocorr()) if len(residuals) > 1 else 0.0
    }

    # 3. 予測分布分析
    analysis['prediction_distribution'] = {
        'pred_mean': float(y_pred.mean()),
        'pred_std': float(y_pred.std()),
        'actual_mean': float(y_test.mean()),
        'actual_std': float(y_test.std()),
        'pred_range': [float(y_pred.min()), float(y_pred.max())],
        'actual_range': [float(y_test.min()), float(y_test.max())]
    }

    # 4. 時間パターン分析
    if hasattr(y_test, 'index'):
        test_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred,
            'residual': residuals
        }, index=y_test.index)

        # 時間別性能
        test_df['hour'] = test_df.index.hour
        hourly_performance = test_df.groupby('hour')['residual'].agg(['mean', 'std']).abs()

        analysis['temporal_patterns'] = {
            'best_hour': int(hourly_performance['mean'].idxmin()),
            'worst_hour': int(hourly_performance['mean'].idxmax()),
            'hourly_variation': float(hourly_performance['mean'].std()),
            'time_dependency': float(test_df['residual'].autocorr())
        }

    # 5. 外れ値分析
    residual_threshold = 2 * residuals.std()
    outliers = residuals.abs() > residual_threshold

    analysis['outlier_analysis'] = {
        'outlier_count': int(outliers.sum()),
        'outlier_rate': float(outliers.mean()),
        'max_error': float(residuals.abs().max()),
        'outlier_threshold': float(residual_threshold)
    }

    print(f"✅ モデル性能分析完了")
    print(f"   - RMSE: {metrics.get('diff_rmse', metrics.get('restoration_rmse', 'N/A')):.4f}")
    print(f"   - 外れ値率: {analysis['outlier_analysis']['outlier_rate']:.1%}")
    print(f"   - 有効特徴量数: {analysis['feature_importance'].get('effective_features', 'N/A')}")

    return analysis


def generate_improvement_recommendations(leak_results, performance_analysis):
    """
    改善提案の生成

    Parameters:
    -----------
    leak_results : dict
        データリークチェック結果
    performance_analysis : dict
        性能分析結果

    Returns:
    --------
    dict
        改善提案
    """
    print("\\n💡 改善提案生成中...")

    recommendations = {
        'data_quality': [],
        'feature_engineering': [],
        'model_improvements': [],
        'validation_improvements': [],
        'priority_actions': []
    }

    # データ品質改善
    if leak_results['potential_leaks']:
        recommendations['data_quality'].append({
            'issue': 'Potential data leakage detected',
            'action': f"Review {len(leak_results['potential_leaks'])} features with leak risk",
            'priority': 'HIGH',
            'features': leak_results['potential_leaks'][:5]  # 最初の5個
        })

    # 時間整合性の問題
    future_temp_corr = leak_results['temporal_consistency'].get('calculated_vs_actual_future')
    if future_temp_corr and future_temp_corr < 0.95:
        recommendations['data_quality'].append({
            'issue': 'Temporal inconsistency detected',
            'action': f"Future temperature correlation is {future_temp_corr:.3f}, should be >0.95",
            'priority': 'HIGH'
        })

    # 特徴量エンジニアリング改善
    importance_analysis = performance_analysis['feature_importance']
    if importance_analysis.get('importance_concentration', 0) > 0.3:
        recommendations['feature_engineering'].append({
            'issue': 'High feature importance concentration',
            'action': 'Add more diverse features to reduce dependency on single feature',
            'priority': 'MEDIUM'
        })

    effective_features = importance_analysis.get('effective_features', 0)
    total_features = importance_analysis.get('total_features', 0)
    if total_features > 0 and effective_features / total_features < 0.3:
        recommendations['feature_engineering'].append({
            'issue': 'Low feature utilization',
            'action': f"Only {effective_features}/{total_features} features are effective",
            'priority': 'MEDIUM'
        })

    # モデル改善
    residual_analysis = performance_analysis['residual_analysis']
    if abs(residual_analysis['residual_skewness']) > 1.0:
        recommendations['model_improvements'].append({
            'issue': 'Skewed residuals',
            'action': 'Consider robust loss functions or data transformation',
            'priority': 'MEDIUM'
        })

    if residual_analysis['residual_autocorr'] > 0.1:
        recommendations['model_improvements'].append({
            'issue': 'Temporal correlation in residuals',
            'action': 'Add temporal features or consider sequence models',
            'priority': 'HIGH'
        })

    # 外れ値対策
    outlier_rate = performance_analysis['outlier_analysis']['outlier_rate']
    if outlier_rate > 0.05:  # 5%以上
        recommendations['model_improvements'].append({
            'issue': f'High outlier rate: {outlier_rate:.1%}',
            'action': 'Implement robust training or better outlier detection',
            'priority': 'MEDIUM'
        })

    # バリデーション改善
    temporal_patterns = performance_analysis.get('temporal_patterns', {})
    if temporal_patterns.get('hourly_variation', 0) > 0.1:
        recommendations['validation_improvements'].append({
            'issue': 'High hourly performance variation',
            'action': 'Implement time-stratified validation',
            'priority': 'MEDIUM'
        })

    # 優先アクション
    high_priority = [r for cat in recommendations.values()
                    for r in (cat if isinstance(cat, list) else [cat])
                    if isinstance(r, dict) and r.get('priority') == 'HIGH']

    recommendations['priority_actions'] = high_priority[:3]  # 最優先3つ

    print(f"✅ 改善提案生成完了")
    print(f"   - 高優先度アクション: {len(high_priority)}")
    print(f"   - 総提案数: {sum(len(v) for v in recommendations.values() if isinstance(v, list))}")

    return recommendations


def create_comprehensive_analysis_report(zone=1, horizon=15):
    """
    包括的分析レポートの作成

    Parameters:
    -----------
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン

    Returns:
    --------
    dict
        包括的分析結果
    """
    print("🔥 包括的モデル分析開始")
    print(f"対象: ゾーン{zone}, 予測ホライゾン{horizon}分")

    # データ読み込み
    data_path = project_root / "AllDayData.csv"
    print(f"\\nデータ読み込み中: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)

    # 時間列の確認と設定
    if 'time_stamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['time_stamp'])
        df = df.set_index('datetime')
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    else:
        raise ValueError("時間列が見つかりません。'time_stamp'または'datetime'列が必要です。")

    time_diff_seconds = df.index.to_series().diff().dropna().value_counts().index[0].total_seconds()

    # 基本前処理
    df = filter_temperature_outliers(df)
    df = prepare_time_features(df)

    # 差分予測用目的変数の作成
    df_with_diff_targets = create_temperature_difference_targets(
        df, [zone], [horizon], pd.Timedelta(seconds=time_diff_seconds)
    )

    # 1. データリークチェック
    leak_results = check_data_leakage(df_with_diff_targets, zone, horizon, time_diff_seconds)

    # 2. 特徴量エンジニアリング
    df_processed, selected_features, feature_info = create_difference_prediction_pipeline(
        df=df_with_diff_targets,
        zone_nums=[zone],
        horizons_minutes=[horizon],
        time_diff_seconds=time_diff_seconds,
        smoothing_window=SMOOTHING_WINDOWS[0] if SMOOTHING_WINDOWS else 5,
        feature_selection_threshold=FEATURE_SELECTION_THRESHOLD
    )

    # 3. データ準備
    diff_target_col = f'temp_diff_{zone}_future_{horizon}'
    feature_cols = [col for col in selected_features if col in df_processed.columns]
    valid_data = df_processed.dropna(subset=[diff_target_col] + feature_cols)

    # 高値フィルタリング（25%ile）
    abs_diff_col = f'abs_temp_diff_{zone}_future_{horizon}'
    valid_data[abs_diff_col] = valid_data[diff_target_col].abs()
    filtered_data, filter_info = filter_high_value_targets(
        valid_data, [abs_diff_col], percentile=25
    )

    # 時系列分割
    train_df, test_df = get_time_based_train_test_split(filtered_data, test_size=TEST_SIZE)

    X_train = train_df[feature_cols]
    y_train_diff = train_df[diff_target_col]
    X_test = test_df[feature_cols]
    y_test_diff = test_df[diff_target_col]
    current_temps_test = test_df[f'sens_temp_{zone}']

    # 4. モデル訓練
    model = train_temperature_difference_model(X_train, y_train_diff)

    # 5. 性能分析
    performance_analysis = analyze_model_performance(
        model, X_test, y_test_diff, feature_cols, current_temps_test
    )

    # 6. 改善提案生成
    recommendations = generate_improvement_recommendations(leak_results, performance_analysis)

    # 結果統合
    comprehensive_results = {
        'metadata': {
            'zone': zone,
            'horizon': horizon,
            'analysis_timestamp': datetime.now().isoformat(),
            'data_shape': {
                'original': df.shape,
                'processed': df_processed.shape,
                'filtered': filtered_data.shape,
                'train': X_train.shape,
                'test': X_test.shape
            }
        },
        'data_leak_analysis': leak_results,
        'performance_analysis': performance_analysis,
        'improvement_recommendations': recommendations,
        'filter_info': filter_info
    }

    # 結果保存
    output_dir = project_root / "Output"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"comprehensive_analysis_zone{zone}_horizon{horizon}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\\n結果を保存: {results_file}")

    return comprehensive_results


def print_analysis_summary(results):
    """
    分析結果のサマリー出力
    """
    print("\\n" + "="*80)
    print("🎯 包括的分析結果サマリー")
    print("="*80)

    # メタデータ
    metadata = results['metadata']
    print(f"📊 基本情報:")
    print(f"   ゾーン: {metadata['zone']}, ホライゾン: {metadata['horizon']}分")
    print(f"   データ形状: {metadata['data_shape']['original']} → {metadata['data_shape']['filtered']}")

    # 性能指標
    performance = results['performance_analysis']['basic_metrics']
    print(f"\\n📈 性能指標:")
    print(f"   RMSE: {performance.get('restoration_rmse', performance.get('diff_rmse', 'N/A')):.4f}℃")
    print(f"   R²: {performance.get('restoration_r2', performance.get('diff_r2', 'N/A')):.4f}")
    print(f"   方向精度: {performance.get('direction_accuracy', 'N/A'):.1f}%")

    # データリーク警告
    leak_analysis = results['data_leak_analysis']
    if leak_analysis['potential_leaks']:
        print(f"\\n⚠️  データリーク警告:")
        print(f"   潜在的リスク特徴量: {len(leak_analysis['potential_leaks'])}個")

    # 改善提案
    recommendations = results['improvement_recommendations']
    priority_actions = recommendations['priority_actions']
    if priority_actions:
        print(f"\\n🎯 優先改善アクション:")
        for i, action in enumerate(priority_actions, 1):
            print(f"   {i}. {action['issue']}")
            print(f"      → {action['action']}")

    print("\\n✅ 分析完了！詳細は保存されたJSONファイルをご確認ください。")


if __name__ == "__main__":
    # コマンドライン引数のパース
    zone = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 15

    # 包括的分析実行
    results = create_comprehensive_analysis_report(zone, horizon)

    # サマリー出力
    print_analysis_summary(results)

    print("\\n✅ 包括的分析が正常に完了しました")
