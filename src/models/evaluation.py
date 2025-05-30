#!/usr/bin/env python
# coding: utf-8

"""
モデル評価モジュール
予測モデルの評価指標の計算や評価結果の分析関数を提供
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def calculate_metrics(y_true, y_pred):
    """
    回帰モデルの評価指標を計算する関数

    Parameters:
    -----------
    y_true : Series
        実際の値
    y_pred : Series or array
        予測値

    Returns:
    --------
    dict
        各種評価指標を含む辞書
    """
    # NaN値の処理：両方のデータで対応する位置にNaNがないデータのみを使用
    valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        print("警告: 有効なデータがありません。すべての値がNaNです。")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'mape': float('nan'),
            'r2': float('nan')
        }

    # NaN値を除外したデータで評価指標を計算
    return {
        'rmse': np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)),
        'mae': mean_absolute_error(y_true_valid, y_pred_valid),
        'mape': mean_absolute_percentage_error(y_true_valid, y_pred_valid) * 100,  # パーセント表示
        'r2': r2_score(y_true_valid, y_pred_valid)
    }


def print_metrics(metrics, zone=None, horizon=None):
    """
    評価指標を整形して表示する関数

    Parameters:
    -----------
    metrics : dict
        評価指標の辞書
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    header = "評価指標"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")


def print_lag_dependency_warning(lag_dependency, threshold=30.0, zone=None, horizon=None):
    """
    LAG特徴量への依存度が高い場合に警告を表示する関数

    Parameters:
    -----------
    lag_dependency : dict
        LAG依存度分析の結果辞書
    threshold : float, optional
        警告を表示するLAG依存度の閾値（パーセント）
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    # 総LAG依存度の計算（lag_temp_percentとrolling_temp_percentの合計）
    total_lag_dependency = lag_dependency['lag_temp_percent'] + lag_dependency['rolling_temp_percent']

    # 現在のセンサー値への依存度（過去データの代替指標として使用）
    current_temp_dependency = lag_dependency.get('current_temp_percent', 0)
    future_temp_dependency = lag_dependency.get('future_temp_percent', 0)

    header = "LAG特徴量依存度分析"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"直接的LAG特徴量依存度: {lag_dependency['lag_temp_percent']:.2f}%")
    print(f"移動平均特徴量依存度: {lag_dependency['rolling_temp_percent']:.2f}%")
    print(f"総LAG依存度: {total_lag_dependency:.2f}%")
    print(f"現在センサー値依存度: {current_temp_dependency:.2f}%")
    print(f"未来制御情報依存度: {future_temp_dependency:.2f}%")

    # 依存度が閾値を超える場合に警告を表示
    if total_lag_dependency > threshold:
        print(f"\n⚠️ 警告: LAG特徴量への依存度が高すぎます ({total_lag_dependency:.2f}% > {threshold:.2f}%)")
        print("  モデルが過去の温度値に過度に依存している可能性があります。")
        print("  以下の対策を検討してください:")
        print("  - より物理的な意味を持つ特徴量を追加")
        print("  - LAG特徴量の重みを下げる")
        print("  - より長期のLAGのみを使用")
    elif current_temp_dependency > threshold * 1.5:
        print(f"\n⚠️ 注意: 現在センサー値への依存度が高めです ({current_temp_dependency:.2f}%)")
        print("  モデルが現在のデータに依存している可能性があります。")
    else:
        print(f"\n✅ LAG依存度は許容範囲内です ({total_lag_dependency:.2f}% <= {threshold:.2f}%)")
        print("  モデルは適切に物理特徴量や未来の説明変数を活用しています。")


def analyze_feature_importance(model, feature_names, top_n=15):
    """
    モデルの特徴量重要度を分析する関数

    Parameters:
    -----------
    model : LGBMRegressor or similar model
        評価対象のモデル（feature_importances_属性を持つモデル）
    feature_names : list
        特徴量名のリスト
    top_n : int, optional
        表示する上位特徴量の数

    Returns:
    --------
    DataFrame
        特徴量重要度を降順にソートしたデータフレーム
    """
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    # 重要度順にソート
    sorted_importance = feature_importance.sort_values('importance', ascending=False)

    # 上位n個の特徴量を表示
    top_features = sorted_importance.head(top_n)
    print(f"\n上位{top_n}個の重要な特徴量:")
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")

    return sorted_importance


# この関数は src/diagnostics/lag_analysis.py に移動されました
# 重複を避けるため、ここでは削除されています


def calculate_physical_validity_metrics(y_true, y_pred, ac_state, ac_mode, horizon):
    """
    予測の物理的妥当性を評価する指標を計算する関数

    Parameters:
    -----------
    y_true : Series or array-like
        実測値
    y_pred : Series or array-like
        予測値
    ac_state : Series or array-like
        空調の状態（0: OFF, 1: ON）
    ac_mode : Series or array-like
        空調のモード（0: 冷房, 1: 暖房）
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    dict
        物理的妥当性の評価指標
    """
    # 温度変化量の計算
    temp_change_true = y_true.diff()
    temp_change_pred = y_pred.diff()

    # 物理的妥当性の評価
    validity_metrics = {
        'cooling_validity': 0.0,  # 冷房時の妥当性
        'heating_validity': 0.0,  # 暖房時の妥当性
        'natural_validity': 0.0,  # 空調OFF時の妥当性
        'direction_accuracy': 0.0,  # 温度変化方向の一致率
        'response_delay': 0.0,  # 応答遅れの評価
    }

    # 冷房時の妥当性評価
    cooling_mask = (ac_state == 1) & (ac_mode == 0)
    if cooling_mask.sum() > 0:
        cooling_valid = (temp_change_pred[cooling_mask] < 0).mean()
        validity_metrics['cooling_validity'] = cooling_valid

    # 暖房時の妥当性評価
    heating_mask = (ac_state == 1) & (ac_mode == 1)
    if heating_mask.sum() > 0:
        heating_valid = (temp_change_pred[heating_mask] > 0).mean()
        validity_metrics['heating_validity'] = heating_valid

    # 空調OFF時の妥当性評価
    off_mask = (ac_state == 0)
    if off_mask.sum() > 0:
        # 外気温との関係を考慮した自然温度変化の妥当性
        natural_valid = (temp_change_pred[off_mask] * temp_change_true[off_mask] > 0).mean()
        validity_metrics['natural_validity'] = natural_valid

    # 温度変化方向の一致率
    direction_accuracy = (temp_change_pred * temp_change_true > 0).mean()
    validity_metrics['direction_accuracy'] = direction_accuracy

    # 応答遅れの評価
    # 空調状態変化後の温度変化の遅れを評価
    ac_state_change = ac_state.diff().abs()
    if ac_state_change.sum() > 0:
        # 状態変化後の温度変化の遅れを計算
        response_delays = []
        for i in range(1, len(ac_state_change)):
            if ac_state_change[i] == 1:
                # 状態変化後の温度変化を観察
                true_change = temp_change_true[i:i+horizon].sum()
                pred_change = temp_change_pred[i:i+horizon].sum()
                if true_change != 0:
                    delay = abs(pred_change - true_change) / abs(true_change)
                    response_delays.append(delay)

        if response_delays:
            validity_metrics['response_delay'] = np.mean(response_delays)

    return validity_metrics


def print_physical_validity_metrics(metrics, zone=None, horizon=None):
    """
    物理的妥当性の評価指標を整形して表示する関数

    Parameters:
    -----------
    metrics : dict
        物理的妥当性の評価指標
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    header = "物理的妥当性の評価"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print(f"冷房時の妥当性: {metrics['cooling_validity']:.4f}")
    print(f"暖房時の妥当性: {metrics['heating_validity']:.4f}")
    print(f"自然温度変化の妥当性: {metrics['natural_validity']:.4f}")
    print(f"温度変化方向の一致率: {metrics['direction_accuracy']:.4f}")
    print(f"応答遅れの評価: {metrics['response_delay']:.4f}")


def evaluate_temperature_difference_model(y_true_diff, y_pred_diff, current_temps=None):
    """
    温度差分予測モデル専用の評価関数

    差分予測の特性に合わせた評価指標を計算する：
    - 変化量の予測精度
    - 変化方向の一致率
    - 小さな変化への感度
    - 温度復元精度（現在温度が提供された場合）

    Parameters:
    -----------
    y_true_diff : Series or array
        実際の温度差分値
    y_pred_diff : Series or array
        予測された温度差分値
    current_temps : Series or array, optional
        現在の温度（温度復元評価用）

    Returns:
    --------
    dict
        差分予測専用の評価指標
    """
    # NaN値の処理
    valid_indices = ~(pd.isna(y_true_diff) | pd.isna(y_pred_diff))
    y_true_valid = y_true_diff[valid_indices]
    y_pred_valid = y_pred_diff[valid_indices]

    if len(y_true_valid) == 0:
        print("警告: 差分予測評価で有効なデータがありません")
        return {
            'diff_rmse': float('nan'),
            'diff_mae': float('nan'),
            'direction_accuracy': float('nan'),
            'small_change_sensitivity': float('nan'),
            'large_change_accuracy': float('nan')
        }

    # 基本的な差分予測評価指標
    diff_rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    diff_mae = mean_absolute_error(y_true_valid, y_pred_valid)

    # 変化方向の一致率
    true_direction = np.sign(y_true_valid)
    pred_direction = np.sign(y_pred_valid)
    direction_accuracy = np.mean(true_direction == pred_direction) * 100

    # 小さな変化（±0.1℃以内）への感度
    small_changes = np.abs(y_true_valid) <= 0.1
    if np.sum(small_changes) > 0:
        small_change_mae = mean_absolute_error(
            y_true_valid[small_changes],
            y_pred_valid[small_changes]
        )
        small_change_sensitivity = 1 / (1 + small_change_mae)  # 感度スコア（高いほど良い）
    else:
        small_change_sensitivity = float('nan')

    # 大きな変化（±0.5℃以上）の予測精度
    large_changes = np.abs(y_true_valid) >= 0.5
    if np.sum(large_changes) > 0:
        large_change_mae = mean_absolute_error(
            y_true_valid[large_changes],
            y_pred_valid[large_changes]
        )
        large_change_accuracy = 1 / (1 + large_change_mae)  # 精度スコア（高いほど良い）
    else:
        large_change_accuracy = float('nan')

    metrics = {
        'diff_rmse': diff_rmse,
        'diff_mae': diff_mae,
        'direction_accuracy': direction_accuracy,
        'small_change_sensitivity': small_change_sensitivity,
        'large_change_accuracy': large_change_accuracy
    }

    # 温度復元評価（現在温度が提供された場合）
    if current_temps is not None:
        current_temps_valid = current_temps[valid_indices]
        if len(current_temps_valid) == len(y_pred_valid):
            # 予測された温度を復元
            restored_temps = current_temps_valid + y_pred_valid
            true_future_temps = current_temps_valid + y_true_valid

            # 復元温度の評価
            restoration_rmse = np.sqrt(mean_squared_error(true_future_temps, restored_temps))
            restoration_mae = mean_absolute_error(true_future_temps, restored_temps)
            restoration_r2 = r2_score(true_future_temps, restored_temps)

            metrics.update({
                'restoration_rmse': restoration_rmse,
                'restoration_mae': restoration_mae,
                'restoration_r2': restoration_r2
            })

    return metrics


def print_difference_metrics(metrics, zone=None, horizon=None):
    """
    差分予測の評価指標を整形して表示する関数

    Parameters:
    -----------
    metrics : dict
        差分予測の評価指標辞書
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    header = "差分予測評価指標"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print("=" * 50)
    print(f"📊 差分予測精度:")
    print(f"   温度差分RMSE: {metrics['diff_rmse']:.4f}℃")
    print(f"   温度差分MAE: {metrics['diff_mae']:.4f}℃")

    print(f"\n🎯 変化パターン分析:")
    print(f"   変化方向一致率: {metrics['direction_accuracy']:.1f}%")

    if not pd.isna(metrics['small_change_sensitivity']):
        print(f"   小変化感度スコア: {metrics['small_change_sensitivity']:.3f}")
    else:
        print(f"   小変化感度スコア: N/A (小変化データなし)")

    if not pd.isna(metrics['large_change_accuracy']):
        print(f"   大変化精度スコア: {metrics['large_change_accuracy']:.3f}")
    else:
        print(f"   大変化精度スコア: N/A (大変化データなし)")

    # 温度復元評価結果がある場合
    if 'restoration_rmse' in metrics:
        print(f"\n🌡️ 温度復元性能:")
        print(f"   復元温度RMSE: {metrics['restoration_rmse']:.4f}℃")
        print(f"   復元温度MAE: {metrics['restoration_mae']:.4f}℃")
        print(f"   復元温度R²: {metrics['restoration_r2']:.4f}")


def restore_temperature_from_difference(current_temp, predicted_diff):
    """
    差分予測結果から実際の温度を復元する関数

    Parameters:
    -----------
    current_temp : float or Series
        現在の温度
    predicted_diff : float or Series
        予測された温度差分

    Returns:
    --------
    float or Series
        復元された将来温度 (現在温度 + 予測差分)
    """
    return current_temp + predicted_diff


def compare_difference_vs_direct_prediction(direct_metrics, diff_metrics, current_temps, y_true_future):
    """
    直接温度予測と差分予測の性能を比較する関数

    Parameters:
    -----------
    direct_metrics : dict
        直接温度予測の評価指標
    diff_metrics : dict
        差分予測の評価指標
    current_temps : Series
        現在温度
    y_true_future : Series
        実際の将来温度

    Returns:
    --------
    dict
        比較結果の要約
    """
    comparison = {
        'direct_prediction': {
            'rmse': direct_metrics['rmse'],
            'mae': direct_metrics['mae'],
            'r2': direct_metrics['r2']
        },
        'difference_prediction': {
            'rmse': diff_metrics.get('restoration_rmse', float('nan')),
            'mae': diff_metrics.get('restoration_mae', float('nan')),
            'r2': diff_metrics.get('restoration_r2', float('nan'))
        }
    }

    # 性能改善の計算
    if not pd.isna(comparison['difference_prediction']['rmse']):
        rmse_improvement = (direct_metrics['rmse'] - diff_metrics['restoration_rmse']) / direct_metrics['rmse'] * 100
        mae_improvement = (direct_metrics['mae'] - diff_metrics['restoration_mae']) / direct_metrics['mae'] * 100
        r2_improvement = diff_metrics['restoration_r2'] - direct_metrics['r2']

        comparison['improvements'] = {
            'rmse_improvement_percent': rmse_improvement,
            'mae_improvement_percent': mae_improvement,
            'r2_improvement_absolute': r2_improvement
        }

    return comparison


def print_prediction_comparison(comparison, zone=None, horizon=None):
    """
    直接予測と差分予測の比較結果を表示する関数

    Parameters:
    -----------
    comparison : dict
        compare_difference_vs_direct_predictionの結果
    zone : int, optional
        ゾーン番号
    horizon : int, optional
        予測ホライゾン（分）
    """
    header = "予測手法比較"
    if zone is not None:
        header += f" (ゾーン{zone}"
        if horizon is not None:
            header += f", {horizon}分後)"
        else:
            header += ")"
    elif horizon is not None:
        header += f" ({horizon}分後)"

    print(f"\n{header}:")
    print("=" * 60)

    direct = comparison['direct_prediction']
    diff = comparison['difference_prediction']

    print(f"🔹 直接温度予測:")
    print(f"   RMSE: {direct['rmse']:.4f}℃, MAE: {direct['mae']:.4f}℃, R²: {direct['r2']:.4f}")

    print(f"🔸 差分予測→温度復元:")
    print(f"   RMSE: {diff['rmse']:.4f}℃, MAE: {diff['mae']:.4f}℃, R²: {diff['r2']:.4f}")

    if 'improvements' in comparison:
        imp = comparison['improvements']
        print(f"\n📈 性能改善:")
        print(f"   RMSE改善: {imp['rmse_improvement_percent']:+.1f}%")
        print(f"   MAE改善: {imp['mae_improvement_percent']:+.1f}%")
        print(f"   R²改善: {imp['r2_improvement_absolute']:+.3f}")

        # 総合判定
        if imp['rmse_improvement_percent'] > 0 and imp['mae_improvement_percent'] > 0:
            print("\n✅ 差分予測手法の方が優秀です！")
        elif imp['rmse_improvement_percent'] < -10 or imp['mae_improvement_percent'] < -10:
            print("\n❌ 直接予測手法の方が優秀です")
        else:
            print("\n⚖️ 両手法の性能は同等です")
