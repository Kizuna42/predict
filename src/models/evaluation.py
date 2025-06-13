#!/usr/bin/env python
# coding: utf-8

"""
モデル評価モジュール（簡素化版）
基本的な評価指標の計算と表示機能を提供
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def calculate_metrics(y_true, y_pred):
    """
    回帰モデルの基本評価指標を計算

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
    # NaN値の処理
    valid_indices = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        print("警告: 有効なデータがありません")
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'mape': float('nan'),
            'r2': float('nan')
        }

    # 評価指標の計算
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    r2 = r2_score(y_true_valid, y_pred_valid)
    
    # MAPEの計算（ゼロ除算対策）
    try:
        mape = mean_absolute_percentage_error(y_true_valid, y_pred_valid)
    except:
        mape = float('nan')

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }


def print_metrics(metrics, zone, horizon):
    """評価指標をわかりやすく表示"""
    print(f"\n📊 Zone {zone} - {horizon}分後予測の評価結果:")
    print(f"  RMSE: {metrics['rmse']:.4f}°C")
    print(f"  MAE:  {metrics['mae']:.4f}°C")
    print(f"  MAPE: {metrics['mape']:.4f}%")
    print(f"  R²:   {metrics['r2']:.4f}")


def analyze_feature_importance(model, feature_names, top_n=15):
    """特徴量重要度を分析"""
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })

    sorted_importance = feature_importance.sort_values('importance', ascending=False)
    top_features = sorted_importance.head(top_n)
    
    print(f"\n上位{top_n}個の重要な特徴量:")
    for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance']), 1):
        print(f"{i}. {feature}: {importance:.4f}")

    return sorted_importance


def evaluate_temperature_difference_model(y_diff_true, y_diff_pred, current_temps):
    """
    温度差分予測モデルの評価

    Parameters:
    -----------
    y_diff_true : Series
        実際の温度差分
    y_diff_pred : Series
        予測された温度差分
    current_temps : Series
        現在の温度

    Returns:
    --------
    dict
        差分予測の評価指標
    """
    # 差分の評価
    diff_metrics = calculate_metrics(y_diff_true, y_diff_pred)
    
    # 復元温度の評価
    y_restored_true = current_temps + y_diff_true
    y_restored_pred = current_temps + y_diff_pred
    restored_metrics = calculate_metrics(y_restored_true, y_restored_pred)
    
    # 結果を統合
    combined_metrics = {
        'diff_rmse': diff_metrics['rmse'],
        'diff_mae': diff_metrics['mae'],
        'diff_mape': diff_metrics['mape'],
        'diff_r2': diff_metrics['r2'],
        'restored_rmse': restored_metrics['rmse'],
        'restored_mae': restored_metrics['mae'],
        'restored_mape': restored_metrics['mape'],
        'restored_r2': restored_metrics['r2']
    }
    
    return combined_metrics


def print_difference_metrics(metrics, zone, horizon):
    """差分予測の評価指標を表示"""
    print(f"\n🔥 Zone {zone} - {horizon}分後差分予測の評価結果:")
    print(f"  差分予測:")
    print(f"    RMSE: {metrics['diff_rmse']:.4f}°C")
    print(f"    MAE:  {metrics['diff_mae']:.4f}°C")
    print(f"    R²:   {metrics['diff_r2']:.4f}")
    print(f"  復元温度:")
    print(f"    RMSE: {metrics['restored_rmse']:.4f}°C")
    print(f"    MAE:  {metrics['restored_mae']:.4f}°C")
    print(f"    R²:   {metrics['restored_r2']:.4f}")


def restore_temperature_from_difference(diff_pred, current_temps):
    """差分予測から温度を復元"""
    return current_temps + diff_pred


def compare_difference_vs_direct_prediction(direct_metrics, diff_metrics, current_temps, y_true):
    """直接予測と差分予測の比較分析"""
    comparison = {
        'direct_rmse': direct_metrics['rmse'],
        'difference_rmse': diff_metrics['restored_rmse'],
        'direct_mae': direct_metrics['mae'],
        'difference_mae': diff_metrics['restored_mae'],
        'direct_r2': direct_metrics['r2'],
        'difference_r2': diff_metrics['restored_r2']
    }
    
    # 改善率の計算
    rmse_improvement = ((direct_metrics['rmse'] - diff_metrics['restored_rmse']) / direct_metrics['rmse']) * 100
    mae_improvement = ((direct_metrics['mae'] - diff_metrics['restored_mae']) / direct_metrics['mae']) * 100
    r2_improvement = ((diff_metrics['restored_r2'] - direct_metrics['r2']) / abs(direct_metrics['r2'])) * 100
    
    comparison.update({
        'rmse_improvement_pct': rmse_improvement,
        'mae_improvement_pct': mae_improvement,
        'r2_improvement_pct': r2_improvement
    })
    
    return comparison


def print_prediction_comparison(comparison, zone, horizon):
    """予測手法の比較結果を表示"""
    print(f"\n📈 Zone {zone} - {horizon}分後予測手法比較:")
    print(f"  直接予測    vs 差分予測")
    print(f"  RMSE: {comparison['direct_rmse']:.4f} vs {comparison['difference_rmse']:.4f} ({comparison['rmse_improvement_pct']:+.1f}%)")
    print(f"  MAE:  {comparison['direct_mae']:.4f} vs {comparison['difference_mae']:.4f} ({comparison['mae_improvement_pct']:+.1f}%)")
    print(f"  R²:   {comparison['direct_r2']:.4f} vs {comparison['difference_r2']:.4f} ({comparison['r2_improvement_pct']:+.1f}%)")


def test_physical_validity(model, feature_names, test_data, zone, horizon, 
                          is_difference_model=False, current_temp_col=None):
    """
    物理的妥当性テスト - サーモ制御による予測温度変化の確認

    Parameters:
    -----------
    model : trained model
        学習済みモデル
    feature_names : list
        特徴量名のリスト
    test_data : DataFrame
        テストデータ
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン
    is_difference_model : bool
        差分予測モデルかどうか
    current_temp_col : str
        現在温度の列名（差分モデルの場合）

    Returns:
    --------
    dict
        物理的妥当性テストの結果
    """
    print(f"\n🔬 Zone {zone} - 物理的妥当性テスト実行中...")
    
    # テスト用のサンプルデータを準備（最新の100サンプル）
    sample_data = test_data.tail(100).copy()
    
    if len(sample_data) == 0:
        print("警告: テストデータが不足しています")
        return None
    
    # AC制御関連の特徴量を特定
    ac_valid_col = f'AC_valid_{zone}'
    ac_mode_col = f'AC_mode_{zone}'
    ac_set_col = f'AC_set_{zone}'
    
    # 必要な列が存在するかチェック
    required_cols = [col for col in [ac_valid_col, ac_mode_col, ac_set_col] 
                    if col in feature_names and col in sample_data.columns]
    
    if not required_cols:
        print(f"警告: AC制御関連の特徴量が見つかりません（Zone {zone}）")
        return None
    
    # ベースライン予測（現在の設定）
    baseline_features = sample_data[feature_names]
    baseline_pred = model.predict(baseline_features)
    
    results = {
        'baseline_pred_mean': np.mean(baseline_pred),
        'baseline_pred_std': np.std(baseline_pred),
        'tests': []
    }
    
    # テスト1: サーモON vs OFF
    if ac_valid_col in required_cols:
        print(f"  🔥 テスト1: サーモON vs OFF")
        
        # サーモON設定
        test_data_on = sample_data.copy()
        test_data_on[ac_valid_col] = 1  # サーモON
        pred_on = model.predict(test_data_on[feature_names])
        
        # サーモOFF設定
        test_data_off = sample_data.copy()
        test_data_off[ac_valid_col] = 0  # サーモOFF
        pred_off = model.predict(test_data_off[feature_names])
        
        # 差分予測の場合は温度に復元
        if is_difference_model and current_temp_col:
            current_temps = sample_data[current_temp_col]
            pred_on_temp = current_temps + pred_on
            pred_off_temp = current_temps + pred_off
            baseline_temp = current_temps + baseline_pred
        else:
            pred_on_temp = pred_on
            pred_off_temp = pred_off
            baseline_temp = baseline_pred
        
        # 結果分析
        temp_diff_on_vs_baseline = np.mean(pred_on_temp - baseline_temp)
        temp_diff_off_vs_baseline = np.mean(pred_off_temp - baseline_temp)
        temp_diff_on_vs_off = np.mean(pred_on_temp - pred_off_temp)
        
        test1_result = {
            'test_name': 'サーモON vs OFF',
            'on_vs_baseline': temp_diff_on_vs_baseline,
            'off_vs_baseline': temp_diff_off_vs_baseline,
            'on_vs_off': temp_diff_on_vs_off,
            'physical_validity': temp_diff_on_vs_off > 0  # ONの方が高い温度予測なら物理的に妥当
        }
        
        results['tests'].append(test1_result)
        
        print(f"    サーモON vs ベースライン: {temp_diff_on_vs_baseline:+.3f}°C")
        print(f"    サーモOFF vs ベースライン: {temp_diff_off_vs_baseline:+.3f}°C")
        print(f"    サーモON vs OFF: {temp_diff_on_vs_off:+.3f}°C")
        print(f"    物理的妥当性: {'✅ OK' if test1_result['physical_validity'] else '❌ NG'}")
    
    # テスト2: モード変更（冷房 vs 暖房）
    if ac_mode_col in required_cols:
        print(f"  ❄️ テスト2: 冷房 vs 暖房モード")
        
        # 冷房モード
        test_data_cool = sample_data.copy()
        test_data_cool[ac_mode_col] = 0  # 冷房
        if ac_valid_col in test_data_cool.columns:
            test_data_cool[ac_valid_col] = 1  # サーモON
        pred_cool = model.predict(test_data_cool[feature_names])
        
        # 暖房モード
        test_data_heat = sample_data.copy()
        test_data_heat[ac_mode_col] = 1  # 暖房
        if ac_valid_col in test_data_heat.columns:
            test_data_heat[ac_valid_col] = 1  # サーモON
        pred_heat = model.predict(test_data_heat[feature_names])
        
        # 差分予測の場合は温度に復元
        if is_difference_model and current_temp_col:
            current_temps = sample_data[current_temp_col]
            pred_cool_temp = current_temps + pred_cool
            pred_heat_temp = current_temps + pred_heat
    else:
            pred_cool_temp = pred_cool
            pred_heat_temp = pred_heat
        
        # 結果分析
        temp_diff_heat_vs_cool = np.mean(pred_heat_temp - pred_cool_temp)
        
        test2_result = {
            'test_name': '暖房 vs 冷房',
            'heat_vs_cool': temp_diff_heat_vs_cool,
            'physical_validity': temp_diff_heat_vs_cool > 0  # 暖房の方が高い温度予測なら物理的に妥当
        }
        
        results['tests'].append(test2_result)
        
        print(f"    暖房 vs 冷房: {temp_diff_heat_vs_cool:+.3f}°C")
        print(f"    物理的妥当性: {'✅ OK' if test2_result['physical_validity'] else '❌ NG'}")
    
    # テスト3: 設定温度変更
    if ac_set_col in required_cols:
        print(f"  🌡️ テスト3: 設定温度変更")
        
        # 現在の平均設定温度
        current_setpoint = sample_data[ac_set_col].mean()
        
        # 高設定温度（+2°C）
        test_data_high = sample_data.copy()
        test_data_high[ac_set_col] = current_setpoint + 2
        if ac_valid_col in test_data_high.columns:
            test_data_high[ac_valid_col] = 1  # サーモON
        pred_high = model.predict(test_data_high[feature_names])
        
        # 低設定温度（-2°C）
        test_data_low = sample_data.copy()
        test_data_low[ac_set_col] = current_setpoint - 2
        if ac_valid_col in test_data_low.columns:
            test_data_low[ac_valid_col] = 1  # サーモON
        pred_low = model.predict(test_data_low[feature_names])
        
        # 差分予測の場合は温度に復元
        if is_difference_model and current_temp_col:
            current_temps = sample_data[current_temp_col]
            pred_high_temp = current_temps + pred_high
            pred_low_temp = current_temps + pred_low
        else:
            pred_high_temp = pred_high
            pred_low_temp = pred_low
        
        # 結果分析
        temp_diff_high_vs_low = np.mean(pred_high_temp - pred_low_temp)
        
        test3_result = {
            'test_name': '設定温度 高 vs 低',
            'high_vs_low': temp_diff_high_vs_low,
            'physical_validity': temp_diff_high_vs_low > 0  # 高設定の方が高い温度予測なら物理的に妥当
        }
        
        results['tests'].append(test3_result)
        
        print(f"    設定温度 高(+2°C) vs 低(-2°C): {temp_diff_high_vs_low:+.3f}°C")
        print(f"    物理的妥当性: {'✅ OK' if test3_result['physical_validity'] else '❌ NG'}")
    
    # 総合評価
    valid_tests = [test for test in results['tests'] if 'physical_validity' in test]
    if valid_tests:
        overall_validity = all(test['physical_validity'] for test in valid_tests)
        validity_score = sum(test['physical_validity'] for test in valid_tests) / len(valid_tests)
        
        results['overall_validity'] = overall_validity
        results['validity_score'] = validity_score
        
        print(f"\n📋 総合評価:")
        print(f"    物理的妥当性スコア: {validity_score:.1%}")
        print(f"    総合判定: {'✅ 物理的に妥当' if overall_validity else '⚠️ 要確認'}")
    
    return results


def test_difference_prediction_behavior(model, feature_names, test_data, zone, horizon, current_temp_col):
    """
    差分予測の挙動テスト - プラス方向でもマイナス予測が出るケースの分析

    Parameters:
    -----------
    model : trained model
        差分予測モデル
    feature_names : list
        特徴量名のリスト
    test_data : DataFrame
        テストデータ
    zone : int
        対象ゾーン
    horizon : int
        予測ホライゾン
    current_temp_col : str
        現在温度の列名

    Returns:
    --------
    dict
        差分予測挙動の分析結果
    """
    print(f"\n🔥 Zone {zone} - 差分予測挙動テスト実行中...")
    
    # 温度差分の予測
    sample_data = test_data.tail(200).copy()
    if len(sample_data) == 0:
        return None
    
    diff_pred = model.predict(sample_data[feature_names])
    current_temps = sample_data[current_temp_col]
    
    # 温度トレンドの分析
    temp_changes = current_temps.diff().fillna(0)  # 現在の温度変化率
    
    # ケース分析
    rising_trend = temp_changes > 0.1  # 上昇トレンド
    falling_trend = temp_changes < -0.1  # 下降トレンド
    stable_trend = abs(temp_changes) <= 0.1  # 安定トレンド
    
    # 予測方向の分析
    pred_positive = diff_pred > 0.05  # 正の差分予測
    pred_negative = diff_pred < -0.05  # 負の差分予測
    pred_neutral = abs(diff_pred) <= 0.05  # ニュートラル予測
    
    # 興味深いケースの特定
    cases = {
        'rising_but_negative_pred': np.sum(rising_trend & pred_negative),
        'falling_but_positive_pred': np.sum(falling_trend & pred_positive),
        'stable_but_large_pred': np.sum(stable_trend & (abs(diff_pred) > 0.2)),
        'total_samples': len(sample_data)
    }
    
    # 統計情報
    stats = {
        'mean_diff_pred': np.mean(diff_pred),
        'std_diff_pred': np.std(diff_pred),
        'mean_temp_change': np.mean(temp_changes),
        'correlation': np.corrcoef(temp_changes, diff_pred)[0, 1] if len(temp_changes) > 1 else 0
    }
    
    results = {
        'cases': cases,
        'stats': stats,
        'examples': {}
    }
    
    # 具体例の抽出
    if cases['rising_but_negative_pred'] > 0:
        rising_negative_idx = np.where(rising_trend & pred_negative)[0][:3]
        examples_rising_negative = []
        for idx in rising_negative_idx:
            examples_rising_negative.append({
                'current_temp': current_temps.iloc[idx],
                'temp_change': temp_changes.iloc[idx],
                'diff_pred': diff_pred[idx],
                'restored_temp': current_temps.iloc[idx] + diff_pred[idx]
            })
        results['examples']['rising_but_negative'] = examples_rising_negative
    
    if cases['falling_but_positive_pred'] > 0:
        falling_positive_idx = np.where(falling_trend & pred_positive)[0][:3]
        examples_falling_positive = []
        for idx in falling_positive_idx:
            examples_falling_positive.append({
                'current_temp': current_temps.iloc[idx],
                'temp_change': temp_changes.iloc[idx],
                'diff_pred': diff_pred[idx],
                'restored_temp': current_temps.iloc[idx] + diff_pred[idx]
            })
        results['examples']['falling_but_positive'] = examples_falling_positive
    
    # 結果表示
    print(f"  📊 差分予測挙動分析:")
    print(f"    上昇トレンドなのに負の予測: {cases['rising_but_negative_pred']}/{cases['total_samples']} ({cases['rising_but_negative_pred']/cases['total_samples']*100:.1f}%)")
    print(f"    下降トレンドなのに正の予測: {cases['falling_but_positive_pred']}/{cases['total_samples']} ({cases['falling_but_positive_pred']/cases['total_samples']*100:.1f}%)")
    print(f"    安定なのに大きな予測: {cases['stable_but_large_pred']}/{cases['total_samples']} ({cases['stable_but_large_pred']/cases['total_samples']*100:.1f}%)")
    print(f"    温度変化と差分予測の相関: {stats['correlation']:.3f}")
    
    # 物理的解釈
    print(f"\n  🧠 物理的解釈:")
    if cases['rising_but_negative_pred'] > 0:
        print(f"    ✅ 上昇中でも負の予測: これは制御により温度上昇が抑制される予測として妥当")
    if cases['falling_but_positive_pred'] > 0:
        print(f"    ✅ 下降中でも正の予測: これは制御により温度低下が抑制される予測として妥当")
    
    return results
