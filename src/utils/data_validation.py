#!/usr/bin/env python
# coding: utf-8

"""
データ検証ユーティリティモジュール
重複チェック、データ品質確認、エラーハンドリングを統一
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


def check_and_remove_duplicate_columns(df: pd.DataFrame, feature_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """
    データフレームの列名重複をチェックし、重複を排除
    
    Parameters:
    -----------
    df : DataFrame
        対象データフレーム
    feature_names : list, optional
        特徴量名のリスト
        
    Returns:
    --------
    df_clean : DataFrame
        重複排除後のデータフレーム
    clean_feature_names : list
        重複排除後の特徴量名
    stats : dict
        処理統計情報
    """
    stats = {
        'original_columns': len(df.columns),
        'duplicates_removed': 0,
        'final_columns': 0
    }
    
    # データフレームの列名重複チェック
    if len(df.columns) != len(set(df.columns)):
        print("⚠️  データフレームの列名に重複を検出 - 重複を排除します")
        
        unique_cols = []
        seen_cols = set()
        
        for col in df.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)
        
        duplicate_count = len(df.columns) - len(unique_cols)
        stats['duplicates_removed'] = duplicate_count
        
        df_clean = df[unique_cols]
        print(f"✅ {duplicate_count}個の重複列を排除 (残り: {len(unique_cols)}列)")
    else:
        df_clean = df.copy()
    
    # 特徴量名リストの重複チェック
    if feature_names is not None:
        unique_features = []
        seen_features = set()
        
        for feature in feature_names:
            if feature not in seen_features:
                unique_features.append(feature)
                seen_features.add(feature)
        
        if len(unique_features) != len(feature_names):
            feature_duplicates = len(feature_names) - len(unique_features)
            print(f"⚠️  特徴量名リストの重複を排除: {feature_duplicates}個")
        
        # データフレームに存在する特徴量のみ保持
        clean_feature_names = [f for f in unique_features if f in df_clean.columns]
    else:
        clean_feature_names = list(df_clean.columns)
    
    stats['final_columns'] = len(df_clean.columns)
    return df_clean, clean_feature_names, stats


def validate_data_quality(df: pd.DataFrame, target_cols: List[str], min_rows: int = 100) -> Dict[str, any]:
    """
    データ品質の基本的なバリデーション
    
    Parameters:
    -----------
    df : DataFrame
        対象データフレーム
    target_cols : list
        必須列のリスト
    min_rows : int
        最小必要行数
        
    Returns:
    --------
    validation_result : dict
        バリデーション結果
    """
    result = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'stats': {
            'total_rows': len(df),
            'missing_cols': [],
            'null_percentages': {}
        }
    }
    
    # 行数チェック
    if len(df) < min_rows:
        result['errors'].append(f"データ行数不足: {len(df)}行 (最小必要: {min_rows}行)")
        result['is_valid'] = False
    
    # 必須列の存在チェック
    missing_cols = [col for col in target_cols if col not in df.columns]
    if missing_cols:
        result['errors'].append(f"必須列が不足: {missing_cols}")
        result['stats']['missing_cols'] = missing_cols
        result['is_valid'] = False
    
    # 欠損値チェック
    for col in target_cols:
        if col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            result['stats']['null_percentages'][col] = null_pct
            
            if null_pct > 50:
                result['warnings'].append(f"{col}: 欠損値が50%以上 ({null_pct:.1f}%)")
            elif null_pct > 20:
                result['warnings'].append(f"{col}: 欠損値が20%以上 ({null_pct:.1f}%)")
    
    # 無限値・異常値チェック
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in target_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                result['warnings'].append(f"{col}: 無限値を検出 ({inf_count}個)")
    
    return result


def safe_data_preparation(df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    安全なデータ準備 - 重複除去と品質チェックを統合
    
    Parameters:
    -----------
    df : DataFrame
        元データ
    feature_cols : list
        特徴量列名
    target_col : str
        目的変数列名
        
    Returns:
    --------
    clean_data : DataFrame
        クリーン済みデータ
    preparation_info : dict
        準備処理の情報
    """
    preparation_info = {
        'original_shape': df.shape,
        'duplicate_removal': {},
        'validation': {},
        'final_shape': None
    }
    
    # 重複除去
    df_clean, clean_features, dup_stats = check_and_remove_duplicate_columns(df, feature_cols)
    preparation_info['duplicate_removal'] = dup_stats
    
    # 必要な列のみ抽出
    required_cols = [col for col in clean_features + [target_col] if col in df_clean.columns]
    df_subset = df_clean[required_cols]
    
    # データ品質チェック
    validation_result = validate_data_quality(df_subset, required_cols)
    preparation_info['validation'] = validation_result
    
    # 欠損値を含む行を削除
    df_final = df_subset.dropna(subset=required_cols)
    preparation_info['final_shape'] = df_final.shape
    
    print(f"📊 データ準備完了: {preparation_info['original_shape']} → {preparation_info['final_shape']}")
    
    return df_final, preparation_info


def print_data_preparation_summary(preparation_info: Dict[str, any]) -> None:
    """
    データ準備処理の結果サマリーを表示
    
    Parameters:
    -----------
    preparation_info : dict
        準備処理の情報
    """
    print("\n📋 データ準備サマリー:")
    print(f"  📥 元データ: {preparation_info['original_shape'][0]:,}行 x {preparation_info['original_shape'][1]}列")
    
    if preparation_info['duplicate_removal']['duplicates_removed'] > 0:
        print(f"  🔧 重複除去: {preparation_info['duplicate_removal']['duplicates_removed']}列")
    
    if preparation_info['final_shape']:
        print(f"  📤 最終データ: {preparation_info['final_shape'][0]:,}行 x {preparation_info['final_shape'][1]}列")
        
        row_reduction = preparation_info['original_shape'][0] - preparation_info['final_shape'][0]
        if row_reduction > 0:
            reduction_pct = (row_reduction / preparation_info['original_shape'][0]) * 100
            print(f"  📉 行数削減: {row_reduction:,}行 ({reduction_pct:.1f}%)")
    
    # バリデーション結果
    validation = preparation_info['validation']
    if validation['warnings']:
        print(f"  ⚠️  警告: {len(validation['warnings'])}件")
    if validation['errors']:
        print(f"  ❌ エラー: {len(validation['errors'])}件")
    if validation['is_valid']:
        print(f"  ✅ データ品質: 良好") 