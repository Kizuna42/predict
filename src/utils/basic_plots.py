#!/usr/bin/env python
# coding: utf-8

"""
基本的な可視化機能
特徴量重要度プロットのみを提供（簡素化版）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from .font_config import setup_japanese_font

# フォント設定を実行
setup_japanese_font()

# グラフ設定
sns.set_theme(style="whitegrid", palette="colorblind")

# 追加のmatplotlib設定
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})


def plot_feature_importance(feature_importance, zone, horizon, save_dir=None, top_n=15, save=True):
    """
    特徴量重要度のプロット

    Parameters:
    -----------
    feature_importance : DataFrame
        特徴量と重要度を含むDataFrame
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_dir : str, optional
        グラフ保存ディレクトリ
    top_n : int, optional
        表示する特徴量数
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # 重要度で降順ソート
    importance_sorted = feature_importance.sort_values('importance', ascending=False)

    # 上位N個の特徴量を抽出
    top_features = importance_sorted.head(top_n)

    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_features, ax=ax)

    ax.set_title(f'Zone {zone} - Feature Importance for {horizon}-min Prediction (Top {top_n})', fontsize=14)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)

    # グラフ保存
    if save and save_dir:
        output_path = os.path.join(save_dir, f'feature_importance_zone_{zone}_horizon_{horizon}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"特徴量重要度グラフ保存: {output_path}")

    return fig


# 公開API
__all__ = [
    'plot_feature_importance'
]
