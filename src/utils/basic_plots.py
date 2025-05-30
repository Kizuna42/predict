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


def plot_feature_importance(model, feature_names, zone, horizon, save_path=None,
                          top_n=15, model_type="予測", save=True):
    """
    特徴量重要度のプロット

    Parameters:
    -----------
    model : LGBMRegressor or similar model
        学習済みモデル（feature_importances_属性を持つモデル）
    feature_names : list
        特徴量名のリスト
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    save_path : str, optional
        保存先パス（Noneの場合は保存しない）
    top_n : int, optional
        表示する特徴量数
    model_type : str, optional
        モデルタイプ（グラフタイトル用）
    save : bool, optional
        グラフを保存するか

    Returns:
    --------
    matplotlib.figure.Figure
        プロットのFigureオブジェクト
    """
    # モデルから特徴量重要度を取得
    try:
        importances = model.feature_importances_
    except AttributeError:
        print(f"警告: モデルに feature_importances_ 属性がありません")
        return None

    # DataFrameを作成
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })

    # 重要度で降順ソート
    importance_sorted = feature_importance_df.sort_values('importance', ascending=False)

    # 上位N個の特徴量を抽出
    top_features = importance_sorted.head(top_n)

    # プロット作成
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(top_features)), top_features['importance'])

    # 色分け（重要度に応じて）
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 軸の設定
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()  # 上位から表示

    ax.set_title(f'ゾーン{zone} - {model_type}モデル特徴量重要度 ({horizon}分後)', fontsize=14, fontweight='bold')
    ax.set_xlabel('重要度', fontsize=12)
    ax.set_ylabel('特徴量', fontsize=12)

    # グリッド追加
    ax.grid(True, alpha=0.3, axis='x')

    # 値をバーの上に表示
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        ax.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # グラフ保存
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特徴量重要度グラフ保存: {save_path}")

    return fig


# 公開API
__all__ = [
    'plot_feature_importance'
]
