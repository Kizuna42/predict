"""
特徴量重要度の可視化機能
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_feature_importance(importance_df, zone, output_dir):
    """
    特徴量重要度を可視化する

    Args:
        importance_df: 特徴量重要度のデータフレーム
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        fig_path: 保存したグラフのパス
    """
    try:
        if importance_df is None or importance_df.empty:
            print(f"Zone {zone}: 特徴量重要度データが空です")
            return None

        top_features = importance_df.head(15).copy()
        total_importance = importance_df['Importance'].sum()

        if total_importance > 0:
            top_features['Importance_Pct'] = top_features['Importance'] / total_importance * 100
        else:
            top_features['Importance_Pct'] = 0

        plt.figure(figsize=(12, 8))
        bars = plt.barh(
            top_features['Feature'],
            top_features['Importance_Pct'],
            color=plt.cm.viridis(np.linspace(0, 0.8, len(top_features))),
            edgecolor='gray',
            alpha=0.8
        )

        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{top_features['Importance_Pct'].iloc[i]:.1f}%",
                va='center'
            )

        plt.xlabel('重要度 (%)')
        plt.ylabel('特徴量')
        plt.title(f'ゾーン {zone} のトップ15特徴量重要度')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        fig_path = os.path.join(output_dir, f'zone_{zone}_feature_importance.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig_path

    except Exception as e:
        print(f"Error in visualize_feature_importance for zone {zone}: {e}")
        import traceback
        traceback.print_exc()
        return None
