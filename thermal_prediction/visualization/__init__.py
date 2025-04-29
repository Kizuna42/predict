"""
可視化モジュール
"""

# 日本語フォント設定
import matplotlib.pyplot as plt
import matplotlib
import sys

try:
    # japanize-matplotlibをインポート
    import japanize_matplotlib
except ImportError:
    print("日本語フォント表示のために japanize-matplotlib パッケージをインストールすることをお勧めします。")
    print("インストール方法: pip install japanize-matplotlib")

    # 代替設定として利用可能なフォントを試す
    import platform
    if platform.system() == 'Darwin':  # macOS
        matplotlib.rcParams['font.family'] = 'Hiragino Sans'
    elif platform.system() == 'Windows':
        matplotlib.rcParams['font.family'] = 'MS Gothic'
    else:  # Linux他
        try:
            matplotlib.rcParams['font.family'] = 'IPAGothic'
        except:
            pass

# マイナス記号などの表示修正
matplotlib.rcParams['axes.unicode_minus'] = False

from .feature_importance import visualize_feature_importance
from .predictions import visualize_zone_with_predictions
from .horizon import (
    visualize_horizon_metrics,
    visualize_horizon_scatter,
    visualize_horizon_timeseries,
    visualize_all_horizons_timeseries,
    visualize_feature_ranks,
    visualize_zones_comparison
)

# インタラクティブな可視化関数
try:
    from .interactive import (
        interactive_horizon_metrics,
        interactive_horizon_scatter,
        interactive_timeseries,
        interactive_feature_importance,
        interactive_all_horizons_timeseries,
        interactive_feature_ranks
    )
except ImportError:
    print("インタラクティブな可視化機能を使用するには plotly パッケージが必要です。")
    print("インストール方法: pip install plotly nbformat kaleido")
