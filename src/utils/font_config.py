#!/usr/bin/env python
# coding: utf-8

"""
フォント設定ユーティリティ
日本語表示のためのフォント設定
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform


def get_font_properties():
    """
    システムに応じた日本語フォントプロパティを取得

    Returns:
    --------
    matplotlib.font_manager.FontProperties
        日本語フォントプロパティ
    """
    system = platform.system()

    # システム別のフォント設定
    if system == "Darwin":  # macOS
        font_candidates = [
            "Hiragino Sans",
            "Hiragino Kaku Gothic Pro",
            "Yu Gothic",
            "Meiryo",
            "Takao Gothic"
        ]
    elif system == "Windows":
        font_candidates = [
            "Yu Gothic",
            "Meiryo",
            "MS Gothic",
            "Takao Gothic"
        ]
    else:  # Linux
        font_candidates = [
            "Takao Gothic",
            "IPAexGothic",
            "Noto Sans CJK JP",
            "DejaVu Sans"
        ]

    # 利用可能なフォントを検索
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    for font_name in font_candidates:
        if font_name in available_fonts:
            return fm.FontProperties(family=font_name)

    # フォールバック: デフォルトフォント
    print("警告: 日本語フォントが見つかりません。デフォルトフォントを使用します。")
    return fm.FontProperties()


def setup_japanese_font():
    """
    matplotlibの日本語フォント設定
    """
    font_prop = get_font_properties()

    # matplotlibの設定
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

    return font_prop
