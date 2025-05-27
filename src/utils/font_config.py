#!/usr/bin/env python
# coding: utf-8

"""
フォント設定モジュール
日本語フォントの設定を管理
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform


def setup_japanese_font():
    """
    日本語フォントの設定を行う（強化版）
    """
    # 利用可能な日本語フォントを検索（優先順位付き）
    japanese_fonts = [
        'Hiragino Sans',
        'Hiragino Kaku Gothic Pro',
        'Yu Gothic',
        'Meiryo',
        'IPAexGothic',
        'IPAGothic',
        'MS Gothic',
        'Takao Gothic',
        'VL Gothic',
        'Noto Sans CJK JP'
    ]

    # システムで利用可能なフォント一覧を取得
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 利用可能な日本語フォントを見つける
    selected_font = None
    for font in japanese_fonts:
        if font in available_fonts:
            selected_font = font
            break

    # フォント設定を強制的に適用
    if selected_font:
        plt.rcParams['font.family'] = [selected_font]
        plt.rcParams['font.sans-serif'] = [selected_font] + japanese_fonts
        print(f"日本語フォント設定: {selected_font}")
    else:
        # macOSの場合のフォールバック
        if platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Hiragino Sans']
            plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Arial Unicode MS']
            print("macOS用フォント設定: Hiragino Sans")
        else:
            # その他のシステム用フォールバック
            plt.rcParams['font.family'] = ['DejaVu Sans']
            plt.rcParams['font.sans-serif'] = japanese_fonts + ['DejaVu Sans', 'Arial']
            print("フォールバック設定を使用")

    # 重要な設定を強制適用
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.autolayout'] = True

    # 設定確認のためのテスト
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, 'テスト', fontsize=12)
        plt.close(fig)
        print("✅ 日本語フォント設定テスト成功")
    except Exception as e:
        print(f"⚠️ フォント設定テストで警告: {e}")


def get_font_properties():
    """
    日本語フォントプロパティを取得

    Returns:
    --------
    matplotlib.font_manager.FontProperties
        日本語フォントプロパティ
    """
    font_prop = fm.FontProperties()
    try:
        # 利用可能な日本語フォントを取得
        japanese_fonts = ['Hiragino Sans', 'Hiragino Kaku Gothic Pro', 'Yu Gothic', 'Meiryo']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        selected_font = None
        for font in japanese_fonts:
            if font in available_fonts:
                selected_font = font
                break

        if selected_font:
            font_prop.set_family(selected_font)
        else:
            # macOSのデフォルト
            font_prop.set_family('Hiragino Sans')
    except:
        pass

    return font_prop
