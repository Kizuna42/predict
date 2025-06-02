#!/usr/bin/env python
# coding: utf-8

"""
フォント設定ユーティリティ
日本語表示のためのフォント設定
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings


def get_system_japanese_fonts():
    """
    システムに応じた日本語フォント候補を取得

    Returns:
    --------
    list
        システム別の日本語フォント候補リスト
    """
    system = platform.system()

    if system == "Darwin":  # macOS
        return [
            "Hiragino Sans",
            "Hiragino Kaku Gothic Pro",
            "Hiragino Kaku Gothic ProN",
            "Hiragino Maru Gothic Pro",
            "Yu Gothic Medium",
            "Yu Gothic",
            "Osaka",
            "TakaoGothic",
            "IPAexGothic",
            "Noto Sans CJK JP"
        ]
    elif system == "Windows":
        return [
            "Yu Gothic UI",
            "Yu Gothic",
            "Meiryo UI",
            "Meiryo",
            "MS UI Gothic",
            "MS Gothic",
            "NSimSun",
            "TakaoGothic",
            "IPAexGothic"
        ]
    else:  # Linux
        return [
            "Noto Sans CJK JP",
            "TakaoGothic",
            "TakaoPGothic",
            "IPAexGothic",
            "IPAGothic",
            "VL Gothic",
            "Sazanami Gothic",
            "Kochi Gothic",
            "DejaVu Sans"
        ]


def find_available_japanese_font():
    """
    利用可能な日本語フォントを検索

    Returns:
    --------
    str or None
        利用可能な日本語フォント名
    """
    # 利用可能なフォント一覧を取得
    available_fonts = set([f.name for f in fm.fontManager.ttflist])

    # システム別候補から検索
    font_candidates = get_system_japanese_fonts()

    for font_name in font_candidates:
        if font_name in available_fonts:
            print(f"日本語フォントを検出: {font_name}")
            return font_name

    # より詳細な検索（部分一致）
    japanese_keywords = ['gothic', 'mincho', 'sans', 'hiragino', 'yu', 'meiryo', 'takao', 'ipa', 'noto']

    for font in available_fonts:
        font_lower = font.lower()
        for keyword in japanese_keywords:
            if keyword in font_lower and ('jp' in font_lower or 'japan' in font_lower or 'cjk' in font_lower):
                print(f"日本語フォントを検出 (部分一致): {font}")
                return font

    return None


def setup_matplotlib_japanese():
    """
    matplotlibの日本語表示設定を行う

    Returns:
    --------
    str
        使用するフォント名
    """
    # 日本語フォントを検索
    japanese_font = find_available_japanese_font()

    if japanese_font:
        # matplotlibの設定
        plt.rcParams['font.family'] = japanese_font
        plt.rcParams['font.sans-serif'] = [japanese_font] + plt.rcParams['font.sans-serif']
    else:
        # フォールバック設定
        warnings.warn("日本語フォントが見つかりません。代替フォントを設定します。")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        japanese_font = 'DejaVu Sans'

    # 共通設定
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
    plt.rcParams['figure.autolayout'] = True    # レイアウト自動調整

    # 日本語表示テスト
    try:
        fig, ax = plt.subplots(figsize=(1, 1))
        ax.text(0.5, 0.5, 'テスト', ha='center', va='center')
        plt.close(fig)
        print(f"日本語フォント設定完了: {japanese_font}")
    except Exception as e:
        warnings.warn(f"日本語フォント設定でエラーが発生: {e}")

    return japanese_font


def get_font_properties():
    """
    日本語フォントプロパティを取得

    Returns:
    --------
    matplotlib.font_manager.FontProperties
        日本語フォントプロパティ
    """
    japanese_font = find_available_japanese_font()

    if japanese_font:
        return fm.FontProperties(family=japanese_font, size=12)
    else:
        return fm.FontProperties(family='DejaVu Sans', size=12)


def setup_japanese_font():
    """
    日本語フォント設定の統合関数

    Returns:
    --------
    matplotlib.font_manager.FontProperties
        設定されたフォントプロパティ
    """
    font_name = setup_matplotlib_japanese()
    return fm.FontProperties(family=font_name, size=12)


# 利用可能フォント一覧表示用の診断関数
def diagnose_fonts():
    """
    利用可能なフォントの診断情報を表示
    """
    print("=== フォント診断情報 ===")
    print(f"OS: {platform.system()}")

    available_fonts = [f.name for f in fm.fontManager.ttflist]
    print(f"利用可能フォント数: {len(available_fonts)}")

    # 日本語っぽいフォントを検索
    japanese_fonts = []
    japanese_keywords = ['gothic', 'mincho', 'hiragino', 'yu', 'meiryo', 'takao', 'ipa', 'noto', 'osaka']

    for font in available_fonts:
        font_lower = font.lower()
        for keyword in japanese_keywords:
            if keyword in font_lower:
                japanese_fonts.append(font)
                break

    print(f"日本語関連フォント: {len(japanese_fonts)}個")
    for font in sorted(set(japanese_fonts))[:10]:  # 最初の10個を表示
        print(f"  - {font}")

    if len(japanese_fonts) > 10:
        print(f"  ... 他 {len(japanese_fonts) - 10}個")

    # 現在の設定
    print(f"現在のfont.family設定: {plt.rcParams['font.family']}")
    print(f"unicode_minus設定: {plt.rcParams['axes.unicode_minus']}")


if __name__ == "__main__":
    # 診断実行
    diagnose_fonts()
    print("\n" + "="*50)

    # 設定実行
    setup_japanese_font()
