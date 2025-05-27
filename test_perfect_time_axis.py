#!/usr/bin/env python
# coding: utf-8

"""
完璧な時間軸修正システムのテストスクリプト
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 完璧な時間軸修正システムのインポート
from perfect_time_axis_main import (
    run_single_zone_perfect_demo,
    run_perfect_time_axis_demonstration,
    demonstrate_time_axis_concepts
)


def quick_test():
    """
    クイックテスト実行
    """
    print("🚀 完璧な時間軸修正システム - クイックテスト")
    print("=" * 80)

    # 1. 概念説明
    print("\n1️⃣ 時間軸修正の概念説明")
    demonstrate_time_axis_concepts()

    # 2. 単一ゾーンデモ
    print("\n2️⃣ 単一ゾーンデモ（ゾーン1, 15分予測）")
    result = run_single_zone_perfect_demo(zone=1, horizon=15)

    if result and result['success']:
        print("✅ 単一ゾーンデモ成功!")
    else:
        print("❌ 単一ゾーンデモ失敗")
        if result:
            print(f"エラー: {result['error_message']}")

    # 3. 複数ゾーンデモ（小規模）
    print("\n3️⃣ 複数ゾーンデモ（ゾーン1-2, 15分・30分予測）")
    comprehensive_result = run_perfect_time_axis_demonstration(
        target_zones=[1, 2],
        target_horizons=[15, 30]
    )

    if comprehensive_result:
        print("✅ 複数ゾーンデモ成功!")
        print(f"成功率: {comprehensive_result['successful_visualizations']}/{comprehensive_result['total_visualizations']}")
    else:
        print("❌ 複数ゾーンデモ失敗")

    print("\n🎉 クイックテスト完了!")


def full_test():
    """
    フルテスト実行
    """
    print("🚀 完璧な時間軸修正システム - フルテスト")
    print("=" * 80)

    # 全ゾーン、全ホライゾンでテスト
    comprehensive_result = run_perfect_time_axis_demonstration()

    if comprehensive_result:
        print("✅ フルテスト成功!")
        summary = comprehensive_result.get('overall_summary', {})
        if summary:
            print(f"全体平均MAE: {summary.get('overall_average_mae', 'N/A'):.3f}°C")
            print(f"全体平均RMSE: {summary.get('overall_average_rmse', 'N/A'):.3f}°C")
            print(f"全体平均相関: {summary.get('overall_average_correlation', 'N/A'):.3f}")
    else:
        print("❌ フルテスト失敗")

    print("\n🎉 フルテスト完了!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='完璧な時間軸修正システムテスト')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='テストモード')

    args = parser.parse_args()

    if args.mode == 'quick':
        quick_test()
    else:
        full_test()
