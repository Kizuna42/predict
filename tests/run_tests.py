#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
テストを実行するためのスクリプト
"""

import unittest
import sys
import os
import pathlib

# プロジェクトのルートディレクトリをパスに追加
current_dir = pathlib.Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def run_all_tests():
    """全てのテストを実行する"""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=current_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()

def run_specific_tests(pattern):
    """特定のパターンに一致するテストを実行する"""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=current_dir, pattern=f'test_{pattern}*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    return result.wasSuccessful()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='テスト実行スクリプト')
    parser.add_argument('--module', type=str, help='テストするモジュール名（例: features, lgbm）', default=None)

    args = parser.parse_args()

    if args.module:
        success = run_specific_tests(args.module)
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)
