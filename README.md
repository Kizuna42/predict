# 空調システム室内温度予測モデル

このプロジェクトは空調システムの室内温度を予測するための機械学習モデルを開発するためのコードを提供します。物理モデルベースの特徴量エンジニアリングと時系列予測を組み合わせて、より精度の高い未来温度予測を実現します。

## 🎯 主な特徴

- **物理法則に基づいた特徴量エンジニアリング**: 熱力学的原理を考慮した特徴量生成
- **完璧な時間軸修正システム**: 予測値と同じ時刻の実測値での正確な比較
- **データリーク防止**: 未来値の適切な扱いによる厳密なデータリーク防止
- **LAG 依存度低減**: 過去の温度履歴への過度な依存を回避
- **マルチゾーン対応**: 複数ゾーンの同時予測
- **複数ホライゾン**: 15 分、30 分、45 分、60 分の予測をサポート
- **包括的な診断機能**: 詳細な分析と可視化

## 📁 ディレクトリ構造

```
.
├── src/                          # ソースコードディレクトリ
│   ├── data/                     # データ処理関連モジュール
│   │   ├── preprocessing.py      # データ前処理
│   │   └── feature_engineering.py # 特徴量生成
│   ├── models/                   # モデル関連モジュール
│   │   ├── training.py           # モデルトレーニング
│   │   └── evaluation.py         # モデル評価
│   ├── utils/                    # ユーティリティ
│   │   ├── visualization.py      # 統合可視化システム
│   │   ├── perfect_time_axis_visualization.py # 完璧な時間軸修正
│   │   ├── advanced_visualization.py # 高度な可視化
│   │   ├── basic_plots.py        # 基本プロット
│   │   └── font_config.py        # フォント設定
│   ├── diagnostics/              # 診断機能
│   │   ├── lag_analysis.py       # LAG依存度分析
│   │   ├── time_validation.py    # 時間軸検証
│   │   ├── feature_analysis.py   # 特徴量分析
│   │   ├── performance_metrics.py # 性能指標
│   │   ├── comprehensive_lag_analysis.py # 包括的LAG分析
│   │   └── time_axis_verification.py # 時間軸整合性検証
│   └── config.py                 # 設定値
├── Output/                       # 出力ディレクトリ
│   ├── models/                   # 保存されたモデル
│   ├── perfect_time_axis/        # 完璧な時間軸修正プロット
│   ├── visualizations/           # 各種可視化ファイル
│   └── lag_diagnosis/            # LAG診断結果
├── main.py                       # メインスクリプト
├── final_perfect_time_axis_demo.py # 完璧な時間軸修正デモ
├── requirements.txt              # 依存パッケージ
├── PERFECT_TIME_AXIS_SOLUTION_REPORT.md # 解決レポート
└── README.md                     # このファイル
```

## 🚀 使い方

### 環境設定

```bash
# 必要なパッケージをインストール
pip install -r requirements.txt
```

### 基本実行

```bash
# 全ゾーン・全ホライゾンで実行
python main.py

# 特定のゾーンとホライゾンで実行（テストモード）
python main.py --test --zones 1 2 --horizons 15 30
```

### 完璧な時間軸修正デモ

```bash
# 包括的なデモンストレーション
python final_perfect_time_axis_demo.py --mode report

# 概念説明のみ
python final_perfect_time_axis_demo.py --mode explain

# デモのみ
python final_perfect_time_axis_demo.py --mode demo
```

### プログラムからの利用

```python
from src.utils.perfect_time_axis_visualization import create_simple_demo

# 簡単なデモの実行
result = create_simple_demo(zone=1, horizon=15, save_dir="output")
```

## 📊 出力

実行すると、以下の出力が生成されます：

### 基本出力

- **学習済みモデル**: `Output/models/lgbm_model_zone_*_horizon_*.pkl`
- **特徴量情報**: `Output/models/features_zone_*_horizon_*.pkl`
- **特徴量重要度**: `Output/feature_importance_zone_*_horizon_*.png`

### 完璧な時間軸修正

- **時間軸修正プロット**: `Output/perfect_time_axis/perfect_time_axis_zone_*_horizon_*.png`
- **比較デモンストレーション**: 3 つの方法（間違った方法、部分修正、完璧な方法）の並列比較

### 診断結果

- **LAG 依存度分析**: `Output/lag_diagnosis/lag_analysis_report_horizon_*.json`
- **時間軸検証**: 詳細な時間軸整合性レポート
- **性能指標**: 包括的な予測性能評価

### 高度な可視化

- **超詳細分刻み分析**: 2 時間〜48 時間の詳細可視化
- **修正済み時系列**: 時間軸修正済みの時系列プロット

## 🎯 完璧な時間軸修正システム

### 解決された問題

従来の予測可視化では「後追い現象」が発生していました：

- **問題**: 予測値が入力時刻に表示され、実測値の後追いに見える
- **解決**: 予測値を予測対象時刻に表示し、同じ時刻の実測値と比較

### 3 つの比較方法

1. **❌ 間違った方法**: 予測値を入力時刻に表示
2. **⚠️ 部分修正**: 予測値の時間軸のみ修正（比較対象が不適切）
3. **✅ 完璧な方法**: 予測値と同じ時刻の実測値で正確な比較

## 🛡️ データリーク防止対策

1. **時系列特性の考慮**: 未来の値を使った特徴量エンジニアリングを禁止
2. **移動平均の適切な設定**: `min_periods=1`で過去データのみ使用
3. **多項式特徴量**: トレーニングデータのみで生成、テストデータには変換のみ適用
4. **時系列分割**: ランダム分割を避け、時間順に分割して評価
5. **LAG 特徴量の排除**: 過去の温度値への直接的な依存を回避

## 📈 性能指標

- **MAE (平均絶対誤差)**: 予測精度の基本指標
- **RMSE (平方根平均二乗誤差)**: 大きな誤差に敏感な指標
- **相関係数**: 予測値と実測値の線形関係
- **R² 決定係数**: モデルの説明力
- **LAG 依存度**: 過去値への依存度（0%が理想）

## ⚠️ 注意事項

- **データファイル**: `AllDayData.csv`を別途用意する必要があります
- **メモリ使用量**: 大規模データセット使用時は注意が必要です
- **フォント設定**: 日本語表示でエラーが発生する場合は、フォント設定を確認してください

## 🔧 トラブルシューティング

### フォントエラー

```bash
# フォント設定エラーの場合、デフォルトフォントが自動使用されます
# 日本語表示が必要な場合は、システムに適切なフォントをインストールしてください
```

### メモリ不足

```bash
# テストモードで実行してメモリ使用量を削減
python main.py --test --zones 1 --horizons 15
```

## 📚 詳細ドキュメント

- **完璧な時間軸修正**: `PERFECT_TIME_AXIS_SOLUTION_REPORT.md`
- **技術的詳細**: ソースコード内の docstring
- **診断機能**: `src/diagnostics/`モジュール
