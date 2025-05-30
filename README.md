# 🌡️ 温度予測システム

環境制御のための高精度温度予測システム。従来の直接温度予測と革新的な差分予測の両方をサポートします。

## 🎯 システム概要

### 解決する課題

- **精密な温度制御**: 環境制御システムのための高精度温度予測
- **ラグ問題の解決**: 従来予測の後追い問題を差分予測で改善
- **リアルタイム予測**: 空調制御への即座の対応

### 予測手法

#### 1. 従来の直接温度予測

- **目的**: 将来の絶対温度値を直接予測
- **特徴**: 安定した予測、既存システムとの互換性
- **用途**: 基本的な温度管理、比較基準

#### 2. 差分予測（革新的手法）

- **目的**: 温度の変化量を予測し、現在温度に加算
- **特徴**: ラグが少ない、高応答性、変化パターンの学習
- **用途**: 精密制御、リアルタイム応答が必要な場面

## 🚀 クイックスタート

### 基本実行

```bash
# 従来の直接温度予測
python main.py

# 差分予測システム
python main.py --mode difference

# 両方の手法を比較実行
python main.py --mode both

# テストモード（特定のゾーン・ホライゾン）
python main.py --test --zones 1 2 --horizons 15 30
```

### Python コードから実行

```python
# 直接予測
from main import run_temperature_prediction
run_temperature_prediction(mode='direct')

# 差分予測
run_temperature_prediction(mode='difference')

# 比較実行
run_temperature_prediction(mode='both')
```

## 📁 プロジェクト構造

```
prediction/
├── main.py                    # 統合メインスクリプト
├── src/
│   ├── config.py             # 設定管理
│   ├── data/
│   │   ├── preprocessing.py   # データ前処理
│   │   └── feature_engineering.py # 特徴量エンジニアリング
│   ├── models/
│   │   ├── training.py       # モデル訓練
│   │   ├── evaluation.py     # モデル評価
│   │   └── prediction.py     # 予測実行
│   ├── utils/
│   │   ├── visualization.py  # 可視化機能
│   │   └── font_config.py    # フォント設定
│   └── diagnostics/
│       ├── performance_metrics.py # 性能分析
│       └── time_validation.py     # 時間軸検証
├── Output/
│   ├── models/               # 学習済みモデル
│   └── visualizations/       # 可視化結果
├── requirements.txt
└── README.md
```

## 🛠️ セットアップ

### 必要要件

- Python 3.8+
- pandas, numpy, matplotlib, lightgbm, scikit-learn

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd prediction

# 依存関係をインストール
pip install -r requirements.txt

# データファイルを配置
# AllDayData.csv をルートディレクトリに配置
```

## 📊 評価指標と出力

### 共通評価指標

- **RMSE**: 予測精度の基本指標
- **MAE**: 平均絶対誤差
- **R²**: 決定係数
- **MAPE**: 平均絶対パーセント誤差

### 差分予測専用指標

- **変化方向一致率**: 温度上昇/下降の予測精度
- **小変化感度**: 微小変化（±0.1℃）への感度
- **大変化精度**: 大きな変化（±0.5℃ 以上）の予測精度

### 出力ファイル

#### モデルファイル

- `Output/models/lgbm_model_zone_*_horizon_*.pkl` - 直接予測モデル
- `Output/models/diff_model_zone_*_horizon_*.pkl` - 差分予測モデル

#### 可視化

- `Output/visualizations/feature_importance_*.png` - 特徴量重要度
- `Output/visualizations/prediction_results_*.png` - 予測結果
- `Output/visualizations/comparison_*.png` - 手法比較

## 🔧 高度な使用方法

### コマンドラインオプション

```bash
# 基本オプション
python main.py --mode [direct|difference|both]    # 予測手法選択
python main.py --test                              # テストモード
python main.py --zones 1 2 3                      # 対象ゾーン指定
python main.py --horizons 15 30 60                # 予測ホライゾン指定

# 詳細オプション
python main.py --no-visualization                 # 可視化スキップ
python main.py --save-models                      # モデル保存
python main.py --comparison-only                  # 比較のみ実行
```

### カスタム設定

`src/config.py`で設定をカスタマイズ可能：

```python
# 予測ホライゾン（分）
HORIZONS = [15, 30, 60, 120]

# 対象ゾーン
ZONES = [1, 2, 3, 4, 5, 6, 7, 8]

# モデルパラメータ
LGBM_PARAMS = {
    'learning_rate': 0.05,
    'n_estimators': 1000,
    # ...
}
```

## 📈 技術的特徴

### データ前処理

- **外れ値除去**: 物理的に不可能な温度値の除去
- **スムージング**: ノイズ除去のための移動平均
- **時間特徴量**: 周期的時間パターンの抽出

### 特徴量エンジニアリング

- **物理ベース特徴量**: 熱力学的関係を考慮した特徴量
- **制御状態特徴量**: 空調システムの状態情報
- **環境特徴量**: 外気温、日射量などの外部条件

### モデル最適化

- **時系列分割**: データリーク防止の厳密な分割
- **重み付け学習**: 重要な変化に対する重み付け
- **アンサンブル**: 複数モデルの組み合わせ

## 🎨 包括的可視化システム

### 詳細な予測精度分析

#### 1. **特徴量重要度分析**

- **グラフ**: `*_feature_importance_zone_*_horizon_*.png`
- **内容**:
  - 上位 20 の重要特徴量を可視化
  - 重要度スコアの数値表示
  - 特徴量の影響力ランキング

#### 2. **時系列比較分析**

- **グラフ**: `*_timeseries_zone_*_horizon_*.png`
- **内容**:
  - 実際値 vs 予測値の時系列プロット
  - 予測誤差の時系列変化
  - 統計情報（RMSE、MAE、R²）をグラフ内表示
  - 最新 7 日間のデータを詳細表示

#### 3. **散布図精度分析**

- **グラフ**: `*_scatter_analysis_zone_*_horizon_*.png`
- **内容**:
  - 実際値 vs 予測値の散布図（理想線付き）
  - 残差プロット（予測誤差の分布）
  - 残差ヒストグラム（誤差の正規性確認）
  - Q-Q プロット（統計的品質検証）

#### 4. **性能サマリー**

- **グラフ**: `*_performance_summary_zone_*_horizon_*.png`
- **内容**:
  - 全評価指標の棒グラフ表示
  - 指標値の数値表示
  - 色分けによる視覚的理解

#### 5. **手法比較分析**

- **グラフ**: `method_comparison_zone_*_horizon_*.png`
- **内容**:
  - 直接予測 vs 差分予測の性能比較
  - 改善率の可視化（%表示）
  - 緑色=改善、赤色=劣化の直感的表示

#### 6. **復元温度時系列**（差分予測専用）

- **グラフ**: `difference_restored_timeseries_zone_*_horizon_*.png`
- **内容**:
  - 差分予測から復元した温度の時系列比較
  - 実際の将来温度との比較
  - 復元精度の確認

### 可視化の自動生成

```bash
# 包括的可視化付きで実行
python main.py --mode both

# 可視化のみテスト
python test_visualization.py
```

### 文字化け対策

- **フォント設定**: 日本語表示対応済み
- **エンコーディング**: UTF-8 完全対応
- **特殊文字**: マイナス記号など適切に表示

## 🚨 注意点とベストプラクティス

### データ品質

- ✅ 十分なデータ量（最低 1 週間以上）
- ✅ 欠損値の適切な処理
- ✅ 外れ値の事前確認

### モデル運用

- ⚠️ 定期的な再学習の実施
- ⚠️ 予測精度の継続的監視
- ⚠️ 季節変動への対応

### 差分予測の考慮点

- 📝 現在温度との加算による温度復元が必要
- 📝 連続予測時の誤差蓄積に注意
- 📝 急激な環境変化時の性能確認

## 🤝 開発・貢献

### 拡張可能性

- **新しい特徴量**: ドメイン知識に基づく特徴量追加
- **モデルアルゴリズム**: 他の機械学習手法の組み込み
- **リアルタイム予測**: ストリーミングデータ対応

### コード品質

- 型ヒントの使用
- 包括的なテストケース
- 詳細なドキュメント

## 📚 参考資料

### 技術文書

- 機械学習による時系列予測
- LightGBM 公式ドキュメント
- 温度制御システム設計原理

### 関連論文

- 建物内温度予測手法
- 機械学習を用いた環境制御
- 時系列データの特徴量エンジニアリング

---

## 🎉 まとめ

本温度予測システムは、従来の直接予測と革新的な差分予測を組み合わせることで、高精度かつ応答性の高い温度制御を実現します。用途に応じて最適な手法を選択し、環境制御システムの性能向上に貢献できます。

**🚀 今すぐ始める**: `python main.py --mode both --test`でサンプル実行
