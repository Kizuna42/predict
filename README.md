# 温度予測システム

## 概要

このプロジェクトは、空調システムの各ゾーンの温度を予測するための機械学習モデルを提供します。予測結果の可視化機能も含まれています。

## 機能

- 特徴量生成と前処理
- LightGBM を使用した温度予測モデル
- 異なる予測ホライゾン（5 分後、10 分後、15 分後など）での評価
- 静的およびインタラクティブな可視化機能
- 全ゾーンの温度予測結果の可視化

## 必要なパッケージ

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib plotly japanize-matplotlib kaleido nbformat
```

## 使用方法

### 1. 全ゾーンの温度予測可視化

全ゾーンの温度予測結果を一度に表示するには、`analyze_horizon.py`を実行します。

```bash
python thermal_prediction/analyze_horizon.py --file_path AllDayData.csv --output_dir ./output/horizon_analysis --zones all --horizons 5,10,15,20,30
```

または、Python スクリプトから呼び出す場合：

```python
from thermal_prediction.analyze_horizon import analyze_prediction_horizons

# 全ゾーンの5分後、10分後、15分後予測を分析
results = analyze_prediction_horizons(
    file_path='AllDayData.csv',
    output_dir='./output/horizon_analysis',
    zones='all',
    horizons=[5, 10, 15]
)
```

### 2. Makefile を使った簡単な実行方法

プロジェクトには Makefile が用意されており、簡単なコマンドで分析と可視化を実行できます。

```bash
# セットアップ
make setup            # 必要なパッケージをインストール
make setup-interactive # インタラクティブ可視化用パッケージをインストール

# 分析実行
make analyze-all      # すべてのゾーンの分析を実行
make analyze-horizons # 予測ホライゾン分析を実行
make analyze-zone-0   # ゾーン0のみの分析を実行

# カスタム分析
make custom ZONES=0,1,2 HORIZONS=5,10,15  # 特定のゾーンとホライゾンで分析

# インタラクティブ可視化
make show-all-zones            # 全ゾーンの予測結果を表示（15分後がデフォルト）
make show-all-zones HORIZON=30 # 30分後の予測結果を表示
make show-zone ZONE=1          # ゾーン1の詳細結果を表示
```

コマンドの使い方を確認するには：

```bash
make help  # 利用可能なコマンドの一覧と説明を表示
```

### 3. インタラクティブ可視化の直接実行

インタラクティブな可視化機能を直接実行することもできます：

```bash
python thermal_prediction/visualize.py --zones 0,1,2 --horizon 15 --open
```

引数の説明：

- `--zones`: 表示するゾーン番号（カンマ区切り、または「all」）
- `--horizon`: 表示する予測ホライゾン（分）
- `--output_dir`: 結果出力ディレクトリ
- `--open`: 生成した HTML ファイルを自動的に開く

### 4. インタラクティブ可視化の仕組み

インタラクティブな可視化機能は以下のファイルに実装されています：

- `thermal_prediction/visualization/interactive.py`

このモジュールには以下の関数が含まれています：

- `interactive_horizon_metrics`: 各予測ホライゾンごとの評価指標を可視化
- `interactive_horizon_scatter`: 予測値と実測値の散布図を表示
- `interactive_timeseries`: 単一ホライゾンの時系列予測を可視化
- `interactive_all_horizons_timeseries`: 全予測ホライゾンの時系列を一つのグラフで可視化
- `interactive_feature_ranks`: 特徴量重要度の順位を可視化
- `interactive_all_zones_timeseries`: 全ゾーンの温度予測結果を一度に可視化

### 5. 全ゾーン可視化の使用例

以下のコードは、全ゾーンの温度予測結果を可視化する方法を示しています：

```python
from thermal_prediction.visualization.interactive import interactive_all_zones_timeseries

# 全ゾーンの15分後予測を可視化
graph_paths = interactive_all_zones_timeseries(
    results=prediction_results,  # ゾーン番号をキーとする辞書
    horizon=15,                 # 予測ホライゾン（分）
    zones=[0, 1, 2, 3, 4, 5],   # 表示するゾーン
    output_dir='./output/all_zones' # 出力ディレクトリ
)

# 結果のHTMLファイルパスが返される
print(f"全ゾーン可視化グラフが保存されました: {graph_paths[0]}")
```

`prediction_results`は以下の形式の辞書である必要があります：

```
{
    '0': DataFrame(columns=['actual', 'pred_15', ...]),
    '1': DataFrame(columns=['actual', 'pred_15', ...]),
    '2': DataFrame(columns=['actual', 'pred_15', ...]),
    ...
}
```

各 DataFrame のインデックスは時間を表します。

## プロジェクト構造

```
thermal_prediction/
├── __init__.py
├── analyze_horizon.py    # 予測ホライゾン分析
├── config.py             # 設定値
├── main.py               # メインエントリポイント
├── visualize.py          # インタラクティブ可視化スクリプト
├── models/               # モデル関連
├── utils/                # ユーティリティ関数
│   ├── __init__.py
│   └── features.py       # 特徴量生成
└── visualization/        # 可視化関連
    ├── __init__.py
    ├── feature_importance.py
    ├── horizon.py
    ├── interactive.py    # インタラクティブ可視化
    └── predictions.py
```

## 実装の詳細

`interactive_all_zones_timeseries`関数は、全ゾーンの温度予測結果をサブプロットとして表示します。各ゾーンのグラフには予測値と実測値が含まれており、グラフは HTML 形式（インタラクティブ）と PNG 形式（静的）の両方で保存されます。

引数の詳細：

- `results`: ゾーン番号をキーとする辞書。各ゾーンの DataFrame には'actual'（実測値）と'pred\_{horizon}'（予測値）の列が必要。
- `horizon`: 予測ホライゾン（分）
- `zones`: 表示するゾーン番号のリスト
- `output_dir`: 出力ディレクトリ

この関数はサブプロットを作成し、各ゾーンの予測結果をプロットします。結果は HTML 形式で保存され、インタラクティブにデータを調査することができます。また、静的な PNG 画像も生成されます。

## 依存関係

- Python 3.6+
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- plotly
- japanize-matplotlib（日本語表示用）
- kaleido（静的画像出力用）
- nbformat（HTML 出力用）
