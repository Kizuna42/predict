# 温度予測モジュール

空調システムの温度予測を行うための Python モジュールです。

## 機能

- 異なる予測ホライゾンでの温度予測モデルの評価
- 特徴量の選択と重要度分析
- 時系列データの可視化

## インストール

リポジトリをクローンして、必要なパッケージをインストールします。

```bash
git clone https://github.com/yourusername/thermal-prediction.git
cd thermal-prediction
pip install -r requirements.txt
```

## 使用方法

### コマンドライン

予測ホライゾン分析を実行する例:

```bash
python analyze_temp.py --file_path ./AllDayData.csv --output_dir ./output --analyze_horizons --zones 0,1,2 --horizons 5,10,15,20
```

### オプション

- `--file_path`: 入力データファイルのパス（デフォルト: `./AllDayData.csv`）
- `--output_dir`: 出力ディレクトリ（デフォルト: `./output/sens_temp_predictions`）
- `--start_date`: 分析開始日（デフォルト: `2024-06-26`）
- `--end_date`: 分析終了日（デフォルト: `2024-09-20`）
- `--analyze_horizons`: 予測ホライゾン分析を実行する
- `--horizons`: 評価する予測ホライゾン（分）をカンマ区切りで指定（デフォルト: `5,10,15,20,30`）
- `--zones`: 分析するゾーン番号（カンマ区切り、または `all` で全ゾーン）（デフォルト: `all`）

## モジュール構成

- `thermal_prediction/`: メインパッケージ
  - `utils/`: ユーティリティ関数
    - `thermo.py`: サーモ状態の計算
    - `features.py`: 特徴量生成
  - `models/`: モデル関連
    - `lgbm.py`: LightGBM モデル
  - `visualization/`: 可視化機能
    - `feature_importance.py`: 特徴量重要度の可視化
    - `predictions.py`: 予測結果の可視化
    - `horizon.py`: 予測ホライゾン分析の可視化
  - `analyze_horizon.py`: 予測ホライゾン分析
  - `main.py`: メインモジュール

## 必要条件

- Python 3.7 以上
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- LightGBM
