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
python -m thermal_prediction.main --file_path ./AllDayData.csv --output_dir ./output --analyze_horizons --zones 0,1,2 --horizons 5,10,15,20
```

### オプション

- `--file_path`: 入力データファイルのパス（デフォルト: `./AllDayData.csv`）
- `--output_dir`: 出力ディレクトリ（デフォルト: `./output/sens_temp_predictions`）
- `--start_date`: 分析開始日（デフォルト: `2024-06-26`）
- `--end_date`: 分析終了日（デフォルト: `2024-09-20`）
- `--analyze_horizons`: 予測ホライゾン分析を実行する
- `--horizons`: 評価する予測ホライゾン（分）をカンマ区切りで指定（デフォルト: `5,10,15,20,30`）
- `--zones`: 分析するゾーン番号（カンマ区切り、または `all` で全ゾーン）（デフォルト: `all`）

## ディレクトリ構造

```
thermal_prediction/
├── __init__.py           # パッケージの初期化ファイル
├── main.py               # メインエントリーポイント
├── analyze_horizon.py    # 予測ホライズン分析モジュール
├── config.py             # 設定パラメータの一元管理
├── utils/                # ユーティリティ機能
│   ├── __init__.py       # ユーティリティのエクスポート定義
│   ├── features.py       # 特徴量生成関連の機能
│   └── thermo.py         # サーモ状態関連の機能
├── models/               # モデル定義・学習
│   ├── __init__.py       # モデルのエクスポート定義
│   └── lgbm.py           # LightGBMモデル実装
└── visualization/        # 可視化機能
    ├── __init__.py       # 可視化機能のエクスポート定義
    ├── horizon.py        # 予測ホライズン関連の可視化
    ├── predictions.py    # 予測結果の可視化
    └── feature_importance.py # 特徴量重要度の可視化
```

## 設定管理

`config.py`ファイルでプロジェクト全体の設定を一元管理しています。主な設定項目：

- データの入出力パス
- モデルのハイパーパラメータ
- 特徴量選択の設定
- ゾーンと室外機の対応関係
- 可視化のデフォルト設定

設定を変更する場合は、このファイルを編集するか、コマンドライン引数で上書きしてください。

## 主要モジュールの説明

- **main.py**: コマンドライン引数を処理し、温度予測モデルの学習・評価を実行するエントリーポイント
- **analyze_horizon.py**: 異なる予測ホライズンでの温度予測モデルの性能を評価
- **utils/features.py**: 特徴量生成のためのユーティリティ関数
- **utils/thermo.py**: サーモ状態の決定などの HVAC 関連ロジック
- **models/lgbm.py**: LightGBM を使用したモデルの訓練と評価
- **visualization/**: 様々な可視化機能を提供するモジュール群

## 特徴量生成プロセス

温度予測モデルでは、以下の特徴量を使用します：

1. **時間特徴量**: 時刻、曜日、週末フラグ、夜間フラグなど
2. **温度ラグ特徴量**: 過去の温度データ（1 分前、5 分前、15 分前）
3. **温度変化率**: 直近 5 分間の温度変化率
4. **電力消費**: 室外機の電力消費データとそのラグ
5. **サーモ状態**: エアコンのサーモスタット状態と持続時間
6. **外気温特徴量**: 外気温とセンサー温度との差
7. **交互作用特徴量**: サーモ状態と温度の掛け合わせなど

特徴量はモデルの訓練前に重要度に基づいて選択され、最適な組み合わせが使用されます。

## 必要条件

- Python 3.7 以上
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- LightGBM
