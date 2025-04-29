# 温度予測モジュール

空調システムの温度予測を行うための Python モジュールです。様々な予測ホライゾン（5 分後、10 分後など）での温度変化を機械学習で予測し、分析結果を可視化します。

## 目次

- [機能概要](#機能概要)
- [クイックスタート](#クイックスタート)
- [インストール方法](#インストール方法)
- [使用方法](#使用方法)
  - [Makefile による実行](#makefileによる実行)
  - [コマンドライン実行](#コマンドライン実行)
  - [Python からの利用](#pythonからの利用)
- [設定方法](#設定方法)
- [モジュール構成](#モジュール構成)
- [主要機能の説明](#主要機能の説明)
- [分析結果の見方](#分析結果の見方)
- [特徴量一覧](#特徴量一覧)
- [必要システム要件](#必要システム要件)

## 機能概要

- **予測ホライゾン分析**: 異なる時間先（5 分、10 分、15 分、20 分、30 分など）の温度を予測し、精度を評価
- **ゾーン別分析**: 建物内の複数ゾーン（最大 12 ゾーン）について個別に予測モデルを構築
- **特徴量選択**: 温度予測に効果的な特徴量を自動選択
- **モデル評価**: RMSE、MAE、R² などの指標による予測精度評価
- **可視化機能**: 予測結果の散布図、時系列グラフ、特徴量重要度などを自動生成

## クイックスタート

```bash
# 1. セットアップ
make setup

# 2. すべてのゾーンで予測ホライゾン分析を実行
make analyze-all

# 3. 結果を確認
# 出力ディレクトリ（デフォルト: ./output）に結果が保存されます
```

## インストール方法

### 依存パッケージのインストール

```bash
# Makefileを使用する場合
make setup

# または手動でインストール
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
pip install -r requirements.txt
```

### 開発モードでのインストール

```bash
pip install -e .
```

## 使用方法

### Makefile による実行

最も簡単な実行方法は、同梱の Makefile を使用することです。

```bash
# ヘルプを表示
make help

# 基本コマンド
make run              # config.pyの設定に基づいて実行
make analyze-horizons # 予測ホライゾン分析を実行
make analyze-all      # すべてのゾーンで分析
make analyze-zone-0   # ゾーン0のみを分析

# カスタム設定での実行
make custom ZONES=0,1,2 HORIZONS=5,10,15
```

### コマンドライン実行

コマンドライン引数を直接指定することもできます。

```bash
# 基本実行形式
python thermal_prediction/main.py --analyze_horizons --file_path ./AllDayData.csv --output_dir ./output

# 特定のゾーンと予測ホライゾンを指定
python thermal_prediction/main.py --analyze_horizons --zones 0,1,2 --horizons 5,10,15,20

# 分析期間を指定
python thermal_prediction/main.py --analyze_horizons --start_date 2024-07-01 --end_date 2024-08-31
```

### Python からの利用

このモジュールは他の Python プログラムから直接インポートして利用できます。

```python
from thermal_prediction.utils import determine_thermo_status, prepare_features_for_sens_temp
from thermal_prediction.models import train_lgbm_model
import pandas as pd

# データを読み込み
df = pd.read_csv('AllDayData.csv')
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# サーモ状態を計算
thermo_df = determine_thermo_status(df)

# 特定のゾーンの特徴量を作成（5分後を予測）
zone = 0
X, y, features_df = prepare_features_for_sens_temp(df, thermo_df, zone, prediction_horizon=5)

# モデルを訓練・評価
model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)

# 上位特徴量を確認
print(f"Top 5 特徴量: {', '.join(importance_df.head(5)['Feature'].tolist())}")
```

## 設定方法

設定は以下の方法で行えます：

1. **config.py**: 基本設定を一元管理
2. **コマンドライン引数**: 実行時に特定の設定を上書き
3. **Makefile 変数**: Make 実行時に環境変数として設定

### 主な設定項目

| 設定項目                    | 説明                 | デフォルト値                   |
| --------------------------- | -------------------- | ------------------------------ |
| DEFAULT_FILE_PATH           | 入力データファイル   | ./AllDayData.csv               |
| DEFAULT_OUTPUT_DIR          | 出力ディレクトリ     | ./output/sens_temp_predictions |
| DEFAULT_START_DATE          | 分析開始日           | 2024-06-26                     |
| DEFAULT_END_DATE            | 分析終了日           | 2024-09-20                     |
| DEFAULT_PREDICTION_HORIZONS | 予測ホライゾン（分） | [5, 10, 15, 20, 30]            |
| ALL_ZONES                   | 全ゾーン番号         | 0〜11                          |
| THERMO_DEADBAND             | サーモ状態不感帯     | 1.0°C                          |

## モジュール構成

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

## 主要機能の説明

- **main.py**: コマンドライン引数を処理し、温度予測モデルの学習・評価を実行するエントリーポイント
- **analyze_horizon.py**: 異なる予測ホライズンでの温度予測モデルの性能を評価
- **utils/features.py**: 特徴量生成のためのユーティリティ関数
- **utils/thermo.py**: サーモ状態の決定などの HVAC 関連ロジック
- **models/lgbm.py**: LightGBM を使用したモデルの訓練と評価
- **visualization/**: 様々な可視化機能を提供するモジュール群

## 分析結果の見方

予測結果の散布図、時系列グラフ、特徴量重要度などを自動生成します。

## 特徴量一覧

温度予測モデルでは、以下の特徴量を使用します：

1. **時間特徴量**: 時刻、曜日、週末フラグ、夜間フラグなど
2. **温度ラグ特徴量**: 過去の温度データ（1 分前、5 分前、15 分前）
3. **温度変化率**: 直近 5 分間の温度変化率
4. **電力消費**: 室外機の電力消費データとそのラグ
5. **サーモ状態**: エアコンのサーモスタット状態と持続時間
6. **外気温特徴量**: 外気温とセンサー温度との差
7. **交互作用特徴量**: サーモ状態と温度の掛け合わせなど

特徴量はモデルの訓練前に重要度に基づいて選択され、最適な組み合わせが使用されます。

## 必要システム要件

- Python 3.7 以上
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- LightGBM
