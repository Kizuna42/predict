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

### 1. 予測ホライゾン分析 (`analyze_horizon.py`)

異なる時間先の温度を予測して精度を比較します。例えば、5分後、10分後、15分後...の温度予測を行い、予測精度の変化を評価します。

```bash
# 実行例（Makefile）
make analyze-horizons

# または直接実行
python thermal_prediction/main.py --analyze_horizons
```

**出力**:
- 各ホライゾンごとのRMSE、MAE、R²のグラフ
- 予測値と実測値の散布図
- 時系列での予測と実測の比較グラフ
- 各ホライゾンでの重要特徴量ランキング

### 2. 特徴量生成 (`utils/features.py`)

温度予測のために様々な特徴量を生成します：

- 時間関連特徴量（時刻、曜日、周期変換）
- 温度履歴特徴量（過去の温度値、変化率）
- 電力消費特徴量（室外機電力消費）
- サーモ状態特徴量（サーモ状態と持続時間）
- 外部環境特徴量（外気温、日射量）

### 3. サーモ状態計算 (`utils/thermo.py`)

空調機のサーモスタット状態（ON/OFF）を推定します。室温と設定温度の関係から、冷房/暖房モードごとにサーモ状態を判定します。

### 4. モデルトレーニング (`models/lgbm.py`)

LightGBMを使用した回帰モデルを訓練します。特徴量選択、交差検証、早期停止などの機能を備えています。

## 分析結果の見方

分析結果は出力ディレクトリに以下のように保存されます：



主要な評価指標：
- **RMSE (Root Mean Squared Error)**: 予測誤差の二乗平均の平方根
- **MAE (Mean Absolute Error)**: 予測誤差の絶対値平均
- **R² (決定係数)**: モデルの説明力（1に近いほど良い）

## 特徴量一覧

このモジュールで使用される主な特徴量：

1. **基本時間特徴量**
   - 時刻（hour）、曜日（day_of_week）
   - 周期変換した時刻（hour_sin, hour_cos）
   - 週末フラグ（is_weekend）
   - 夜間フラグ（is_night）

2. **温度関連特徴量**
   - 過去の温度値（sens_temp_lag_1, sens_temp_lag_5, ...）
   - 温度変化率（sens_temp_change_5）
   - 移動平均温度（sens_temp_roll_15）

3. **サーモ状態特徴量**
   - サーモ状態（thermo_X）
   - サーモ状態変化（thermo_change）
   - サーモ状態持続時間（thermo_duration）

4. **電力消費特徴量**
   - 室外機電力消費値（power_L, power_M, power_R）
   - 過去の電力消費値（power_X_lag_1）
   - 電力消費移動平均（power_X_roll_15）

5. **外部環境特徴量**
   - 外気温（outdoor_temp）
   - 室内外温度差（temp_diff_outdoor）
   - 日射量関連特徴量（is_sunny_day）

## 必要システム要件

- Python 3.7以上
- 主要パッケージ:
  - pandas
  - numpy
  - scikit-learn
  - LightGBM
  - matplotlib
  - seaborn
