# 温度予測モジュール用Makefile

# デフォルト設定
FILE_PATH := ./AllDayData.csv
OUTPUT_DIR := ./output
START_DATE := 2024-06-26
END_DATE := 2024-09-20
ZONES := all
HORIZONS := 5,10,15,20,30

# Python仮想環境のパス
VENV := venv

# ヘルプコマンドを表示
help:
	@echo "温度予測モジュール - 使用可能なコマンド"
	@echo ""
	@echo "基本コマンド:"
	@echo "  make setup            - 必要なパッケージをインストール"
	@echo "  make run              - configで設定された内容で分析を実行"
	@echo "  make clean            - 出力ディレクトリを削除"
	@echo ""
	@echo "分析コマンド:"
	@echo "  make analyze-horizons - 予測ホライゾン分析を実行"
	@echo "  make analyze-all      - すべてのゾーンで分析を実行"
	@echo "  make analyze-zone-0   - ゾーン0のみで分析を実行"
	@echo ""
	@echo "カスタム分析:"
	@echo "  make custom ZONES=0,1,2 HORIZONS=5,10,15 - カスタム設定で分析"
	@echo ""
	@echo "環境変数:"
	@echo "  FILE_PATH  - 入力データファイル (デフォルト: $(FILE_PATH))"
	@echo "  OUTPUT_DIR - 出力ディレクトリ (デフォルト: $(OUTPUT_DIR))"
	@echo "  START_DATE - 分析開始日 (デフォルト: $(START_DATE))"
	@echo "  END_DATE   - 分析終了日 (デフォルト: $(END_DATE))"
	@echo "  ZONES      - 分析対象ゾーン (デフォルト: $(ZONES))"
	@echo "  HORIZONS   - 予測ホライゾン (デフォルト: $(HORIZONS))"

# 初期セットアップ
setup:
	@echo "環境セットアップを開始します..."
	python -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt
	@echo "セットアップ完了"

# 日本語フォントのインストールターゲットを追加
setup-fonts:
	@echo "日本語フォントをインストールしています..."
	@if [ "$(shell uname)" = "Darwin" ]; then \
		pip install matplotlib; \
		pip install japanize-matplotlib; \
		echo "macOSに日本語フォントをインストールしました"; \
	elif [ "$(shell uname)" = "Linux" ]; then \
		sudo apt-get update && sudo apt-get install -y fonts-ipafont fonts-ipaexfont; \
		echo "Linuxに日本語フォントをインストールしました"; \
	else \
		echo "Windows環境では手動でフォントをインストールしてください"; \
	fi

# 基本実行
run:
	@echo "温度予測分析を実行します..."
	python thermal_prediction/main.py

# 予測ホライゾン分析
analyze-horizons:
	@echo "予測ホライゾン分析を実行します..."
	python thermal_prediction/main.py \
		--file_path $(FILE_PATH) \
		--output_dir $(OUTPUT_DIR) \
		--analyze_horizons \
		--zones $(ZONES) \
		--horizons $(HORIZONS) \
		--start_date $(START_DATE) \
		--end_date $(END_DATE)

# すべてのゾーンで分析
analyze-all:
	@echo "すべてのゾーンで分析を実行します..."
	python thermal_prediction/main.py \
		--file_path $(FILE_PATH) \
		--output_dir $(OUTPUT_DIR) \
		--analyze_horizons \
		--zones all \
		--horizons $(HORIZONS) \
		--start_date $(START_DATE) \
		--end_date $(END_DATE)

# ゾーン0のみで分析
analyze-zone-0:
	@echo "ゾーン0のみで分析を実行します..."
	python thermal_prediction/main.py \
		--file_path $(FILE_PATH) \
		--output_dir $(OUTPUT_DIR) \
		--analyze_horizons \
		--zones 0 \
		--horizons $(HORIZONS) \
		--start_date $(START_DATE) \
		--end_date $(END_DATE)

# カスタム分析
custom:
	@echo "カスタム設定で分析を実行します..."
	python thermal_prediction/main.py \
		--file_path $(FILE_PATH) \
		--output_dir $(OUTPUT_DIR) \
		--analyze_horizons \
		--zones $(ZONES) \
		--horizons $(HORIZONS) \
		--start_date $(START_DATE) \
		--end_date $(END_DATE)

# 出力ディレクトリをクリーン
clean:
	@echo "出力ディレクトリを削除しています..."
	rm -rf $(OUTPUT_DIR)
	@echo "削除完了"

# テスト実行ターゲット
test: ## 全てのテストを実行
	@echo "全てのテストを実行します..."
	@python tests/run_tests.py

test-features: ## 特徴量生成のテストを実行
	@echo "特徴量生成のテストを実行します..."
	@python tests/run_tests.py --module features

test-feature-selection: ## 特徴量選択のテストを実行
	@echo "特徴量選択のテストを実行します..."
	@python tests/run_tests.py --module feature_selection

test-models: ## モデルのテストを実行
	@echo "モデルのテストを実行します..."
	@python tests/run_tests.py --module lgbm

.PHONY: help setup run analyze-horizons analyze-all analyze-zone-0 custom clean setup-fonts test test-features test-feature-selection test-models
