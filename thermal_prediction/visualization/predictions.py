"""
予測結果の可視化機能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ..utils.features import prepare_features_for_prediction_without_dropna

def visualize_zone_with_predictions(df, thermo_df, zone, power_col, model, features_df, output_dir,
                                   start_date=None, end_date=None, days_to_show=7):
    """
    ゾーンの予測結果を可視化する

    Args:
        df: 入力データフレーム
        thermo_df: サーモ状態データフレーム
        zone: ゾーン番号
        power_col: 電力列名
        model: 訓練済みモデル
        features_df: 特徴量データフレーム
        output_dir: 出力ディレクトリ
        start_date: 開始日
        end_date: 終了日
        days_to_show: 表示日数

    Returns:
        fig, fig_scatter: 生成した図のタプル
    """
    try:
        valid_col = f'AC_valid_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        sens_temp_col = f'sens_temp_{zone}'
        set_col = f'AC_set_{zone}'

        if valid_col not in df.columns or sens_temp_col not in df.columns:
            print(f"Missing required columns for visualization of zone {zone}")
            return None

        # 表示期間の設定
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        else:
            start = df['time_stamp'].min()
            end = start + pd.Timedelta(days=days_to_show)

        time_mask_df = (df['time_stamp'] >= start) & (df['time_stamp'] <= end)
        time_mask_thermo = (thermo_df['time_stamp'] >= start) & (thermo_df['time_stamp'] <= end)

        if time_mask_df.sum() == 0 or time_mask_thermo.sum() == 0:
            print(f"No data available for zone {zone} in the selected period")
            return None

        # 必要な列を抽出
        required_cols_df = ['time_stamp', valid_col, mode_col, sens_temp_col, power_col]
        if set_col in df.columns:
            required_cols_df.append(set_col)

        filtered_df = df.loc[time_mask_df, required_cols_df].copy()

        required_cols_thermo = ['time_stamp', thermo_col]
        thermo_or_col = f'thermo_{power_col}_or'
        if thermo_or_col in thermo_df.columns:
            required_cols_thermo.append(thermo_or_col)

        filtered_thermo = thermo_df.loc[time_mask_thermo, required_cols_thermo].copy()

        # データフレームの結合
        merged_df = pd.merge(filtered_df, filtered_thermo, on='time_stamp', how='left')

        # 予測用特徴量の準備
        model_features = model.feature_name()
        temp_features = prepare_features_for_prediction_without_dropna(merged_df, zone, power_col)

        if temp_features is None:
            print(f"Could not prepare features for prediction for zone {zone}")
            return None

        # モデルに必要な特徴量が全て揃っているか確認
        missing_features = [feat for feat in model_features if feat not in temp_features.columns]
        if missing_features:
            for feat in missing_features:
                temp_features[feat] = 0

            if len(missing_features) <= 3:
                print(f"Added missing features for zone {zone}: {', '.join(missing_features)}")
            else:
                print(f"Added {len(missing_features)} missing features for zone {zone}")

        # 予測用データの準備
        X_for_pred = temp_features[model_features].copy()
        valid_indices = ~X_for_pred.isna().any(axis=1)
        valid_rows = valid_indices.sum()

        if valid_rows == 0:
            print(f"No valid rows without NaN for prediction in zone {zone}")
            return None

        X_for_pred = X_for_pred[valid_indices].copy()

        # 予測の実行
        try:
            predictions = model.predict(X_for_pred, predict_disable_shape_check=True)
            merged_df[f'{sens_temp_col}_pred'] = np.nan
            merged_df.loc[merged_df.index[valid_indices], f'{sens_temp_col}_pred'] = predictions
        except Exception as e:
            print(f"Error making predictions for zone {zone}: {e}")
            if len(model_features) != X_for_pred.shape[1]:
                print(f"Feature count mismatch: model={len(model_features)}, data={X_for_pred.shape[1]}")
            return None

        # 時系列グラフの描画
        fig, ax = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1], sharex=True)

        # 上段: 温度と運転状態
        ax[0].plot(merged_df['time_stamp'], merged_df[sens_temp_col], 'c-', linewidth=2, label='実測温度')
        ax[0].plot(merged_df['time_stamp'], merged_df[f'{sens_temp_col}_pred'], 'm--', linewidth=2, label='予測温度')

        if set_col in merged_df.columns:
            ax[0].plot(merged_df['time_stamp'], merged_df[set_col], 'g-', linewidth=1.5, label='設定温度')
            ax[0].fill_between(
                merged_df['time_stamp'],
                merged_df[set_col] - 1.0,
                merged_df[set_col] + 1.0,
                color='gray',
                alpha=0.2,
                label='不感帯(±1.0°C)'
            )

        # サーモ状態の表示
        ax0_twin = ax[0].twinx()
        ax0_twin.plot(merged_df['time_stamp'], merged_df[thermo_col], 'r-', linewidth=1.5, label='サーモ状態')

        if mode_col in merged_df.columns:
            modes = merged_df[mode_col].copy()
            cooling_mask = (modes == 1)
            heating_mask = (modes == 2)

            if cooling_mask.any():
                cooling_df = merged_df[cooling_mask].copy()
                ax0_twin.plot(cooling_df['time_stamp'], cooling_df[thermo_col], 'b-', linewidth=1.5, label='冷房ON')

            if heating_mask.any():
                heating_df = merged_df[heating_mask].copy()
                ax0_twin.plot(heating_df['time_stamp'], heating_df[thermo_col], 'r-', linewidth=1.5, label='暖房ON')

        # Y軸の設定
        ax0_twin.set_ylim(-0.1, 1.1)
        ax0_twin.set_ylabel('運転状態', fontsize=12)
        ax0_twin.tick_params(axis='y', labelsize=10)

        # 温度範囲の設定
        temp_min = merged_df[sens_temp_col].min()
        temp_max = merged_df[sens_temp_col].max()
        padding = (temp_max - temp_min) * 0.1
        ax[0].set_ylim(temp_min - padding, temp_max + padding)

        # 凡例の設定
        lines0, labels0 = ax[0].get_legend_handles_labels()
        lines0_twin, labels0_twin = ax0_twin.get_legend_handles_labels()
        ax[0].legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper right', fontsize=10)

        # タイトルと軸ラベルの設定
        ax[0].set_ylabel('温度 (°C)', fontsize=12)
        ax[0].tick_params(axis='y', labelsize=10)
        ax[0].set_title(f'ゾーン {zone} - 温度予測と実測値', fontsize=14)
        ax[0].grid(True, alpha=0.3)

        # 下段: 消費電力と空調状態
        if power_col in merged_df.columns:
            ax[1].plot(merged_df['time_stamp'], merged_df[power_col], 'b-', linewidth=1.5, label='消費電力')
            ax[1].set_ylabel('消費電力 (kW)', fontsize=12)
            ax[1].tick_params(axis='y', labelsize=10)

            if valid_col in merged_df.columns:
                ax1_twin = ax[1].twinx()
                ax1_twin.plot(merged_df['time_stamp'], merged_df[valid_col], 'g--', linewidth=1.5, label='空調ON/OFF')
                ax1_twin.set_ylim(-0.1, 1.1)
                ax1_twin.set_ylabel('空調状態', fontsize=12)
                ax1_twin.tick_params(axis='y', labelsize=10)

                lines1, labels1 = ax[1].get_legend_handles_labels()
                lines1_twin, labels1_twin = ax1_twin.get_legend_handles_labels()
                ax[1].legend(lines1 + lines1_twin, labels1 + labels1_twin, loc='upper right', fontsize=10)
            else:
                ax[1].legend(loc='upper right', fontsize=10)

            ax[1].set_title(f'ゾーン {zone} - 消費電力と空調状態', fontsize=14)
            ax[1].grid(True, alpha=0.3)
        else:
            ax[1].text(0.5, 0.5, f'ゾーン {zone} の電力データがありません',
                      ha='center', va='center', fontsize=12)
            ax[1].set_title(f'ゾーン {zone}', fontsize=14)

        # X軸の設定
        fig.autofmt_xdate()
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        fig.tight_layout()

        # 散布図の作成
        fig_scatter = plt.figure(figsize=(12, 10))
        ax_scatter = fig_scatter.add_subplot(111)

        valid_data = merged_df.dropna(subset=[sens_temp_col, f'{sens_temp_col}_pred'])
        if not valid_data.empty:
            # 時間帯ごとに色分け
            timestamps = pd.to_datetime(valid_data['time_stamp'])
            hours = timestamps.dt.hour
            time_periods = pd.cut(
                hours,
                bins=[0, 6, 12, 18, 24],
                labels=['夜間\n(0-6時)', '午前\n(6-12時)', '午後\n(12-18時)', '夕方/夜\n(18-24時)'],
                include_lowest=True
            )

            cmap = plt.cm.viridis
            colors = {
                '夜間\n(0-6時)': cmap(0.1),
                '午前\n(6-12時)': cmap(0.4),
                '午後\n(12-18時)': cmap(0.7),
                '夕方/夜\n(18-24時)': cmap(0.9)
            }

            for period in np.unique(time_periods):
                mask = (time_periods == period)
                ax_scatter.scatter(
                    valid_data[sens_temp_col][mask],
                    valid_data[f'{sens_temp_col}_pred'][mask],
                    c=[colors[period]],
                    label=period,
                    alpha=0.7,
                    s=50,
                    edgecolor='k',
                    linewidth=0.3
                )

            # グラフの設定
            min_val = min(valid_data[sens_temp_col].min(), valid_data[f'{sens_temp_col}_pred'].min())
            max_val = max(valid_data[sens_temp_col].max(), valid_data[f'{sens_temp_col}_pred'].max())
            padding = (max_val - min_val) * 0.05

            # 完全一致線
            ax_scatter.plot(
                [min_val-padding, max_val+padding],
                [min_val-padding, max_val+padding],
                'k--',
                alpha=0.7,
                label='完全一致線'
            )

            ax_scatter.set_xlim(min_val-padding, max_val+padding)
            ax_scatter.set_ylim(min_val-padding, max_val+padding)
            ax_scatter.set_xlabel('実測温度 (°C)', fontsize=14)
            ax_scatter.set_ylabel('予測温度 (°C)', fontsize=14)
            ax_scatter.tick_params(axis='both', labelsize=12)

            # 評価指標の計算
            actual = valid_data[sens_temp_col]
            predicted = valid_data[f'{sens_temp_col}_pred']
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
            max_error = np.max(np.abs(actual - predicted))

            # 統計情報の表示
            stats_text = (
                f'評価指標:\n'
                f'RMSE: {rmse:.3f}°C\n'
                f'MAE: {mae:.3f}°C\n'
                f'R²: {r2:.3f}\n'
                f'最大誤差: {max_error:.3f}°C\n'
                f'データ数: {len(actual)}点'
            )

            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax_scatter.text(
                0.05, 0.95, stats_text,
                transform=ax_scatter.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=props
            )

            # 傾向線
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val-padding, max_val+padding, 100)
            ax_scatter.plot(
                x_trend,
                p(x_trend),
                "r-",
                linewidth=1.5,
                alpha=0.6,
                label=f'傾向線 (y={z[0]:.2f}x+{z[1]:.2f})'
            )

            ax_scatter.legend(loc='lower right', fontsize=12)
            ax_scatter.grid(True, alpha=0.3)

            title = f'ゾーン {zone} - 実測温度 vs 予測温度の比較'
            subtitle = f'期間: {start.strftime("%Y-%m-%d")} から {end.strftime("%Y-%m-%d")}'
            ax_scatter.set_title(f'{title}\n{subtitle}', fontsize=16)
        else:
            ax_scatter.text(0.5, 0.5, '有効なデータがありません', ha='center', va='center', fontsize=14)
            ax_scatter.set_title(f'ゾーン {zone} - データなし', fontsize=16)

        fig_scatter.tight_layout()

        return fig, fig_scatter

    except Exception as e:
        print(f"Error in visualize_zone_with_predictions for zone {zone}: {e}")
        import traceback
        traceback.print_exc()
        return None, None
