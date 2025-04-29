"""
インタラクティブな可視化機能
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import json
import datetime

# 出力設定
pio.templates.default = "plotly_white"

def interactive_horizon_metrics(metrics_df, zone, output_dir):
    """
    予測ホライゾンごとの評価指標をインタラクティブに可視化する

    Args:
        metrics_df: 評価指標のデータフレーム
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        metrics_path: 保存したグラフのパス
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # RMSEのプロット
    fig.add_trace(
        go.Scatter(
            x=metrics_df['Horizon'],
            y=metrics_df['RMSE'],
            name="RMSE",
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10)
        )
    )

    # MAEのプロット
    fig.add_trace(
        go.Scatter(
            x=metrics_df['Horizon'],
            y=metrics_df['MAE'],
            name="MAE",
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        )
    )

    # R²のプロット（二次軸）
    fig.add_trace(
        go.Scatter(
            x=metrics_df['Horizon'],
            y=metrics_df['R²'],
            name="R²",
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=10)
        ),
        secondary_y=True
    )

    # レイアウトの設定
    fig.update_layout(
        title=f'ゾーン {zone} - 予測ホライゾンに対する評価指標の変化',
        xaxis=dict(
            title='予測ホライゾン（分）',
            tickmode='array',
            tickvals=metrics_df['Horizon']
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        width=900,
        height=600
    )

    # Y軸のタイトル設定
    fig.update_yaxes(title_text="RMSE / MAE (°C)", secondary_y=False)
    fig.update_yaxes(title_text="R²", secondary_y=True)

    # グリッド線の設定
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # ファイルの保存
    os.makedirs(output_dir, exist_ok=True)
    metrics_path_html = os.path.join(output_dir, f'zone_{zone}_metrics_interactive.html')
    metrics_path_png = os.path.join(output_dir, f'zone_{zone}_metrics_interactive.png')

    # HTML形式で保存（インタラクティブ）
    fig.write_html(metrics_path_html, include_plotlyjs='cdn')  # CDNを使用する

    # PNG形式でも保存（静的画像）
    try:
        fig.write_image(metrics_path_png)
    except Exception as e:
        print(f"静的画像の保存に失敗しました: {e}")
        # グラフだけ生成できていれば成功とみなす

    return metrics_path_html

def interactive_horizon_scatter(results, horizons, zone, output_dir):
    """
    予測値と実測値の散布図をインタラクティブに可視化

    Args:
        results: 結果辞書
        horizons: 予測ホライゾンのリスト
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        scatter_path: 保存したグラフのパス
    """
    # 表示するホライゾン数に応じて行数を決定
    n_horizons = len(horizons)
    rows = (n_horizons + 1) // 2  # 切り上げ除算

    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=[f'{h}分後予測' for h in horizons],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    # 各ホライゾンの散布図を追加
    for i, horizon in enumerate(horizons):
        if horizon not in results['predictions'] or horizon not in results['actual']:
            continue

        y_pred = results['predictions'][horizon]
        y_actual = results['actual'][horizon]

        row = i // 2 + 1
        col = i % 2 + 1

        # 予測値と実測値の範囲を取得
        min_val = min(np.min(y_pred), np.min(y_actual))
        max_val = max(np.max(y_pred), np.max(y_actual))
        padding = (max_val - min_val) * 0.05

        # 散布図
        fig.add_trace(
            go.Scatter(
                x=y_actual,
                y=y_pred,
                mode='markers',
                marker=dict(
                    size=8,
                    color='rgba(0, 0, 255, 0.5)',
                    line=dict(
                        color='rgba(0, 0, 255, 1.0)',
                        width=1
                    )
                ),
                name=f'{horizon}分後',
                text=[f'実測: {a:.2f}°C<br>予測: {p:.2f}°C<br>誤差: {p-a:.2f}°C'
                      for a, p in zip(y_actual, y_pred)],
                hoverinfo='text'
            ),
            row=row, col=col
        )

        # 45度線（理想的な予測）
        fig.add_trace(
            go.Scatter(
                x=[min_val-padding, max_val+padding],
                y=[min_val-padding, max_val+padding],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='理想線',
                showlegend=False
            ),
            row=row, col=col
        )

        # 傾向線
        z = np.polyfit(y_actual, y_pred, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min_val-padding, max_val+padding, 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                line=dict(color='green', width=2),
                name=f'傾向線 (y={z[0]:.2f}x+{z[1]:.2f})',
                showlegend=False
            ),
            row=row, col=col
        )

        # 評価指標を計算して注釈として表示
        rmse = np.sqrt(np.mean((y_actual - y_pred) ** 2))
        mae = np.mean(np.abs(y_actual - y_pred))
        r2 = 1 - np.sum((y_actual - y_pred) ** 2) / np.sum((y_actual - np.mean(y_actual)) ** 2)

        # 注釈テキスト
        annotation_text = (
            f'RMSE: {rmse:.3f}°C<br>'
            f'MAE: {mae:.3f}°C<br>'
            f'R²: {r2:.3f}'
        )

        # 注釈を追加
        fig.add_annotation(
            x=min_val + padding,
            y=max_val - padding,
            text=annotation_text,
            showarrow=False,
            bordercolor='black',
            borderwidth=1,
            borderpad=4,
            bgcolor='white',
            opacity=0.8,
            row=row, col=col
        )

        # 軸の範囲設定
        fig.update_xaxes(
            range=[min_val-padding, max_val+padding],
            row=row, col=col
        )
        fig.update_yaxes(
            range=[min_val-padding, max_val+padding],
            row=row, col=col
        )

    # 全体のレイアウト設定
    fig.update_layout(
        title=f'ゾーン {zone} - 各予測ホライゾンにおける実測値と予測値の比較',
        showlegend=False,
        width=1000,
        height=300 * rows,
    )

    # 全てのサブプロットに対する軸ラベル設定
    fig.update_xaxes(title_text='実測温度 (°C)')
    fig.update_yaxes(title_text='予測温度 (°C)')

    # ファイルの保存
    os.makedirs(output_dir, exist_ok=True)
    scatter_path_html = os.path.join(output_dir, f'zone_{zone}_scatter_interactive.html')
    scatter_path_png = os.path.join(output_dir, f'zone_{zone}_scatter_interactive.png')

    # HTML形式で保存（インタラクティブ）
    fig.write_html(scatter_path_html, include_plotlyjs='cdn')

    # PNG形式でも保存（静的画像）
    try:
        fig.write_image(scatter_path_png)
    except Exception as e:
        print(f"静的画像の保存に失敗しました: {e}")

    return scatter_path_html

def interactive_timeseries(results, horizon, zone, output_dir):
    """
    単一ホライゾンの時系列予測結果をインタラクティブに可視化

    Args:
        results: 結果辞書
        horizon: 予測ホライゾン（分）
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        path: 保存したグラフのパス
    """
    if horizon not in results['predictions'] or horizon not in results['actual'] or horizon not in results['timestamps']:
        return None

    # 予測値と実測値を取得
    y_pred = results['predictions'][horizon]
    y_actual = results['actual'][horizon]
    timestamps = results['timestamps'][horizon]

    # 実測値と予測値の差分
    y_diff = y_pred - y_actual

    # サブプロットを作成
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"ゾーン {zone} - {horizon}分後の温度予測と実測値", "予測誤差"),
        row_heights=[0.7, 0.3]
    )

    # 温度の実測値と予測値のプロット
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_actual,
            mode='lines',
            name='実測値',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_pred,
            mode='lines',
            name='予測値',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # 予測誤差のプロット
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=y_diff,
            mode='lines',
            name='予測誤差',
            line=dict(color='green', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.1)'
        ),
        row=2, col=1
    )

    # ゼロラインを追加（誤差なしの基準線）
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        row=2, col=1
    )

    # 全体のレイアウト設定
    fig.update_layout(
        title=f'ゾーン {zone} - {horizon}分後予測の時系列比較',
        xaxis_title="時刻",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=1000,
        height=600
    )

    # Y軸ラベルの設定
    fig.update_yaxes(title_text="温度 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="予測誤差 (°C)", row=2, col=1)

    # X軸の日付フォーマット設定
    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64, datetime.datetime)):
        fig.update_xaxes(
            tickformat="%m/%d %H:%M",
            tickangle=45,
            row=2, col=1
        )

    # ファイルの保存
    os.makedirs(output_dir, exist_ok=True)
    timeseries_path_html = os.path.join(output_dir, f'zone_{zone}_horizon_{horizon}_timeseries_interactive.html')
    timeseries_path_png = os.path.join(output_dir, f'zone_{zone}_horizon_{horizon}_timeseries_interactive.png')

    # HTML形式で保存（インタラクティブ）
    fig.write_html(timeseries_path_html, include_plotlyjs='cdn')

    # PNG形式でも保存（静的画像）
    try:
        fig.write_image(timeseries_path_png)
    except Exception as e:
        print(f"静的画像の保存に失敗しました: {e}")

    return timeseries_path_html

def interactive_feature_importance(importance_df, zone, output_dir):
    """
    特徴量重要度をインタラクティブに可視化

    Args:
        importance_df: 重要度のデータフレーム
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        importance_path: 保存したグラフのパス
    """
    if importance_df is None or len(importance_df) == 0:
        return None

    # 重要度の降順でソート
    importance_df = importance_df.sort_values('importance', ascending=False)

    # プロット作成
    fig = go.Figure()

    # 水平棒グラフで表示
    fig.add_trace(go.Bar(
        y=importance_df['feature_name'],
        x=importance_df['importance'],
        orientation='h',
        marker=dict(
            color='rgba(58, 71, 80, 0.6)',
            line=dict(color='rgba(58, 71, 80, 1.0)', width=3)
        )
    ))

    # レイアウト設定
    fig.update_layout(
        title=f'ゾーン {zone} - 特徴量重要度',
        xaxis_title='重要度',
        yaxis_title='特徴量',
        height=max(600, len(importance_df) * 20),  # 特徴量の数に応じて高さを調整
        width=900,
        margin=dict(l=200, r=20, t=50, b=50)  # 左マージンを広げて特徴量名を表示
    )

    # ファイルの保存
    os.makedirs(output_dir, exist_ok=True)
    importance_path_html = os.path.join(output_dir, f'zone_{zone}_feature_importance_interactive.html')
    importance_path_png = os.path.join(output_dir, f'zone_{zone}_feature_importance_interactive.png')

    # HTML形式で保存（インタラクティブ）
    fig.write_html(importance_path_html, include_plotlyjs='cdn')

    # PNG形式でも保存（静的画像）
    try:
        fig.write_image(importance_path_png)
    except Exception as e:
        print(f"静的画像の保存に失敗しました: {e}")

    return importance_path_html

def interactive_all_horizons_timeseries(results, horizons, zone, output_dir):
    """
    全予測ホライゾンの時系列予測結果をインタラクティブに可視化

    Args:
        results: 結果辞書
        horizons: 予測ホライゾンのリスト
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        path: 保存したグラフのパス
    """
    fig = go.Figure()

    # 最初のホライゾンのデータを使って時間軸を設定
    if horizons[0] in results['timestamps'] and horizons[0] in results['actual']:
        timestamps = results['timestamps'][horizons[0]]
        y_actual = results['actual'][horizons[0]]

        # タイムスタンプの確認とフォーマット
        if isinstance(timestamps[0], np.datetime64) or isinstance(timestamps[0], pd.Timestamp):
            x_time = timestamps
        elif isinstance(timestamps[0], str):
            try:
                x_time = [pd.to_datetime(ts) for ts in timestamps]
            except:
                x_time = np.arange(len(y_actual))
        else:
            x_time = np.arange(len(y_actual))

        # 実測値のプロット
        fig.add_trace(
            go.Scatter(
                x=x_time,
                y=y_actual,
                mode='lines',
                name='実測値',
                line=dict(color='black', width=3)
            )
        )

        # 各ホライゾンの予測値をプロット
        colors = px.colors.qualitative.Plotly
        for i, horizon in enumerate(horizons):
            if horizon in results['predictions']:
                y_pred = results['predictions'][horizon]
                color_idx = i % len(colors)

                fig.add_trace(
                    go.Scatter(
                        x=x_time,
                        y=y_pred,
                        mode='lines',
                        name=f'{horizon}分後予測',
                        line=dict(color=colors[color_idx], width=2)
                    )
                )

        # レイアウト設定
        fig.update_layout(
            title=f'ゾーン {zone} - 全予測ホライゾンの時系列比較',
            xaxis=dict(
                title='時間',
            ),
            yaxis=dict(
                title='温度 (°C)',
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            width=1200,
            height=600
        )

        # X軸設定（時間表示を整形）
        if isinstance(x_time[0], (pd.Timestamp, np.datetime64, datetime.datetime)):
            fig.update_xaxes(
                rangebreaks=[dict(pattern="hour", bounds=[23, 9])],  # 夜間を省略
                tickformat="%m/%d %H:%M"
            )

        # ファイルの保存
        os.makedirs(output_dir, exist_ok=True)
        combined_path_html = os.path.join(output_dir, f'zone_{zone}_all_horizons_interactive.html')
        combined_path_png = os.path.join(output_dir, f'zone_{zone}_all_horizons_interactive.png')

        # HTML形式で保存（インタラクティブ）
        fig.write_html(combined_path_html)

        # PNG形式でも保存（静的画像）
        fig.write_image(combined_path_png)

        return combined_path_html

    return None

def interactive_feature_ranks(results, zone, output_dir):
    """
    各ホライゾンにおける特徴量の重要度順位をインタラクティブに可視化

    Args:
        results: 結果辞書
        zone: ゾーン番号
        output_dir: 出力ディレクトリ

    Returns:
        path: 保存したグラフのパス
    """
    # 各ホライゾンの上位特徴量をまとめる
    top_features_unique = []
    for features in results['top_features']:
        for feature in features:
            if feature not in top_features_unique:
                top_features_unique.append(feature)

    if not top_features_unique:
        return None

    # 上位10件のみ表示
    top_features_unique = top_features_unique[:10]

    # 各ホライゾンごとに特徴量の順位を記録
    feature_ranks = pd.DataFrame(index=top_features_unique, columns=results['horizon'])
    for horizon_idx, horizon in enumerate(results['horizon']):
        features = results['top_features'][horizon_idx]
        for rank, feature in enumerate(features):
            if feature in top_features_unique:
                feature_ranks.loc[feature, horizon] = rank + 1

    # 欠損値をNaNで埋める
    feature_ranks = feature_ranks.fillna(np.nan)

    # Plotlyヒートマップの作成
    z_values = feature_ranks.values
    # NaNを-1に変換して特別な色で表示
    z_values_mod = np.where(np.isnan(z_values), -1, z_values)

    # テキスト作成（NaNの場合は空文字）
    text = [[str(int(val)) if not np.isnan(val) else "" for val in row] for row in z_values]

    fig = go.Figure(data=go.Heatmap(
        z=z_values_mod,
        x=feature_ranks.columns,
        y=feature_ranks.index,
        colorscale=[
            [0, 'rgba(255,255,255,0)'],  # NaNの色（透明）
            [0.01, 'rgb(0,0,255)'],      # 1位の色（青色）
            [1, 'rgb(255,255,0)']        # 最下位の色（黄色）
        ],
        text=text,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(
            title='順位',
            tickvals=list(range(1, int(np.nanmax(z_values)) + 1)),
            ticktext=list(range(1, int(np.nanmax(z_values)) + 1))
        ),
        hovertemplate='特徴量: %{y}<br>予測ホライゾン: %{x}分<br>順位: %{text}<extra></extra>'
    ))

    # レイアウト設定
    fig.update_layout(
        title=f'ゾーン {zone} - 異なる予測ホライゾンにおける特徴量重要度の順位',
        xaxis=dict(
            title='予測ホライゾン（分）',
            tickmode='array',
            tickvals=results['horizon']
        ),
        yaxis=dict(
            title='特徴量'
        ),
        width=900,
        height=600
    )

    # ファイルの保存
    os.makedirs(output_dir, exist_ok=True)
    ranks_path_html = os.path.join(output_dir, f'zone_{zone}_feature_ranks_interactive.html')
    ranks_path_png = os.path.join(output_dir, f'zone_{zone}_feature_ranks_interactive.png')

    # HTML形式で保存（インタラクティブ）
    fig.write_html(ranks_path_html)

    # PNG形式でも保存（静的画像）
    fig.write_image(ranks_path_png)

    return ranks_path_html

def interactive_all_zones_timeseries(results, horizon, zones, output_dir):
    """
    全ゾーンの温度予測結果をインタラクティブに可視化

    Args:
        results: 予測結果が含まれる辞書（キー: ゾーン番号）
        horizon: 予測ホライゾン（分）
        zones: 可視化対象のゾーン番号リスト
        output_dir: 出力ディレクトリ

    Returns:
        graph_paths: 保存したグラフのパスのリスト
    """
    graph_paths = []

    # 全ゾーンのサブプロットを作成
    # 1行に最大3つのゾーンを表示
    n_zones = len(zones)
    cols = min(3, n_zones)
    rows = (n_zones + cols - 1) // cols

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f'ゾーン {zone}' for zone in zones],
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # 各ゾーンの予測結果をプロット
    for i, zone in enumerate(zones):
        row = i // cols + 1
        col = i % cols + 1

        zone_str = str(zone)
        if zone_str not in results:
            print(f"ゾーン {zone} のデータがありません")
            continue

        zone_results = results[zone_str]

        # 予測値と実測値が存在するか確認
        horizon_key = f'pred_{horizon}'
        if horizon_key not in zone_results.columns or 'actual' not in zone_results.columns:
            print(f"ゾーン {zone} のホライゾン {horizon} の予測データまたは実測データがありません")
            continue

        # 予測値をプロット
        fig.add_trace(
            go.Scatter(
                x=zone_results.index,
                y=zone_results[horizon_key],
                mode='lines',
                name=f'予測 ({horizon}分)',
                line=dict(color='red'),
                showlegend=i==0  # 最初のゾーンのみ凡例表示
            ),
            row=row, col=col
        )

        # 実測値をプロット
        fig.add_trace(
            go.Scatter(
                x=zone_results.index,
                y=zone_results['actual'],
                mode='lines',
                name='実測',
                line=dict(color='blue'),
                showlegend=i==0  # 最初のゾーンのみ凡例表示
            ),
            row=row, col=col
        )

        # 軸ラベル
        fig.update_yaxes(title_text='温度 (℃)', row=row, col=col)
        if row == rows:  # 最終行の場合
            fig.update_xaxes(title_text='時間', row=row, col=col)

    # レイアウト設定
    fig.update_layout(
        title=f'全ゾーン温度予測 (ホライゾン: {horizon}分)',
        height=300 * rows,
        width=300 * cols,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, l=50, r=50, b=50)
    )

    # ファイルの保存
    os.makedirs(output_dir, exist_ok=True)
    graph_path_html = os.path.join(output_dir, f'all_zones_horizon_{horizon}_interactive.html')
    graph_path_png = os.path.join(output_dir, f'all_zones_horizon_{horizon}_interactive.png')

    # HTML形式で保存（インタラクティブ）
    fig.write_html(graph_path_html, include_plotlyjs='cdn')

    # PNG形式でも保存（静的画像）
    try:
        fig.write_image(graph_path_png)
    except Exception as e:
        print(f"静的画像の保存に失敗しました: {e}")

    graph_paths.append(graph_path_html)
    return graph_paths
