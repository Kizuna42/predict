def determine_thermo_status(df, deadband=1.0):
    """
    Calculate thermostat status based on set temperature, actual temperature and mode
    ベクトル化によりパフォーマンスを向上
    """
    df = df.reset_index(drop=True)
    df['time_stamp'] = pd.to_datetime(df['time_stamp'])

    min_date = df['time_stamp'].min()
    max_date = df['time_stamp'].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='1min')
    result_df = pd.DataFrame({'time_stamp': date_range})

    # 事前に全てのサーモ列を0で初期化
    thermo_cols = [f'thermo_{zone}' for zone in range(12)]
    result_df[thermo_cols] = 0

    for zone in range(12):
        valid_col = f'AC_valid_{zone}'  # Valid flag
        set_col = f'AC_set_{zone}'      # Set temperature
        temp_col = f'AC_temp_{zone}'    # Room temperature
        mode_col = f'AC_mode_{zone}'    # Mode (1=cooling, 2=heating)
        thermo_col = f'thermo_{zone}'   # Thermo state

        # 必要な列が存在するか確認
        if all(col in df.columns for col in [valid_col, set_col, temp_col, mode_col]):
            # NaN値を0で埋める
            df_zone = df[[valid_col, set_col, temp_col, mode_col, 'time_stamp']].copy()
            df_zone.fillna({valid_col: 0, set_col: 0, temp_col: 0, mode_col: 0}, inplace=True)

            # サーモ状態計算（ループを使用せずにベクトル化）
            df_zone[thermo_col] = 0
            mask = (df_zone[valid_col] > 0) & (df_zone[mode_col].isin([1, 2]))

            # 冷暖房モードとサーモ状態を追跡するために全行を走査する必要があるが、
            # NumPyでより効率的に実装
            rows = len(df_zone)
            thermo_values = np.zeros(rows)

            for i in range(1, rows):
                if not mask.iloc[i]:
                    thermo_values[i] = 0
                    continue

                current_mode = df_zone[mode_col].iloc[i]
                prev_thermo = thermo_values[i-1]

                if current_mode == 2:  # Heating mode
                    if prev_thermo == 0 and df_zone[temp_col].iloc[i] < df_zone[set_col].iloc[i] - deadband:
                        thermo_values[i] = 1  # OFF → ON
                    elif prev_thermo == 1 and df_zone[temp_col].iloc[i] > df_zone[set_col].iloc[i] + deadband:
                        thermo_values[i] = 0  # ON → OFF
                    else:
                        thermo_values[i] = prev_thermo  # 状態維持
                elif current_mode == 1:  # Cooling mode
                    if prev_thermo == 0 and df_zone[temp_col].iloc[i] > df_zone[set_col].iloc[i] + deadband:
                        thermo_values[i] = 1  # OFF → ON
                    elif prev_thermo == 1 and df_zone[temp_col].iloc[i] < df_zone[set_col].iloc[i] - deadband:
                        thermo_values[i] = 0  # ON → OFF
                    else:
                        thermo_values[i] = prev_thermo  # 状態維持

            df_zone[thermo_col] = thermo_values

            # 結果をマージ
            result_df = pd.merge(
                result_df,
                df_zone[['time_stamp', thermo_col]],
                on='time_stamp',
                how='left',
                suffixes=('', '_new')
            )
            # NaN値を0で埋めて整数型に変換
            col_name = f"{thermo_col}_new" if f"{thermo_col}_new" in result_df.columns else thermo_col
            result_df[thermo_col] = result_df[col_name].fillna(0).astype(int)

            # 重複列があれば削除
            if f"{thermo_col}_new" in result_df.columns:
                result_df = result_df.drop(columns=[f'{thermo_col}_new'])
        else:
            print(f"Warning: Required columns for zone {zone} not found. Setting thermo_{zone} to 0.")

    # 各外気ユニットのOR値を計算
    # ブール演算を使用して高速化
    result_df['thermo_L_or'] = (
        result_df['thermo_0'].astype(bool) |
        result_df['thermo_1'].astype(bool) |
        result_df['thermo_6'].astype(bool) |
        result_df['thermo_7'].astype(bool)
    ).astype(int)

    result_df['thermo_M_or'] = (
        result_df['thermo_2'].astype(bool) |
        result_df['thermo_3'].astype(bool) |
        result_df['thermo_8'].astype(bool) |
        result_df['thermo_9'].astype(bool)
    ).astype(int)

    result_df['thermo_R_or'] = (
        result_df['thermo_4'].astype(bool) |
        result_df['thermo_5'].astype(bool) |
        result_df['thermo_10'].astype(bool) |
        result_df['thermo_11'].astype(bool)
    ).astype(int)

    return result_df

def prepare_features_for_sens_temp(df, thermo_df, zone, look_back=60):
    """
    効率化された特徴量エンジニアリング関数
    未来データのリーケージを防止するよう修正
    """
    # ゾーン別カラム定義
    valid_col = f'AC_valid_{zone}'
    mode_col = f'AC_mode_{zone}'
    thermo_col = f'thermo_{zone}'
    sens_temp_col = f'sens_temp_{zone}'

    # ゾーンが属する室外機を特定
    L_zones = [0, 1, 6, 7]
    M_zones = [2, 3, 8, 9]
    R_zones = [4, 5, 10, 11]

    if zone in L_zones:
        power_col = 'L'
        thermo_or_col = 'thermo_L_or'
    elif zone in M_zones:
        power_col = 'M'
        thermo_or_col = 'thermo_M_or'
    elif zone in R_zones:
        power_col = 'R'
        thermo_or_col = 'thermo_R_or'
    else:
        print(f"Zone {zone} not assigned to any outdoor unit")
        return None, None, None

    # 必要最小限のカラムだけをマージしてメモリ使用量を削減
    required_thermo_cols = ['time_stamp', thermo_col, thermo_or_col]
    thermo_subset = thermo_df[required_thermo_cols].copy()

    required_df_cols = ['time_stamp', valid_col, mode_col, sens_temp_col, power_col]
    optional_cols = ['outdoor_temp', 'solar_radiation', 'humidity']
    for col in optional_cols:
        if col in df.columns:
            required_df_cols.append(col)

    df_subset = df[required_df_cols].copy()

    # マージ
    merged_df = pd.merge(df_subset, thermo_subset, on='time_stamp', how='left')

    # 必要な列が存在するか確認
    required_cols = [valid_col, mode_col, thermo_col, sens_temp_col, power_col, thermo_or_col]
    missing_cols = [col for col in required_cols if col not in merged_df.columns]

    if missing_cols:
        print(f"Missing required columns for zone {zone}: {missing_cols}")
        return None, None, None

    # 時間関連特徴量を一度に計算
    merged_df['hour'] = merged_df['time_stamp'].dt.hour
    merged_df['day_of_week'] = merged_df['time_stamp'].dt.dayofweek
    merged_df['month'] = merged_df['time_stamp'].dt.month
    merged_df['is_weekend'] = (merged_df['day_of_week'] >= 5).astype(int)
    merged_df['is_night'] = ((merged_df['hour'] >= 19) | (merged_df['hour'] <= 6)).astype(int)
    merged_df['is_morning'] = ((merged_df['hour'] > 6) & (merged_df['hour'] <= 12)).astype(int)
    merged_df['is_afternoon'] = ((merged_df['hour'] > 12) & (merged_df['hour'] < 19)).astype(int)

    # 三角関数特徴量をベクトル化
    hour_rad = 2 * np.pi * merged_df['hour'] / 24
    merged_df['hour_sin'] = np.sin(hour_rad)
    merged_df['hour_cos'] = np.cos(hour_rad)

    day_rad = 2 * np.pi * merged_df['time_stamp'].dt.day / 31
    merged_df['day_sin'] = np.sin(day_rad)
    merged_df['day_cos'] = np.cos(day_rad)

    week_rad = 2 * np.pi * merged_df['day_of_week'] / 7
    merged_df['week_sin'] = np.sin(week_rad)
    merged_df['week_cos'] = np.cos(week_rad)

    # ラグ特徴量をまとめて計算
    lag_cols = {}

    # 重要: シフト方向を修正（未来の漏洩を防止）
    # センサー温度のラグ（過去の値）
    for lag in [1, 5, 15, 30, 60]:
        if lag <= look_back:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            # 修正: shift(-lag)で過去のデータを参照
            merged_df[lag_col] = merged_df[sens_temp_col].shift(-lag)
            lag_cols[lag_col] = True

    # 温度変化率（過去データのみを使用）
    for lag in [1, 5, 15]:
        if lag <= look_back:
            change_col = f'{sens_temp_col}_change_{lag}'
            # 修正: 過去データ同士の差分を計算
            merged_df[change_col] = (merged_df[f'{sens_temp_col}_lag_1'] - merged_df[f'{sens_temp_col}_lag_{lag+1}']) / lag
            lag_cols[change_col] = True

    # 電力消費のラグ（過去の値）
    for lag in [1, 5, 15, 30]:
        if lag <= look_back:
            lag_col = f'{power_col}_lag_{lag}'
            # 修正: shift(-lag)で過去のデータを参照
            merged_df[lag_col] = merged_df[power_col].shift(-lag)
            lag_cols[lag_col] = True

    # 移動平均（ローリング）特徴量 - 過去データのみ使用
    windows = [5, 15, 30, 60]
    for window in windows:
        if window <= look_back:
            # 修正: まず過去の値を参照してから移動平均を計算
            # センサー温度の過去データから移動平均を計算
            roll_temp = f'{sens_temp_col}_roll_{window}'
            temp_past = merged_df[sens_temp_col].shift(-1)  # t-1の値
            merged_df[roll_temp] = temp_past.rolling(window=window, min_periods=1).mean()
            lag_cols[roll_temp] = True

            # 電力消費の過去データから移動平均を計算
            roll_power = f'{power_col}_roll_{window}'
            power_past = merged_df[power_col].shift(-1)  # t-1の値
            merged_df[roll_power] = power_past.rolling(window=window, min_periods=1).mean()
            lag_cols[roll_power] = True

            # 温度の標準偏差も同様に過去データから計算
            std_temp = f'{sens_temp_col}_std_{window}'
            merged_df[std_temp] = temp_past.rolling(window=window, min_periods=1).std()
            lag_cols[std_temp] = True

    # サーモ状態変化と持続時間を計算
    merged_df['thermo_change'] = merged_df[thermo_col].diff(-1).fillna(0)  # 修正: 前の時刻との差分

    # サーモ状態持続時間を効率的に計算（過去方向に計算）
    # 修正: サーモ状態が変わったところを特定し、過去方向にリセット
    reset_points = (merged_df[thermo_col] != merged_df[thermo_col].shift(-1)).astype(int)
    reset_points.iloc[-1] = 1  # 最後の点もリセットポイント

    # リセットポイントから各グループ識別子を作成
    group_id = reset_points.iloc[::-1].cumsum().iloc[::-1]  # 過去方向に累積

    # 各グループ内でのインデックスを取得（これが持続時間）
    merged_df['thermo_duration'] = merged_df.groupby(group_id).cumcount().iloc[::-1]  # 過去方向にカウント

    # 同様に、サーモ状態変化からの経過時間を計算（過去方向に）
    change_points = (merged_df['thermo_change'] != 0).astype(int)
    change_group_id = change_points.iloc[::-1].cumsum().iloc[::-1]
    merged_df['time_since_thermo_change'] = merged_df.groupby(change_group_id).cumcount().iloc[::-1]

    # 外気温特徴量（存在する場合）- 過去のデータのみ使用
    if 'outdoor_temp' in merged_df.columns:
        merged_df['outdoor_temp_lag_1'] = merged_df['outdoor_temp'].shift(-1)  # 修正: t-1の外気温
        # 過去の内外温度差
        merged_df['temp_diff_outdoor'] = merged_df[sens_temp_col].shift(-1) - merged_df['outdoor_temp_lag_1']
        lag_cols['outdoor_temp_lag_1'] = True
        lag_cols['temp_diff_outdoor'] = True

        for window in [15, 60]:
            if window <= look_back:
                outdoor_roll = f'outdoor_temp_roll_{window}'
                # 修正: 過去データの移動平均
                outdoor_past = merged_df['outdoor_temp'].shift(-1)  # t-1の値
                merged_df[outdoor_roll] = outdoor_past.rolling(window=window, min_periods=1).mean()
                lag_cols[outdoor_roll] = True

    # 日射量特徴量（存在する場合）- 過去のデータのみ使用
    if 'solar_radiation' in merged_df.columns:
        merged_df['solar_radiation_lag_1'] = merged_df['solar_radiation'].shift(-1)  # 修正: t-1の日射量
        lag_cols['solar_radiation_lag_1'] = True

        for window in [15, 60]:
            if window <= look_back:
                solar_roll = f'solar_radiation_roll_{window}'
                # 修正: 過去データの移動平均
                solar_past = merged_df['solar_radiation'].shift(-1)  # t-1の値
                merged_df[solar_roll] = solar_past.rolling(window=window, min_periods=1).mean()
                lag_cols[solar_roll] = True

    # 湿度特徴量（存在する場合）- 過去のデータのみ使用
    if 'humidity' in merged_df.columns:
        merged_df['humidity_lag_1'] = merged_df['humidity'].shift(-1)  # 修正: t-1の湿度
        lag_cols['humidity_lag_1'] = True

        for window in [15, 60]:
            if window <= look_back:
                humidity_roll = f'humidity_roll_{window}'
                # 修正: 過去データの移動平均
                humidity_past = merged_df['humidity'].shift(-1)  # t-1の値
                merged_df[humidity_roll] = humidity_past.rolling(window=window, min_periods=1).mean()
                lag_cols[humidity_roll] = True

    # インタラクション特徴量 - 過去データを使用
    # 修正: 過去の温度データを使用
    merged_df['thermo_x_temp'] = merged_df[thermo_col] * merged_df[sens_temp_col].shift(-1)
    merged_df['thermo_duration_x_temp'] = merged_df['thermo_duration'] * merged_df[sens_temp_col].shift(-1)

    # 修正: 過去の温度変化率を使用
    if f'{sens_temp_col}_change_5' in merged_df.columns:
        merged_df['thermo_on_temp_change'] = merged_df[thermo_col] * merged_df[f'{sens_temp_col}_change_5']
    else:
        merged_df['thermo_on_temp_change'] = 0

    # 新たな特徴量: 時間帯と気象条件の組み合わせ
    if 'outdoor_temp' in merged_df.columns and 'solar_radiation' in merged_df.columns:
        # 日中で日射量が高い（晴れ）
        merged_df['is_sunny_day'] = ((merged_df['hour'] >= 9) &
                                    (merged_df['hour'] <= 17) &
                                    (merged_df['solar_radiation'].shift(-1) >
                                     merged_df['solar_radiation'].shift(-1).mean())).astype(int)

        # 夜間で外気温が低い
        merged_df['is_cold_night'] = ((merged_df['is_night'] == 1) &
                                     (merged_df['outdoor_temp'].shift(-1) <
                                      merged_df['outdoor_temp'].shift(-1).mean())).astype(int)

    # NaN値を含む行を削除
    merged_df = merged_df.dropna()

    # モデルに使用する特徴量を定義
    feature_columns = [
        # 時間特徴量
        'hour', 'day_of_week', 'month', 'is_weekend',
        'is_night', 'is_morning', 'is_afternoon', 'hour_sin', 'hour_cos',
        'day_sin', 'day_cos', 'week_sin', 'week_cos',

        # HVAC操作特徴量
        valid_col, mode_col, thermo_col, thermo_or_col,
        'thermo_duration', 'time_since_thermo_change', 'thermo_change',

        # 電力消費
        power_col,

        # インタラクション特徴量
        'thermo_x_temp', 'thermo_duration_x_temp', 'thermo_on_temp_change'
    ]

    # 新しい特徴量があれば追加
    if 'is_sunny_day' in merged_df.columns:
        feature_columns.append('is_sunny_day')
    if 'is_cold_night' in merged_df.columns:
        feature_columns.append('is_cold_night')

    # 先に計算したラグ特徴量を追加
    feature_columns.extend([col for col in lag_cols.keys() if col in merged_df.columns])

    # 目標変数
    target = sens_temp_col

    # Xとyを準備
    X = merged_df[feature_columns]
    y = merged_df[target]

    return X, y, merged_df

def train_lgbm_model(X, y, test_size=0.2, random_state=42):
    """
    LightGBMモデルのトレーニング関数
    時系列データに適した分割方法に修正
    """
    try:
        # データ分割 - 時系列を考慮した分割方法に変更
        # 単純なランダム分割ではなく、時系列順に分割
        train_size = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        # LightGBM用のデータセット作成
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        # 最適化パラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'lambda_l1': 0.1,  # L1正則化
            'lambda_l2': 0.1,  # L2正則化
            'force_col_wise': True,  # 列単位での処理を強制（メモリ使用量削減）
        }

        # モデルトレーニング（早期停止あり）
        callbacks = [
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=100, show_stdv=False)  # 進捗ログ
        ]

        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, test_data],
            callbacks=callbacks
        )

        # テストセットでの予測
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

        # 複数の評価指標を計算
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))

        # 0による除算を防止（小さな値を加算）
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

        # 評価指標を表示
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")

        # 特徴量重要度分析
        importance = model.feature_importance(importance_type='gain')
        feature_names = model.feature_name()

        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # トップ10の特徴量を表示
        print("\nTop 10 Feature Importance:")
        print(importance_df.head(10))

        return model, X_test, y_test, y_pred, importance_df

    except Exception as e:
        print(f"Error in train_lgbm_model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def visualize_feature_importance(importance_df, zone, output_dir):
    """
    特徴量重要度の可視化
    パフォーマンス最適化とエラーハンドリング強化
    """
    try:
        # データチェック
        if importance_df is None or importance_df.empty:
            print(f"Zone {zone}: 特徴量重要度データが空です")
            return None

        # トップ15の特徴量を抽出
        top_features = importance_df.head(15).copy()

        # 重要度の合計を計算して割合に変換
        total_importance = importance_df['Importance'].sum()
        if total_importance > 0:  # ゼロ除算を防ぐ
            top_features['Importance_Pct'] = top_features['Importance'] / total_importance * 100
        else:
            top_features['Importance_Pct'] = 0

        # 図の作成
        plt.figure(figsize=(12, 8))

        # 水平棒グラフ（パーセント表示）
        bars = plt.barh(
            top_features['Feature'],
            top_features['Importance_Pct'],
            color=plt.cm.viridis(np.linspace(0, 0.8, len(top_features))),  # カラーグラデーション
            edgecolor='gray',
            alpha=0.8
        )

        # 各バーに値を表示
        for i, bar in enumerate(bars):
            plt.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{top_features['Importance_Pct'].iloc[i]:.1f}%",
                va='center'
            )

        # グラフのスタイル設定
        plt.xlabel('重要度 (%)')
        plt.ylabel('特徴量')
        plt.title(f'ゾーン {zone} のトップ15特徴量重要度')
        plt.gca().invert_yaxis()  # 最も重要な特徴量を上に表示
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # ファイル保存
        fig_path = os.path.join(output_dir, f'zone_{zone}_feature_importance.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        return fig_path

    except Exception as e:
        print(f"Error in visualize_feature_importance for zone {zone}: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_features_for_prediction_without_dropna(df, zone, power_col):
    """
    欠損値を削除せずに特徴量を準備する関数
    未来データのリーケージを防止するよう修正
    """
    try:
        # ゾーン別カラム定義
        valid_col = f'AC_valid_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        sens_temp_col = f'sens_temp_{zone}'

        # サーモORカラムを決定
        thermo_or_map = {'L': 'thermo_L_or', 'M': 'thermo_M_or', 'R': 'thermo_R_or'}
        if power_col not in thermo_or_map:
            raise ValueError(f"Invalid power column: {power_col}")

        thermo_or_col = thermo_or_map[power_col]

        # 必要なカラムが存在するか確認
        required_cols = [valid_col, mode_col, thermo_col, sens_temp_col, power_col, thermo_or_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for zone {zone}: {', '.join(missing_cols)}")

        # カラムをコピーせずに処理の高速化のためにビューを使用
        features_df = df

        # 時間関連特徴量を一度に計算
        hour = features_df['time_stamp'].dt.hour
        day_of_week = features_df['time_stamp'].dt.dayofweek
        day_of_month = features_df['time_stamp'].dt.day
        month = features_df['time_stamp'].dt.month

        # 基本時間特徴量
        features_df['hour'] = hour
        features_df['day_of_week'] = day_of_week
        features_df['month'] = month
        features_df['is_weekend'] = (day_of_week >= 5).astype(int)

        # 時間帯特徴量
        features_df['is_night'] = ((hour >= 19) | (hour <= 6)).astype(int)
        features_df['is_morning'] = ((hour > 6) & (hour <= 12)).astype(int)
        features_df['is_afternoon'] = ((hour > 12) & (hour < 19)).astype(int)

        # 周期性を捉える特徴量（三角関数）- ベクトル化
        hour_rad = 2 * np.pi * hour / 24
        features_df['hour_sin'] = np.sin(hour_rad)
        features_df['hour_cos'] = np.cos(hour_rad)

        day_rad = 2 * np.pi * day_of_month / 31
        features_df['day_sin'] = np.sin(day_rad)
        features_df['day_cos'] = np.cos(day_rad)

        week_rad = 2 * np.pi * day_of_week / 7
        features_df['week_sin'] = np.sin(week_rad)
        features_df['week_cos'] = np.cos(week_rad)

        # ラグ特徴量の計算 - 過去データを正しく参照
        # センサー温度のラグ
        for lag in [1, 5, 15, 30, 60]:
            features_df[f'{sens_temp_col}_lag_{lag}'] = features_df[sens_temp_col].shift(-lag)

        # 温度変化率を計算 - 過去データ同士の比較
        for lag in [1, 5, 15]:
            features_df[f'{sens_temp_col}_change_{lag}'] = (
                features_df[f'{sens_temp_col}_lag_1'] - features_df[f'{sens_temp_col}_lag_{lag+1}']
            ) / lag

        # 電力消費のラグ - 過去データを参照
        for lag in [1, 5, 15, 30]:
            features_df[f'{power_col}_lag_{lag}'] = features_df[power_col].shift(-lag)

        # 移動平均特徴量 - 過去データのみを使用
        windows = [5, 15, 30, 60]
        for window in windows:
            # センサー温度の移動平均
            temp_past = features_df[sens_temp_col].shift(-1)  # t-1の値
            features_df[f'{sens_temp_col}_roll_{window}'] = temp_past.rolling(
                window=window, min_periods=1).mean()

            # 電力消費の移動平均
            power_past = features_df[power_col].shift(-1)  # t-1の値
            features_df[f'{power_col}_roll_{window}'] = power_past.rolling(
                window=window, min_periods=1).mean()

            # センサー温度の標準偏差
            features_df[f'{sens_temp_col}_std_{window}'] = temp_past.rolling(
                window=window, min_periods=1).std()

        # サーモ状態変化と持続時間 - 過去方向に計算
        features_df['thermo_change'] = features_df[thermo_col].diff(-1).fillna(0)

        # サーモ状態持続時間を効率的に計算 - 過去方向
        reset_points = (features_df[thermo_col] != features_df[thermo_col].shift(-1)).astype(int)
        reset_points.iloc[-1] = 1  # 最後の点もリセットポイント
        group_id = reset_points.iloc[::-1].cumsum().iloc[::-1]
        features_df['thermo_duration'] = features_df.groupby(group_id).cumcount().iloc[::-1]

        # サーモ状態変化からの経過時間 - 過去方向
        change_points = (features_df['thermo_change'] != 0).astype(int)
        change_group_id = change_points.iloc[::-1].cumsum().iloc[::-1]
        features_df['time_since_thermo_change'] = features_df.groupby(change_group_id).cumcount().iloc[::-1]

        # 外気温特徴量（存在する場合）- 過去データのみ使用
        if 'outdoor_temp' in features_df.columns:
            features_df['outdoor_temp_lag_1'] = features_df['outdoor_temp'].shift(-1)
            features_df['temp_diff_outdoor'] = features_df[sens_temp_col].shift(-1) - features_df['outdoor_temp_lag_1']

            for window in [15, 60]:
                outdoor_past = features_df['outdoor_temp'].shift(-1)  # t-1の値
                features_df[f'outdoor_temp_roll_{window}'] = outdoor_past.rolling(
                    window=window, min_periods=1).mean()

        # インタラクション特徴量 - 過去データを使用
        if thermo_col in features_df.columns:
            # サーモ状態と過去の温度の相互作用
            features_df['thermo_x_temp'] = features_df[thermo_col] * features_df[sens_temp_col].shift(-1)

            # サーモON持続時間と過去の温度の相互作用
            features_df['thermo_duration_x_temp'] = features_df['thermo_duration'] * features_df[sens_temp_col].shift(-1)

        # 時間帯と気象条件の組み合わせ特徴量
        if 'outdoor_temp' in features_df.columns and 'solar_radiation' in features_df.columns:
            # 日中で日射量が高い（晴れ）
            features_df['is_sunny_day'] = ((features_df['hour'] >= 9) &
                                         (features_df['hour'] <= 17) &
                                         (features_df['solar_radiation'].shift(-1) >
                                          features_df['solar_radiation'].shift(-1).mean())).astype(int)

            # 夜間で外気温が低い
            features_df['is_cold_night'] = ((features_df['is_night'] == 1) &
                                          (features_df['outdoor_temp'].shift(-1) <
                                           features_df['outdoor_temp'].shift(-1).mean())).astype(int)

        # 特徴量リスト
        feature_columns = [
            # 時間特徴量
            'hour', 'day_of_week', 'month', 'is_weekend',
            'is_night', 'is_morning', 'is_afternoon', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos', 'week_sin', 'week_cos',

            # HVAC操作特徴量
            valid_col, mode_col, thermo_col, thermo_or_col,
            'thermo_duration', 'time_since_thermo_change', 'thermo_change',

            # 電力消費
            power_col
        ]

        # 新しい特徴量を追加
        if 'is_sunny_day' in features_df.columns:
            feature_columns.append('is_sunny_day')
        if 'is_cold_night' in features_df.columns:
            feature_columns.append('is_cold_night')

        # ラグ特徴量
        for lag in [1, 5, 15, 30, 60]:
            lag_col = f'{sens_temp_col}_lag_{lag}'
            if lag_col in features_df.columns:
                feature_columns.append(lag_col)

        # 温度変化率特徴量
        for lag in [1, 5, 15]:
            change_col = f'{sens_temp_col}_change_{lag}'
            if change_col in features_df.columns:
                feature_columns.append(change_col)

        # 電力ラグ特徴量
        for lag in [1, 5, 15, 30]:
            lag_col = f'{power_col}_lag_{lag}'
            if lag_col in features_df.columns:
                feature_columns.append(lag_col)

        # 移動平均特徴量
        for window in windows:
            roll_temp = f'{sens_temp_col}_roll_{window}'
            roll_power = f'{power_col}_roll_{window}'
            std_temp = f'{sens_temp_col}_std_{window}'

            for col in [roll_temp, roll_power, std_temp]:
                if col in features_df.columns:
                    feature_columns.append(col)

        # インタラクション特徴量
        for col in ['thermo_x_temp', 'thermo_duration_x_temp']:
            if col in features_df.columns:
                feature_columns.append(col)

        # 外気温特徴量
        if 'outdoor_temp_lag_1' in features_df.columns:
            feature_columns.append('outdoor_temp_lag_1')
            feature_columns.append('temp_diff_outdoor')

            for window in [15, 60]:
                outdoor_roll = f'outdoor_temp_roll_{window}'
                if outdoor_roll in features_df.columns:
                    feature_columns.append(outdoor_roll)

        # 存在しない特徴量を除外（安全対策）
        final_columns = [col for col in feature_columns if col in features_df.columns]

        return features_df[final_columns]

    except Exception as e:
        print(f"Error in prepare_features_for_prediction_without_dropna: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_zone_with_predictions(df, thermo_df, zone, power_col, model, features_df, output_dir,
                                   start_date=None, end_date=None, days_to_show=7):
    """
    ゾーンデータの可視化関数
    パフォーマンス最適化とエラーハンドリング強化
    """
    try:
        # 必要なカラムを定義
        valid_col = f'AC_valid_{zone}'
        mode_col = f'AC_mode_{zone}'
        thermo_col = f'thermo_{zone}'
        sens_temp_col = f'sens_temp_{zone}'
        set_col = f'AC_set_{zone}'

        # 必須カラムが存在するか確認
        if valid_col not in df.columns or sens_temp_col not in df.columns:
            print(f"Missing required columns for visualization of zone {zone}")
            return None

        # 日付範囲の設定
        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
        else:
            start = df['time_stamp'].min()
            end = start + pd.Timedelta(days=days_to_show)

        # データのフィルタリング - コピーを最小限にしてメモリ使用を削減
        time_mask_df = (df['time_stamp'] >= start) & (df['time_stamp'] <= end)
        time_mask_thermo = (thermo_df['time_stamp'] >= start) & (thermo_df['time_stamp'] <= end)

        if time_mask_df.sum() == 0 or time_mask_thermo.sum() == 0:
            print(f"No data available for zone {zone} in the selected period")
            return None

        # 必要な列だけを抽出してメモリ使用量を削減
        required_cols_df = ['time_stamp', valid_col, mode_col, sens_temp_col, power_col]
        if set_col in df.columns:
            required_cols_df.append(set_col)

        filtered_df = df.loc[time_mask_df, required_cols_df].copy()

        required_cols_thermo = ['time_stamp', thermo_col]
        thermo_or_col = f'thermo_{power_col}_or'
        if thermo_or_col in thermo_df.columns:
            required_cols_thermo.append(thermo_or_col)

        filtered_thermo = thermo_df.loc[time_mask_thermo, required_cols_thermo].copy()

        # データのマージ
        merged_df = pd.merge(filtered_df, filtered_thermo, on='time_stamp', how='left')

        # モデルの特徴量リストを取得
        model_features = model.feature_name()

        # 予測用の特徴量を準備
        temp_features = prepare_features_for_prediction_without_dropna(merged_df, zone, power_col)
        if temp_features is None:
            print(f"Could not prepare features for prediction for zone {zone}")
            return None

        # モデルの特徴量と一致するようにする
        missing_features = [feat for feat in model_features if feat not in temp_features.columns]
        if missing_features:
            # 足りない特徴量を0で埋める
            for feat in missing_features:
                temp_features[feat] = 0

            if len(missing_features) <= 3:  # 少数の場合だけ詳細をログ
                print(f"Added missing features for zone {zone}: {', '.join(missing_features)}")
            else:
                print(f"Added {len(missing_features)} missing features for zone {zone}")

        # モデルの特徴量順に並べ替え
        X_for_pred = temp_features[model_features].copy()

        # 欠損値を含む行を除外
        valid_indices = ~X_for_pred.isna().any(axis=1)
        valid_rows = valid_indices.sum()

        if valid_rows == 0:
            print(f"No valid rows without NaN for prediction in zone {zone}")
            return None

        X_for_pred = X_for_pred[valid_indices].copy()

        # 予測実行
        try:
            # NaN値のない行だけを予測
            predictions = model.predict(X_for_pred, predict_disable_shape_check=True)

            # 予測値をマージしたデータフレームに格納
            merged_df[f'{sens_temp_col}_pred'] = np.nan
            merged_df.loc[merged_df.index[valid_indices], f'{sens_temp_col}_pred'] = predictions

        except Exception as e:
            print(f"Error making predictions for zone {zone}: {e}")
            print(f"Model features: {len(model_features)}, Prediction features: {X_for_pred.shape[1]}")

            # より詳細なデバッグ情報
            if len(model_features) != X_for_pred.shape[1]:
                print(f"Feature count mismatch: model={len(model_features)}, data={X_for_pred.shape[1]}")

                # 最初の数個の特徴量だけを表示
                max_show = min(5, len(model_features))
                print(f"First {max_show} model features: {model_features[:max_show]}")
                print(f"First {max_show} data features: {list(X_for_pred.columns[:max_show])}")
            return None

        # 2パネルのグラフを作成（各パネルの高さを調整）
        fig, ax = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1], sharex=True)

        # 1. 温度データと予測
        ax[0].plot(merged_df['time_stamp'], merged_df[sens_temp_col], 'c-', linewidth=2, label='実測温度')
        ax[0].plot(merged_df['time_stamp'], merged_df[f'{sens_temp_col}_pred'], 'm--', linewidth=2, label='予測温度')

        # 設定温度を追加（存在する場合）
        if set_col in merged_df.columns:
            ax[0].plot(merged_df['time_stamp'], merged_df[set_col], 'g-', linewidth=1.5, label='設定温度')

            # デッドバンドを視覚化
            ax[0].fill_between(merged_df['time_stamp'], merged_df[set_col] - 1.0, merged_df[set_col] + 1.0,
                          color='gray', alpha=0.2, label='不感帯(±1.0°C)')

        # サーモ状態をオーバーレイ表示
        ax0_twin = ax[0].twinx()

        # サーモ状態と空調モードを一緒に表示
        ax0_twin.plot(merged_df['time_stamp'], merged_df[thermo_col], 'r-', linewidth=1.5, label='サーモ状態')

        # モードを追加（存在する場合）
        if mode_col in merged_df.columns:
            # モードに基づいた色分け
            modes = merged_df[mode_col].copy()
            cooling_mask = (modes == 1)
            heating_mask = (modes == 2)

            # 冷房運転時にはサーモ状態を青で表示
            if cooling_mask.any():
                cooling_df = merged_df[cooling_mask].copy()
                ax0_twin.plot(cooling_df['time_stamp'], cooling_df[thermo_col], 'b-', linewidth=1.5, label='冷房ON')

            # 暖房運転時にはサーモ状態を赤で表示
            if heating_mask.any():
                heating_df = merged_df[heating_mask].copy()
                ax0_twin.plot(heating_df['time_stamp'], heating_df[thermo_col], 'r-', linewidth=1.5, label='暖房ON')

        ax0_twin.set_ylim(-0.1, 1.1)
        ax0_twin.set_ylabel('運転状態', fontsize=12)
        ax0_twin.tick_params(axis='y', labelsize=10)

        # 軸の範囲を設定して見やすくする
        temp_min = merged_df[sens_temp_col].min()
        temp_max = merged_df[sens_temp_col].max()
        padding = (temp_max - temp_min) * 0.1
        ax[0].set_ylim(temp_min - padding, temp_max + padding)

        # 凡例を結合
        lines0, labels0 = ax[0].get_legend_handles_labels()
        lines0_twin, labels0_twin = ax0_twin.get_legend_handles_labels()
        ax[0].legend(lines0 + lines0_twin, labels0 + labels0_twin, loc='upper right', fontsize=10)

        ax[0].set_ylabel('温度 (°C)', fontsize=12)
        ax[0].tick_params(axis='y', labelsize=10)
        ax[0].set_title(f'ゾーン {zone} - 温度予測と実測値', fontsize=14)
        ax[0].grid(True, alpha=0.3)

        # 2. 電力消費とサーモOR
        if power_col in merged_df.columns:
            # 電力消費
            ax[1].plot(merged_df['time_stamp'], merged_df[power_col], 'b-', linewidth=1.5, label='消費電力')
            ax[1].set_ylabel('消費電力 (kW)', fontsize=12)
            ax[1].tick_params(axis='y', labelsize=10)

            # AC_valid状態も表示（空調のON/OFF）
            if valid_col in merged_df.columns:
                ax1_twin = ax[1].twinx()
                ax1_twin.plot(merged_df['time_stamp'], merged_df[valid_col], 'g--', linewidth=1.5, label='空調ON/OFF')
                ax1_twin.set_ylim(-0.1, 1.1)
                ax1_twin.set_ylabel('空調状態', fontsize=12)
                ax1_twin.tick_params(axis='y', labelsize=10)

                # 凡例を結合
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

        # X軸のフォーマット設定
        fig.autofmt_xdate()
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))

        # レイアウト調整
        fig.tight_layout()

        # 散布図を作成（実測値vs予測値）
        fig_scatter = plt.figure(figsize=(12, 10))
        ax_scatter = fig_scatter.add_subplot(111)

        # 有効なデータポイントのみ抽出
        valid_data = merged_df.dropna(subset=[sens_temp_col, f'{sens_temp_col}_pred'])

        if not valid_data.empty:
            # データポイントの時間情報を抽出
            timestamps = pd.to_datetime(valid_data['time_stamp'])

            # 時間帯に基づいて色分けするための値を計算
            # 0-23の時間を得る
            hours = timestamps.dt.hour

            # 時間帯を定義（朝・昼・夕方・夜）
            time_periods = pd.cut(
                hours,
                bins=[0, 6, 12, 18, 24],
                labels=['夜間\n(0-6時)', '午前\n(6-12時)', '午後\n(12-18時)', '夕方/夜\n(18-24時)'],
                include_lowest=True
            )

            # 時間帯ごとにカラーマップを定義
            cmap = plt.cm.viridis
            colors = {
                '夜間\n(0-6時)': cmap(0.1),
                '午前\n(6-12時)': cmap(0.4),
                '午後\n(12-18時)': cmap(0.7),
                '夕方/夜\n(18-24時)': cmap(0.9)
            }

            # 時間帯ごとにデータを分割して散布図を作成
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

            # 対角線（完全一致の場合）
            min_val = min(valid_data[sens_temp_col].min(), valid_data[f'{sens_temp_col}_pred'].min())
            max_val = max(valid_data[sens_temp_col].max(), valid_data[f'{sens_temp_col}_pred'].max())
            padding = (max_val - min_val) * 0.05  # 5%のパディング
            ax_scatter.plot([min_val-padding, max_val+padding], [min_val-padding, max_val+padding], 'k--', alpha=0.7, label='完全一致線')

            # 軸範囲を調整
            ax_scatter.set_xlim(min_val-padding, max_val+padding)
            ax_scatter.set_ylim(min_val-padding, max_val+padding)

            # 軸ラベル
            ax_scatter.set_xlabel('実測温度 (°C)', fontsize=14)
            ax_scatter.set_ylabel('予測温度 (°C)', fontsize=14)
            ax_scatter.tick_params(axis='both', labelsize=12)

            # 誤差指標を計算
            actual = valid_data[sens_temp_col]
            predicted = valid_data[f'{sens_temp_col}_pred']

            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            mae = np.mean(np.abs(actual - predicted))
            r2 = 1 - np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2)
            max_error = np.max(np.abs(actual - predicted))

            # 誤差指標のテキストを作成して表示
            stats_text = (
                f'評価指標:\n'
                f'RMSE: {rmse:.3f}°C\n'
                f'MAE: {mae:.3f}°C\n'
                f'R²: {r2:.3f}\n'
                f'最大誤差: {max_error:.3f}°C\n'
                f'データ数: {len(actual)}点'
            )

            # テキストボックスのスタイル設定
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')

            # 左上にテキストを配置
            ax_scatter.text(
                0.05, 0.95, stats_text,
                transform=ax_scatter.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=props
            )

            # トレンドラインを追加（線形回帰）
            z = np.polyfit(actual, predicted, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min_val-padding, max_val+padding, 100)
            ax_scatter.plot(x_trend, p(x_trend), "r-", linewidth=1.5, alpha=0.6, label=f'傾向線 (y={z[0]:.2f}x+{z[1]:.2f})')

            # 凡例の配置
            ax_scatter.legend(loc='lower right', fontsize=12)

            # グリッド
            ax_scatter.grid(True, alpha=0.3)

            # タイトル
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

def main():
    """
    メイン関数: 予測モデル作成と評価
    パフォーマンス最適化とエラーハンドリング強化
    """
    import pandas as pd
    import numpy as np
    import os
    import time
    import logging
    from datetime import timedelta
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings

    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # 警告を抑制（必要に応じて）
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # 処理時間計測開始
    start_time = time.time()

    # 設定
    file_path = '/content/drive/Shareddrives/Shared_engineer/500_Soft/Data_Algo/Data/202405_錦糸町7F実証/DayData/combined_merged_all_data.csv'
    output_dir = '/content/drive/Shareddrives/Shared_engineer/500_Soft/Data_Algo/Data/202405_錦糸町7F実証/DayData/sens_temp_predictions3'
    model_dir = '/content/drive/Shareddrives/Shared_engineer/500_Soft/Data_Algo/Data/202405_錦糸町7F実証/DayData/sens_temp_models3'

    # ゾーンと室外機のマッピング
    zone_to_power = {
        0: 'L', 1: 'L', 6: 'L', 7: 'L',  # L系統
        2: 'M', 3: 'M', 8: 'M', 9: 'M',  # M系統
        4: 'R', 5: 'R', 10: 'R', 11: 'R' # R系統
    }

    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    try:
        # データ読み込み
        logger.info(f"データを読み込んでいます: {file_path}")
        read_start = time.time()

        try:
            df = pd.read_csv(file_path)
            read_time = time.time() - read_start
            logger.info(f"データ読み込み完了: {len(df)}行 × {len(df.columns)}列 ({read_time:.2f}秒)")
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return

        # データの前処理
        preprocess_start = time.time()

        # 'algo'列がある場合、NaNを含む行を削除
        if 'algo' in df.columns:
            before_rows = len(df)
            df = df.dropna(subset=['algo'])
            logger.info(f"'algo'列のNaN行を削除しました: {before_rows} → {len(df)}行")

        # 時間型への変換
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])

        # 分析期間を設定
        start_date = '2024-06-26'
        end_date = '2024-09-20'
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        filtered_df = df[(df['time_stamp'] >= start) & (df['time_stamp'] <= end)]
        logger.info(f"分析期間: {start_date} から {end_date} ({len(filtered_df)}行)")

        # サーモ状態計算
        logger.info("サーモ状態を計算しています...")
        thermo_start = time.time()
        thermo_df = determine_thermo_status(filtered_df)

        # 必要なカラムを確保
        wanted_columns = ['time_stamp'] + [f'thermo_{i}' for i in range(12)] + ['thermo_L_or', 'thermo_M_or', 'thermo_R_or']
        for col in wanted_columns:
            if col not in thermo_df.columns and col != 'time_stamp':
                thermo_df[col] = 0
        thermo_df = thermo_df[wanted_columns]

        thermo_time = time.time() - thermo_start
        logger.info(f"サーモ状態の計算が完了しました ({thermo_time:.2f}秒)")

        preprocess_time = time.time() - preprocess_start
        logger.info(f"前処理完了 ({preprocess_time:.2f}秒)")

        # 結果保存用のデータフレーム
        results_df = pd.DataFrame(columns=['Zone', 'RMSE', 'MAE', 'MAPE', 'R2', 'Top_Features', 'Training_Time'])

        # 各ゾーンの処理を並列化するために必要なデータを事前準備
        # (実際の並列化はこのサンプルでは実装されていませんが、マルチプロセッシングで実装可能です)

        # 各ゾーンを順に処理
        for zone in range(12):
            zone_start = time.time()
            logger.info(f"\n===== ゾーン {zone} の処理開始 =====")

            # このゾーンが属する室外機を特定
            if zone not in zone_to_power:
                logger.warning(f"ゾーン {zone} はどの室外機にも割り当てられていません")
                continue

            power_col = zone_to_power[zone]

            # 必要なカラムが存在するか確認
            valid_col = f'AC_valid_{zone}'
            mode_col = f'AC_mode_{zone}'
            sens_temp_col = f'sens_temp_{zone}'

            required_cols = [valid_col, mode_col, sens_temp_col, power_col]
            missing_cols = [col for col in required_cols if col not in filtered_df.columns]

            if missing_cols:
                logger.warning(f"ゾーン {zone} に必要なカラムがありません: {', '.join(missing_cols)}")
                continue

            try:
                # 特徴量の作成
                feature_start = time.time()
                logger.info(f"ゾーン {zone} の特徴量を作成中...")

                X, y, features_df = prepare_features_for_sens_temp(filtered_df, thermo_df, zone)

                if X is None or y is None or features_df is None:
                    logger.warning(f"ゾーン {zone} の特徴量を作成できませんでした")
                    continue

                feature_time = time.time() - feature_start
                logger.info(f"特徴量作成完了: {X.shape} ({feature_time:.2f}秒)")

                # モデルのトレーニング
                train_start = time.time()
                logger.info(f"ゾーン {zone} の LightGBM モデルをトレーニングしています...")

                model, X_test, y_test, y_pred, importance_df = train_lgbm_model(X, y)

                train_time = time.time() - train_start
                logger.info(f"モデルトレーニング完了 ({train_time:.2f}秒)")

                # モデルの保存
                model_path = os.path.join(model_dir, f'sens_temp_model_zone_{zone}.txt')
                model.save_model(model_path)
                logger.info(f"モデルを保存しました: {model_path}")

                # 特徴量重要度の可視化
                viz_start = time.time()
                importance_path = visualize_feature_importance(importance_df, zone, output_dir)
                logger.info(f"特徴量重要度グラフを保存しました: {importance_path}")

                # 評価指標の計算
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = np.mean(np.abs(y_test - y_pred))
                mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100  # 0除算防止
                r2 = r2_score(y_test, y_pred)
                top_features = ', '.join(importance_df.head(3)['Feature'].tolist())

                # 結果をデータフレームに追加
                results_df = pd.concat([
                    results_df,
                    pd.DataFrame({
                        'Zone': [zone],
                        'RMSE': [rmse],
                        'MAE': [mae],
                        'MAPE': [mape],
                        'R2': [r2],
                        'Top_Features': [top_features],
                        'Training_Time': [train_time]
                    })
                ], ignore_index=True)

                # 評価指標をログ出力
                logger.info(f"評価指標: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R²={r2:.4f}")

                # 時間帯による予測精度分析
                time_analysis = analyze_prediction_by_time(X_test, y_test, y_pred, features_df)
                time_analysis_path = os.path.join(output_dir, f'zone_{zone}_time_analysis.png')
                time_analysis.savefig(time_analysis_path, dpi=150)
                plt.close(time_analysis)
                logger.info(f"時間帯分析を保存しました: {time_analysis_path}")

                # 可視化期間を設定
                first_week_start = start
                first_week_end = first_week_start + timedelta(days=7)
                last_week_end = end
                last_week_start = last_week_end - timedelta(days=7)

                # 最初の週の予測を可視化
                logger.info(f"ゾーン {zone} の最初の週の予測を可視化しています...")
                first_week_figs = visualize_zone_with_predictions(
                    filtered_df, thermo_df, zone, power_col, model, features_df, output_dir,
                    start_date=first_week_start, end_date=first_week_end
                )

                if first_week_figs:
                    first_week_fig, first_week_scatter = first_week_figs

                    if first_week_fig:
                        first_week_path = os.path.join(output_dir, f'zone_{zone}_first_week.png')
                        first_week_fig.savefig(first_week_path, dpi=150)
                        plt.close(first_week_fig)
                        logger.info(f"最初の週の時系列可視化を保存しました: {first_week_path}")

                    if first_week_scatter:
                        first_week_scatter_path = os.path.join(output_dir, f'zone_{zone}_first_week_scatter.png')
                        first_week_scatter.savefig(first_week_scatter_path, dpi=150)
                        plt.close(first_week_scatter)
                        logger.info(f"最初の週の散布図を保存しました: {first_week_scatter_path}")

                # 最後の週の予測を可視化
                logger.info(f"ゾーン {zone} の最後の週の予測を可視化しています...")
                last_week_figs = visualize_zone_with_predictions(
                    filtered_df, thermo_df, zone, power_col, model, features_df, output_dir,
                    start_date=last_week_start, end_date=last_week_end
                )

                if last_week_figs:
                    last_week_fig, last_week_scatter = last_week_figs

                    if last_week_fig:
                        last_week_path = os.path.join(output_dir, f'zone_{zone}_last_week.png')
                        last_week_fig.savefig(last_week_path, dpi=150)
                        plt.close(last_week_fig)
                        logger.info(f"最後の週の時系列可視化を保存しました: {last_week_path}")

                    if last_week_scatter:
                        last_week_scatter_path = os.path.join(output_dir, f'zone_{zone}_last_week_scatter.png')
                        last_week_scatter.savefig(last_week_scatter_path, dpi=150)
                        plt.close(last_week_scatter)
                        logger.info(f"最後の週の散布図を保存しました: {last_week_scatter_path}")

                viz_time = time.time() - viz_start
                logger.info(f"可視化完了 ({viz_time:.2f}秒)")

                # ゾーン処理時間
                zone_time = time.time() - zone_start
                logger.info(f"ゾーン {zone} の処理が完了しました (合計: {zone_time:.2f}秒)")

            except Exception as e:
                logger.error(f"ゾーン {zone} の処理中にエラーが発生しました: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # 結果サマリーの保存
        if not results_df.empty:
            results_path = os.path.join(output_dir, 'model_results_summary.csv')
            results_df.to_csv(results_path, index=False)
            logger.info(f"モデル結果サマリーを保存しました: {results_path}")

            # 全体結果の可視化
            logger.info("全体結果を可視化しています...")
            visualize_overall_results(results_df, output_dir)
            logger.info("全体結果の可視化が完了しました")
        else:
            logger.warning("処理結果がありません。結果サマリーは保存されませんでした。")

        # 合計処理時間
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"\n全処理が完了しました！ 合計処理時間: {int(hours)}時間 {int(minutes)}分 {seconds:.2f}秒")

    except Exception as e:
        logger.error(f"メイン処理中に予期せぬエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

# def analyze_prediction_by_time(X_test, y_test, y_pred, features_df):
#     """時間帯ごとの予測精度を分析"""
#     # X_testとfeatures_dfのインデックスを一致させる
#     common_idx = X_test.index.intersection(features_df.index)
#     X_test = X_test.loc[common_idx]
#     y_test = y_test.loc[common_idx]
#     y_pred = y_pred[common_idx.isin(X_test.index)]

#     # 予測誤差を計算
#     errors = np.abs(y_test - y_pred)

#     # 時間帯情報を取得
#     hours = features_df.loc[common_idx, 'hour']

#     # データフレームを作成
#     analysis_df = pd.DataFrame({
#         'hour': hours,
#         'error': errors
#     })

#     # 時間帯ごとの平均誤差と標準偏差を計算
#     hourly_stats = analysis_df.groupby('hour')['error'].agg(['mean', 'std', 'count']).reset_index()

#     # 可視化
#     fig, ax = plt.subplots(figsize=(12, 6))

#     # 棒グラフとエラーバー
#     ax.bar(hourly_stats['hour'], hourly_stats['mean'],
#            yerr=hourly_stats['std'], alpha=0.7, capsize=5)

#     # グラフ設定
#     ax.set_xlabel('時間帯 (時)')
#     ax.set_ylabel('平均絶対誤差 (°C)')
#     ax.set_title('時間帯ごとの予測精度')
#     ax.set_xticks(range(0, 24))
#     ax.grid(axis='y', linestyle='--', alpha=0.7)

#     # サンプル数を表示
#     for i, (_, row) in enumerate(hourly_stats.iterrows()):
#         ax.text(row['hour'], row['mean'] + row['std'] + 0.05,
#                 f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8)

#     plt.tight_layout()
#     return fig

def visualize_overall_results(results_df, output_dir):
    """すべてのゾーンの結果を可視化"""
    # 1. 精度指標のバープロット
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # RMSE
    sns.barplot(x='Zone', y='RMSE', data=results_df, ax=axes[0, 0])
    axes[0, 0].set_title('RMSE by Zone')
    axes[0, 0].set_xlabel('Zone')
    axes[0, 0].set_ylabel('RMSE (°C)')
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # MAE
    sns.barplot(x='Zone', y='MAE', data=results_df, ax=axes[0, 1])
    axes[0, 1].set_title('MAE by Zone')
    axes[0, 1].set_xlabel('Zone')
    axes[0, 1].set_ylabel('MAE (°C)')
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # MAPE
    sns.barplot(x='Zone', y='MAPE', data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title('MAPE by Zone')
    axes[1, 0].set_xlabel('Zone')
    axes[1, 0].set_ylabel('MAPE (%)')
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # R2
    sns.barplot(x='Zone', y='R2', data=results_df, ax=axes[1, 1])
    axes[1, 1].set_title('R² by Zone')
    axes[1, 1].set_xlabel('Zone')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_metrics.png'), dpi=150)
    plt.close()

    # 2. トップ特徴量の可視化
    # 各特徴量の出現回数をカウント
    all_features = []
    for features in results_df['Top_Features']:
        all_features.extend([f.strip() for f in features.split(',')])

    feature_counts = pd.Series(all_features).value_counts().head(15)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=feature_counts.values, y=feature_counts.index)
    plt.title('Most Common Top Features Across All Zones')
    plt.xlabel('Frequency')
    plt.ylabel('Feature')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'top_features_overall.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
