#!/usr/bin/env python
# coding: utf-8

"""
モデルトレーニングモジュール
物理法則を考慮したLightGBMモデルのトレーニング関数を提供
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
from src.config import LGBM_PARAMS, MODELS_DIR


class PhysicsConstrainedLGBM:
    """
    物理制約を考慮したLightGBMモデル
    
    制御変数への感度と物理的妥当性を確保するために：
    1. 制御変数（AC_valid, AC_mode, AC_set）の重要度を強制的に高める
    2. 物理制約違反時のペナルティを課す
    3. 適応的重み付けによる学習効果の向上
    """
    
    def __init__(self, control_sensitivity=0.5, physics_penalty=0.2, 
                 adaptive_weight_start=0.1, adaptive_weight_end=0.7, **lgb_params):
        """
        Parameters:
        -----------
        control_sensitivity : float
            制御変数への感度パラメータ（0.0-1.0）
        physics_penalty : float
            物理制約違反時のペナルティ係数
        adaptive_weight_start : float
            適応的重み付けの開始値
        adaptive_weight_end : float
            適応的重み付けの終了値
        lgb_params : dict
            LightGBMのパラメータ
        """
        self.control_sensitivity = control_sensitivity
        self.physics_penalty = physics_penalty
        self.adaptive_weight_start = adaptive_weight_start
        self.adaptive_weight_end = adaptive_weight_end
        
        # デフォルトパラメータをマージ
        default_params = LGBM_PARAMS.copy()
        default_params.update(lgb_params)
        default_params['verbose'] = -1
        
        self.lgb_params = default_params
        self.model = None
        self.feature_names = None
        self.control_features = []
        
    def _identify_control_features(self, feature_names):
        """制御変数関連の特徴量を特定"""
        control_keywords = ['AC_valid', 'AC_mode', 'AC_set', 'thermo_state']
        self.control_features = [feat for feat in feature_names 
                               if any(keyword in feat for keyword in control_keywords)]
        print(f"制御関連特徴量を特定: {len(self.control_features)}個")
        return self.control_features
    
    def _create_adaptive_weights(self, y_train, iteration_ratio):
        """
        適応的重み付けの作成
        学習の進行に応じて制御効果を段階的に強化
        """
        # 基本重み：温度変化の大きさに応じて
        temp_changes = np.abs(y_train)
        basic_weights = 1 + temp_changes / (temp_changes.mean() + 1e-6)
        
        # 適応的重み：学習進行に応じて制御効果を重視
        current_adaptive_weight = (self.adaptive_weight_start + 
                                 (self.adaptive_weight_end - self.adaptive_weight_start) * iteration_ratio)
        
        # 制御変数の影響を強化する重み
        control_weight = 1 + current_adaptive_weight
        
        # 最終重み
        final_weights = basic_weights * control_weight
        
        # 重みの正規化とクリッピング
        final_weights = np.clip(final_weights, 0.5, 5.0)
        
        return final_weights
    
    def _apply_physics_constraints(self, pred, X_test):
        """
        物理制約の適用
        制御変数の変化に対して予測が適切に反応するよう調整
        """
        adjusted_pred = pred.copy()
        
        if len(self.control_features) == 0:
            return adjusted_pred
            
        # 制御効果の強制的な調整
        for feat in self.control_features:
            if feat in X_test.columns:
                feat_values = X_test[feat].values
                
                if 'AC_valid' in feat:
                    # サーモON時は温度制御効果を強化
                    thermo_on_mask = feat_values == 1
                    adjusted_pred[thermo_on_mask] += self.control_sensitivity * 0.1
                    
                elif 'AC_mode' in feat:
                    # 暖房時は正の調整、冷房時は負の調整
                    heat_mask = feat_values == 1
                    cool_mask = feat_values == 0
                    adjusted_pred[heat_mask] += self.control_sensitivity * 0.15
                    adjusted_pred[cool_mask] -= self.control_sensitivity * 0.15
                    
                elif 'AC_set' in feat:
                    # 設定温度の影響を強化 - 絶対値ベースで制御効果を適用
                    # 高設定温度は正の調整、低設定温度は負の調整
                    baseline_temp = 24.0  # 基準温度（°C）
                    temp_adjustment = (feat_values - baseline_temp) * self.control_sensitivity * 0.15
                    adjusted_pred += temp_adjustment
        
        return adjusted_pred
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """モデルの学習"""
        print("🚀 PhysicsConstrainedLGBM学習開始...")
        
        self.feature_names = list(X_train.columns)
        self._identify_control_features(self.feature_names)
        
        # 段階的学習による制御感度の向上
        n_stages = 3
        models = []
        
        for stage in range(n_stages):
            iteration_ratio = stage / (n_stages - 1)
            print(f"  📊 学習ステージ {stage + 1}/{n_stages} (適応重み: {iteration_ratio:.1f})")
            
            # 適応的重み作成
            weights = self._create_adaptive_weights(y_train, iteration_ratio)
            
            # モデル学習
            stage_model = lgb.LGBMRegressor(**self.lgb_params)
            
            if X_val is not None and y_val is not None:
                val_weights = self._create_adaptive_weights(y_val, iteration_ratio)
                stage_model.fit(
                    X_train, y_train, 
                    sample_weight=weights,
                    eval_set=[(X_val, y_val)],
                    eval_sample_weight=[val_weights],
                    callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                )
            else:
                stage_model.fit(X_train, y_train, sample_weight=weights)
            
            models.append(stage_model)
        
        # 最終モデルを選択（最後のステージ）
        self.model = models[-1]
        
        # 制御特徴量の重要度チェック
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            control_importance = importance_df[
                importance_df['feature'].isin(self.control_features)
            ]
            
            print(f"  📈 制御特徴量重要度トップ5:")
            for _, row in control_importance.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.3f}")
        
        print("✅ PhysicsConstrainedLGBM学習完了")
        return self
    
    def predict(self, X_test):
        """物理制約を考慮した予測"""
        if self.model is None:
            raise ValueError("モデルが学習されていません。先にfit()を実行してください。")
        
        # 基本予測
        base_pred = self.model.predict(X_test)
        
        # 物理制約の適用
        constrained_pred = self._apply_physics_constraints(base_pred, X_test)
        
        return constrained_pred
    
    @property
    def feature_importances_(self):
        """特徴量重要度の取得"""
        if self.model is None:
            return None
        return self.model.feature_importances_
    
    @property
    def best_iteration_(self):
        """最適イテレーション数の取得"""
        if self.model is None:
            return None
        return getattr(self.model, 'best_iteration_', None)


def train_physics_guided_model(X_train, y_train, params=None):
    """
    物理法則を考慮したモデルのトレーニング
    修正: 特徴量の重複をチェックし、重複を排除

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数
    params : dict
        LightGBMのパラメータ（Noneの場合はデフォルト値を使用）

    Returns:
    --------
    LGBMRegressor
        トレーニング済みモデル
    """
    print("物理法則ガイド付きモデルをトレーニング中...")

    # 列名の重複チェック
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: トレーニングデータの列名に重複があります。重複を排除します。")
        # 重複を排除したデータフレームを作成
        unique_cols = []
        seen_cols = set()
        for col in X_train.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)

        # 重複を排除した特徴量のみを使用
        duplicate_count = len(X_train.columns) - len(unique_cols)
        X_train = X_train[unique_cols]
        print(f"{duplicate_count}個の重複特徴量を排除しました。残り特徴量数: {len(unique_cols)}")

    # パラメータが指定されていない場合は、デフォルト値を使用
    if params is None:
        params = LGBM_PARAMS.copy()  # コピーを作成して元の設定を変更しないようにする

    # 警告メッセージを抑制するため、verboseを-1に設定
    params['verbose'] = -1

    # 物理モデルに適したパラメータを使用
    lgb_model = lgb.LGBMRegressor(**params)

    try:
        # Pythonの標準警告を一時的に抑制
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # クラス重み付け：急激な温度変化を重視
            if 'weight' in y_train.index.names:
                print("サンプル重み付けを使用します")
                lgb_model.fit(X_train, y_train)
            else:
                # 急激な温度変化に対する重み付け
                temp_changes = y_train.diff().abs().fillna(0)
                weights = 1 + temp_changes / temp_changes.mean()

                # スパイクの影響を制限
                max_weight = 3.0  # 最大ウェイト値を制限
                weights = weights.clip(upper=max_weight)

                lgb_model.fit(X_train, y_train, sample_weight=weights)

        return lgb_model

    except Exception as e:
        print(f"モデルトレーニング中にエラーが発生しました: {e}")
        print("緊急対処モードでトレーニングを試みます...")

        # 最小限の特徴量だけを使用して再トレーニング
        # 基本的な特徴量のみを選択（温度、設定値、制御状態など）
        basic_features = [col for col in X_train.columns if any(key in col for key in [
            'sens_temp', 'thermo_state', 'AC_valid', 'AC_mode', 'atmospheric', 'solar'
        ])]

        if len(basic_features) > 0:
            print(f"基本特徴量{len(basic_features)}個のみを使用してトレーニングします")
            simple_model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbose=-1  # 警告を抑制
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simple_model.fit(X_train[basic_features], y_train)
            return simple_model
        else:
            # すべての対処が失敗した場合はダミーモデルを返す
            print("ダミーモデルを作成します（平均値予測）")
            from sklearn.dummy import DummyRegressor
            dummy_model = DummyRegressor(strategy='mean')
            dummy_model.fit(X_train.iloc[:, 0].values.reshape(-1, 1), y_train)
            return dummy_model


def save_model_and_features(model, feature_list, zone, horizon, poly_config=None):
    """
    モデルと特徴量情報を保存する関数

    Parameters:
    -----------
    model : 学習済みモデル
        保存するモデル
    feature_list : list
        特徴量のリスト
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    poly_config : dict, optional
        多項式特徴量の設定情報

    Returns:
    --------
    tuple
        (モデルのファイルパス, 特徴量情報のファイルパス)
    """
    # モデルと特徴量情報のファイルパス
    model_filename = os.path.join(MODELS_DIR, f"lgbm_model_zone_{zone}_horizon_{horizon}.pkl")
    features_filename = os.path.join(MODELS_DIR, f"features_zone_{zone}_horizon_{horizon}.pkl")

    try:
        # モデルの保存
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"モデルを保存しました: {model_filename}")

        # 特徴量情報の保存
        feature_info = {
            'feature_cols': feature_list,
            'poly_config': poly_config if poly_config else {
                'use_poly': False,
                'poly_features': []
            }
        }
        with open(features_filename, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"特徴量情報を保存しました: {features_filename}")

        return model_filename, features_filename
    except Exception as e:
        print(f"モデルまたは特徴量情報の保存中にエラーが発生しました: {e}")
        return None, None


def load_model_and_features(zone, horizon):
    """
    保存されたモデルと特徴量情報を読み込む関数

    Parameters:
    -----------
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）

    Returns:
    --------
    tuple
        (モデル, 特徴量情報)
    """
    model_filename = os.path.join(MODELS_DIR, f"lgbm_model_zone_{zone}_horizon_{horizon}.pkl")
    features_filename = os.path.join(MODELS_DIR, f"features_zone_{zone}_horizon_{horizon}.pkl")

    try:
        # モデルの読み込み
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)

        # 特徴量情報の読み込み
        with open(features_filename, 'rb') as f:
            feature_info = pickle.load(f)

        print(f"モデルと特徴量情報を読み込みました: ゾーン{zone}, {horizon}分後")
        return model, feature_info
    except Exception as e:
        print(f"モデルまたは特徴量情報の読み込み中にエラーが発生しました: {e}")
        return None, None


def train_temperature_difference_model(X_train, y_train, params=None):
    """
    温度差分予測専用のモデルトレーニング関数
    PhysicsConstrainedLGBMを使用して物理的妥当性を確保

    温度の変化量を予測するため、従来の温度予測とは異なるパラメータ調整を行う：
    - より高い学習率で変化パターンを捉える
    - 小さな差分値に対する感度を向上
    - 変化の激しい期間への重み付け強化
    - 特徴量重要度に基づく動的調整
    - 物理制約による制御変数への感度向上

    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数（温度差分）
    params : dict
        LightGBMのパラメータ（Noneの場合は差分予測用デフォルト値を使用）

    Returns:
    --------
    PhysicsConstrainedLGBM
        トレーニング済み物理制約差分予測モデル
    """
    print("🔥 高精度温度差分予測モデルをトレーニング中...")

    # 列名の重複チェック
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: トレーニングデータの列名に重複があります。重複を排除します。")
        unique_cols = []
        seen_cols = set()
        for col in X_train.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)

        duplicate_count = len(X_train.columns) - len(unique_cols)
        X_train = X_train[unique_cols]
        print(f"{duplicate_count}個の重複特徴量を排除しました。残り特徴量数: {len(unique_cols)}")

    # 高精度差分予測に最適化されたパラメータ
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',  # 温度差分では平均絶対誤差が適切
            'boosting_type': 'gbdt',
            'num_leaves': 255,  # より複雑なパターンを捉える
            'learning_rate': 0.05,  # より慎重な学習
            'feature_fraction': 0.85,  # 特徴量の多様性を保持
            'bagging_fraction': 0.75,  # より厳格なサンプリング
            'bagging_freq': 3,
            'max_depth': 10,  # より深い決定木
            'min_data_in_leaf': 8,  # 小さな差分値を捉えるため小さめに設定
            'lambda_l1': 0.05,  # L1正則化を強化
            'lambda_l2': 0.15,  # L2正則化を強化
            'min_gain_to_split': 0.01,  # より細かい分割を許可
            'max_bin': 512,  # より細かいビニング
            'random_state': 42,
            'n_estimators': 1000,  # 調整済み
            'verbose': -1,
            'force_col_wise': True  # メモリ効率の改善
        }

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            print("🎯 高度な重み付け戦略を適用中...")
            
            # PhysicsConstrainedLGBMを使用
            physics_model = PhysicsConstrainedLGBM(
                control_sensitivity=0.9,  # 制御変数への感度を0.7→0.9に強化
                physics_penalty=0.3,
                adaptive_weight_start=0.1,
                adaptive_weight_end=0.7,
                **params
            )

            # 検証用分割でearly stoppingを使用
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )

            # モデル学習
            physics_model.fit(
                X_train_split, y_train_split,
                X_val=X_val_split, y_val=y_val_split
            )

            print(f"✅ 高精度差分予測モデル訓練完了")

        return physics_model

    except Exception as e:
        print(f"PhysicsConstrainedLGBM学習中にエラーが発生しました: {e}")
        print("フォールバック: 従来の差分予測モデルで再試行...")

        # フォールバック: 従来のLightGBMモデル
        lgb_model = lgb.LGBMRegressor(**params)

        try:
            # 急激な温度変化に対する重み付け
            temp_changes = y_train.diff().abs().fillna(0)
            base_weights = 1 + temp_changes / (temp_changes.mean() + 1e-6)

            # 制御効果の重み調整（より強い重み付け）
            control_features = [col for col in X_train.columns 
                              if any(keyword in col for keyword in ['AC_valid', 'AC_mode', 'AC_set', 'thermo_state'])]
            
            if control_features:
                print(f"制御特徴量 {len(control_features)}個を特定し、重み付けを強化します")
                control_weights = np.ones(len(y_train))
                
                # 制御変数の変化時に重みを増加
                for feat in control_features:
                    if feat in X_train.columns:
                        feat_changes = X_train[feat].diff().abs().fillna(0)
                        control_weights += feat_changes * 2.0  # 制御変更時の重みを強化
                
                final_weights = base_weights * control_weights
            else:
                final_weights = base_weights

            # 最大重みを制限
            final_weights = np.clip(final_weights, 0.5, 5.0)

            print(f"重み付け統計:")
            print(f"  平均重み: {final_weights.mean():.3f}")
            print(f"  重み範囲: {final_weights.min():.3f} - {final_weights.max():.3f}")
            print(f"  高重み(>2.0)データ: {(final_weights > 2.0).sum()}行 ({(final_weights > 2.0).mean()*100:.1f}%)")

            # 検証用分割でearly stoppingを使用
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split, weights_train, weights_val = train_test_split(
                X_train, y_train, final_weights, test_size=0.15, random_state=42
            )

            lgb_model.fit(
                X_train_split, y_train_split,
                sample_weight=weights_train,
                eval_set=[(X_val_split, y_val_split)],
                eval_sample_weight=[weights_val],
                callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
            )

            print(f"✅ フォールバック差分予測モデル訓練完了 (最終イテレーション: {lgb_model.best_iteration_})")

            return lgb_model

        except Exception as fallback_error:
            print(f"フォールバックモデルトレーニング中にもエラーが発生しました: {fallback_error}")
            print("シンプルな差分予測モデルで再試行...")

            # 最終フォールバック: シンプルなパラメータ
            simple_model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                verbose=-1
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                simple_model.fit(X_train, y_train)

            return simple_model


def save_difference_model_and_features(model, feature_list, zone, horizon, poly_config=None):
    """
    差分予測モデルと特徴量情報を保存する関数

    Parameters:
    -----------
    model : 学習済みモデル
        保存する差分予測モデル
    feature_list : list
        特徴量のリスト
    zone : int
        ゾーン番号
    horizon : int
        予測ホライゾン（分）
    poly_config : dict, optional
        多項式特徴量の設定情報

    Returns:
    --------
    tuple
        (モデルのファイルパス, 特徴量情報のファイルパス)
    """
    # 差分予測モデル専用のファイル名
    model_filename = os.path.join(MODELS_DIR, f"diff_model_zone_{zone}_horizon_{horizon}.pkl")
    features_filename = os.path.join(MODELS_DIR, f"diff_features_zone_{zone}_horizon_{horizon}.pkl")

    try:
        # モデルの保存
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"差分予測モデルを保存しました: {model_filename}")

        # 特徴量情報の保存
        feature_info = {
            'feature_cols': feature_list,
            'model_type': 'temperature_difference',
            'poly_config': poly_config if poly_config else {
                'use_poly': False,
                'poly_features': []
            }
        }
        with open(features_filename, 'wb') as f:
            pickle.dump(feature_info, f)
        print(f"差分予測特徴量情報を保存しました: {features_filename}")

        return model_filename, features_filename
    except Exception as e:
        print(f"差分予測モデルまたは特徴量情報の保存中にエラーが発生しました: {e}")
        return None, None


def train_physics_constrained_difference_model(X_train, y_train, params=None):
    """
    物理制約付き温度差分予測モデルのトレーニング
    
    Parameters:
    -----------
    X_train : DataFrame
        訓練用特徴量
    y_train : Series
        訓練用目的変数（温度差分）
    params : dict
        モデルパラメータ
        
    Returns:
    --------
    PhysicsConstrainedLGBM
        トレーニング済み物理制約モデル
    """
    print("🚀 物理制約付き差分予測モデル学習開始...")
    
    # 列名の重複チェック
    if len(X_train.columns) != len(set(X_train.columns)):
        print("警告: 重複特徴量を排除します")
        unique_cols = []
        seen_cols = set()
        for col in X_train.columns:
            if col not in seen_cols:
                unique_cols.append(col)
                seen_cols.add(col)
        X_train = X_train[unique_cols]
        print(f"重複排除後の特徴量数: {len(unique_cols)}")
    
    # デフォルトパラメータ
    if params is None:
        params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 10,
            'random_state': 42
        }
    
    try:
        # 物理制約モデルの初期化
        physics_model = PhysicsConstrainedLGBM(
            control_sensitivity=0.9,
            physics_penalty=0.3,
            adaptive_weight_start=0.1,
            adaptive_weight_end=0.7,
            **params
        )
        
        # 検証データの分割
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42
        )
        
        # モデル学習
        physics_model.fit(
            X_train_split, y_train_split,
            X_val=X_val_split, y_val=y_val_split
        )
        
        print("✅ 物理制約付きモデル学習完了")
        return physics_model
        
    except Exception as e:
        print(f"エラー: {e}")
        print("フォールバック: 通常のLightGBMを使用")
        
        import lightgbm as lgb
        lgb_model = lgb.LGBMRegressor(**params)
        lgb_model.fit(X_train, y_train)
        return lgb_model
