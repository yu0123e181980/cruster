"""
LightGBM分類器（学習データ≥500件用）
"""
from typing import Dict, List
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import logging

from .base import BaseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LightGBMClassifier(BaseClassifier):
    """
    LightGBMを使用した商品分類器
    学習データが500件以上の場合に推奨
    """

    def __init__(self):
        """LightGBM分類器の初期化"""
        super().__init__()
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def train(self, train_df: pd.DataFrame) -> None:
        """
        LightGBMモデルの学習

        Args:
            train_df: 学習データ
        """
        try:
            logger.info(f"LightGBM学習開始: {len(train_df)}件")

            # 特徴量ベクトル化器を作成
            self.vectorizers['main'] = TfidfVectorizer(
                analyzer='char',
                ngram_range=(1, 3),
                max_features=5000,
                min_df=1,
                sublinear_tf=True
            )

            # 特徴量作成
            X_text = self._create_features(train_df)
            X_train = self.vectorizers['main'].fit_transform(X_text).toarray()

            # 特徴量名を保存
            feature_names = self.vectorizers['main'].get_feature_names_out().tolist()

            # 各階層ごとにモデル作成
            for pred_col, ans_col in zip(self.HIERARCHY_COLUMNS, self.ANSWER_COLUMNS):
                if ans_col not in train_df.columns:
                    logger.warning(f"{ans_col} カラムが見つかりません")
                    continue

                logger.info(f"{pred_col} のモデル学習中...")

                # ラベルエンコーディング
                y_train = train_df[ans_col].fillna('').astype(str)
                le = LabelEncoder()
                y_encoded = le.fit_transform(y_train)
                self.label_encoders[pred_col] = le

                # クラス数を取得
                num_classes = len(le.classes_)

                # LightGBMデータセット作成
                train_data = lgb.Dataset(
                    X_train,
                    label=y_encoded,
                    feature_name=feature_names
                )

                # パラメータ設定
                params = {
                    'objective': 'multiclass' if num_classes > 2 else 'binary',
                    'num_class': num_classes if num_classes > 2 else 1,
                    'metric': 'multi_logloss' if num_classes > 2 else 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.9,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'random_state': 42
                }

                # モデル学習
                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=100,
                    valid_sets=[train_data],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )

                self.models[pred_col] = model
                self.feature_names[pred_col] = feature_names

            self.is_trained = True
            logger.info("LightGBM学習完了")

        except Exception as e:
            logger.error(f"LightGBM学習エラー: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        LightGBMによる予測

        Args:
            X: 予測対象データ

        Returns:
            予測結果を含むDataFrame
        """
        try:
            if not self.is_trained:
                raise ValueError("モデルが学習されていません")

            logger.info(f"LightGBM予測開始: {len(X)}件")

            # 結果格納用DataFrame
            result = X.copy()

            # 特徴量作成
            X_text = self._create_features(X)
            X_features = self.vectorizers['main'].transform(X_text).toarray()

            # 階層制約付き予測
            prev_pred = None

            for pred_col in self.HIERARCHY_COLUMNS:
                if pred_col not in self.models:
                    result[pred_col] = ''
                    result[f'{pred_col}_confidence'] = 0.0
                    continue

                model = self.models[pred_col]
                le = self.label_encoders[pred_col]

                # 予測（確率）
                y_pred_proba = model.predict(X_features)

                # 2クラスの場合の処理
                if len(y_pred_proba.shape) == 1:
                    y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])

                # 最も確率の高いクラスを選択
                y_pred_encoded = np.argmax(y_pred_proba, axis=1)
                predictions = le.inverse_transform(y_pred_encoded)

                # 信頼度（最大確率）
                confidence = np.max(y_pred_proba, axis=1)

                # 階層制約適用
                if prev_pred is not None:
                    low_confidence_mask = result[f'{prev_pred}_confidence'] < 0.5
                    confidence[low_confidence_mask] *= 0.5

                result[pred_col] = predictions
                result[f'{pred_col}_confidence'] = confidence

                prev_pred = pred_col

            # 全体の信頼度（4階層の平均）
            confidence_cols = [f'{col}_confidence' for col in self.HIERARCHY_COLUMNS]
            result['confidence'] = result[confidence_cols].mean(axis=1)

            logger.info("LightGBM予測完了")
            return result

        except Exception as e:
            logger.error(f"LightGBM予測エラー: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        LightGBMの特徴量重要度を取得

        Returns:
            階層ごとの特徴量重要度辞書
        """
        try:
            importance_dict = {}

            for pred_col, model in self.models.items():
                # 特徴量重要度を取得
                importance = model.feature_importance(importance_type='gain')
                feature_names = self.feature_names.get(pred_col, [])

                if len(feature_names) != len(importance):
                    logger.warning(f"{pred_col}: 特徴量数が一致しません")
                    continue

                # 上位20個を抽出
                indices = np.argsort(importance)[::-1][:20]
                importance_dict[pred_col] = {
                    feature_names[i]: float(importance[i])
                    for i in indices
                }

            return importance_dict

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {str(e)}")
            return {}

    def _create_features(self, df: pd.DataFrame) -> List[str]:
        """
        特徴量文字列リストを作成

        Args:
            df: データフレーム

        Returns:
            特徴量文字列のリスト
        """
        features = []

        for _, row in df.iterrows():
            parts = []

            # 商品名
            if 'product_name' in df.columns:
                parts.append(str(row.get('product_name', '')))

            # 規格
            if 'standard' in df.columns:
                parts.append(str(row.get('standard', '')))

            # メーカー
            if 'manufacturer' in df.columns:
                parts.append(str(row.get('manufacturer', '')))

            features.append(' '.join(parts))

        return features

    def get_algorithm_name(self) -> str:
        """アルゴリズム名を取得"""
        return "LightGBM"
