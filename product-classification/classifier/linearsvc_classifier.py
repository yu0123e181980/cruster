"""
LinearSVC分類器（学習データ<500件用）
"""
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import logging

from .base import BaseClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearSVCClassifier(BaseClassifier):
    """
    LinearSVCを使用した商品分類器
    学習データが500件未満の場合に推奨
    """

    def __init__(self):
        """LinearSVC分類器の初期化"""
        super().__init__()
        self.pipelines: Dict[str, Pipeline] = {}

    def train(self, train_df: pd.DataFrame) -> None:
        """
        LinearSVCモデルの学習

        Args:
            train_df: 学習データ
        """
        try:
            logger.info(f"LinearSVC学習開始: {len(train_df)}件")

            # 各階層ごとにパイプライン作成
            for pred_col, ans_col in zip(self.HIERARCHY_COLUMNS, self.ANSWER_COLUMNS):
                if ans_col not in train_df.columns:
                    logger.warning(f"{ans_col} カラムが見つかりません")
                    continue

                logger.info(f"{pred_col} のモデル学習中...")

                # 特徴量作成
                X_train = self._create_features(train_df)
                y_train = train_df[ans_col].fillna('').astype(str)

                # パイプライン作成（TF-IDF + LinearSVC）
                pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(
                        analyzer='char',
                        ngram_range=(1, 3),  # 文字1-3グラム
                        max_features=5000,
                        min_df=1,
                        sublinear_tf=True
                    )),
                    ('clf', LinearSVC(
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                        dual=False
                    ))
                ])

                # 学習
                pipeline.fit(X_train, y_train)
                self.pipelines[pred_col] = pipeline

                # 特徴量名を保存
                self.feature_names[pred_col] = pipeline.named_steps['tfidf'].get_feature_names_out().tolist()

            self.is_trained = True
            logger.info("LinearSVC学習完了")

        except Exception as e:
            logger.error(f"LinearSVC学習エラー: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        LinearSVCによる予測

        Args:
            X: 予測対象データ

        Returns:
            予測結果を含むDataFrame
        """
        try:
            if not self.is_trained:
                raise ValueError("モデルが学習されていません")

            logger.info(f"LinearSVC予測開始: {len(X)}件")

            # 結果格納用DataFrame
            result = X.copy()

            # 特徴量作成
            X_features = self._create_features(X)

            # 階層制約付き予測
            prev_pred = None

            for pred_col in self.HIERARCHY_COLUMNS:
                if pred_col not in self.pipelines:
                    result[pred_col] = ''
                    result[f'{pred_col}_confidence'] = 0.0
                    continue

                pipeline = self.pipelines[pred_col]

                # 予測
                predictions = pipeline.predict(X_features)

                # 信頼度取得（decision_functionを正規化）
                decision_values = pipeline.decision_function(X_features)

                # 多クラスの場合は最大値を使用
                if len(decision_values.shape) > 1:
                    confidence = np.max(decision_values, axis=1)
                else:
                    confidence = decision_values

                # 0-1に正規化
                confidence = 1 / (1 + np.exp(-confidence))

                # 階層制約適用（上位階層が低信頼度の場合、下位も低信頼度にする）
                if prev_pred is not None:
                    low_confidence_mask = result[f'{prev_pred}_confidence'] < 0.5
                    confidence[low_confidence_mask] *= 0.5

                result[pred_col] = predictions
                result[f'{pred_col}_confidence'] = confidence

                prev_pred = pred_col

            # 全体の信頼度（4階層の平均）
            confidence_cols = [f'{col}_confidence' for col in self.HIERARCHY_COLUMNS]
            result['confidence'] = result[confidence_cols].mean(axis=1)

            logger.info("LinearSVC予測完了")
            return result

        except Exception as e:
            logger.error(f"LinearSVC予測エラー: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        LinearSVCの特徴量重要度を取得

        Returns:
            階層ごとの特徴量重要度辞書
        """
        try:
            importance_dict = {}

            for pred_col, pipeline in self.pipelines.items():
                # 係数の絶対値を重要度とする
                clf = pipeline.named_steps['clf']
                coef = clf.coef_

                # 多クラスの場合は各クラスの平均を取る
                if len(coef.shape) > 1:
                    importance = np.abs(coef).mean(axis=0)
                else:
                    importance = np.abs(coef)

                # 特徴量名と重要度をマッピング
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
        return "LinearSVC"
