"""
商品分類器の基底クラス
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseClassifier(ABC):
    """
    商品分類器の抽象基底クラス
    LinearSVCとLightGBMの共通インターフェースを定義
    """

    # 4階層の分類カラム
    HIERARCHY_COLUMNS = [
        'predicted_category',
        'predicted_subcategory',
        'predicted_segment',
        'predicted_subsegment'
    ]

    # 対応する正解カラム
    ANSWER_COLUMNS = [
        'category',
        'subcategory',
        'segment',
        'subsegment'
    ]

    def __init__(self):
        """分類器の初期化"""
        self.models: Dict[str, Any] = {}
        self.vectorizers: Dict[str, Any] = {}
        self.jan_dict: Dict[str, Dict[str, str]] = {}
        self.is_trained = False
        self.feature_names: Dict[str, List[str]] = {}

    def create_jan_dict(self, trial_df: pd.DataFrame) -> None:
        """
        トライアルマスターからJAN辞書を作成

        Args:
            trial_df: トライアルマスターデータ
        """
        try:
            logger.info(f"JAN辞書作成開始: {len(trial_df)}件")

            # JANをキーとした辞書を作成
            for _, row in trial_df.iterrows():
                jan = str(row.get('jan', '')).strip()
                if jan and jan != 'nan':
                    self.jan_dict[jan] = {
                        'category': row.get('category', ''),
                        'subcategory': row.get('subcategory', ''),
                        'segment': row.get('segment', ''),
                        'subsegment': row.get('subsegment', ''),
                        'product_name': row.get('product_name', ''),
                        'standard': row.get('standard', ''),
                        'manufacturer': row.get('manufacturer', '')
                    }

            logger.info(f"JAN辞書作成完了: {len(self.jan_dict)}件")

        except Exception as e:
            logger.error(f"JAN辞書作成エラー: {str(e)}")
            raise

    @abstractmethod
    def train(self, train_df: pd.DataFrame) -> None:
        """
        モデルの学習（サブクラスで実装）

        Args:
            train_df: 学習データ
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        予測実行（サブクラスで実装）

        Args:
            X: 予測対象データ

        Returns:
            予測結果を含むDataFrame
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        特徴量重要度を取得（サブクラスで実装）

        Returns:
            階層ごとの特徴量重要度辞書
        """
        pass

    def validate_accuracy(
        self,
        data: pd.DataFrame,
        train_ratios: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        9段階クロスバリデーション（学習データ10%-90%）

        Args:
            data: 検証用データ
            train_ratios: 学習データ比率リスト（デフォルト: 0.1-0.9の9段階）

        Returns:
            検証結果辞書
        """
        try:
            if train_ratios is None:
                train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            logger.info(f"精度検証開始: {len(train_ratios)}段階")

            results = {
                'train_ratios': train_ratios,
                'accuracy_scores': {col: [] for col in self.HIERARCHY_COLUMNS}
            }

            for ratio in train_ratios:
                logger.info(f"学習データ比率 {ratio*100:.0f}% で検証中...")

                # データ分割
                train_data, test_data = train_test_split(
                    data,
                    train_size=ratio,
                    random_state=42,
                    stratify=data['category'] if 'category' in data.columns else None
                )

                # モデル学習
                self.train(train_data)

                # 予測
                predictions = self.predict(test_data)

                # 各階層の精度計算
                for pred_col, ans_col in zip(self.HIERARCHY_COLUMNS, self.ANSWER_COLUMNS):
                    if ans_col in test_data.columns and pred_col in predictions.columns:
                        accuracy = accuracy_score(
                            test_data[ans_col],
                            predictions[pred_col]
                        )
                        results['accuracy_scores'][pred_col].append(accuracy)
                    else:
                        results['accuracy_scores'][pred_col].append(0.0)

            logger.info("精度検証完了")
            return results

        except Exception as e:
            logger.error(f"精度検証エラー: {str(e)}")
            raise

    def predict_single(self, product_data: pd.Series) -> Dict[str, Any]:
        """
        単一商品の予測

        Args:
            product_data: 商品データ（1行）

        Returns:
            予測結果辞書
        """
        try:
            # DataFrameに変換
            df = pd.DataFrame([product_data])

            # 予測実行
            result = self.predict(df)

            # 最初の行を辞書で返す
            return result.iloc[0].to_dict()

        except Exception as e:
            logger.error(f"予測エラー: {str(e)}")
            raise

    def _prepare_features(self, df: pd.DataFrame) -> str:
        """
        特徴量文字列を作成

        Args:
            df: データフレーム

        Returns:
            結合された特徴量文字列
        """
        features = []

        # 商品名
        if 'product_name' in df.columns:
            features.append(df['product_name'].fillna('').astype(str))

        # 規格
        if 'standard' in df.columns:
            features.append(df['standard'].fillna('').astype(str))

        # メーカー
        if 'manufacturer' in df.columns:
            features.append(df['manufacturer'].fillna('').astype(str))

        # 全て結合
        return ' '.join(features) if features else ''

    def get_algorithm_name(self) -> str:
        """
        アルゴリズム名を取得

        Returns:
            アルゴリズム名
        """
        return self.__class__.__name__
