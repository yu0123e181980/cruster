"""
商品分類サービス
"""
import pandas as pd
from typing import Dict, Tuple, Optional, Any
import logging

from classifier import LinearSVCClassifier, LightGBMClassifier, BaseClassifier
from utils.preprocessing import (
    preprocess_dataframe,
    create_jan_dict,
    match_by_jan,
    combine_results
)
from utils.multiprocess_utils import predict_multiprocess, predict_sequential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationService:
    """
    商品分類サービス
    """

    def __init__(self):
        """初期化"""
        self.classifier: Optional[BaseClassifier] = None
        self.algorithm_type: Optional[str] = None
        self.jan_dict: Dict[str, Dict[str, str]] = {}

    def select_algorithm(
        self,
        algorithm_choice: str,
        train_data_size: int
    ) -> str:
        """
        アルゴリズムを選択

        Args:
            algorithm_choice: ユーザー選択（'自動選択（推奨）', 'LinearSVC', 'LightGBM'）
            train_data_size: 学習データサイズ

        Returns:
            選択されたアルゴリズム名
        """
        try:
            if algorithm_choice == 'LinearSVC':
                return 'LinearSVC'
            elif algorithm_choice == 'LightGBM':
                return 'LightGBM'
            else:  # 自動選択
                if train_data_size < 500:
                    logger.info(f"学習データ{train_data_size}件 < 500件 → LinearSVC選択")
                    return 'LinearSVC'
                else:
                    logger.info(f"学習データ{train_data_size}件 ≥ 500件 → LightGBM選択")
                    return 'LightGBM'

        except Exception as e:
            logger.error(f"アルゴリズム選択エラー: {str(e)}")
            return 'LinearSVC'  # デフォルト

    def create_classifier(self, algorithm_type: str) -> BaseClassifier:
        """
        分類器を作成

        Args:
            algorithm_type: アルゴリズムタイプ

        Returns:
            分類器インスタンス
        """
        try:
            self.algorithm_type = algorithm_type

            if algorithm_type == 'LinearSVC':
                self.classifier = LinearSVCClassifier()
            elif algorithm_type == 'LightGBM':
                self.classifier = LightGBMClassifier()
            else:
                raise ValueError(f"不明なアルゴリズム: {algorithm_type}")

            logger.info(f"分類器作成完了: {algorithm_type}")
            return self.classifier

        except Exception as e:
            logger.error(f"分類器作成エラー: {str(e)}")
            raise

    def prepare_data(
        self,
        market_df: pd.DataFrame,
        trial_df: pd.DataFrame,
        market_column_mapping: Dict[str, str],
        trial_column_mapping: Dict[str, str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        データの前処理

        Args:
            market_df: 市場データ
            trial_df: トライアルマスター
            market_column_mapping: 市場データのカラムマッピング
            trial_column_mapping: トライアルマスターのカラムマッピング

        Returns:
            (前処理済み市場データ, 前処理済みトライアルマスター)
        """
        try:
            logger.info("データ前処理開始")

            # 市場データの前処理
            market_processed = preprocess_dataframe(market_df, market_column_mapping)

            # トライアルマスターの前処理
            trial_processed = preprocess_dataframe(trial_df, trial_column_mapping)

            logger.info("データ前処理完了")
            return market_processed, trial_processed

        except Exception as e:
            logger.error(f"データ前処理エラー: {str(e)}")
            raise

    def classify(
        self,
        market_df: pd.DataFrame,
        trial_df: pd.DataFrame,
        use_multiprocess: bool = True,
        n_processes: Optional[int] = None,
        progress_callback: Any = None
    ) -> pd.DataFrame:
        """
        商品分類を実行

        Args:
            market_df: 市場データ（前処理済み）
            trial_df: トライアルマスター（前処理済み）
            use_multiprocess: マルチプロセス使用フラグ
            n_processes: プロセス数
            progress_callback: 進捗コールバック

        Returns:
            分類結果DataFrame
        """
        try:
            logger.info("商品分類開始")

            # JAN辞書作成
            self.jan_dict = create_jan_dict(trial_df)
            self.classifier.create_jan_dict(trial_df)

            # JAN一致判定
            jan_matched, jan_unmatched = match_by_jan(market_df, self.jan_dict)

            logger.info(f"JAN一致: {len(jan_matched)}件")
            logger.info(f"JAN不一致（機械学習予測対象）: {len(jan_unmatched)}件")

            # 機械学習予測が必要な場合
            ml_predictions = pd.DataFrame()

            if len(jan_unmatched) > 0:
                # モデル学習
                logger.info("モデル学習開始")

                # 学習データは JAN一致データ + トライアルマスター
                if len(jan_matched) > 0:
                    train_data = pd.concat([jan_matched, trial_df], ignore_index=True)
                    train_data = train_data.drop_duplicates(subset=['jan'])
                else:
                    train_data = trial_df

                self.classifier.train(train_data)
                logger.info("モデル学習完了")

                # 予測実行
                if use_multiprocess and n_processes and n_processes > 1:
                    logger.info(f"マルチプロセス予測: {n_processes}プロセス")
                    ml_predictions = predict_multiprocess(
                        self.classifier,
                        jan_unmatched,
                        n_processes
                    )
                else:
                    logger.info("逐次予測")
                    ml_predictions = predict_sequential(
                        self.classifier,
                        jan_unmatched,
                        progress_callback=progress_callback
                    )

            # 結果結合
            results = combine_results(jan_matched, ml_predictions)

            logger.info(f"商品分類完了: {len(results)}件")
            return results

        except Exception as e:
            logger.error(f"商品分類エラー: {str(e)}")
            raise

    def validate_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        精度検証

        Args:
            data: 検証用データ

        Returns:
            検証結果辞書
        """
        try:
            if not self.classifier:
                raise ValueError("分類器が初期化されていません")

            logger.info("精度検証開始")
            results = self.classifier.validate_accuracy(data)
            logger.info("精度検証完了")

            return results

        except Exception as e:
            logger.error(f"精度検証エラー: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        特徴量重要度を取得

        Returns:
            階層ごとの特徴量重要度辞書
        """
        try:
            if not self.classifier:
                raise ValueError("分類器が初期化されていません")

            logger.info("特徴量重要度取得開始")
            importance = self.classifier.get_feature_importance()
            logger.info("特徴量重要度取得完了")

            return importance

        except Exception as e:
            logger.error(f"特徴量重要度取得エラー: {str(e)}")
            raise

    def get_statistics(self, results: pd.DataFrame) -> Dict[str, Any]:
        """
        分類結果の統計情報を取得

        Args:
            results: 分類結果DataFrame

        Returns:
            統計情報辞書
        """
        try:
            stats = {
                'total': len(results),
                'jan_matched': 0,
                'ml_high': 0,
                'ml_medium': 0,
                'ml_low': 0,
                'jan_matched_rate': 0.0,
                'ml_high_rate': 0.0,
                'ml_medium_rate': 0.0,
                'ml_low_rate': 0.0,
                'average_confidence': 0.0
            }

            if len(results) == 0:
                return stats

            # ステータスごとの件数
            if 'status' in results.columns:
                status_counts = results['status'].value_counts()
                stats['jan_matched'] = status_counts.get('jan_matched', 0)
                stats['ml_high'] = status_counts.get('ml_high', 0)
                stats['ml_medium'] = status_counts.get('ml_medium', 0)
                stats['ml_low'] = status_counts.get('ml_low', 0)

            # 比率計算
            total = stats['total']
            stats['jan_matched_rate'] = stats['jan_matched'] / total if total > 0 else 0.0
            stats['ml_high_rate'] = stats['ml_high'] / total if total > 0 else 0.0
            stats['ml_medium_rate'] = stats['ml_medium'] / total if total > 0 else 0.0
            stats['ml_low_rate'] = stats['ml_low'] / total if total > 0 else 0.0

            # 平均信頼度
            if 'confidence' in results.columns:
                stats['average_confidence'] = results['confidence'].mean()

            return stats

        except Exception as e:
            logger.error(f"統計情報取得エラー: {str(e)}")
            raise
