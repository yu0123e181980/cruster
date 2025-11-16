"""
特徴量重要度分析サービス
"""
import pandas as pd
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_feature_importance_data(
    importance_dict: Dict[str, Dict[str, float]],
    top_n: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    特徴量重要度をDataFrame形式で取得

    Args:
        importance_dict: 階層ごとの特徴量重要度辞書
        top_n: 上位何件を取得するか

    Returns:
        階層ごとのDataFrame辞書
    """
    try:
        logger.info("特徴量重要度データ変換開始")

        result = {}

        for hierarchy, features in importance_dict.items():
            if not features:
                result[hierarchy] = pd.DataFrame()
                continue

            # DataFrameに変換
            df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in features.items()
            ])

            # 重要度でソート
            df = df.sort_values('importance', ascending=False).head(top_n)

            # インデックスをリセット
            df = df.reset_index(drop=True)

            result[hierarchy] = df

        logger.info("特徴量重要度データ変換完了")
        return result

    except Exception as e:
        logger.error(f"特徴量重要度データ変換エラー: {str(e)}")
        raise


def format_feature_name(feature: str, max_length: int = 30) -> str:
    """
    特徴量名を表示用にフォーマット

    Args:
        feature: 元の特徴量名
        max_length: 最大文字数

    Returns:
        フォーマット済み特徴量名
    """
    try:
        if len(feature) <= max_length:
            return feature

        # 長い場合は省略
        return feature[:max_length-3] + '...'

    except Exception as e:
        logger.warning(f"特徴量名フォーマットエラー: {str(e)}")
        return str(feature)


def get_top_features_summary(
    importance_dict: Dict[str, Dict[str, float]],
    top_n: int = 5
) -> Dict[str, List[Tuple[str, float]]]:
    """
    各階層の上位特徴量サマリーを取得

    Args:
        importance_dict: 階層ごとの特徴量重要度辞書
        top_n: 上位何件を取得するか

    Returns:
        階層ごとの上位特徴量リスト
    """
    try:
        summary = {}

        for hierarchy, features in importance_dict.items():
            if not features:
                summary[hierarchy] = []
                continue

            # 重要度でソート
            sorted_features = sorted(
                features.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]

            summary[hierarchy] = sorted_features

        return summary

    except Exception as e:
        logger.error(f"上位特徴量サマリー取得エラー: {str(e)}")
        raise


def create_importance_comparison(
    importance_dict: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    階層間での特徴量重要度を比較するDataFrameを作成

    Args:
        importance_dict: 階層ごとの特徴量重要度辞書

    Returns:
        比較用DataFrame
    """
    try:
        logger.info("特徴量重要度比較データ作成開始")

        # 全ての特徴量を収集
        all_features = set()
        for features in importance_dict.values():
            all_features.update(features.keys())

        # DataFrameを作成
        comparison_data = []

        for feature in all_features:
            row = {'feature': feature}
            for hierarchy, features in importance_dict.items():
                row[hierarchy] = features.get(feature, 0.0)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        # 平均重要度でソート
        hierarchy_cols = list(importance_dict.keys())
        df['average_importance'] = df[hierarchy_cols].mean(axis=1)
        df = df.sort_values('average_importance', ascending=False)
        df = df.drop('average_importance', axis=1)

        logger.info("特徴量重要度比較データ作成完了")
        return df

    except Exception as e:
        logger.error(f"特徴量重要度比較データ作成エラー: {str(e)}")
        raise


def normalize_importance(
    importance_dict: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, float]]:
    """
    特徴量重要度を正規化（0-1）

    Args:
        importance_dict: 階層ごとの特徴量重要度辞書

    Returns:
        正規化された重要度辞書
    """
    try:
        normalized = {}

        for hierarchy, features in importance_dict.items():
            if not features:
                normalized[hierarchy] = {}
                continue

            # 最大値を取得
            max_importance = max(features.values()) if features else 1.0

            # 正規化
            if max_importance > 0:
                normalized[hierarchy] = {
                    k: v / max_importance
                    for k, v in features.items()
                }
            else:
                normalized[hierarchy] = features.copy()

        return normalized

    except Exception as e:
        logger.error(f"特徴量重要度正規化エラー: {str(e)}")
        raise


def export_importance_to_csv(
    importance_dict: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """
    特徴量重要度をCSVにエクスポート

    Args:
        importance_dict: 階層ごとの特徴量重要度辞書
        output_path: 出力ファイルパス
    """
    try:
        logger.info(f"特徴量重要度CSV出力開始: {output_path}")

        # 比較DataFrameを作成
        df = create_importance_comparison(importance_dict)

        # CSV出力
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        logger.info("特徴量重要度CSV出力完了")

    except Exception as e:
        logger.error(f"特徴量重要度CSV出力エラー: {str(e)}")
        raise
