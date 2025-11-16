"""
SHAP値による説明性分析サービス
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import shap

from classifier import BaseClassifier, LinearSVCClassifier, LightGBMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_shap_values(
    classifier: BaseClassifier,
    product_data: pd.Series,
    hierarchy: str = 'predicted_category'
) -> Optional[shap.Explanation]:
    """
    SHAP値を計算

    Args:
        classifier: 分類器
        product_data: 商品データ（1行）
        hierarchy: 階層（'predicted_category', 'predicted_subcategory', など）

    Returns:
        SHAP Explanationオブジェクト
    """
    try:
        logger.info(f"SHAP値計算開始: {hierarchy}")

        # LinearSVCの場合
        if isinstance(classifier, LinearSVCClassifier):
            return _get_shap_values_linearsvc(classifier, product_data, hierarchy)

        # LightGBMの場合
        elif isinstance(classifier, LightGBMClassifier):
            return _get_shap_values_lightgbm(classifier, product_data, hierarchy)

        else:
            logger.warning(f"未対応の分類器タイプ: {type(classifier)}")
            return None

    except Exception as e:
        logger.error(f"SHAP値計算エラー: {str(e)}")
        return None


def _get_shap_values_linearsvc(
    classifier: LinearSVCClassifier,
    product_data: pd.Series,
    hierarchy: str
) -> Optional[shap.Explanation]:
    """
    LinearSVCのSHAP値を計算

    Args:
        classifier: LinearSVC分類器
        product_data: 商品データ
        hierarchy: 階層

    Returns:
        SHAP Explanationオブジェクト
    """
    try:
        if hierarchy not in classifier.pipelines:
            logger.warning(f"{hierarchy} のモデルが見つかりません")
            return None

        pipeline = classifier.pipelines[hierarchy]

        # 特徴量作成
        feature_text = _create_feature_text(product_data)

        # TF-IDFベクトル化
        X_vec = pipeline.named_steps['tfidf'].transform([feature_text])

        # LinearExplainerを使用
        explainer = shap.LinearExplainer(
            pipeline.named_steps['clf'],
            X_vec,
            feature_perturbation="interventional"
        )

        # SHAP値計算
        shap_values = explainer(X_vec)

        logger.info("LinearSVC SHAP値計算完了")
        return shap_values

    except Exception as e:
        logger.error(f"LinearSVC SHAP値計算エラー: {str(e)}")
        return None


def _get_shap_values_lightgbm(
    classifier: LightGBMClassifier,
    product_data: pd.Series,
    hierarchy: str
) -> Optional[shap.Explanation]:
    """
    LightGBMのSHAP値を計算

    Args:
        classifier: LightGBM分類器
        product_data: 商品データ
        hierarchy: 階層

    Returns:
        SHAP Explanationオブジェクト
    """
    try:
        if hierarchy not in classifier.models:
            logger.warning(f"{hierarchy} のモデルが見つかりません")
            return None

        model = classifier.models[hierarchy]

        # 特徴量作成
        feature_text = _create_feature_text(product_data)

        # TF-IDFベクトル化
        X_vec = classifier.vectorizers['main'].transform([feature_text]).toarray()

        # TreeExplainerを使用
        explainer = shap.TreeExplainer(model)

        # SHAP値計算
        shap_values = explainer(X_vec)

        logger.info("LightGBM SHAP値計算完了")
        return shap_values

    except Exception as e:
        logger.error(f"LightGBM SHAP値計算エラー: {str(e)}")
        return None


def _create_feature_text(product_data: pd.Series) -> str:
    """
    商品データから特徴量文字列を作成

    Args:
        product_data: 商品データ

    Returns:
        特徴量文字列
    """
    parts = []

    if 'product_name' in product_data.index:
        parts.append(str(product_data.get('product_name', '')))

    if 'standard' in product_data.index:
        parts.append(str(product_data.get('standard', '')))

    if 'manufacturer' in product_data.index:
        parts.append(str(product_data.get('manufacturer', '')))

    return ' '.join(parts)


def create_shap_waterfall(
    shap_values: shap.Explanation,
    max_display: int = 10
) -> plt.Figure:
    """
    SHAP Waterfall Plotを作成

    Args:
        shap_values: SHAP値
        max_display: 表示する特徴量の最大数

    Returns:
        matplotlib Figure
    """
    try:
        logger.info("SHAP Waterfall Plot作成開始")

        # 新しいfigureを作成
        fig, ax = plt.subplots(figsize=(10, 6))

        # Waterfall Plotを描画
        shap.waterfall_plot(
            shap_values[0],
            max_display=max_display,
            show=False
        )

        plt.tight_layout()

        logger.info("SHAP Waterfall Plot作成完了")
        return fig

    except Exception as e:
        logger.error(f"SHAP Waterfall Plot作成エラー: {str(e)}")
        raise


def create_shap_summary(
    classifier: BaseClassifier,
    sample_data: pd.DataFrame,
    hierarchy: str = 'predicted_category',
    max_display: int = 20
) -> Optional[plt.Figure]:
    """
    SHAP Summary Plotを作成

    Args:
        classifier: 分類器
        sample_data: サンプルデータ
        hierarchy: 階層
        max_display: 表示する特徴量の最大数

    Returns:
        matplotlib Figure
    """
    try:
        logger.info(f"SHAP Summary Plot作成開始: {hierarchy}")

        # LinearSVCの場合
        if isinstance(classifier, LinearSVCClassifier):
            if hierarchy not in classifier.pipelines:
                logger.warning(f"{hierarchy} のモデルが見つかりません")
                return None

            pipeline = classifier.pipelines[hierarchy]

            # 特徴量作成
            features = classifier._create_features(sample_data)
            X_vec = pipeline.named_steps['tfidf'].transform(features)

            # LinearExplainer
            explainer = shap.LinearExplainer(
                pipeline.named_steps['clf'],
                X_vec,
                feature_perturbation="interventional"
            )

            shap_values = explainer(X_vec)

        # LightGBMの場合
        elif isinstance(classifier, LightGBMClassifier):
            if hierarchy not in classifier.models:
                logger.warning(f"{hierarchy} のモデルが見つかりません")
                return None

            model = classifier.models[hierarchy]

            # 特徴量作成
            features = classifier._create_features(sample_data)
            X_vec = classifier.vectorizers['main'].transform(features).toarray()

            # TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_vec)

        else:
            logger.warning(f"未対応の分類器タイプ: {type(classifier)}")
            return None

        # Summary Plotを作成
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()

        logger.info("SHAP Summary Plot作成完了")
        return fig

    except Exception as e:
        logger.error(f"SHAP Summary Plot作成エラー: {str(e)}")
        return None


def get_shap_feature_importance(
    shap_values: shap.Explanation,
    top_n: int = 10
) -> pd.DataFrame:
    """
    SHAP値から特徴量重要度を取得

    Args:
        shap_values: SHAP値
        top_n: 上位何件を取得するか

    Returns:
        特徴量重要度のDataFrame
    """
    try:
        # SHAP値の絶対値の平均を計算
        importance = np.abs(shap_values.values).mean(axis=0)

        # 特徴量名を取得
        if hasattr(shap_values, 'feature_names'):
            feature_names = shap_values.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(len(importance))]

        # DataFrameに変換
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        # ソート
        df = df.sort_values('importance', ascending=False).head(top_n)
        df = df.reset_index(drop=True)

        return df

    except Exception as e:
        logger.error(f"SHAP特徴量重要度取得エラー: {str(e)}")
        raise
