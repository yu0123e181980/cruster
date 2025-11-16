"""
データ前処理ユーティリティ
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_dataframe(
    df: pd.DataFrame,
    column_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    データフレームの前処理

    Args:
        df: 元のDataFrame
        column_mapping: カラムマッピング辞書

    Returns:
        前処理済みDataFrame
    """
    try:
        logger.info(f"データ前処理開始: {len(df)}行")

        # コピーを作成
        processed = df.copy()

        # カラム名を標準化
        rename_dict = {}
        for standard_name, original_name in column_mapping.items():
            if original_name and original_name in processed.columns:
                rename_dict[original_name] = standard_name

        processed = processed.rename(columns=rename_dict)

        # 必須カラムのチェック
        required_columns = ['jan', 'product_name']
        for col in required_columns:
            if col not in processed.columns:
                raise ValueError(f"必須カラム '{col}' が見つかりません")

        # JANコードの正規化
        if 'jan' in processed.columns:
            processed['jan'] = processed['jan'].apply(normalize_jan)

        # 文字列カラムの正規化
        string_columns = ['product_name', 'standard', 'manufacturer']
        for col in string_columns:
            if col in processed.columns:
                processed[col] = processed[col].fillna('').astype(str).apply(normalize_text)

        # 空白行を削除
        processed = processed[processed['jan'].notna() & (processed['jan'] != '')]

        logger.info(f"データ前処理完了: {len(processed)}行")
        return processed

    except Exception as e:
        logger.error(f"前処理エラー: {str(e)}")
        raise


def normalize_jan(jan: any) -> str:
    """
    JANコードの正規化

    Args:
        jan: JANコード

    Returns:
        正規化されたJANコード
    """
    try:
        if pd.isna(jan):
            return ''

        # 文字列に変換
        jan_str = str(jan).strip()

        # 数値の場合、小数点以下を削除
        if '.' in jan_str:
            jan_str = jan_str.split('.')[0]

        # 数字以外を削除
        jan_str = re.sub(r'\D', '', jan_str)

        # 前ゼロ埋め（13桁または8桁）
        if len(jan_str) > 0:
            if len(jan_str) <= 8:
                jan_str = jan_str.zfill(8)
            else:
                jan_str = jan_str.zfill(13)

        return jan_str

    except:
        return ''


def normalize_text(text: str) -> str:
    """
    テキストの正規化

    Args:
        text: 元のテキスト

    Returns:
        正規化されたテキスト
    """
    try:
        if pd.isna(text):
            return ''

        text = str(text).strip()

        # 全角スペースを半角に
        text = text.replace('　', ' ')

        # 連続するスペースを1つに
        text = re.sub(r'\s+', ' ', text)

        # 特殊文字の正規化
        text = text.replace('\u3000', ' ')  # 全角スペース
        text = text.replace('\xa0', ' ')    # ノーブレークスペース

        return text.strip()

    except:
        return ''


def create_jan_dict(trial_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    トライアルマスターからJAN辞書を作成

    Args:
        trial_df: トライアルマスターデータ

    Returns:
        JAN辞書
    """
    try:
        logger.info(f"JAN辞書作成開始: {len(trial_df)}件")

        jan_dict = {}

        required_columns = ['jan', 'category', 'subcategory', 'segment', 'subsegment']
        for col in required_columns:
            if col not in trial_df.columns:
                logger.warning(f"カラム '{col}' が見つかりません")

        for _, row in trial_df.iterrows():
            jan = str(row.get('jan', '')).strip()

            if jan and jan != 'nan' and jan != '':
                jan_dict[jan] = {
                    'category': str(row.get('category', '')),
                    'subcategory': str(row.get('subcategory', '')),
                    'segment': str(row.get('segment', '')),
                    'subsegment': str(row.get('subsegment', '')),
                    'product_name': str(row.get('product_name', '')),
                    'standard': str(row.get('standard', '')),
                    'manufacturer': str(row.get('manufacturer', ''))
                }

        logger.info(f"JAN辞書作成完了: {len(jan_dict)}件")
        return jan_dict

    except Exception as e:
        logger.error(f"JAN辞書作成エラー: {str(e)}")
        raise


def match_by_jan(
    market_df: pd.DataFrame,
    jan_dict: Dict[str, Dict[str, str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    JANコードで直接マッチング

    Args:
        market_df: 市場データ
        jan_dict: JAN辞書

    Returns:
        (マッチしたデータ, マッチしなかったデータ)
    """
    try:
        logger.info(f"JAN一致判定開始: {len(market_df)}件")

        matched_list = []
        unmatched_list = []

        for idx, row in market_df.iterrows():
            jan = str(row.get('jan', '')).strip()

            if jan in jan_dict:
                # マッチした場合
                matched_row = row.to_dict()
                jan_info = jan_dict[jan]

                matched_row.update({
                    'predicted_category': jan_info['category'],
                    'predicted_subcategory': jan_info['subcategory'],
                    'predicted_segment': jan_info['segment'],
                    'predicted_subsegment': jan_info['subsegment'],
                    'confidence': 1.0,
                    'status': 'jan_matched',
                    'method': 'JAN直接マッチ',
                    'category': jan_info['category'],
                    'subcategory': jan_info['subcategory'],
                    'segment': jan_info['segment'],
                    'subsegment': jan_info['subsegment']
                })

                matched_list.append(matched_row)
            else:
                # マッチしなかった場合
                unmatched_list.append(row.to_dict())

        matched_df = pd.DataFrame(matched_list) if matched_list else pd.DataFrame()
        unmatched_df = pd.DataFrame(unmatched_list) if unmatched_list else pd.DataFrame()

        logger.info(f"JAN一致: {len(matched_df)}件, 不一致: {len(unmatched_df)}件")

        return matched_df, unmatched_df

    except Exception as e:
        logger.error(f"JAN一致判定エラー: {str(e)}")
        raise


def assign_status(confidence: float) -> str:
    """
    信頼度に基づいてステータスを割り当て

    Args:
        confidence: 信頼度（0-1）

    Returns:
        ステータス文字列
    """
    if confidence >= 0.8:
        return 'ml_high'
    elif confidence >= 0.5:
        return 'ml_medium'
    else:
        return 'ml_low'


def combine_results(
    jan_matched: pd.DataFrame,
    ml_predicted: pd.DataFrame
) -> pd.DataFrame:
    """
    JAN一致結果と機械学習予測結果を結合

    Args:
        jan_matched: JAN一致データ
        ml_predicted: 機械学習予測データ

    Returns:
        結合されたDataFrame
    """
    try:
        logger.info("結果結合開始")

        # 機械学習予測結果にステータスとメソッドを追加
        if len(ml_predicted) > 0:
            ml_predicted['status'] = ml_predicted['confidence'].apply(assign_status)
            ml_predicted['method'] = '機械学習予測'

        # 結合
        if len(jan_matched) > 0 and len(ml_predicted) > 0:
            combined = pd.concat([jan_matched, ml_predicted], ignore_index=True)
        elif len(jan_matched) > 0:
            combined = jan_matched
        elif len(ml_predicted) > 0:
            combined = ml_predicted
        else:
            combined = pd.DataFrame()

        logger.info(f"結果結合完了: {len(combined)}件")
        return combined

    except Exception as e:
        logger.error(f"結果結合エラー: {str(e)}")
        raise


def validate_dataframe(df: pd.DataFrame, data_type: str) -> List[str]:
    """
    データフレームの妥当性検証

    Args:
        df: 検証対象のDataFrame
        data_type: データタイプ（'market' or 'trial'）

    Returns:
        エラーメッセージのリスト
    """
    errors = []

    # 空チェック
    if len(df) == 0:
        errors.append(f"{data_type}データが空です")
        return errors

    # 必須カラムチェック
    if data_type == 'market':
        required = ['jan', 'product_name']
    else:  # trial
        required = ['jan', 'product_name', 'category', 'subcategory', 'segment', 'subsegment']

    for col in required:
        if col not in df.columns:
            errors.append(f"必須カラム '{col}' が見つかりません")

    # データ件数チェック
    if data_type == 'trial' and len(df) < 10:
        errors.append(f"トライアルマスターが少なすぎます（最低10件必要、現在{len(df)}件）")

    return errors
