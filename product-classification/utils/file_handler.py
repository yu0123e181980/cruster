"""
ファイル読み書きユーティリティ
"""
import pandas as pd
import numpy as np
from typing import Optional, Union
import chardet
import logging
import os
from datetime import datetime
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_file(
    file_obj: Union[str, io.BytesIO],
    encoding: Optional[str] = None
) -> pd.DataFrame:
    """
    CSV/Excelファイルを読み込み
    複数エンコーディングを試行して日本語に対応

    Args:
        file_obj: ファイルパスまたはファイルオブジェクト
        encoding: エンコーディング（指定なしの場合は自動検出）

    Returns:
        読み込んだDataFrame
    """
    try:
        # ファイル名を取得
        if isinstance(file_obj, str):
            file_path = file_obj
            file_name = os.path.basename(file_path)
        else:
            file_name = getattr(file_obj, 'name', 'uploaded_file')

        logger.info(f"ファイル読み込み開始: {file_name}")

        # Excelファイルの場合
        if file_name.endswith(('.xlsx', '.xls')):
            if isinstance(file_obj, str):
                df = pd.read_excel(file_path)
            else:
                df = pd.read_excel(file_obj)
            logger.info(f"Excel読み込み完了: {len(df)}行")
            return df

        # CSVファイルの場合
        encodings = ['utf-8', 'utf-8-sig', 'shift-jis', 'cp932', 'euc-jp', 'iso-2022-jp']

        if encoding:
            encodings.insert(0, encoding)

        # バイナリデータを取得
        if isinstance(file_obj, str):
            with open(file_path, 'rb') as f:
                raw_data = f.read()
        else:
            raw_data = file_obj.read()
            file_obj.seek(0)  # ポインタをリセット

        # エンコーディング自動検出
        detected = chardet.detect(raw_data)
        if detected['encoding'] and detected['confidence'] > 0.7:
            detected_encoding = detected['encoding']
            if detected_encoding not in encodings:
                encodings.insert(0, detected_encoding)
            logger.info(f"検出されたエンコーディング: {detected_encoding} (信頼度: {detected['confidence']})")

        # 各エンコーディングで試行
        last_error = None
        for enc in encodings:
            try:
                if isinstance(file_obj, str):
                    df = pd.read_csv(file_path, encoding=enc)
                else:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding=enc)

                logger.info(f"CSV読み込み成功: {enc}, {len(df)}行")
                return df

            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_error = e
                continue

        # 全てのエンコーディングで失敗
        raise ValueError(f"ファイル読み込み失敗: {last_error}")

    except Exception as e:
        logger.error(f"ファイル読み込みエラー: {str(e)}")
        raise


def save_results(
    df: pd.DataFrame,
    process_id: str,
    output_format: str = 'csv',
    output_dir: str = 'product-classification/data/results'
) -> str:
    """
    結果をCSV/Excelで保存

    Args:
        df: 保存するDataFrame
        process_id: プロセスID
        output_format: 出力形式（'csv' or 'excel'）
        output_dir: 出力ディレクトリ

    Returns:
        保存したファイルパス
    """
    try:
        # ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)

        # タイムスタンプ付きファイル名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if output_format.lower() in ['csv', 'csv（推奨）']:
            file_path = os.path.join(
                output_dir,
                f'classification_results_{timestamp}_{process_id}.csv'
            )

            # 大量データの場合はストリーミング書き込み
            if len(df) > 100000:
                logger.info(f"大量データ検出: ストリーミング書き込み開始")
                save_csv_streaming(df, file_path)
            else:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')

            logger.info(f"CSV保存完了: {file_path}")

        else:  # Excel
            file_path = os.path.join(
                output_dir,
                f'classification_results_{timestamp}_{process_id}.xlsx'
            )

            # Excelの最大行数チェック
            if len(df) > 1048576:
                raise ValueError("Excelの最大行数（1,048,576行）を超えています。CSV形式を使用してください。")

            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='分類結果')

            logger.info(f"Excel保存完了: {file_path}")

        return file_path

    except Exception as e:
        logger.error(f"ファイル保存エラー: {str(e)}")
        raise


def save_csv_streaming(
    df: pd.DataFrame,
    file_path: str,
    chunksize: int = 1000
) -> None:
    """
    大量データをストリーミングでCSV出力
    メモリ効率を重視

    Args:
        df: 保存するDataFrame
        file_path: 出力ファイルパス
        chunksize: チャンクサイズ
    """
    try:
        logger.info(f"ストリーミングCSV書き込み開始: {len(df)}行")

        # ヘッダー書き込み
        df.iloc[:0].to_csv(file_path, index=False, encoding='utf-8-sig', mode='w')

        # チャンクごとに書き込み
        total_chunks = (len(df) + chunksize - 1) // chunksize

        for i in range(0, len(df), chunksize):
            chunk = df.iloc[i:i+chunksize]
            chunk.to_csv(
                file_path,
                index=False,
                encoding='utf-8-sig',
                mode='a',
                header=False
            )

            if (i // chunksize + 1) % 10 == 0:
                logger.info(f"進捗: {i//chunksize + 1}/{total_chunks} チャンク")

        logger.info(f"ストリーミングCSV書き込み完了")

    except Exception as e:
        logger.error(f"ストリーミング書き込みエラー: {str(e)}")
        raise


def detect_column_mapping(df: pd.DataFrame) -> dict:
    """
    カラム名を自動検出

    Args:
        df: データフレーム

    Returns:
        カラムマッピング辞書
    """
    mapping = {
        'jan': None,
        'product_name': None,
        'standard': None,
        'manufacturer': None
    }

    columns = df.columns.tolist()

    # JAN
    for col in columns:
        col_lower = col.lower()
        if 'jan' in col_lower or 'code' in col_lower or 'バーコード' in col:
            mapping['jan'] = col
            break

    # 商品名
    for col in columns:
        col_lower = col.lower()
        if '商品名' in col or 'product' in col_lower or '品名' in col:
            mapping['product_name'] = col
            break

    # 規格
    for col in columns:
        if '規格' in col or 'standard' in col.lower() or 'spec' in col.lower():
            mapping['standard'] = col
            break

    # メーカー
    for col in columns:
        col_lower = col.lower()
        if 'メーカー' in col or 'manufacturer' in col_lower or '製造' in col:
            mapping['manufacturer'] = col
            break

    return mapping
