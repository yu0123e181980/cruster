"""
マルチプロセス処理ユーティリティ
"""
import pandas as pd
import numpy as np
from typing import List, Any
from multiprocessing import Pool, cpu_count
import logging
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_dataframe(df: pd.DataFrame, n_splits: int) -> List[pd.DataFrame]:
    """
    データフレームをn分割

    Args:
        df: 分割対象のDataFrame
        n_splits: 分割数

    Returns:
        分割されたDataFrameのリスト
    """
    try:
        if len(df) == 0:
            return []

        chunk_size = max(1, len(df) // n_splits)
        chunks = []

        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunks.append(chunk)

        logger.info(f"データフレーム分割完了: {len(chunks)}個のチャンク")
        return chunks

    except Exception as e:
        logger.error(f"データフレーム分割エラー: {str(e)}")
        raise


def predict_chunk(args: tuple) -> pd.DataFrame:
    """
    チャンクごとの予測処理（マルチプロセス用）

    Args:
        args: (classifier, chunk_df)のタプル

    Returns:
        予測結果のDataFrame
    """
    try:
        classifier, chunk = args

        # 予測実行
        result = classifier.predict(chunk)

        # メモリ解放
        del chunk
        gc.collect()

        return result

    except Exception as e:
        logger.error(f"チャンク予測エラー: {str(e)}")
        # エラーの場合は空のDataFrameを返す
        return pd.DataFrame()


def predict_multiprocess(
    classifier: Any,
    data: pd.DataFrame,
    n_processes: int = None,
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    マルチプロセスで予測実行

    Args:
        classifier: 分類器
        data: 予測対象データ
        n_processes: プロセス数（Noneの場合はCPUコア数-1）
        batch_size: バッチサイズ

    Returns:
        予測結果のDataFrame
    """
    try:
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)

        logger.info(f"マルチプロセス予測開始: {len(data)}件, {n_processes}プロセス")

        # データが少ない場合は通常予測
        if len(data) < batch_size:
            logger.info("データ量が少ないため、シングルプロセスで実行")
            return classifier.predict(data)

        # データ分割
        chunks = split_dataframe(data, n_processes)

        # 引数のリストを作成
        args_list = [(classifier, chunk) for chunk in chunks]

        # マルチプロセス実行
        with Pool(processes=n_processes) as pool:
            results = pool.map(predict_chunk, args_list)

        # 結果を結合
        combined_results = pd.concat(results, ignore_index=True)

        # メモリ解放
        del results
        del chunks
        gc.collect()

        logger.info(f"マルチプロセス予測完了: {len(combined_results)}件")
        return combined_results

    except Exception as e:
        logger.error(f"マルチプロセス予測エラー: {str(e)}")
        logger.warning("シングルプロセスにフォールバック")
        # エラーの場合はシングルプロセスで実行
        return classifier.predict(data)


def predict_sequential(
    classifier: Any,
    data: pd.DataFrame,
    batch_size: int = 1000,
    progress_callback: Any = None
) -> pd.DataFrame:
    """
    バッチ処理で逐次予測（メモリ効率重視）

    Args:
        classifier: 分類器
        data: 予測対象データ
        batch_size: バッチサイズ
        progress_callback: 進捗コールバック関数

    Returns:
        予測結果のDataFrame
    """
    try:
        logger.info(f"逐次予測開始: {len(data)}件, バッチサイズ={batch_size}")

        results = []
        total_batches = (len(data) + batch_size - 1) // batch_size

        for i in range(0, len(data), batch_size):
            batch = data.iloc[i:i+batch_size].copy()

            # 予測実行
            batch_result = classifier.predict(batch)
            results.append(batch_result)

            # 進捗報告
            current_batch = i // batch_size + 1
            if progress_callback:
                progress = current_batch / total_batches
                progress_callback(progress)

            if current_batch % 10 == 0:
                logger.info(f"進捗: {current_batch}/{total_batches} バッチ")

            # メモリ解放
            del batch
            del batch_result
            gc.collect()

        # 結合
        combined = pd.concat(results, ignore_index=True)

        # メモリ解放
        del results
        gc.collect()

        logger.info(f"逐次予測完了: {len(combined)}件")
        return combined

    except Exception as e:
        logger.error(f"逐次予測エラー: {str(e)}")
        raise


def get_optimal_process_count() -> int:
    """
    最適なプロセス数を取得

    Returns:
        推奨プロセス数
    """
    try:
        n_cpu = cpu_count()

        # 最小1、最大でCPU数-1
        optimal = max(1, n_cpu - 1)

        logger.info(f"検出されたCPU数: {n_cpu}, 推奨プロセス数: {optimal}")
        return optimal

    except Exception as e:
        logger.warning(f"CPU数取得エラー: {str(e)}, デフォルト値1を使用")
        return 1
