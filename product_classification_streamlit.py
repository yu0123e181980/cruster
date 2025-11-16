#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品分類システム - Streamlit版
階層型機械学習（LinearSVC/LightGBM自動切り替え）+ 特徴量重要度 + SHAP分析
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
import re
import logging
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from functools import partial

# LightGBM import (optional)
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available. Only LinearSVC will be used.")

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def normalize_jan(jan_value) -> str:
    """JANコードを正規化（.0除去、文字列化、トリミング）"""
    if pd.isna(jan_value) or jan_value == '':
        return ''

    jan_str = str(jan_value).strip()

    if jan_str.endswith('.0'):
        jan_str = jan_str[:-2]

    if jan_str.lower() in ['nan', 'none', '']:
        return ''

    return jan_str


class BaseClassifier(ABC):
    """分類器の基底クラス"""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """モデルを訓練"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を実行"""
        pass

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """決定関数（信頼度計算用）"""
        pass

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """特徴量重要度を取得"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """分類器の名前"""
        pass


class LinearSVCClassifier(BaseClassifier):
    """LinearSVCベースの分類器"""

    def __init__(self):
        self.model = LinearSVC(
            class_weight='balanced',
            random_state=42,
            max_iter=2000,
            dual=False
        )
        self._feature_importance = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        # 特徴量重要度を係数の絶対値の平均として計算
        if len(self.model.coef_.shape) > 1:
            self._feature_importance = np.abs(self.model.coef_).mean(axis=0)
        else:
            self._feature_importance = np.abs(self.model.coef_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)

    def get_feature_importance(self) -> np.ndarray:
        return self._feature_importance if self._feature_importance is not None else np.array([])

    @property
    def name(self) -> str:
        return "LinearSVC"


class LightGBMClassifier(BaseClassifier):
    """LightGBMベースの分類器"""

    def __init__(self):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        self.model = None
        self._feature_importance = None
        self._n_classes = 0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # クラス数を確認
        self._n_classes = len(np.unique(y))

        params = {
            'objective': 'multiclass' if self._n_classes > 2 else 'binary',
            'num_class': self._n_classes if self._n_classes > 2 else 1,
            'metric': 'multi_logloss' if self._n_classes > 2 else 'binary_logloss',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1
        }

        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(0)]
        )

        self._feature_importance = self.model.feature_importance(importance_type='gain')

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X)

        if self._n_classes > 2:
            return np.argmax(predictions, axis=1)
        else:
            return (predictions > 0.5).astype(int).flatten()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        predictions = self.model.predict(X)

        if self._n_classes > 2:
            # 多クラス: 最大確率値を返す
            return np.max(predictions, axis=1)
        else:
            # 2クラス: 確率値をロジット変換
            probs = predictions.flatten()
            return np.log(probs / (1 - probs + 1e-10))

    def get_feature_importance(self) -> np.ndarray:
        return self._feature_importance if self._feature_importance is not None else np.array([])

    @property
    def name(self) -> str:
        return "LightGBM"


class ProductClassifierStreamlit:
    """Streamlit用商品分類エンジン（階層型学習、自動アルゴリズム選択対応）"""

    def __init__(self, algorithm: str = 'auto'):
        """
        Args:
            algorithm: 'auto', 'linearsvc', 'lightgbm'
        """
        self.algorithm = algorithm
        self.vectorizer = None
        self.models: Dict[str, BaseClassifier] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.jan_dict = {}
        self.is_trained = False
        self.hierarchy_mapping = {}
        self.use_hierarchical = True
        self.feature_names = []
        self.training_data_size = 0

    def _get_classifier(self, n_samples: int) -> BaseClassifier:
        """データサイズに応じて分類器を選択"""
        if self.algorithm == 'linearsvc':
            return LinearSVCClassifier()
        elif self.algorithm == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                logger.warning("LightGBM not available, using LinearSVC")
                return LinearSVCClassifier()
            return LightGBMClassifier()
        else:  # auto
            if n_samples >= 500 and LIGHTGBM_AVAILABLE:
                return LightGBMClassifier()
            else:
                return LinearSVCClassifier()

    def normalize_text(self, text: str) -> str:
        """日本語テキストの正規化処理"""
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        return text.strip()

    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """数値特徴量を抽出（容量・入数・価格）"""
        result_df = df.copy()

        if 'avg_price' in df.columns:
            def extract_price(price_str):
                if pd.isna(price_str) or price_str == '':
                    return 0
                matches = re.findall(r'[\d,]+', str(price_str))
                if matches:
                    try:
                        return float(matches[0].replace(',', ''))
                    except:
                        return 0
                return 0

            result_df['price_numeric'] = df['avg_price'].apply(extract_price)

        if 'standard' in df.columns:
            def extract_volume(standard_str):
                if pd.isna(standard_str) or standard_str == '':
                    return 0
                matches = re.findall(r'(\d+(?:\.\d+)?)\s*(ml|g|kg|l)', str(standard_str).lower())
                if matches:
                    value, unit = matches[0]
                    value = float(value)
                    if unit in ['kg']:
                        value *= 1000
                    elif unit in ['l']:
                        value *= 1000
                    return value
                return 0

            def extract_count(standard_str):
                if pd.isna(standard_str) or standard_str == '':
                    return 1
                matches = re.findall(r'(\d+)\s*個|(\d+)\s*本|(\d+)\s*袋', str(standard_str))
                if matches:
                    for match in matches:
                        for count in match:
                            if count:
                                return int(count)
                return 1

            result_df['volume_numeric'] = df['standard'].apply(extract_volume)
            result_df['count_numeric'] = df['standard'].apply(extract_count)

        if 'jan' in df.columns:
            def extract_maker_code(jan_str):
                if pd.isna(jan_str) or jan_str == '':
                    return ''
                jan_str = str(jan_str).strip()
                if len(jan_str) >= 7:
                    return jan_str[:7]
                return ''

            result_df['jan_maker_code'] = df['jan'].apply(extract_maker_code)

        return result_df

    def create_hierarchy_mapping(self, trial_df: pd.DataFrame):
        """階層構造マッピングを作成"""
        try:
            trial_df.columns = trial_df.columns.str.strip()

            required_columns = ['カテゴリー名', 'サブカテゴリー名', 'セグメント名', 'サブセグメント名']
            missing_columns = [col for col in required_columns if col not in trial_df.columns]

            if missing_columns:
                logger.error(f"必要なカラムが見つかりません: {missing_columns}")
                raise ValueError(f"必要なカラムが見つかりません: {missing_columns}")

            self.hierarchy_mapping = {
                'category_to_subcategory': defaultdict(set),
                'subcategory_to_segment': defaultdict(set),
                'segment_to_subsegment': defaultdict(set)
            }

            for _, row in trial_df.iterrows():
                category = row['カテゴリー名']
                subcategory = row['サブカテゴリー名']
                segment = row['セグメント名']
                subsegment = row['サブセグメント名']

                if pd.notna(category) and pd.notna(subcategory):
                    self.hierarchy_mapping['category_to_subcategory'][category].add(subcategory)
                if pd.notna(subcategory) and pd.notna(segment):
                    self.hierarchy_mapping['subcategory_to_segment'][subcategory].add(segment)
                if pd.notna(segment) and pd.notna(subsegment):
                    self.hierarchy_mapping['segment_to_subsegment'][segment].add(subsegment)

            for key in self.hierarchy_mapping:
                for parent_key in self.hierarchy_mapping[key]:
                    self.hierarchy_mapping[key][parent_key] = list(self.hierarchy_mapping[key][parent_key])

            logger.info(f"階層マッピング作成完了")

        except Exception as e:
            logger.error(f"階層マッピング作成エラー: {e}")
            raise

    def prepare_features(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        """特徴量を作成（文字n-gram + 数値特徴量）"""
        try:
            text_features = []
            for _, row in df.iterrows():
                text_parts = []
                for col in feature_cols:
                    if col in ['price_numeric', 'volume_numeric', 'count_numeric']:
                        continue
                    text_parts.append(self.normalize_text(row.get(col, '')))
                text_features.append(' '.join(text_parts))

            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(2, 4),
                    max_features=2000,
                    min_df=2,
                    max_df=0.9
                )
                text_matrix = self.vectorizer.fit_transform(text_features).toarray()
                # 特徴量名を保存
                self.feature_names = list(self.vectorizer.get_feature_names_out())
            else:
                text_matrix = self.vectorizer.transform(text_features).toarray()

            numeric_features = []
            for col in ['price_numeric', 'volume_numeric', 'count_numeric']:
                if col in df.columns:
                    values = df[col].fillna(0).values
                    if np.max(values) > 0:
                        values = values / np.max(values)
                    numeric_features.append(values.reshape(-1, 1))
                    if self.vectorizer is None or col not in self.feature_names:
                        self.feature_names.append(col)

            if numeric_features:
                numeric_matrix = np.hstack(numeric_features)
                features = np.hstack([text_matrix, numeric_matrix])
            else:
                features = text_matrix

            return features
        except Exception as e:
            logger.error(f"特徴量作成エラー: {e}")
            raise

    def create_jan_dict(self, trial_df: pd.DataFrame):
        """JAN辞書を作成"""
        try:
            trial_df.columns = trial_df.columns.str.strip()

            self.jan_dict = {}
            for _, row in trial_df.iterrows():
                jan = normalize_jan(row['JAN'])

                if jan:
                    self.jan_dict[jan] = {
                        'カテゴリー名': row['カテゴリー名'],
                        'サブカテゴリー名': row['サブカテゴリー名'],
                        'セグメント名': row['セグメント名'],
                        'サブセグメント名': row['サブセグメント名']
                    }
            logger.info(f"JAN辞書作成完了: {len(self.jan_dict)}件")
        except Exception as e:
            logger.error(f"JAN辞書作成エラー: {e}")
            raise

    def train(self, train_df: pd.DataFrame, feature_cols: list, target_cols: list,
              use_hierarchical: bool = True):
        """モデルを訓練"""
        try:
            train_df = train_df.dropna(subset=target_cols)
            if len(train_df) == 0:
                raise ValueError("訓練データが空です")

            self.training_data_size = len(train_df)

            if use_hierarchical and not self.hierarchy_mapping:
                logger.warning("階層マッピングが未作成のため従来型学習に切り替えます")
                use_hierarchical = False

            self.use_hierarchical = use_hierarchical

            train_df = self.extract_numeric_features(train_df)
            X = self.prepare_features(train_df, feature_cols)

            for target_col in target_cols:
                le = LabelEncoder()
                y = le.fit_transform(train_df[target_col].fillna('不明'))
                self.label_encoders[target_col] = le

                # 分類器を選択
                classifier = self._get_classifier(len(train_df))
                classifier.fit(X, y)
                self.models[target_col] = classifier

                logger.info(f"{target_col}: {classifier.name} 使用")

            self.is_trained = True
            algorithm_used = list(self.models.values())[0].name if self.models else "Unknown"
            logger.info(f"モデル訓練完了: {len(train_df)}件のデータで学習（{algorithm_used}）")

        except Exception as e:
            logger.error(f"モデル訓練エラー: {e}")
            raise

    def get_feature_importance(self, target_col: str, top_n: int = 20) -> pd.DataFrame:
        """指定されたターゲットの特徴量重要度を取得"""
        if target_col not in self.models:
            return pd.DataFrame()

        classifier = self.models[target_col]
        importance = classifier.get_feature_importance()

        if len(importance) == 0:
            return pd.DataFrame()

        # 特徴量名と重要度を結合
        feature_names = self.feature_names[:len(importance)]

        df = pd.DataFrame({
            '特徴量': feature_names,
            '重要度': importance
        })

        # 降順ソートしてトップNを取得
        df = df.sort_values('重要度', ascending=False).head(top_n)

        return df

    def predict_batch(self, df: pd.DataFrame, feature_cols: list,
                     target_cols: list, use_hierarchical: bool = True,
                     batch_size: int = 1000, progress_callback=None) -> pd.DataFrame:
        """バッチ予測を実行（大量データ対応）"""
        try:
            if not self.is_trained:
                raise ValueError("モデルが訓練されていません")

            use_hierarchical = use_hierarchical and bool(self.hierarchy_mapping)

            df = self.extract_numeric_features(df)

            results = []
            total_rows = len(df)

            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i+batch_size]
                X = self.prepare_features(batch, feature_cols)

                batch_results = batch.copy()

                if use_hierarchical:
                    batch_results = self._hierarchical_predict(batch_results, X, target_cols)
                else:
                    batch_results = self._standard_predict(batch_results, X, target_cols)

                results.append(batch_results)

                # 進捗コールバック
                if progress_callback:
                    progress = min((i + batch_size) / total_rows, 1.0)
                    progress_callback(progress)

            logger.info(f"予測完了: {total_rows}件")

            return pd.concat(results, ignore_index=True)

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            raise

    def _standard_predict(self, batch_results: pd.DataFrame, X: np.ndarray,
                         target_cols: list) -> pd.DataFrame:
        """従来型予測"""
        confidences = []

        for target_col in target_cols:
            model = self.models[target_col]
            le = self.label_encoders[target_col]

            predictions = model.predict(X)
            decision_scores = model.decision_function(X)

            if decision_scores.ndim > 1:
                max_scores = np.max(decision_scores, axis=1)
                confidences_normalized = 1 / (1 + np.exp(-max_scores))
            else:
                confidences_normalized = 1 / (1 + np.exp(-np.abs(decision_scores)))

            decoded_predictions = le.inverse_transform(predictions)
            batch_results[target_col] = decoded_predictions

            confidences.append(confidences_normalized)

        batch_results['confidence'] = np.mean(confidences, axis=0)

        return batch_results

    def _hierarchical_predict(self, batch_results: pd.DataFrame, X: np.ndarray,
                             target_cols: list) -> pd.DataFrame:
        """階層型予測"""
        hierarchy_order = [
            ('predicted_category', None),
            ('predicted_subcategory', 'subcategory'),
            ('predicted_segment', 'segment'),
            ('predicted_subsegment', 'subsegment')
        ]

        batch_confidences = []

        for idx in range(len(batch_results)):
            row_predictions = {}
            row_confidences = []

            for target_col, hierarchy_level in hierarchy_order:
                model = self.models[target_col]
                le = self.label_encoders[target_col]

                X_single = X[idx:idx+1]
                predictions = model.predict(X_single)
                decision_scores = model.decision_function(X_single)

                if decision_scores.ndim > 1:
                    max_score = np.max(decision_scores)
                    base_confidence = 1 / (1 + np.exp(-max_score))
                else:
                    base_confidence = 1 / (1 + np.exp(-np.abs(decision_scores[0])))

                decoded_prediction = le.inverse_transform(predictions)[0]

                if hierarchy_level:
                    parent_col = hierarchy_order[hierarchy_order.index((target_col, hierarchy_level)) - 1][0]
                    parent_prediction = row_predictions[parent_col]

                    valid_choices = self._get_valid_choices(parent_prediction, hierarchy_level)

                    if valid_choices and decoded_prediction not in valid_choices:
                        decoded_prediction = valid_choices[0] if valid_choices else decoded_prediction
                        base_confidence *= 0.5

                row_predictions[target_col] = decoded_prediction
                row_confidences.append(base_confidence)

            for target_col, _ in hierarchy_order:
                batch_results.loc[batch_results.index[idx], target_col] = row_predictions[target_col]

            batch_confidences.append(np.mean(row_confidences))

        batch_results['confidence'] = batch_confidences

        return batch_results

    def _get_valid_choices(self, parent_prediction: str, hierarchy_level: str) -> list:
        """階層制約に基づく有効な選択肢を取得"""
        mapping_key = {
            'subcategory': 'category_to_subcategory',
            'segment': 'subcategory_to_segment',
            'subsegment': 'segment_to_subsegment'
        }.get(hierarchy_level)

        if mapping_key and parent_prediction in self.hierarchy_mapping[mapping_key]:
            return self.hierarchy_mapping[mapping_key][parent_prediction]
        return []

    def validate_accuracy(self, matched_data: pd.DataFrame, feature_cols: list,
                         target_cols: list, progress_callback=None) -> dict:
        """精度検証を実行（9段階）"""
        try:
            train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            validation_results = {
                'train_ratios': train_ratios,
                'accuracy_scores': {col: [] for col in target_cols},
                'sample_counts': []
            }

            logger.info("精度検証開始...")

            for i, train_ratio in enumerate(train_ratios):
                logger.info(f"検証 {i+1}/9: 学習データ{int(train_ratio*100)}%")

                n_samples = len(matched_data)
                n_train = int(n_samples * train_ratio)

                shuffled_data = matched_data.sample(frac=1, random_state=42+i).reset_index(drop=True)
                train_data = shuffled_data.iloc[:n_train]
                test_data = shuffled_data.iloc[n_train:]

                subsegment_col = 'predicted_subsegment'
                if subsegment_col in train_data.columns:
                    unique_subsegments = train_data[subsegment_col].nunique()
                    if unique_subsegments < 2:
                        logger.warning(f"検証 {i+1}/9 スキップ: サブセグメントが{unique_subsegments}クラスのみ")
                        validation_results['sample_counts'].append({
                            'train': len(train_data),
                            'test': len(test_data)
                        })
                        for target_col in target_cols:
                            validation_results['accuracy_scores'][target_col].append(0.0)
                        continue

                validation_results['sample_counts'].append({
                    'train': len(train_data),
                    'test': len(test_data)
                })

                temp_classifier = ProductClassifierStreamlit(algorithm=self.algorithm)
                temp_classifier.hierarchy_mapping = self.hierarchy_mapping.copy()
                temp_classifier.use_hierarchical = self.use_hierarchical

                try:
                    temp_classifier.train(train_data, feature_cols, target_cols, use_hierarchical=False)
                except Exception as e:
                    logger.error(f"検証 {i+1}/9 訓練エラー: {str(e)}")
                    for target_col in target_cols:
                        validation_results['accuracy_scores'][target_col].append(0.0)
                    continue

                try:
                    predictions = temp_classifier.predict_batch(test_data, feature_cols, target_cols,
                                                              use_hierarchical=self.use_hierarchical,
                                                              batch_size=50)
                except Exception as e:
                    logger.error(f"検証 {i+1}/9 予測エラー: {str(e)}")
                    for target_col in target_cols:
                        validation_results['accuracy_scores'][target_col].append(0.0)
                    continue

                for target_col in target_cols:
                    if target_col in predictions.columns and target_col in test_data.columns:
                        y_true = test_data[target_col]
                        y_pred = predictions[target_col]

                        accuracy = accuracy_score(y_true, y_pred)
                        validation_results['accuracy_scores'][target_col].append(round(accuracy, 4))
                    else:
                        validation_results['accuracy_scores'][target_col].append(0.0)

                logger.info(f"検証 {i+1}/9 完了")

                if progress_callback:
                    progress_callback((i + 1) / len(train_ratios))

            logger.info("精度検証完了")
            return validation_results

        except Exception as e:
            logger.error(f"精度検証エラー: {e}")
            return None
