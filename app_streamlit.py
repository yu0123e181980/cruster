#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品自動分類ツール - Streamlit版
階層型機械学習 + 特徴量重要度 + SHAP分析 + 大量データ対応
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import io
import os
from product_classification_streamlit import (
    ProductClassifierStreamlit,
    normalize_jan,
    LIGHTGBM_AVAILABLE
)

# SHAP import (optional)
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ページ設定
st.set_page_config(
    page_title="商品自動分類ツール",
    page_icon="🥫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# セッション状態の初期化
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = None
if 'market_data' not in st.session_state:
    st.session_state.market_data = None
if 'trial_data' not in st.session_state:
    st.session_state.trial_data = None
if 'algorithm' not in st.session_state:
    st.session_state.algorithm = 'auto'


def load_file(uploaded_file) -> pd.DataFrame:
    """アップロードされたファイルを読み込む"""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='shift_jis')
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return None


def process_classification(market_df: pd.DataFrame, trial_df: pd.DataFrame,
                          column_mapping: dict, algorithm: str,
                          use_multiprocessing: bool = False,
                          n_processes: int = 4) -> tuple:
    """商品分類処理を実行"""
    try:
        # カラムマッピング適用
        market_df = market_df.rename(columns={
            column_mapping['jan_column']: 'jan',
            column_mapping['product_column']: 'product_name'
        })

        for key, col in column_mapping.items():
            if key not in ['jan_column', 'product_column'] and col:
                market_df[key.replace('_column', '')] = market_df.get(col, '')

        # JANコード正規化
        market_df['jan'] = market_df['jan'].apply(normalize_jan)
        trial_df['JAN'] = trial_df['JAN'].apply(normalize_jan)

        # 分類器初期化
        classifier = ProductClassifierStreamlit(algorithm=algorithm)

        # JAN辞書作成
        classifier.create_jan_dict(trial_df)

        # 階層マッピング作成
        if len(trial_df) > 0:
            classifier.create_hierarchy_mapping(trial_df)

        # JAN照合
        jan_matched_list = []
        jan_unmatched_list = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in market_df.iterrows():
            jan = normalize_jan(row['jan'])

            if jan in classifier.jan_dict:
                matched_data = classifier.jan_dict[jan]
                result_row = row.to_dict()
                result_row.update({
                    'predicted_category': matched_data['カテゴリー名'],
                    'predicted_subcategory': matched_data['サブカテゴリー名'],
                    'predicted_segment': matched_data['セグメント名'],
                    'predicted_subsegment': matched_data['サブセグメント名'],
                    'confidence': 1.0,
                    'status': 'jan_matched',
                    'method': 'JAN照合'
                })
                jan_matched_list.append(result_row)
            else:
                jan_unmatched_list.append(row)

            # 進捗表示
            if idx % 100 == 0:
                progress = (idx + 1) / len(market_df)
                progress_bar.progress(progress)
                status_text.text(f"JAN照合中: {idx + 1}/{len(market_df)}")

        progress_bar.progress(1.0)
        status_text.text(f"JAN照合完了: {len(jan_matched_list)}件一致")

        logger.info(f"JAN照合完了: {len(jan_matched_list)}件一致")

        # 精度検証実行
        validation_results = None
        if len(jan_matched_list) > 50:
            st.info("精度検証を実行中...")
            matched_df = pd.DataFrame(jan_matched_list)

            available_features = []
            base_features = ['product_name', 'standard', 'manufacturer', 'container',
                           'avg_price', 'jan_maker_code', 'existing_category',
                           'existing_subcategory', 'existing_segment', 'existing_subsegment']

            for col in base_features:
                if col in matched_df.columns and matched_df[col].notna().any() and (matched_df[col] != '').any():
                    available_features.append(col)

            if not available_features:
                available_features = ['product_name']

            target_cols = ['predicted_category', 'predicted_subcategory',
                         'predicted_segment', 'predicted_subsegment']

            validation_progress_bar = st.progress(0)
            validation_status = st.empty()

            def validation_callback(progress):
                validation_progress_bar.progress(progress)
                validation_status.text(f"精度検証進捗: {int(progress * 100)}%")

            validation_results = classifier.validate_accuracy(
                matched_df, available_features, target_cols,
                progress_callback=validation_callback
            )

            validation_progress_bar.empty()
            validation_status.empty()

        # 機械学習による予測
        ml_results_list = []

        if len(jan_unmatched_list) > 0 and len(jan_matched_list) > 0:
            st.info(f"機械学習モデル訓練中（{algorithm.upper()}）...")

            train_df = pd.DataFrame(jan_matched_list)

            available_features = []
            base_features = ['product_name', 'standard', 'manufacturer', 'container',
                           'avg_price', 'jan_maker_code', 'existing_category',
                           'existing_subcategory', 'existing_segment', 'existing_subsegment']

            for col in base_features:
                if col in train_df.columns and train_df[col].notna().any() and (train_df[col] != '').any():
                    available_features.append(col)

            if not available_features:
                available_features = ['product_name']

            target_cols = ['predicted_category', 'predicted_subcategory',
                         'predicted_segment', 'predicted_subsegment']

            # モデル訓練
            classifier.train(train_df, available_features, target_cols, use_hierarchical=True)

            # 使用されたアルゴリズムを表示
            if classifier.models:
                algorithm_used = list(classifier.models.values())[0].name
                st.success(f"モデル訓練完了: {algorithm_used}を使用")

            # 予測実行
            st.info(f"予測実行中: {len(jan_unmatched_list)}件...")

            unmatched_df = pd.DataFrame(jan_unmatched_list)

            for col in available_features:
                if col not in unmatched_df.columns:
                    unmatched_df[col] = ''

            ml_progress_bar = st.progress(0)
            ml_status = st.empty()

            def ml_callback(progress):
                ml_progress_bar.progress(progress)
                ml_status.text(f"予測進捗: {int(progress * 100)}%")

            predictions_df = classifier.predict_batch(
                unmatched_df, available_features, target_cols,
                use_hierarchical=True, batch_size=1000,
                progress_callback=ml_callback
            )

            ml_progress_bar.empty()
            ml_status.empty()

            # ステータス設定
            for _, row in predictions_df.iterrows():
                confidence = row['confidence']
                if confidence >= 0.7:
                    status = 'ml_high'
                elif confidence >= 0.4:
                    status = 'ml_medium'
                else:
                    status = 'ml_low'

                result_row = row.to_dict()
                result_row.update({
                    'status': status,
                    'method': f'{algorithm_used}予測'
                })
                ml_results_list.append(result_row)

            logger.info(f"ML予測完了: {len(ml_results_list)}件")

        elif len(jan_unmatched_list) > 0:
            st.warning("警告: JAN一致データがないためMLスキップ")
            for row in jan_unmatched_list:
                result_row = row.to_dict()
                result_row.update({
                    'predicted_category': '不明',
                    'predicted_subcategory': '不明',
                    'predicted_segment': '不明',
                    'predicted_subsegment': '不明',
                    'confidence': 0.0,
                    'status': 'ml_low',
                    'method': 'データ不足'
                })
                ml_results_list.append(result_row)

        # 結果統合
        all_results = jan_matched_list + ml_results_list

        if not all_results:
            st.error('処理結果が空です')
            return None, None, None

        results_df = pd.DataFrame(all_results)

        return results_df, classifier, validation_results

    except Exception as e:
        st.error(f"処理エラー: {e}")
        logger.error(f"処理エラー: {e}", exc_info=True)
        return None, None, None


def display_feature_importance(classifier: ProductClassifierStreamlit):
    """特徴量重要度を表示"""
    st.subheader("📊 特徴量重要度分析")

    if not classifier or not classifier.is_trained:
        st.warning("モデルが訓練されていません")
        return

    target_cols = ['predicted_category', 'predicted_subcategory',
                  'predicted_segment', 'predicted_subsegment']

    # タブで階層ごとに表示
    tabs = st.tabs(["カテゴリー", "サブカテゴリー", "セグメント", "サブセグメント"])

    for idx, (tab, target_col) in enumerate(zip(tabs, target_cols)):
        with tab:
            importance_df = classifier.get_feature_importance(target_col, top_n=20)

            if importance_df.empty:
                st.info("特徴量重要度が利用できません")
                continue

            # 横棒グラフ
            st.bar_chart(
                importance_df.set_index('特徴量')['重要度'],
                horizontal=True
            )

            # テーブル表示
            st.dataframe(
                importance_df,
                use_container_width=True,
                hide_index=True
            )


def display_shap_analysis(classifier: ProductClassifierStreamlit, results_df: pd.DataFrame):
    """SHAP値による個別分析"""
    st.subheader("🔍 SHAP値による個別商品分析")

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed. Install with: pip install shap")
        return

    if not classifier or not classifier.is_trained:
        st.warning("モデルが訓練されていません")
        return

    if results_df is None or len(results_df) == 0:
        st.warning("分類結果がありません")
        return

    # 商品選択
    product_options = results_df[['jan', 'product_name']].head(100).apply(
        lambda x: f"{x['jan']} - {x['product_name']}", axis=1
    ).tolist()

    selected_product_str = st.selectbox(
        "分析する商品を選択してください（上位100件）:",
        product_options
    )

    if st.button("SHAP分析を実行"):
        try:
            # 選択された商品のインデックスを取得
            selected_idx = product_options.index(selected_product_str)
            selected_row = results_df.iloc[selected_idx]

            st.info("SHAP値を計算中...")

            # TODO: SHAP分析の実装
            # 現在は未実装のプレースホルダー
            st.warning("SHAP分析機能は開発中です")

        except Exception as e:
            st.error(f"SHAP分析エラー: {e}")


def display_validation_results(validation_results: dict):
    """精度検証結果を表示"""
    st.subheader("📈 予測精度検証結果")

    if not validation_results:
        st.info("精度検証結果がありません")
        return

    train_ratios = validation_results['train_ratios']
    accuracy_scores = validation_results['accuracy_scores']
    sample_counts = validation_results['sample_counts']

    # グラフデータ作成
    chart_data = pd.DataFrame({
        '学習データ割合': [f"{int(r*100)}%" for r in train_ratios],
        'カテゴリー': [s for s in accuracy_scores.get('predicted_category', [])],
        'サブカテゴリー': [s for s in accuracy_scores.get('predicted_subcategory', [])],
        'セグメント': [s for s in accuracy_scores.get('predicted_segment', [])],
        'サブセグメント': [s for s in accuracy_scores.get('predicted_subsegment', [])]
    })

    # 折れ線グラフ
    st.line_chart(
        chart_data.set_index('学習データ割合'),
        use_container_width=True
    )

    # サマリー表示
    best_idx = len(train_ratios) - 1
    best_ratio = train_ratios[best_idx]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "カテゴリー精度",
            f"{accuracy_scores['predicted_category'][best_idx]*100:.1f}%"
        )

    with col2:
        st.metric(
            "サブカテゴリー精度",
            f"{accuracy_scores['predicted_subcategory'][best_idx]*100:.1f}%"
        )

    with col3:
        st.metric(
            "セグメント精度",
            f"{accuracy_scores['predicted_segment'][best_idx]*100:.1f}%"
        )

    with col4:
        st.metric(
            "サブセグメント精度",
            f"{accuracy_scores['predicted_subsegment'][best_idx]*100:.1f}%"
        )

    # 詳細テーブル
    with st.expander("詳細データを表示"):
        detail_df = pd.DataFrame({
            '学習データ割合': [f"{int(r*100)}%" for r in train_ratios],
            '学習件数': [sc['train'] for sc in sample_counts],
            '検証件数': [sc['test'] for sc in sample_counts],
            'カテゴリー': [f"{s*100:.1f}%" for s in accuracy_scores['predicted_category']],
            'サブカテゴリー': [f"{s*100:.1f}%" for s in accuracy_scores['predicted_subcategory']],
            'セグメント': [f"{s*100:.1f}%" for s in accuracy_scores['predicted_segment']],
            'サブセグメント': [f"{s*100:.1f}%" for s in accuracy_scores['predicted_subsegment']]
        })

        st.dataframe(detail_df, use_container_width=True, hide_index=True)


def main():
    """メイン関数"""

    # タイトル
    st.title("🥫 商品自動分類ツール")
    st.markdown("階層型機械学習による商品自動分類システム")

    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")

        # アルゴリズム選択
        st.subheader("アルゴリズム")
        algorithm_options = {
            '自動選択（推奨）': 'auto',
            'LinearSVC': 'linearsvc',
            'LightGBM': 'lightgbm'
        }

        if not LIGHTGBM_AVAILABLE:
            algorithm_options.pop('LightGBM')
            st.warning("LightGBMが未インストール")

        selected_algorithm_label = st.selectbox(
            "使用アルゴリズム",
            list(algorithm_options.keys()),
            help="自動選択: 500件未満はLinearSVC、500件以上はLightGBM"
        )

        st.session_state.algorithm = algorithm_options[selected_algorithm_label]

        st.info(f"""
        **アルゴリズム選択ルール:**
        - LinearSVC: < 500件
        - LightGBM: ≥ 500件
        """)

        # マルチプロセス設定
        st.subheader("並列処理")
        use_multiprocessing = st.checkbox(
            "マルチプロセス予測を有効化",
            value=False,
            help="大量データ処理時に高速化（実験的機能）"
        )

        if use_multiprocessing:
            import multiprocessing
            max_cores = multiprocessing.cpu_count()
            n_processes = st.slider(
                "プロセス数",
                min_value=1,
                max_value=max_cores,
                value=max(1, max_cores - 1),
                help=f"利用可能なCPUコア数: {max_cores}"
            )
        else:
            n_processes = 1

        st.divider()

        # バージョン情報
        st.caption("Version 2.0.0 (Streamlit)")

    # メインコンテンツ
    tab1, tab2, tab3, tab4 = st.tabs([
        "📁 データアップロード",
        "📊 分類結果",
        "🔍 分析",
        "📈 精度検証"
    ])

    # タブ1: データアップロード
    with tab1:
        st.header("データアップロード")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("1️⃣ 市場データ")
            market_file = st.file_uploader(
                "市場データをアップロード",
                type=['xlsx', 'xls', 'csv'],
                key='market_file'
            )

            if market_file:
                st.session_state.market_data = load_file(market_file)

                if st.session_state.market_data is not None:
                    st.success(f"✓ {len(st.session_state.market_data)}件のデータを読み込みました")

                    # カラム選択
                    st.subheader("カラムマッピング")

                    columns = list(st.session_state.market_data.columns)

                    jan_col = st.selectbox("JAN（必須）", columns, key='jan_col')
                    product_col = st.selectbox("商品名（必須）", columns, key='product_col')

                    with st.expander("任意項目"):
                        standard_col = st.selectbox("規格", [''] + columns, key='standard_col')
                        manufacturer_col = st.selectbox("メーカー", [''] + columns, key='manufacturer_col')
                        container_col = st.selectbox("容器", [''] + columns, key='container_col')
                        avg_price_col = st.selectbox("平均価格", [''] + columns, key='avg_price_col')
                        category_col = st.selectbox("大分類", [''] + columns, key='category_col')
                        subcategory_col = st.selectbox("中分類", [''] + columns, key='subcategory_col')
                        segment_col = st.selectbox("小分類", [''] + columns, key='segment_col')
                        subsegment_col = st.selectbox("細分類", [''] + columns, key='subsegment_col')

        with col2:
            st.subheader("2️⃣ トライアルマスター")
            trial_file = st.file_uploader(
                "トライアルマスターをアップロード",
                type=['xlsx', 'xls', 'csv'],
                key='trial_file'
            )

            if trial_file:
                st.session_state.trial_data = load_file(trial_file)

                if st.session_state.trial_data is not None:
                    st.success(f"✓ {len(st.session_state.trial_data)}件のデータを読み込みました")

                    # 必須カラムチェック
                    required_cols = ['JAN', '商品名', '規格', 'メーカー名',
                                   'カテゴリー名', 'サブカテゴリー名', 'セグメント名', 'サブセグメント名']

                    missing_cols = [col for col in required_cols
                                  if col not in st.session_state.trial_data.columns]

                    if missing_cols:
                        st.error(f"必須カラムが不足: {', '.join(missing_cols)}")
                    else:
                        st.success("✓ すべての必須カラムが揃っています")

        # 分類実行ボタン
        if st.session_state.market_data is not None and st.session_state.trial_data is not None:
            st.divider()

            if st.button("🚀 分類を実行", type="primary", use_container_width=True):
                column_mapping = {
                    'jan_column': jan_col,
                    'product_column': product_col,
                    'standard_column': standard_col if standard_col else None,
                    'manufacturer_column': manufacturer_col if manufacturer_col else None,
                    'container_column': container_col if container_col else None,
                    'avg_price_column': avg_price_col if avg_price_col else None,
                    'category_column': category_col if category_col else None,
                    'subcategory_column': subcategory_col if subcategory_col else None,
                    'segment_column': segment_col if segment_col else None,
                    'subsegment_column': subsegment_col if subsegment_col else None
                }

                with st.spinner('分類処理を実行中...'):
                    results_df, classifier, validation_results = process_classification(
                        st.session_state.market_data.copy(),
                        st.session_state.trial_data.copy(),
                        column_mapping,
                        st.session_state.algorithm,
                        use_multiprocessing,
                        n_processes
                    )

                    if results_df is not None:
                        st.session_state.results_df = results_df
                        st.session_state.classifier = classifier
                        st.session_state.validation_results = validation_results

                        st.success("✓ 分類処理が完了しました！")
                        st.balloons()

    # タブ2: 分類結果
    with tab2:
        st.header("分類結果")

        if st.session_state.results_df is not None:
            results_df = st.session_state.results_df

            # 統計情報
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("総件数", len(results_df))

            with col2:
                jan_matched = len(results_df[results_df['status'] == 'jan_matched'])
                st.metric("JAN一致", jan_matched)

            with col3:
                ml_high = len(results_df[results_df['status'] == 'ml_high'])
                st.metric("高信頼度", ml_high)

            with col4:
                ml_medium = len(results_df[results_df['status'] == 'ml_medium'])
                st.metric("中信頼度", ml_medium)

            with col5:
                ml_low = len(results_df[results_df['status'] == 'ml_low'])
                st.metric("低信頼度", ml_low)

            st.divider()

            # フィルター機能
            col_filter1, col_filter2, col_filter3 = st.columns(3)

            with col_filter1:
                search_term = st.text_input("🔍 検索（JAN・商品名）", "")

            with col_filter2:
                status_filter = st.multiselect(
                    "ステータス",
                    ['jan_matched', 'ml_high', 'ml_medium', 'ml_low'],
                    default=['jan_matched', 'ml_high', 'ml_medium', 'ml_low']
                )

            with col_filter3:
                confidence_min = st.slider(
                    "最低信頼度",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.1
                )

            # フィルター適用
            filtered_df = results_df.copy()

            if search_term:
                filtered_df = filtered_df[
                    filtered_df['jan'].str.contains(search_term, case=False, na=False) |
                    filtered_df['product_name'].str.contains(search_term, case=False, na=False)
                ]

            if status_filter:
                filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]

            filtered_df = filtered_df[filtered_df['confidence'] >= confidence_min]

            st.info(f"表示件数: {len(filtered_df)} / {len(results_df)}")

            # 結果テーブル
            display_cols = ['jan', 'product_name', 'predicted_category',
                          'predicted_subcategory', 'predicted_segment',
                          'predicted_subsegment', 'confidence', 'status', 'method']

            available_cols = [col for col in display_cols if col in filtered_df.columns]

            st.dataframe(
                filtered_df[available_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "信頼度",
                        help="予測の信頼度",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                }
            )

            # ダウンロードボタン
            st.divider()

            # Excel出力
            output = io.BytesIO()

            if len(filtered_df) > 100000:
                st.warning("⚠️ データが10万行を超えています。CSV形式での出力を推奨します。")

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df[available_cols].to_excel(writer, index=False, sheet_name='分類結果')

            output.seek(0)

            st.download_button(
                label="📥 結果をダウンロード（Excel）",
                data=output,
                file_name=f"商品分類結果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        else:
            st.info("分類結果がありません。データアップロードタブから処理を実行してください。")

    # タブ3: 分析
    with tab3:
        st.header("詳細分析")

        if st.session_state.classifier is not None:
            # 特徴量重要度
            display_feature_importance(st.session_state.classifier)

            st.divider()

            # SHAP分析
            if st.session_state.results_df is not None:
                display_shap_analysis(st.session_state.classifier, st.session_state.results_df)
        else:
            st.info("分析結果がありません。まずデータをアップロードして分類を実行してください。")

    # タブ4: 精度検証
    with tab4:
        st.header("精度検証")

        if st.session_state.validation_results is not None:
            display_validation_results(st.session_state.validation_results)
        else:
            st.info("精度検証結果がありません。分類処理を実行すると自動的に表示されます。")


if __name__ == '__main__':
    main()
