"""
商品自動分類ツール - Streamlitメインアプリ
"""
import streamlit as st
import pandas as pd
import numpy as np
import uuid
import io
import sys
from pathlib import Path
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import shap

# パスの設定
sys.path.insert(0, str(Path(__file__).parent))

from services.classification_service import ClassificationService
from services.feature_importance import get_feature_importance_data
from services.shap_explainer import get_shap_values, create_shap_waterfall
from utils.file_handler import load_file, detect_column_mapping
from utils.preprocessing import validate_dataframe
from utils.multiprocess_utils import get_optimal_process_count


# ページ設定
st.set_page_config(
    page_title="商品自動分類ツール",
    page_icon="🥫",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """セッション状態の初期化"""
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'process_id' not in st.session_state:
        st.session_state.process_id = None
    if 'classifier_service' not in st.session_state:
        st.session_state.classifier_service = None
    if 'algorithm_type' not in st.session_state:
        st.session_state.algorithm_type = None
    if 'statistics' not in st.session_state:
        st.session_state.statistics = None


def render_sidebar():
    """サイドバーを描画"""
    with st.sidebar:
        st.header("⚙️ 設定")

        # アルゴリズム選択
        algorithm = st.radio(
            "アルゴリズム",
            ["自動選択（推奨）", "LinearSVC", "LightGBM"],
            help="自動選択: 学習データ500件未満→LinearSVC、500件以上→LightGBM"
        )

        st.markdown("---")

        # マルチプロセス設定
        st.subheader("マルチプロセス設定")
        use_multiprocess = st.checkbox("マルチプロセス使用", value=True)

        n_processes = 1
        if use_multiprocess:
            max_processes = cpu_count()
            n_processes = st.slider(
                "プロセス数",
                min_value=1,
                max_value=max_processes,
                value=max(1, max_processes - 1),
                help=f"CPUコア数: {max_processes}"
            )

        st.markdown("---")

        # 出力形式
        st.subheader("出力形式")
        output_format = st.radio(
            "形式",
            ["CSV（推奨）", "Excel"],
            help="大量データの場合はCSVを推奨"
        )

        st.markdown("---")

        # システム情報
        st.subheader("システム情報")
        st.info(f"""
        **CPUコア数**: {cpu_count()}
        **推奨プロセス数**: {get_optimal_process_count()}
        """)

        return algorithm, use_multiprocess, n_processes, output_format


def render_file_upload():
    """ファイルアップロード部分を描画"""
    st.subheader("📁 ステップ1: ファイルアップロード")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**市場データ**")
        market_file = st.file_uploader(
            "分類したい商品データをアップロード",
            type=['csv', 'xlsx', 'xls'],
            help="JAN、商品名などを含むCSV/Excelファイル",
            key="market_file"
        )

    with col2:
        st.markdown("**トライアルマスター**")
        trial_file = st.file_uploader(
            "既に分類済みの教師データをアップロード",
            type=['csv', 'xlsx', 'xls'],
            help="JAN、商品名、カテゴリーなどを含むCSV/Excelファイル",
            key="trial_file"
        )

    return market_file, trial_file


def render_column_mapping(df, data_type):
    """カラムマッピングを描画"""
    st.subheader(f"🔗 ステップ2: カラムマッピング ({data_type})")

    columns = df.columns.tolist()

    # 自動検出
    auto_mapping = detect_column_mapping(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        jan_idx = columns.index(auto_mapping['jan']) if auto_mapping['jan'] in columns else 0
        jan_col = st.selectbox(
            "JAN",
            columns,
            index=jan_idx,
            key=f"{data_type}_jan"
        )

    with col2:
        product_idx = columns.index(auto_mapping['product_name']) if auto_mapping['product_name'] in columns else 0
        product_col = st.selectbox(
            "商品名",
            columns,
            index=product_idx,
            key=f"{data_type}_product"
        )

    with col3:
        standard_options = ['なし'] + columns
        standard_idx = columns.index(auto_mapping['standard']) + 1 if auto_mapping['standard'] in columns else 0
        standard_col = st.selectbox(
            "規格（任意）",
            standard_options,
            index=standard_idx,
            key=f"{data_type}_standard"
        )

    with col4:
        manufacturer_options = ['なし'] + columns
        manufacturer_idx = columns.index(auto_mapping['manufacturer']) + 1 if auto_mapping['manufacturer'] in columns else 0
        manufacturer_col = st.selectbox(
            "メーカー（任意）",
            manufacturer_options,
            index=manufacturer_idx,
            key=f"{data_type}_manufacturer"
        )

    # トライアルマスターの場合、分類カラムも指定
    category_cols = {}
    if data_type == 'trial':
        st.markdown("**分類カラム**")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            category_cols['category'] = st.selectbox(
                "カテゴリー",
                columns,
                key=f"{data_type}_category"
            )

        with col2:
            category_cols['subcategory'] = st.selectbox(
                "サブカテゴリー",
                columns,
                key=f"{data_type}_subcategory"
            )

        with col3:
            category_cols['segment'] = st.selectbox(
                "セグメント",
                columns,
                key=f"{data_type}_segment"
            )

        with col4:
            category_cols['subsegment'] = st.selectbox(
                "サブセグメント",
                columns,
                key=f"{data_type}_subsegment"
            )

    # マッピング辞書を作成
    mapping = {
        'jan': jan_col,
        'product_name': product_col,
        'standard': None if standard_col == 'なし' else standard_col,
        'manufacturer': None if manufacturer_col == 'なし' else manufacturer_col
    }

    if data_type == 'trial':
        mapping.update(category_cols)

    return mapping


def execute_classification(
    market_df,
    trial_df,
    market_mapping,
    trial_mapping,
    algorithm,
    use_multiprocess,
    n_processes
):
    """分類処理を実行"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # サービス初期化
        service = ClassificationService()

        # ステップ1: データ検証
        status_text.info("📋 データ検証中...")
        progress_bar.progress(5)

        market_errors = validate_dataframe(market_df, 'market')
        trial_errors = validate_dataframe(trial_df, 'trial')

        if market_errors:
            for error in market_errors:
                st.error(f"市場データエラー: {error}")
            return False

        if trial_errors:
            for error in trial_errors:
                st.error(f"トライアルマスターエラー: {error}")
            return False

        # ステップ2: データ前処理
        status_text.info("🔄 データ前処理中...")
        progress_bar.progress(10)

        market_processed, trial_processed = service.prepare_data(
            market_df,
            trial_df,
            market_mapping,
            trial_mapping
        )

        # ステップ3: アルゴリズム選択
        status_text.info("🤖 アルゴリズム選択中...")
        progress_bar.progress(15)

        algorithm_type = service.select_algorithm(algorithm, len(trial_processed))
        st.session_state.algorithm_type = algorithm_type

        st.info(f"選択されたアルゴリズム: **{algorithm_type}**")

        # ステップ4: 分類器作成
        status_text.info("🔧 分類器初期化中...")
        progress_bar.progress(20)

        service.create_classifier(algorithm_type)

        # ステップ5: 分類実行
        status_text.info("🚀 分類処理実行中...")
        progress_bar.progress(30)

        def progress_callback(p):
            progress_bar.progress(int(30 + p * 40))

        results = service.classify(
            market_processed,
            trial_processed,
            use_multiprocess=use_multiprocess,
            n_processes=n_processes if use_multiprocess else 1,
            progress_callback=progress_callback
        )

        progress_bar.progress(70)

        # ステップ6: 精度検証
        status_text.info("📊 精度検証中（9段階）...")

        # JAN一致データで検証
        jan_matched = results[results['status'] == 'jan_matched']

        if len(jan_matched) >= 100:
            validation_results = service.validate_accuracy(jan_matched)
            st.session_state.validation_results = validation_results
        else:
            st.warning("精度検証には100件以上のJAN一致データが必要です")
            st.session_state.validation_results = None

        progress_bar.progress(80)

        # ステップ7: 特徴量重要度
        status_text.info("🔍 特徴量重要度計算中...")

        feature_importance = service.get_feature_importance()
        st.session_state.feature_importance = feature_importance

        progress_bar.progress(90)

        # ステップ8: 統計情報
        status_text.info("📈 統計情報計算中...")

        statistics = service.get_statistics(results)
        st.session_state.statistics = statistics

        progress_bar.progress(95)

        # ステップ9: 結果保存
        status_text.info("💾 結果をセッションに保存中...")

        process_id = str(uuid.uuid4())
        st.session_state.results = results
        st.session_state.process_id = process_id
        st.session_state.classifier_service = service

        progress_bar.progress(100)
        status_text.success("✅ 分類完了!")

        return True

    except Exception as e:
        st.error(f"❌ エラーが発生しました: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def render_statistics_tab():
    """統計情報タブを描画"""
    if st.session_state.statistics is None:
        st.warning("統計情報がありません")
        return

    stats = st.session_state.statistics

    st.subheader("📊 分類結果サマリー")

    # メトリクス表示
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("総件数", f"{stats['total']:,}")

    with col2:
        st.metric(
            "JAN一致",
            f"{stats['jan_matched']:,}",
            f"{stats['jan_matched_rate']*100:.1f}%"
        )

    with col3:
        st.metric(
            "高信頼度",
            f"{stats['ml_high']:,}",
            f"{stats['ml_high_rate']*100:.1f}%"
        )

    with col4:
        st.metric(
            "中信頼度",
            f"{stats['ml_medium']:,}",
            f"{stats['ml_medium_rate']*100:.1f}%"
        )

    with col5:
        st.metric(
            "低信頼度",
            f"{stats['ml_low']:,}",
            f"{stats['ml_low_rate']*100:.1f}%"
        )

    # アルゴリズム情報
    if st.session_state.algorithm_type:
        st.info(f"使用アルゴリズム: **{st.session_state.algorithm_type}**")

    # 平均信頼度
    st.metric("平均信頼度", f"{stats['average_confidence']*100:.1f}%")


def render_validation_tab():
    """精度検証タブを描画"""
    if st.session_state.validation_results is None:
        st.warning("精度検証結果がありません")
        return

    validation = st.session_state.validation_results

    st.subheader("📈 学習データ量と精度の関係")

    # DataFrameに変換
    df_validation = pd.DataFrame({
        '学習データ比率': [f"{int(r*100)}%" for r in validation['train_ratios']],
        'カテゴリー': [v*100 for v in validation['accuracy_scores']['predicted_category']],
        'サブカテゴリー': [v*100 for v in validation['accuracy_scores']['predicted_subcategory']],
        'セグメント': [v*100 for v in validation['accuracy_scores']['predicted_segment']],
        'サブセグメント': [v*100 for v in validation['accuracy_scores']['predicted_subsegment']]
    })

    # 折れ線グラフ
    st.line_chart(df_validation.set_index('学習データ比率'))

    # データテーブル
    st.dataframe(df_validation, use_container_width=True)

    # サマリー
    best_idx = len(validation['train_ratios']) - 1
    st.success(f"""
    **最高精度（学習データ{int(validation['train_ratios'][best_idx]*100)}%時）**
    - カテゴリー: {validation['accuracy_scores']['predicted_category'][best_idx]*100:.1f}%
    - サブカテゴリー: {validation['accuracy_scores']['predicted_subcategory'][best_idx]*100:.1f}%
    - セグメント: {validation['accuracy_scores']['predicted_segment'][best_idx]*100:.1f}%
    - サブセグメント: {validation['accuracy_scores']['predicted_subsegment'][best_idx]*100:.1f}%
    """)


def render_feature_importance_tab():
    """特徴量重要度タブを描画"""
    if st.session_state.feature_importance is None:
        st.warning("特徴量重要度データがありません")
        return

    importance = st.session_state.feature_importance

    st.subheader("🔍 特徴量重要度分析")

    # サブタブで各階層を表示
    sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
        "カテゴリー", "サブカテゴリー", "セグメント", "サブセグメント"
    ])

    hierarchies = [
        ('predicted_category', sub_tab1, "カテゴリー"),
        ('predicted_subcategory', sub_tab2, "サブカテゴリー"),
        ('predicted_segment', sub_tab3, "セグメント"),
        ('predicted_subsegment', sub_tab4, "サブセグメント")
    ]

    for hierarchy, tab, name in hierarchies:
        with tab:
            if hierarchy in importance and importance[hierarchy]:
                st.markdown(f"**{name}予測の特徴量重要度**")

                # DataFrameに変換
                df_imp = pd.DataFrame([
                    {'特徴量': k, '重要度': v}
                    for k, v in importance[hierarchy].items()
                ]).sort_values('重要度', ascending=True)  # 横棒グラフ用に昇順

                # 横棒グラフ
                st.bar_chart(df_imp.set_index('特徴量'))

                # データテーブル
                df_imp_sorted = df_imp.sort_values('重要度', ascending=False)
                st.dataframe(df_imp_sorted, use_container_width=True)
            else:
                st.info(f"{name}の特徴量重要度データがありません")


def render_results_tab(output_format):
    """結果一覧タブを描画"""
    if st.session_state.results is None:
        st.warning("分類結果がありません")
        return

    results = st.session_state.results

    st.subheader("📋 分類結果一覧")

    # フィルタ・検索
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        search_keyword = st.text_input("🔍 検索（JAN、商品名）", "")

    with col2:
        status_filter = st.multiselect(
            "ステータス",
            ['jan_matched', 'ml_high', 'ml_medium', 'ml_low'],
            default=['jan_matched', 'ml_high', 'ml_medium', 'ml_low']
        )

    with col3:
        confidence_min = st.slider("最低信頼度", 0.0, 1.0, 0.0, 0.05)

    # フィルタ適用
    filtered = results.copy()

    if search_keyword:
        mask = (
            filtered['jan'].astype(str).str.contains(search_keyword, na=False, case=False) |
            filtered['product_name'].astype(str).str.contains(search_keyword, na=False, case=False)
        )
        filtered = filtered[mask]

    if 'status' in filtered.columns:
        filtered = filtered[filtered['status'].isin(status_filter)]

    if 'confidence' in filtered.columns:
        filtered = filtered[filtered['confidence'] >= confidence_min]

    st.info(f"表示件数: {len(filtered):,} / {len(results):,}")

    # データフレーム表示
    display_columns = [
        'jan', 'product_name', 'predicted_category', 'predicted_subcategory',
        'predicted_segment', 'predicted_subsegment', 'confidence', 'status', 'method'
    ]

    # オプションカラムを追加
    if 'standard' in filtered.columns:
        display_columns.insert(2, 'standard')
    if 'manufacturer' in filtered.columns:
        display_columns.insert(3, 'manufacturer')

    # 存在するカラムのみフィルタ
    display_columns = [col for col in display_columns if col in filtered.columns]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        height=500
    )

    # ダウンロードボタン
    st.markdown("---")
    col1, col2 = st.columns([1, 4])

    with col1:
        if output_format == "CSV（推奨）":
            csv = filtered.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 CSV ダウンロード",
                csv,
                f"classification_results_{st.session_state.process_id}.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            # Excel出力
            if len(filtered) > 100000:
                st.warning("⚠️ 10万行を超えています。CSV形式を推奨します。")

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered.to_excel(writer, index=False, sheet_name='分類結果')

            st.download_button(
                "📥 Excel ダウンロード",
                buffer.getvalue(),
                f"classification_results_{st.session_state.process_id}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # SHAP分析
    render_shap_analysis(filtered)


def render_shap_analysis(filtered_df):
    """SHAP分析セクションを描画"""
    st.markdown("---")
    st.subheader("🔬 個別商品のSHAP分析")

    if st.session_state.classifier_service is None:
        st.warning("分類器が初期化されていません")
        return

    if len(filtered_df) == 0:
        st.info("表示対象の商品がありません")
        return

    # 機械学習予測の商品のみ対象
    ml_products = filtered_df[filtered_df['status'] != 'jan_matched']

    if len(ml_products) == 0:
        st.info("機械学習予測の商品がありません（JAN一致のみ）")
        return

    # 商品選択
    selected_idx = st.selectbox(
        "商品を選択",
        range(len(ml_products)),
        format_func=lambda i: f"{ml_products.iloc[i]['product_name']} (JAN: {ml_products.iloc[i]['jan']})"
    )

    selected_product = ml_products.iloc[selected_idx]

    # 商品情報表示
    st.info(f"""
    **商品**: {selected_product['product_name']}
    **JAN**: {selected_product['jan']}
    **予測カテゴリー**: {selected_product['predicted_category']}
    **信頼度**: {selected_product['confidence']*100:.1f}%
    """)

    # 階層選択
    hierarchy = st.selectbox(
        "分析対象階層",
        [
            "predicted_category",
            "predicted_subcategory",
            "predicted_segment",
            "predicted_subsegment"
        ],
        format_func=lambda x: {
            "predicted_category": "カテゴリー",
            "predicted_subcategory": "サブカテゴリー",
            "predicted_segment": "セグメント",
            "predicted_subsegment": "サブセグメント"
        }[x]
    )

    # SHAP実行ボタン
    if st.button("SHAP分析実行"):
        with st.spinner("SHAP値計算中..."):
            try:
                classifier = st.session_state.classifier_service.classifier
                shap_values = get_shap_values(classifier, selected_product, hierarchy)

                if shap_values is not None:
                    # SHAP Waterfall Plot
                    fig = create_shap_waterfall(shap_values, max_display=15)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("SHAP値の計算に失敗しました")

            except Exception as e:
                st.error(f"SHAP分析エラー: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def main():
    """メイン関数"""
    # セッション状態の初期化
    init_session_state()

    # タイトル
    st.title("🥫 商品自動分類ツール")
    st.markdown("---")

    # サイドバー
    algorithm, use_multiprocess, n_processes, output_format = render_sidebar()

    # ファイルアップロード
    market_file, trial_file = render_file_upload()

    # 両方のファイルがアップロードされた場合
    if market_file and trial_file:
        try:
            # ファイル読み込み
            with st.spinner("ファイル読み込み中..."):
                market_df = load_file(market_file)
                trial_df = load_file(trial_file)

            st.success(f"✅ 市場データ: {len(market_df):,}行、トライアルマスター: {len(trial_df):,}行")

            # カラムマッピング
            st.markdown("---")

            with st.expander("市場データのカラムマッピング", expanded=True):
                market_mapping = render_column_mapping(market_df, 'market')

            with st.expander("トライアルマスターのカラムマッピング", expanded=True):
                trial_mapping = render_column_mapping(trial_df, 'trial')

            # 処理実行ボタン
            st.markdown("---")
            st.subheader("🚀 ステップ3: 分類実行")

            if st.button("分類開始", type="primary", use_container_width=False):
                success = execute_classification(
                    market_df,
                    trial_df,
                    market_mapping,
                    trial_mapping,
                    algorithm,
                    use_multiprocess,
                    n_processes
                )

                if success:
                    st.balloons()

        except Exception as e:
            st.error(f"エラー: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # 結果表示
    if st.session_state.results is not None:
        st.markdown("---")
        st.header("📊 分類結果")

        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 統計情報",
            "📈 精度検証",
            "🔍 特徴量重要度",
            "📋 結果一覧"
        ])

        with tab1:
            render_statistics_tab()

        with tab2:
            render_validation_tab()

        with tab3:
            render_feature_importance_tab()

        with tab4:
            render_results_tab(output_format)


if __name__ == "__main__":
    main()
