#!/bin/bash
# 商品自動分類ツール（Streamlit版）起動スクリプト

echo "========================================="
echo "  商品自動分類ツール - Streamlit版"
echo "========================================="
echo ""

# Pythonの確認
if ! command -v python3 &> /dev/null
then
    echo "エラー: Python3が見つかりません"
    exit 1
fi

echo "Python3のバージョン: $(python3 --version)"
echo ""

# 依存関係のチェック
echo "依存関係を確認中..."
python3 -c "import streamlit, pandas, numpy, sklearn, openpyxl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "必要なライブラリがインストールされていません"
    echo "以下のコマンドでインストールしてください:"
    echo "  pip install -r requirements_streamlit.txt"
    echo ""
    read -p "今すぐインストールしますか？ (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        pip install -r requirements_streamlit.txt
    else
        echo "インストールをスキップしました"
        exit 1
    fi
fi

# オプションライブラリのチェック
echo ""
echo "オプションライブラリを確認中..."
python3 -c "import lightgbm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  LightGBMが未インストール（LinearSVCのみ利用可能）"
fi

python3 -c "import shap" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  SHAPが未インストール（SHAP分析機能は利用不可）"
fi

echo ""
echo "========================================="
echo "  Streamlitサーバーを起動します"
echo "  ブラウザで自動的に開きます"
echo "  終了するには Ctrl+C を押してください"
echo "========================================="
echo ""

# Streamlitアプリケーションを起動
streamlit run app_streamlit.py
