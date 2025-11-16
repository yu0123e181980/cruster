#!/bin/bash
# 商品自動分類ツール起動スクリプト

echo "========================================="
echo "  商品自動分類ツール"
echo "========================================="
echo ""

# 必要なフォルダを作成
echo "必要なフォルダを作成中..."
mkdir -p templates temp_uploads temp_results

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
python3 -c "import flask, pandas, numpy, sklearn, openpyxl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "必要なライブラリがインストールされていません"
    echo "以下のコマンドでインストールしてください:"
    echo "  pip install -r requirements_classification.txt"
    echo ""
    read -p "今すぐインストールしますか？ (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        pip install -r requirements_classification.txt
    else
        echo "インストールをスキップしました"
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "  サーバーを起動します"
echo "  URL: http://localhost:5000"
echo "  終了するには Ctrl+C を押してください"
echo "========================================="
echo ""

# アプリケーションを起動
python3 app_classification.py
