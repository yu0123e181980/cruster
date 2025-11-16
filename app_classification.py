#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
商品自動分類ツール - スタンドアロン版
階層型機械学習による商品分類システム
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# メインアプリケーション
app = Flask(__name__)
app.config['SECRET_KEY'] = 'product-classification-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB制限

# メインページ（商品分類ページに直接アクセス）
@app.route('/')
def index():
    """商品分類ページ"""
    return render_template('product_classification.html')

# 商品分類システムAPI
@app.route('/api/product-classification/columns', methods=['POST'])
def product_classification_columns():
    """市場データの列情報を取得"""
    try:
        from product_classification_web import get_file_columns
        return get_file_columns()
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '商品分類モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"列情報取得エラー: {str(e)}")
        return jsonify({'error': True, 'message': '列情報の取得中にエラーが発生しました'}), 500

@app.route('/api/product-classification/process', methods=['POST'])
def product_classification_process():
    """商品分類処理API"""
    try:
        from product_classification_web import process_product_classification
        return process_product_classification()
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '商品分類モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"商品分類処理エラー: {str(e)}")
        return jsonify({'error': True, 'message': '処理中にエラーが発生しました'}), 500

@app.route('/api/product-classification/download/<download_id>')
def product_classification_download(download_id):
    """商品分類結果ダウンロード"""
    try:
        from product_classification_web import download_classification_results
        return download_classification_results(download_id)
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '商品分類モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"ダウンロードエラー: {str(e)}")
        return jsonify({'error': True, 'message': 'ダウンロード中にエラーが発生しました'}), 500

@app.route('/api/product-classification/cleanup/<download_id>', methods=['DELETE'])
def product_classification_cleanup(download_id):
    """商品分類一時ファイルクリーンアップ"""
    try:
        from product_classification_web import cleanup_classification_results
        return cleanup_classification_results(download_id)
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '商品分類モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"クリーンアップエラー: {str(e)}")
        return jsonify({'error': True, 'message': 'クリーンアップ中にエラーが発生しました'}), 500

# ヘルスチェック
@app.route('/api/health')
def health_check():
    """システムヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'service': 'product-classification'
    })

# エラーハンドラー
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'ページが見つかりません'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'サーバー内部エラーが発生しました'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'ファイルサイズが大きすぎます（50MB以下にしてください）'}), 413

# テンプレートフォルダの設定
app.template_folder = 'templates'
app.static_folder = 'static'

# 必要なフォルダ作成
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('temp_uploads', exist_ok=True)
os.makedirs('temp_results', exist_ok=True)

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("商品自動分類ツール 起動中...")
    logger.info("=" * 60)
    logger.info("機能: 階層型機械学習による商品自動分類")
    logger.info("URL: http://localhost:5000")
    logger.info("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
