from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# メインアプリケーション
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB制限

# メインメニューページ
@app.route('/')
def index():
    """メインメニューページ"""
    return render_template('index.html')

# 機能1: 棚割マスタ処理ページ
@app.route('/feature/1')
def shelf_allocation_page():
    """棚割マスタ処理ページ"""
    return render_template('shelf_allocation.html')

# 機能2: 商品分類システムページ
@app.route('/feature/2')
def product_classification_page():
    """商品分類システムページ"""
    return render_template('product_classification.html')

# 棚割処理API
@app.route('/api/shelf-allocation/process', methods=['POST'])
def shelf_allocation_process():
    """棚割処理API"""
    try:
        from shelf_allocation import process_shelf_allocation
        return process_shelf_allocation()
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '棚割処理モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"棚割処理エラー: {str(e)}")
        return jsonify({'error': True, 'message': '処理中にエラーが発生しました'}), 500

@app.route('/api/shelf-allocation/download/<download_id>')
def shelf_allocation_download(download_id):
    """棚割結果ダウンロード"""
    try:
        from shelf_allocation import download_results
        return download_results(download_id)
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '棚割処理モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"ダウンロードエラー: {str(e)}")
        return jsonify({'error': True, 'message': 'ダウンロード中にエラーが発生しました'}), 500

@app.route('/api/shelf-allocation/cleanup/<download_id>', methods=['DELETE'])
def shelf_allocation_cleanup(download_id):
    """一時ファイルクリーンアップ"""
    try:
        from shelf_allocation import cleanup_results
        return cleanup_results(download_id)
    except ImportError as e:
        logger.error(f"モジュールインポートエラー: {str(e)}")
        return jsonify({'error': True, 'message': '棚割処理モジュールが見つかりません'}), 500
    except Exception as e:
        logger.error(f"クリーンアップエラー: {str(e)}")
        return jsonify({'error': True, 'message': 'クリーンアップ中にエラーが発生しました'}), 500

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

# その他の機能（未実装）- index.htmlからのリンクのみ対応
@app.route('/feature/<int:feature_id>')
def feature_page(feature_id):
    """その他の機能ページ（未実装）"""
    if feature_id == 1:
        return redirect(url_for('shelf_allocation_page'))
    elif feature_id == 2:
        return redirect(url_for('product_classification_page'))
    else:
        # 未実装の機能はメインメニューにリダイレクト
        return redirect(url_for('index'))

# ヘルスチェック
@app.route('/api/health')
def health_check():
    """システムヘルスチェック"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'features': {
            'shelf_allocation': 'active',
            'product_classification': 'active',
            'other_features': 'development'
        }
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
os.makedirs('uploads', exist_ok=True)
os.makedirs('temp_uploads', exist_ok=True)
os.makedirs('temp_results', exist_ok=True)

if __name__ == '__main__':
    logger.info("カテゴリーマネジメントツール起動中...")
    logger.info("利用可能な機能:")
    logger.info("1. 棚割マスタ処理 - /feature/1 [実装済み]")
    logger.info("2. 商品自動分類 - /feature/2 [実装済み]")
    logger.info("3-8. その他の機能 [開発中]")
    
    app.run(debug=True, host='0.0.0.0', port=5000)