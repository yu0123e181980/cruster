#!/usr/bin/env python
"""
セットアップ確認スクリプト
"""
import sys

print("=" * 60)
print("商品自動分類ツール - セットアップ確認")
print("=" * 60)
print()

# Pythonバージョン確認
print(f"✓ Python バージョン: {sys.version}")
print()

# 必要なモジュールの確認
required_modules = [
    ('streamlit', 'Streamlit'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('sklearn', 'scikit-learn'),
    ('lightgbm', 'LightGBM'),
    ('shap', 'SHAP'),
    ('openpyxl', 'openpyxl'),
    ('chardet', 'chardet'),
    ('matplotlib', 'matplotlib'),
]

print("依存関係チェック:")
print("-" * 60)

all_ok = True
for module_name, display_name in required_modules:
    try:
        __import__(module_name)
        print(f"✓ {display_name:20s} ... OK")
    except ImportError:
        print(f"✗ {display_name:20s} ... NG (インストールされていません)")
        all_ok = False

print()

# カスタムモジュールの確認
print("カスタムモジュールチェック:")
print("-" * 60)

custom_modules = [
    'classifier',
    'services.classification_service',
    'services.feature_importance',
    'services.shap_explainer',
    'utils.file_handler',
    'utils.preprocessing',
    'utils.multiprocess_utils',
]

for module_name in custom_modules:
    try:
        __import__(module_name)
        print(f"✓ {module_name:35s} ... OK")
    except ImportError as e:
        print(f"✗ {module_name:35s} ... NG ({e})")
        all_ok = False

print()
print("=" * 60)

if all_ok:
    print("✓ すべての確認が完了しました！")
    print()
    print("次のコマンドでアプリを起動してください:")
    print("  streamlit run app.py")
else:
    print("✗ いくつかの問題が見つかりました。")
    print()
    print("依存関係をインストールしてください:")
    print("  pip install -r requirements.txt")

print("=" * 60)
