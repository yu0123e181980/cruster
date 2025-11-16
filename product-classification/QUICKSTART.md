# クイックスタートガイド

## 1. セットアップ確認

```bash
python check_setup.py
```

すべて `✓` が表示されればOKです。

## 2. アプリ起動

```bash
streamlit run app.py
```

## 3. 起動確認

ターミナルに以下のようなメッセージが表示されます：

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

## 4. ブラウザでアクセス

自動的にブラウザが開かない場合は、手動で以下のURLにアクセスしてください：

```
http://localhost:8501
```

## トラブルシューティング

### 画面が真っ白の場合

#### 1. ブラウザのキャッシュをクリア

**Chrome:**
- Ctrl + Shift + Delete (Windows/Linux)
- Cmd + Shift + Delete (Mac)
- 「キャッシュされた画像とファイル」にチェック
- 「データを削除」をクリック

**Firefox:**
- Ctrl + Shift + Delete (Windows/Linux)
- Cmd + Shift + Delete (Mac)
- 「キャッシュ」にチェック
- 「今すぐ消去」をクリック

#### 2. シークレットモード/プライベートブラウジングで開く

- Chrome: Ctrl + Shift + N (Windows/Linux) / Cmd + Shift + N (Mac)
- Firefox: Ctrl + Shift + P (Windows/Linux) / Cmd + Shift + P (Mac)

そして `http://localhost:8501` にアクセス

#### 3. ポート番号を変更して起動

```bash
streamlit run app.py --server.port 8502
```

ブラウザで `http://localhost:8502` にアクセス

#### 4. ターミナルのエラーメッセージを確認

Streamlitを起動したターミナルに赤いエラーメッセージが表示されていないか確認してください。

エラーがある場合は、そのメッセージを報告してください。

#### 5. プロセスを完全に停止して再起動

```bash
# Ctrl+C で停止
# もし停止しない場合:
pkill -f streamlit

# 再起動
streamlit run app.py
```

### ポート8501が既に使用されている場合

```bash
# 使用中のポートを確認
lsof -i :8501

# プロセスを停止
kill -9 <PID>

# または別のポートで起動
streamlit run app.py --server.port 8502
```

### 依存関係のエラーが出る場合

```bash
# 仮想環境を再作成
deactivate  # 既存の仮想環境から退出
rm -rf venv  # 既存の仮想環境を削除
python3 -m venv venv  # 新しい仮想環境を作成
source venv/bin/activate  # 仮想環境をアクティベート
pip install -r requirements.txt  # 依存関係を再インストール
```

## 正常に起動した場合の画面

以下の要素が表示されていればOKです：

- 🥫 商品自動分類ツール（タイトル）
- 左側にサイドバー（⚙️ 設定）
- ステップ1: ファイルアップロード
- 市場データとトライアルマスターのアップロードボタン

## サンプルデータでテスト

### 市場データ (market_sample.csv)

```csv
JAN,商品名,規格,メーカー
4902102072557,コカ・コーラ,500ml,コカ・コーラ
4902430569514,ペプシコーラ,500ml,サントリー
```

### トライアルマスター (trial_sample.csv)

```csv
JAN,商品名,規格,メーカー,category,subcategory,segment,subsegment
4902102072557,コカ・コーラ,500ml,コカ・コーラ,飲料,炭酸飲料,コーラ,レギュラーコーラ
4902430569514,ペプシコーラ,500ml,サントリー,飲料,炭酸飲料,コーラ,レギュラーコーラ
```

## 問い合わせ

問題が解決しない場合は、以下の情報を含めて報告してください：

1. `python check_setup.py` の出力
2. ターミナルのエラーメッセージ（あれば）
3. ブラウザのコンソールエラー（F12 → Console）
4. OS とPythonバージョン
5. 起動コマンド
