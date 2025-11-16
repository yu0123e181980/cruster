import streamlit as st

st.set_page_config(page_title="テスト", layout="wide")

st.title("🥫 商品自動分類ツール - テスト")
st.write("このメッセージが見えたら、Streamlitは正常に動作しています。")

if st.button("クリックしてテスト"):
    st.success("ボタンが動作しました！")
    st.balloons()
