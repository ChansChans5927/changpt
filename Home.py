import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon=None,
)

# 페이지 선택 섹션
st.markdown("## Select a model")

# HTML 링크를 사용하여 동일한 창에서 페이지로 이동
st.markdown("""
    <a href="/changpt" target="_self" style="text-decoration: none;">
        <p>📈 changpt</p>
    </a>
    <a href="/llama" target="_self" style="text-decoration: none;">
        <p>🦙 llama3</p>
    </a>
    """, unsafe_allow_html=True)
