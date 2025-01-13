import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

# Streamlit 페이지 설정
st.set_page_config(page_title="Financial Chart", layout="wide")

# Sidebar Enhancements
st.sidebar.title("📊 Financial Chart")
st.sidebar.write("Explore financial data of major indices.")

# 지수 선택 드롭다운 메뉴
index_choice = st.sidebar.selectbox("Select Index:", ["S&P 500", "Nasdaq", "Dow Jones"])
ticker_mapping = {"S&P 500": "^GSPC", "Nasdaq": "^IXIC", "Dow Jones": "^DJI"}
ticker = ticker_mapping[index_choice]

# 주식 데이터 가져오는 함수
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1mo', interval='1d')
    return data

# 선택한 지수의 데이터 가져오기
data = fetch_stock_data(ticker)

# 최신 지수 값 및 전일 대비 변동 계산
latest_index_value = data['Close'].iloc[-1]
previous_index_value = data['Close'].iloc[-2]
absolute_change = latest_index_value - previous_index_value
percentage_change = (absolute_change / previous_index_value) * 100

# 사이드바에 변동 정보 표시
st.sidebar.metric(
    label="📊 Current Index Value",
    value=f"{latest_index_value:,.2f}",
    delta=f"{absolute_change:,.2f} ({percentage_change:.2f}%)"
)

# 데이터의 마지막 업데이트 날짜 표시
last_date = data.index[-1].strftime('%Y-%m-%d')
st.sidebar.write(f"📅 Last Updated: **{last_date}**")

# Matplotlib 차트 스타일링 및 시각화 함수
def plot_stock_data(data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#2E2E2E')
    ax.set_facecolor('#1E1E1E')
    ax.plot(data.index, data['Close'], color='#1f77b4', marker='o', linestyle='-', linewidth=2, markersize=5)
    ax.set_title(title, fontsize=20, color='white')
    ax.set_xlabel('Date', fontsize=14, color='white')
    ax.set_ylabel('Index Value', fontsize=14, color='white')
    ax.tick_params(axis='x', colors='white', rotation=45)
    ax.tick_params(axis='y', colors='white')
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

# 차트 표시
st.subheader(f"{index_choice} Index")
fig = plot_stock_data(data, f"{index_choice}")
st.pyplot(fig)
