
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="AlphaQuant", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp {background:#050b16;}
.company-title{font-size:2.8rem;font-weight:800;color:white;}
.metric-card{background:#0d1628;padding:18px;border-radius:18px;border:1px solid #22395d;}
.disclaimer{font-size:12px;color:#7e8ea8;text-align:center;margin-top:80px;}
</style>
""", unsafe_allow_html=True)

def get_currency_config(ticker):
    if ticker.upper().endswith((".NS",".BO")):
        return "₹"
    return "$"

@st.cache_data(ttl=3600)
def fetch_data(ticker):
    end = datetime.today()
    start = end - timedelta(days=5*365)
    df = yf.download(ticker, start=start, end=end, progress=False)
    info = {}
    try:
        info = yf.Ticker(ticker).info
    except:
        pass
    return df, info

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

horizon_map = {
    "1 Month":21,"2 Months":42,"3 Months":63,
    "6 Months":126,"1 Year":252,"2 Years":504,"3 Years":756
}

st.markdown("<h1 style='text-align:center'>ALPHA<span style='color:#00aaff'>QUANT</span></h1>", unsafe_allow_html=True)

c1,c2,c3 = st.columns([1.4,5,1])

with c1:
    horizon_choice = st.selectbox("Forecast", list(horizon_map.keys()))
    engine_choice = st.selectbox("Model",["Meta Prophet","Manual ARIMA"])
    p=d=q=None
    if engine_choice=="Manual ARIMA":
        p = st.number_input("p",0,10,5)
        d = st.number_input("d",0,3,1)
        q = st.number_input("q",0,10,0)

with c2:
    ticker = st.text_input("", placeholder="Search Stock (RELIANCE.NS, AAPL, NVDA...)").upper()

with c3:
    st.write("")
    st.write("")
    run = st.button("GO", use_container_width=True)

if run and ticker:
    st.session_state.analysis_complete = True
    st.session_state.ticker = ticker
    st.session_state.horizon = horizon_choice
    st.session_state.engine = engine_choice
    st.session_state.params = (p,d,q)

if not st.session_state.analysis_complete:
    st.markdown("<div class='disclaimer'>Research purposes only. Not investment advice.</div>", unsafe_allow_html=True)
    st.stop()

ticker = st.session_state.ticker
days = horizon_map[st.session_state.horizon]

df, info = fetch_data(ticker)
close = df["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:,0]
close = close.dropna()
if close.empty:
    st.error(
        f"No historical price data found for '{ticker}'. "
        "Verify the ticker symbol and exchange suffix."
    )
    st.stop()
    
current_price = float(close.iloc[-1])
sym = get_currency_config(ticker)

st.markdown(f"<div class='company-title'>{info.get('longName', ticker)}</div>", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs([
    "Executive Summary",
    "Technical Dashboard",
    "Predictive Forecasting",
    "Raw Data"
])

with tab1:
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Current Price", f"{sym}{current_price:,.2f}")
    m2.metric("Market Cap", str(info.get("marketCap","N/A")))
    m3.metric("PE Ratio", str(info.get("trailingPE","N/A")))
    m4.metric("Beta", str(info.get("beta","N/A")))
    st.subheader("Company Overview")
    st.write(info.get("longBusinessSummary","No description available"))

with tab2:
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    fig = go.Figure()
    fig.add_scatter(x=close.index,y=close,name="Price")
    fig.add_scatter(x=sma50.index,y=sma50,name="SMA50")
    fig.add_scatter(x=sma200.index,y=sma200,name="SMA200")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    if st.session_state.engine == "Meta Prophet":
        pdf = close.reset_index()
        pdf.columns=["ds","y"]
        model = Prophet()
        model.fit(pdf)
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)
        pred = forecast["yhat"].tail(days)
        dates = forecast["ds"].tail(days)
    else:
        p,d,q = st.session_state.params
        model = ARIMA(np.log(close), order=(p,d,q))
        fit = model.fit()
        pred = np.exp(fit.forecast(days))
        dates = pd.bdate_range(close.index[-1], periods=days+1)[1:]

    target = float(pred.iloc[-1] if hasattr(pred,"iloc") else pred[-1])
    roi = ((target-current_price)/current_price)*100

    k1,k2,k3 = st.columns(3)
    k1.metric("Target Price", f"{sym}{target:,.2f}")
    k2.metric("Expected Return", f"{roi:.2f}%")
    k3.metric("Current Price", f"{sym}{current_price:,.2f}")

    fig2 = go.Figure()
    fig2.add_scatter(x=close.index[-250:], y=close[-250:], name="Historical")
    fig2.add_scatter(x=dates, y=pred, name="Forecast")
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    if 'dates' in locals():
        export_df = pd.DataFrame({"Date":dates,"Forecast":pred})
        st.dataframe(export_df, use_container_width=True, height=600)
        st.download_button(
            "Download CSV",
            export_df.to_csv(index=False).encode(),
            file_name=f"{ticker}_forecast.csv"
        )
