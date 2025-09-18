import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import json, time, pytz
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict

# -------------------------------------------------------
# Timezone & Page Config
# -------------------------------------------------------
ist = pytz.timezone("Asia/Kolkata")
st.set_page_config(
    page_title="📈 NSE 10-Filter Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Global Styles
# -------------------------------------------------------
st.markdown("""
<style>
section[data-testid="stSidebar"]{background-color:#f8f9fa;}
.main-header{
    background:linear-gradient(90deg,#001f3f 0%,#0074D9 100%);
    padding:1.2rem;border-radius:12px;margin-bottom:1.2rem;
    text-align:center;color:white;box-shadow:0 4px 14px rgba(0,0,0,.15);
}
.stButton>button{
    width:100%;border-radius:10px;font-weight:600;
    border:2px solid #0074D9;background:#0074D9;color:white;
    transition:all .2s;
}
.stButton>button:hover{background:white;color:#0074D9;}
.metric-card{
    background:white;border-radius:14px;padding:1rem;
    box-shadow:0 4px 10px rgba(0,0,0,.08);text-align:center;
}
.metric-card h3{margin:0;color:#001f3f;font-size:1.1rem;}
.metric-card p{margin:0;font-size:1.6rem;font-weight:700;color:#0074D9;}
.results-header{
    font-size:1.6rem;font-weight:bold;color:#001f3f;
    margin-top:1.5rem;border-bottom:2px solid #0074D9;padding-bottom:.4rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown('<div class="main-header"><h1>🇮🇳 NSE 10-Filter Stock Screener</h1></div>', unsafe_allow_html=True)

# -------------------------------------------------------
# Data Utilities
# -------------------------------------------------------
@st.cache_data(ttl=43200)
def load_nse_stocks() -> Tuple[Dict, str]:
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df = df[df["SERIES"]=="EQ"].dropna(subset=["SYMBOL","NAME OF COMPANY"])
        return dict(zip(df["SYMBOL"], df["NAME OF COMPANY"])), f"Loaded {len(df)} NSE stocks."
    except Exception:
        p = Path("indian_stocks.json")
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return {i["symbol"].replace(".NS",""): i["name"] for i in data}, "Loaded stocks from local file."
        return {"RELIANCE":"Reliance Industries","TCS":"Tata Consultancy Services"}, "Fallback stock list."

@st.cache_data(ttl=1800)
def download_all_data(tickers):
    start = datetime.now(ist) - pd.DateOffset(years=2)
    return yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                       group_by="ticker", auto_adjust=True,
                       threads=False, ignore_tz=True)

@st.cache_data(ttl=300)
def download_current_data(tickers):
    today = datetime.now(ist)
    return yf.download(
        tickers,
        start=today.strftime("%Y-%m-%d"),
        end=(today+timedelta(days=1)).strftime("%Y-%m-%d"),
        group_by="ticker",
        auto_adjust=True,
        threads=False,
        ignore_tz=True
    )

def passes_filters(df, filters, rsi_d=50, rsi_d_cross=50, rsi_w=45, rsi_w_cross=59):
    try:
        if df is None or df.empty or len(df) < 30: return False
        df = df.copy(); latest = df.iloc[-1]
        if filters.get("Close > Open") and latest["Close"] <= latest["Open"]: return False
        if filters.get("Volume > 500k") and latest["Volume"] < 500000: return False
        if len(df) >= 5:
            df["Range"] = df["High"] - df["Low"]
            for i in range(1,5):
                if filters.get(f"Range > {i}d") and df["Range"].iloc[-1] <= df["Range"].iloc[-(i+1)]: return False
        if filters.get("Close > Weekly Open"):
            w_open = df["Open"].resample("W-MON").first().iloc[-1]
            if pd.isna(w_open) or latest["Close"] <= w_open: return False
        if filters.get("Close > Monthly Open"):
            m_open = df["Open"].resample("MS").first().iloc[-1]
            if pd.isna(m_open) or latest["Close"] <= m_open: return False
        # Daily RSI
        if filters.get("Daily RSI >") or filters.get("Daily RSI crossed"):
            r = ta.RSI(df["Close"],14).dropna()
            if r.empty: return False
            if filters.get("Daily RSI >") and r.iloc[-1] <= rsi_d: return False
            if filters.get("Daily RSI crossed") and (len(r)<2 or not (r.iloc[-2]<rsi_d_cross<r.iloc[-1])): return False
        # Weekly RSI
        w = df.resample("W-MON").agg({"Close":"last"}).dropna()
        if filters.get("Weekly RSI >") or filters.get("Weekly RSI crossed"):
            r = ta.RSI(w["Close"],14).dropna()
            if r.empty: return False
            if filters.get("Weekly RSI >") and r.iloc[-1] <= rsi_w: return False
            if filters.get("Weekly RSI crossed") and (len(r)<2 or not (r.iloc[-2]<rsi_w_cross<r.iloc[-1])): return False
        return True
    except Exception:
        return False

# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------
st.sidebar.header("📊 Filter Conditions")
filters = {}
with st.sidebar.expander("📈 Daily / Range", expanded=True):
    filters["Close > Open"] = st.checkbox("Close > Open", True)
    for i in range(1,5):
        filters[f"Range > {i}d"] = st.checkbox(f"Range > {i} Day(s) Ago", True)

with st.sidebar.expander("🗓️ Crossovers", expanded=True):
    filters["Close > Weekly Open"] = st.checkbox("Close > Weekly Open", True)
    filters["Close > Monthly Open"] = st.checkbox("Close > Monthly Open", True)

with st.sidebar.expander("💹 Volume & RSI", expanded=True):
    filters["Volume > 500k"] = st.checkbox("Volume > 500,000", True)
    col1, col2 = st.columns(2)
    with col1: filters["Daily RSI >"] = st.checkbox("Daily RSI >", True)
    with col2: rsi_d = st.number_input("Daily RSI Threshold",0.0,100.0,50.0,0.1,label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1: filters["Daily RSI crossed"] = st.checkbox("Daily RSI Crossed", True)
    with col2: rsi_d_cross = st.number_input("Crossed Threshold",0.0,100.0,50.0,0.1,label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1: filters["Weekly RSI >"] = st.checkbox("Weekly RSI >", True)
    with col2: rsi_w = st.number_input("Weekly RSI Threshold",0.0,100.0,45.0,0.1,label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1: filters["Weekly RSI crossed"] = st.checkbox("Weekly RSI Crossed", True)
    with col2: rsi_w_cross = st.number_input("Weekly Crossed Threshold",0.0,100.0,59.0,0.1,label_visibility="collapsed")

# -------------------------------------------------------
# Dashboard KPIs
# -------------------------------------------------------
stocks, msg = load_nse_stocks()
st.toast(msg, icon="✅")
col1, col2, col3 = st.columns(3)
with col1: st.markdown(f'<div class="metric-card"><h3>Total Stocks</h3><p>{len(stocks)}</p></div>', unsafe_allow_html=True)
with col2: st.markdown(f'<div class="metric-card"><h3>Filters Applied</h3><p>{sum(filters.values())}</p></div>', unsafe_allow_html=True)
with col3: st.markdown(f'<div class="metric-card"><h3>Ready</h3><p>✔</p></div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Scan Logic
# -------------------------------------------------------
if st.button("🚀 Run Scan on All NSE Stocks", use_container_width=True):
    start = datetime.now(ist)
    tickers = [f"{s}.NS" for s in stocks.keys()]
    with st.spinner("Downloading historical data…"):
        hist = download_all_data(tickers)
    with st.spinner("Fetching current market data…"):
        curr = download_current_data(tickers)

    results, status = [], st.empty()
    progress = st.progress(0)

    for i, (sym, name) in enumerate(stocks.items()):
        status.text(f"Scanning {i+1}/{len(stocks)}: {sym}")
        progress.progress((i+1)/len(stocks))
        try:
            df = hist.get(f"{sym}.NS", pd.DataFrame()).dropna(how="all")
            if df.empty: continue
            if df.index.tz is None: df.index = df.index.tz_localize("UTC")
            df.index = df.index.tz_convert("Asia/Kolkata")

            cdf = curr.get(f"{sym}.NS", pd.DataFrame())
            if not cdf.empty:
                if cdf.index.tz is None: cdf.index = cdf.index.tz_localize("UTC")
                cdf.index = cdf.index.tz_convert("Asia/Kolkata")
                latest = cdf.iloc[-1]
                new_idx = pd.Timestamp(datetime.now(ist).date()).tz_localize("Asia/Kolkata")
                row = pd.DataFrame({
                    "Open":[latest["Open"]], "High":[latest["High"]],
                    "Low":[latest["Low"]], "Close":[latest["Close"]],
                    "Volume":[latest["Volume"]],
                }, index=[new_idx])
                if df.index[-1].date() == new_idx.date():
                    df.iloc[-1] = row.iloc[0]
                else:
                    df = pd.concat([df, row])

            if passes_filters(df, filters, rsi_d, rsi_d_cross, rsi_w, rsi_w_cross):
                last = df.iloc[-1]
                pct = ((last["Close"]-last["Open"])/last["Open"]*100) if last["Open"] else 0
                d_rsi = ta.RSI(df["Close"],14).iloc[-1] if len(df)>14 else np.nan
                w_rsi = ta.RSI(df.resample("W-MON")["Close"].last(),14).iloc[-1] if len(df)>14 else np.nan
                results.append({
                    "Symbol": sym,
                    "Name": name,
                    "Close Price": f"₹{last['Close']:.2f}",
                    "% Change": f"{pct:+.2f}%",
                    "Volume": f"{int(last['Volume']):,}",
                    "Daily RSI": f"{d_rsi:.2f}" if pd.notna(d_rsi) else "N/A",
                    "Weekly RSI": f"{w_rsi:.2f}" if pd.notna(w_rsi) else "N/A",
                })
        except Exception:
            continue

    progress.empty(); status.empty()
    end = datetime.now(ist)
    st.markdown('<p class="results-header">Scan Results</p>', unsafe_allow_html=True)
    stamp = f"Scan finished **{end.strftime('%Y-%m-%d %I:%M %p')}** (Duration: {str(end-start).split('.')[0]})"

    if results:
        st.success(f"✅ {len(results)} stocks matched all filters. {stamp}")
        df_out = pd.DataFrame(results)
        df_out["Scan Time"] = end.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df_out, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV",
                           df_out.to_csv(index=False).encode("utf-8"),
                           f"scan_{end.strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv")
    else:
        st.warning(f"⚠️ No stocks matched. {stamp}")
