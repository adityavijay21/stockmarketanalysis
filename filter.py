import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import json, time
from datetime import datetime, timedelta
from pathlib import Path
import pytz
from typing import Dict, Tuple

# ---------------- Page Setup ---------------- #
st.set_page_config(
    page_title="📊 NSE 10-Filter Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- Custom CSS for Dashboard look ----------- #
st.markdown("""
<style>
[data-testid="stSidebar"] {background-color:#f9fafb;}
.main-header{
  background:linear-gradient(90deg,#001f3f 0%,#0074D9 100%);
  padding:1.2rem;border-radius:12px;margin-bottom:1rem;
  text-align:center;color:white;font-size:1.7rem;
  box-shadow:0 4px 12px rgba(0,0,0,0.15);
}
.stButton>button{
  width:100%;border-radius:10px;font-weight:600;
  border:2px solid #0074D9;background:#0074D9;color:white;
  transition:0.2s;
}
.stButton>button:hover{background:white;color:#0074D9;}
.results-header{
  font-size:1.5rem;font-weight:700;color:#001f3f;
  margin-top:1rem;border-bottom:2px solid #0074D9;padding-bottom:.3rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🇮🇳 NSE 10-Filter Stock Screener</div>', unsafe_allow_html=True)

IST = pytz.timezone("Asia/Kolkata")

# ---------------- Helpers ---------------- #
@st.cache_data(ttl=43200)
def load_nse_stocks() -> Tuple[Dict, str]:
    """Return {symbol:name} dict with fallback."""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df = df[(df["SERIES"]=="EQ") & df["SYMBOL"].notna()]
        return dict(zip(df["SYMBOL"], df["NAME OF COMPANY"])), f"Loaded {len(df)} NSE stocks."
    except Exception:
        p = Path("indian_stocks.json")
        if p.exists():
            with open(p) as f: d=json.load(f)
            dct={i['symbol'].replace('.NS',''):i['name'] for i in d}
            return dct,f"Loaded {len(dct)} from local JSON."
        return {"RELIANCE":"Reliance Industries","TCS":"Tata Consultancy"}, "Fallback list used."

@st.cache_data(ttl=1800)
def download_hist(tickers):
    start=(datetime.now(IST)-pd.DateOffset(years=2)).strftime('%Y-%m-%d')
    return yf.download(tickers,start=start,auto_adjust=True,
                       group_by='ticker',threads=False,ignore_tz=True)

@st.cache_data(ttl=300)
def download_intraday(tickers):
    today=datetime.now(IST)
    return yf.download(
        tickers,
        start=today.strftime('%Y-%m-%d'),
        end=(today+timedelta(days=1)).strftime('%Y-%m-%d'),
        auto_adjust=True,group_by='ticker',threads=False,ignore_tz=True
    )

def apply_filters(df:pd.DataFrame, f:dict,
                  rsi_d_gt, rsi_d_cross,
                  rsi_w_gt, rsi_w_cross) -> bool:
    """Return True if stock passes all active filters."""
    try:
        if df is None or df.empty or len(df)<30: return False
        df=df.copy()
        df.index=pd.to_datetime(df.index)

        latest=df.iloc[-1]
        if f["Close > Open"] and latest.Close<=latest.Open: return False
        if f["Volume > 500k"] and latest.Volume<5e5: return False

        # Range comparisons
        if len(df)>=5:
            df['Range']=df.High-df.Low
            for i in range(1,5):
                if f[f"Range > {i}d ago"] and df.Range.iloc[-1]<=df.Range.iloc[-(i+1)]: return False

        if f["Close > Weekly Open"]:
            wk=df.Open.resample('W-MON').first().iloc[-1]
            if pd.isna(wk) or latest.Close<=wk: return False
        if f["Close > Monthly Open"]:
            mo=df.Open.resample('MS').first().iloc[-1]
            if pd.isna(mo) or latest.Close<=mo: return False

        # RSI daily
        if f["Daily RSI >"] or f["Daily RSI crossed"]:
            rsi=ta.RSI(df.Close,14).dropna()
            if rsi.empty: return False
            if f["Daily RSI >"] and rsi.iloc[-1]<=rsi_d_gt: return False
            if f["Daily RSI crossed"] and not (rsi.iloc[-2]<rsi_d_cross<rsi.iloc[-1]): return False

        # RSI weekly
        if f["Weekly RSI >"] or f["Weekly RSI crossed"]:
            wdf=df.resample('W-MON').Close.last().dropna()
            if len(wdf)<15: return False
            wrsi=ta.RSI(wdf,14).dropna()
            if wrsi.empty: return False
            if f["Weekly RSI >"] and wrsi.iloc[-1]<=rsi_w_gt: return False
            if f["Weekly RSI crossed"] and not (wrsi.iloc[-2]<rsi_w_cross<wrsi.iloc[-1]): return False
        return True
    except Exception:
        return False

# ---------------- Sidebar Filters ---------------- #
st.sidebar.header("📊 Filter Conditions")
active={}
with st.sidebar.expander("Daily Price / Range", True):
    active["Close > Open"]=st.checkbox("Close > Open",True)
    for i in range(1,5):
        active[f"Range > {i}d ago"]=st.checkbox(f"Range > {i} Day Ago",True)
with st.sidebar.expander("Periodic Levels", True):
    active["Close > Weekly Open"]=st.checkbox("Close > Weekly Open",True)
    active["Close > Monthly Open"]=st.checkbox("Close > Monthly Open",True)
with st.sidebar.expander("Volume & RSI", True):
    active["Volume > 500k"]=st.checkbox("Volume > 500k",True)
    col1,col2=st.columns(2)
    with col1: active["Daily RSI >"]=st.checkbox("Daily RSI >",True)
    with col2: rsi_d_gt=st.number_input("Daily >",0.0,100.0,50.0,0.1,label_visibility="collapsed")
    col1,col2=st.columns(2)
    with col1: active["Daily RSI crossed"]=st.checkbox("Daily RSI crossed",True)
    with col2: rsi_d_cross=st.number_input("Cross >",0.0,100.0,50.0,0.1,label_visibility="collapsed")
    col1,col2=st.columns(2)
    with col1: active["Weekly RSI >"]=st.checkbox("Weekly RSI >",True)
    with col2: rsi_w_gt=st.number_input("Weekly >",0.0,100.0,45.0,0.1,label_visibility="collapsed")
    col1,col2=st.columns(2)
    with col1: active["Weekly RSI crossed"]=st.checkbox("Weekly RSI crossed",True)
    with col2: rsi_w_cross=st.number_input("Weekly Cross >",0.0,100.0,59.0,0.1,label_visibility="collapsed")

# ---------------- Main Scan ---------------- #
stocks,msg=load_nse_stocks()
st.toast(msg, icon="✅")
st.info(f"Ready to scan **{len(stocks)}** NSE stocks with selected filters.")

if st.button("🚀 Run Scan"):
    start=datetime.now(IST)
    tickers=[f"{s}.NS" for s in stocks]
    with st.spinner("Downloading 2-year historical data…"):
        hist=download_hist(tickers)
    with st.spinner("Fetching current market data…"):
        curr=download_intraday(tickers)

    results=[]
    prog=st.progress(0)
    for i,(sym,name) in enumerate(stocks.items()):
        prog.progress((i+1)/len(stocks))
        h=hist.get(f"{sym}.NS",pd.DataFrame()).dropna(how='all').copy()
        c=curr.get(f"{sym}.NS",pd.DataFrame())
        # merge today’s candle
        if not c.empty:
            c = c.tz_localize('UTC').tz_convert(IST) if c.index.tz is None else c.tz_convert(IST)
            today=datetime.now(IST).date()
            if not h.empty:
                h.index=pd.to_datetime(h.index).tz_localize('UTC').tz_convert(IST)
            if not h.empty and h.index[-1].date()==today:
                h.iloc[-1]=c.iloc[-1]
            else:
                c.index=[pd.Timestamp(today, tz=IST)]
                h=pd.concat([h,c])

        if apply_filters(h,active,rsi_d_gt,rsi_d_cross,rsi_w_gt,rsi_w_cross):
            last=h.iloc[-1]
            pct=(last.Close-last.Open)/last.Open*100 if last.Open else 0
            d_rsi=ta.RSI(h.Close,14).iloc[-1] if len(h)>14 else np.nan
            w_rsi=ta.RSI(h.Close.resample('W-MON').last().dropna(),14).iloc[-1] if len(h)>14 else np.nan
            results.append({
                "Symbol":sym,
                "Name":name,
                "Close":f"₹{last.Close:.2f}",
                "% Change":f"{pct:+.2f}%",
                "Volume":f"{int(last.Volume):,}",
                "Daily RSI":f"{d_rsi:.2f}" if pd.notna(d_rsi) else "N/A",
                "Weekly RSI":f"{w_rsi:.2f}" if pd.notna(w_rsi) else "N/A"
            })

    end=datetime.now(IST)
    st.markdown('<p class="results-header">Scan Results</p>', unsafe_allow_html=True)
    dur=str(end-start).split('.')[0]
    if results:
        st.success(f"✅ {len(results)} stocks passed. Completed {end.strftime('%Y-%m-%d %H:%M:%S')} (took {dur}).")
        df=pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV",
                           df.to_csv(index=False).encode(),
                           f"scan_{end.strftime('%Y%m%d_%H%M')}.csv",
                           "text/csv")
    else:
        st.warning(f"⚠️ No stocks met all criteria. Completed in {dur}.")
