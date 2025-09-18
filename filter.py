import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import talib as ta
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict
import time
import pytz

# --- Timezone Configuration ---
ist = pytz.timezone('Asia/Kolkata')

# --- Page Configuration ---
st.set_page_config(
    page_title="📈 NSE 10-Filter Stock Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Modern CSS Styling ---
st.markdown("""
<style>
    body, .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Inter', sans-serif;
        color: #e0e0e0;
    }
    .main-header {
        background: rgba(0,0,0,0.25);
        padding: 1.8rem;
        border-radius: 18px;
        text-align: center;
        color: #ffffff;
        backdrop-filter: blur(10px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
        margin-bottom: 2rem;
    }
    .stSidebar, .css-1d391kg {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(12px);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        border: none;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 198, 255, 0.4);
    }
    .results-header {
        font-size: 2rem;
        font-weight: 700;
        color: #00c6ff;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    .stDataFrame, .stAlert {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(
    '<div class="main-header">'
    '<h1>🇮🇳 NSE 10-Filter Stock Screener</h1>'
    '<p style="opacity:0.85;">Find top stocks that pass all selected conditions in real-time.</p>'
    '</div>',
    unsafe_allow_html=True
)

# --- Data Loading & Caching ---
@st.cache_data(ttl=43200)
def load_nse_stocks() -> Tuple[Dict, str]:
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url)
        df.dropna(subset=['SYMBOL', 'NAME OF COMPANY'], inplace=True)
        df = df[df['SERIES'] == 'EQ']
        stock_dict = dict(zip(df['SYMBOL'], df['NAME OF COMPANY']))
        return stock_dict, f"Successfully loaded {len(stock_dict)} stocks from NSE."
    except Exception:
        pass
    try:
        json_path = Path("indian_stocks.json")
        if json_path.exists():
            with open(json_path, "r") as f:
                stock_list = json.load(f)
                stock_dict = {item['symbol'].replace('.NS', ''): item['name'] for item in stock_list}
                return stock_dict, f"Loaded {len(stock_dict)} stocks from local file."
        else:
            raise FileNotFoundError
    except Exception:
        fallback_stocks = {
            "RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services"
        }
        return fallback_stocks, "Using a small fallback list."


@st.cache_data(ttl=1800)
def download_all_data(tickers):
    today = datetime.now(ist)
    start_date = today - pd.DateOffset(years=2)
    data = yf.download(
        tickers,
        start=start_date.strftime('%Y-%m-%d'),
        group_by='ticker',
        auto_adjust=True,
        threads=False,
        ignore_tz=True
    )
    return data


@st.cache_data(ttl=300)
def download_current_data(tickers):
    try:
        today = datetime.now(ist)
        start_str = today.strftime('%Y-%m-%d')
        end_str = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        data = yf.download(
            tickers,
            start=start_str,
            end=end_str,
            group_by='ticker',
            auto_adjust=True,
            threads=False,
            ignore_tz=True
        )
        return data
    except Exception as e:
        st.error(f"Error downloading current data: {e}")
        return pd.DataFrame()

def passes_filters(df, filters,
                   rsi_daily_above_threshold=50.0,
                   rsi_daily_crossed_threshold=50.0,
                   rsi_weekly_above_threshold=45.0,
                   rsi_weekly_crossed_threshold=59.0):
    try:
        if df is None or df.empty or len(df) < 30: return False
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        latest = df.iloc[-1]
        if filters.get("Close > Open", False) and latest['Close'] <= latest['Open']: return False
        if filters.get("Volume > 500k", False) and latest['Volume'] < 500000: return False
        if len(df) >= 5:
            df['Range'] = df['High'] - df['Low']
            for i in range(1, 5):
                if filters.get(f"Range > {i}d ago", False):
                    if df['Range'].iloc[-1] <= df['Range'].iloc[-(i + 1)]: return False
        if filters.get("Close > Weekly Open", False):
            weekly_open = df['Open'].resample('W-MON').first().iloc[-1]
            if pd.isna(weekly_open) or latest['Close'] <= weekly_open: return False
        if filters.get("Close > Monthly Open", False):
            monthly_open = df['Open'].resample('MS').first().iloc[-1]
            if pd.isna(monthly_open) or latest['Close'] <= monthly_open: return False
        if filters.get("Daily RSI >", False) or filters.get("Daily RSI crossed", False):
            d_rsi_series = ta.RSI(df['Close'], timeperiod=14)
            if d_rsi_series.dropna().shape[0] < 1: return False
            latest_d_rsi = d_rsi_series.iloc[-1]
            if pd.isna(latest_d_rsi): return False
            if filters.get("Daily RSI >", False) and latest_d_rsi <= rsi_daily_above_threshold: return False
            if filters.get("Daily RSI crossed", False):
                if d_rsi_series.dropna().shape[0] < 2: return False
                prev_d_rsi = d_rsi_series.iloc[-2]
                if pd.isna(prev_d_rsi): return False
                if not (prev_d_rsi < rsi_daily_crossed_threshold and latest_d_rsi > rsi_daily_crossed_threshold): return False
        weekly_data = df.resample('W-MON').agg({'Close': 'last'}).dropna()
        if filters.get("Weekly RSI >", False) or filters.get("Weekly RSI crossed", False):
            if len(weekly_data) < 15: return False
            w_rsi_series = ta.RSI(weekly_data['Close'], timeperiod=14)
            if w_rsi_series.dropna().shape[0] < 1: return False
            latest_w_rsi = w_rsi_series.iloc[-1]
            if pd.isna(latest_w_rsi): return False
            if filters.get("Weekly RSI >", False) and latest_w_rsi <= rsi_weekly_above_threshold: return False
            if filters.get("Weekly RSI crossed", False):
                if w_rsi_series.dropna().shape[0] < 2: return False
                prev_w_rsi = w_rsi_series.iloc[-2]
                if pd.isna(prev_w_rsi): return False
                if not (prev_w_rsi < rsi_weekly_crossed_threshold and latest_w_rsi > rsi_weekly_crossed_threshold): return False
        return True
    except (IndexError, KeyError, Exception):
        return False

# --- Sidebar Filters ---
st.sidebar.title("📊 Filter Conditions")
st.sidebar.caption("Select criteria to find stocks that meet all chosen filters.")

active_filters = {}
with st.sidebar.expander("📈 Daily Price/Range Filters", expanded=True):
    active_filters["Close > Open"] = st.checkbox("Daily Close > Daily Open", True)
    for i in range(1, 5):
        active_filters[f"Range > {i}d ago"] = st.checkbox(f"Daily Range > {i} Day(s) Ago", True)

with st.sidebar.expander("🗓️ Periodical Crossover Filters", expanded=True):
    active_filters["Close > Weekly Open"] = st.checkbox("Daily Close > Weekly Open", True)
    active_filters["Close > Monthly Open"] = st.checkbox("Daily Close > Monthly Open", True)

with st.sidebar.expander("💹 Volume & RSI Filters", expanded=True):
    active_filters["Volume > 500k"] = st.checkbox("Daily Volume > 500,000", True)
    col1, col2 = st.columns(2)
    with col1:
        active_filters["Daily RSI >"] = st.checkbox("Daily RSI >", True)
    with col2:
        rsi_daily_above_threshold = st.number_input("Daily RSI > Threshold", 0.0, 100.0, 50.0, 0.1, label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        active_filters["Daily RSI crossed"] = st.checkbox("Daily RSI Crossed Above", True)
    with col2:
        rsi_daily_crossed_threshold = st.number_input("Daily Crossed Threshold", 0.0, 100.0, 50.0, 0.1, label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        active_filters["Weekly RSI >"] = st.checkbox("Weekly RSI >", True)
    with col2:
        rsi_weekly_above_threshold = st.number_input("Weekly RSI > Threshold", 0.0, 100.0, 45.0, 0.1, label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        active_filters["Weekly RSI crossed"] = st.checkbox("Weekly RSI Crossed Above", True)
    with col2:
        rsi_weekly_crossed_threshold = st.number_input("Weekly Crossed Threshold", 0.0, 100.0, 59.0, 0.1, label_visibility="collapsed")

st.sidebar.markdown("---")

# --- Main Logic ---
all_stocks, status_message = load_nse_stocks()
st.toast(status_message, icon="✅")
total_stocks_count = len(all_stocks)
st.info(f"Ready to scan **{total_stocks_count}** stocks based on your selected filters.")

if st.button("🚀 Run Scan on All NSE Stocks"):
    start_time = datetime.now(ist)
    with st.spinner(f"Downloading historical data for {total_stocks_count} stocks..."):
        tickers = [f"{symbol}.NS" for symbol in all_stocks.keys()]
        data = download_all_data(tickers)
    with st.spinner("Downloading current market data..."):
        current_data = download_current_data(tickers)
        retries = 3
        while current_data.empty and retries > 0:
            st.warning("Rate limit hit on current data download. Retrying after delay...")
            time.sleep(60)
            current_data = download_current_data(tickers)
            retries -= 1

    results = []
    status_text = st.empty()
    progress_bar = st.progress(0)

    for i, (symbol, name) in enumerate(all_stocks.items()):
        status_text.text(f"Scanning... {i + 1}/{total_stocks_count} - {symbol}")
        progress_bar.progress((i + 1) / total_stocks_count)
        try:
            stock_df = data.get(f"{symbol}.NS", pd.DataFrame()).dropna(how='all').copy()
            if not stock_df.empty:
                if stock_df.index.tz is None:
                    stock_df.index = stock_df.index.tz_localize('UTC')
                stock_df.index = stock_df.index.tz_convert('Asia/Kolkata')
            current_df = current_data.get(f"{symbol}.NS", pd.DataFrame())
            if not current_df.empty:
                if current_df.index.tz is None:
                    current_df.index = current_df.index.tz_localize('UTC')
                current_df.index = current_df.index.tz_convert('Asia/Kolkata')
                latest_current = current_df.iloc[-1]
                day_open = latest_current['Open']
                day_high = latest_current['High']
                day_low = latest_current['Low']
                current_price = latest_current['Close']
                volume = latest_current['Volume']
                if volume > 0 and day_open > 0:
                    current_date = datetime.now(ist).date()
                    if not stock_df.empty:
                        last_date = stock_df.index[-1].date()
                        if last_date == current_date:
                            last_index = stock_df.index[-1]
                            stock_df.at[last_index, 'Close'] = current_price
                            stock_df.at[last_index, 'High'] = max(stock_df.at[last_index, 'High'], day_high)
                            stock_df.at[last_index, 'Low'] = min(stock_df.at[last_index, 'Low'], day_low)
                            stock_df.at[last_index, 'Volume'] = volume
                        elif last_date < current_date:
                            new_index = pd.Timestamp(current_date).tz_localize('Asia/Kolkata')
                            new_row = pd.DataFrame({
                                'Open': [day_open],
                                'High': [day_high],
                                'Low': [day_low],
                                'Close': [current_price],
                                'Volume': [volume],
                            }, index=[new_index])
                            stock_df = pd.concat([stock_df, new_row])
                    else:
                        new_index = pd.Timestamp(current_date).tz_localize('Asia/Kolkata')
                        new_row = pd.DataFrame({
                            'Open': [day_open],
                            'High': [day_high],
                            'Low': [day_low],
                            'Close': [current_price],
                            'Volume': [volume],
                        }, index=[new_index])
                        stock_df = pd.concat([stock_df, new_row])
            if passes_filters(stock_df, active_filters,
                              rsi_daily_above_threshold,
                              rsi_daily_crossed_threshold,
                              rsi_weekly_above_threshold,
                              rsi_weekly_crossed_threshold):
                latest = stock_df.iloc[-1]
                pct_change = ((latest['Close'] - latest['Open']) / latest['Open'] * 100) if latest['Open'] != 0 else 0
                d_rsi = ta.RSI(stock_df['Close'], timeperiod=14).iloc[-1] if len(stock_df) > 14 else np.nan
                weekly_data = stock_df.resample('W-MON').agg({'Close': 'last'}).dropna()
                w_rsi = ta.RSI(weekly_data['Close'], timeperiod=14).iloc[-1] if len(weekly_data) > 14 else np.nan
                results.append({
                    "Symbol": symbol,
                    "Name": name,
                    "Close Price": f"₹{latest['Close']:.2f}",
                    "% Change": f"{pct_change:+.2f}%",
                    "Volume": f"{int(latest['Volume']):,}",
                    "Daily RSI": f"{d_rsi:.2f}" if not pd.isna(d_rsi) else "N/A",
                    "Weekly RSI": f"{w_rsi:.2f}" if not pd.isna(w_rsi) else "N/A"
                })
        except KeyError:
            continue

    end_time = datetime.now(ist)
    scan_duration = end_time - start_time
    status_text.empty()
    progress_bar.empty()

    # --- Results ---
    st.markdown(f'<p class="results-header">Scan Results</p>', unsafe_allow_html=True)
    scan_timestamp = f"Scan completed on **{end_time.strftime('%Y-%m-%d at %I:%M:%S %p')}** (Duration: **{str(scan_duration).split('.')[0]}**)."

    if results:
        st.success(f"✅ Found **{len(results)}** stocks that passed all filters. {scan_timestamp}")
        results_df = pd.DataFrame(results)
        results_df['Scan Date & Time'] = end_time.strftime('%Y-%m-%d %H:%M')
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"stock_scan_{end_time.strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )
    else:
        st.warning(f"⚠️ No stocks met all your criteria. Try removing some filters. {scan_timestamp}")
