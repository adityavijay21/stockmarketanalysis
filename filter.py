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
    page_title="📈 NSE 10-Filter Stock Screener — Advanced Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Polished CSS + card styles ---
st.markdown(
    """
    <style>
    /* Header */
    .main-header {
        background: linear-gradient(90deg,#081b36 0%, #0b5394 50%, #0086d1 100%);
        color: white;
        padding: 18px 24px;
        border-radius: 12px;
        box-shadow: 0 6px 18px rgba(3,10,30,0.35);
        margin-bottom: 20px;
    }
    .main-sub {
        color: rgba(255,255,255,0.9);
        margin-top: 6px;
        font-size: 0.95rem;
    }

    /* Buttons & inputs */
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        padding: 8px 12px;
        font-weight: 600;
        border: 2px solid rgba(255,255,255,0.08);
    }

    /* Card */
    .result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%);
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 6px 16px rgba(12,45,90,0.06);
        margin-bottom: 12px;
    }

    /* KPI */
    .kpi {
        background: rgba(255,255,255,0.04);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }

    /* Table tweaks */
    .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
    }
    .dataframe thead th {
      text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# --- Data Loading & Caching (UNCHANGED LOGIC) ---
# I'm intentionally keeping these function names and logic as-is
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

def passes_filters(df, filters, rsi_daily_above_threshold=50.0, rsi_daily_crossed_threshold=50.0, rsi_weekly_above_threshold=45.0, rsi_weekly_crossed_threshold=59.0):
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

        # Daily RSI calculations
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

        # Weekly RSI calculations
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

# ------------------- UI Layout & Controls (ENHANCED UX) ------------------- #

# Top header
st.markdown(
    f'<div class="main-header"><h2>📈 NSE 10-Filter Stock Screener — Advanced Dashboard</h2>'
    f'<div class="main-sub">Scan NSE universe with your 10 filters. Fast scans, clear KPIs, and visual insights — logic untouched.</div></div>',
    unsafe_allow_html=True
)

# Load stocks
all_stocks, status_message = load_nse_stocks()
total_stocks_count = len(all_stocks)

# Sidebar: filters + presets
st.sidebar.header("⚙️ Filters & Scan Controls")
st.sidebar.write("Pick filters (exactly as your logic expects). Use presets to quickly toggle common combos.")

# Presets (UI helpers only)
if "preset" not in st.session_state:
    st.session_state.preset = "Custom"

def apply_preset_aggressive():
    st.session_state["Close > Open"] = True
    st.session_state["Volume > 500k"] = True
    for i in range(1,5):
        st.session_state[f"Range > {i}d ago"] = True
    st.session_state["Close > Weekly Open"] = True
    st.session_state["Close > Monthly Open"] = False
    st.session_state["Daily RSI >"] = True
    st.session_state["Daily RSI crossed"] = True
    st.session_state["Weekly RSI >"] = True
    st.session_state["Weekly RSI crossed"] = True
    st.session_state.preset = "Aggressive"

def apply_preset_relaxed():
    st.session_state["Close > Open"] = True
    st.session_state["Volume > 500k"] = False
    for i in range(1,5):
        st.session_state[f"Range > {i}d ago"] = False
    st.session_state["Close > Weekly Open"] = False
    st.session_state["Close > Monthly Open"] = False
    st.session_state["Daily RSI >"] = False
    st.session_state["Daily RSI crossed"] = False
    st.session_state["Weekly RSI >"] = False
    st.session_state["Weekly RSI crossed"] = False
    st.session_state.preset = "Relaxed"

preset_cols = st.sidebar.columns([1,1,1])
preset_cols[0].button("Aggressive", on_click=apply_preset_aggressive)
preset_cols[1].button("Relaxed", on_click=apply_preset_relaxed)
preset_cols[2].button("Custom", on_click=lambda: st.session_state.update({"preset":"Custom"}))

st.sidebar.markdown("---")
st.sidebar.header("📊 Filter Conditions (same as original)")
st.sidebar.write("Selecting/unselecting here DOES NOT change logic — it's just the same checkboxes you had before.")

# Build the same checkboxes but using session state to allow presets
active_filters = {}
with st.sidebar.expander("📈 Daily Price/Range Filters", expanded=True):
    active_filters["Close > Open"] = st.checkbox("Daily Close > Daily Open", value=True, key="Close > Open")
    for i in range(1, 5):
        active_filters[f"Range > {i}d ago"] = st.checkbox(f"Daily Range > {i} Day(s) Ago", value=True, key=f"Range > {i}d ago")

with st.sidebar.expander("🗓️ Periodical Crossover Filters", expanded=True):
    active_filters["Close > Weekly Open"] = st.checkbox("Daily Close > Weekly Open", value=True, key="Close > Weekly Open")
    active_filters["Close > Monthly Open"] = st.checkbox("Daily Close > Monthly Open", value=True, key="Close > Monthly Open")
    
with st.sidebar.expander("💹 Volume & RSI Filters", expanded=True):
    active_filters["Volume > 500k"] = st.checkbox("Daily Volume > 500,000", value=True, key="Volume > 500k")
    
    col1, col2 = st.columns(2)
    with col1:
        active_filters["Daily RSI >"] = st.checkbox("Daily RSI >", value=True, key="Daily RSI >")
    with col2:
        rsi_daily_above_threshold = st.number_input("Daily RSI > Threshold", min_value=0.0, max_value=100.0, value=50.0, step=0.1, label_visibility="collapsed", key="rsi_daily_above")

    col1, col2 = st.columns(2)
    with col1:
        active_filters["Daily RSI crossed"] = st.checkbox("Daily RSI Crossed Above", value=True, key="Daily RSI crossed")
    with col2:
        rsi_daily_crossed_threshold = st.number_input("Daily Crossed Threshold", min_value=0.0, max_value=100.0, value=50.0, step=0.1, label_visibility="collapsed", key="rsi_daily_crossed")

    col1, col2 = st.columns(2)
    with col1:
        active_filters["Weekly RSI >"] = st.checkbox("Weekly RSI >", value=True, key="Weekly RSI >")
    with col2:
        rsi_weekly_above_threshold = st.number_input("Weekly RSI > Threshold", min_value=0.0, max_value=100.0, value=45.0, step=0.1, label_visibility="collapsed", key="rsi_weekly_above")

    col1, col2 = st.columns(2)
    with col1:
        active_filters["Weekly RSI crossed"] = st.checkbox("Weekly RSI Crossed Above", value=True, key="Weekly RSI crossed")
    with col2:
        rsi_weekly_crossed_threshold = st.number_input("Weekly Crossed Threshold", min_value=0.0, max_value=100.0, value=59.0, step=0.1, label_visibility="collapsed", key="rsi_weekly_crossed")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use presets to quickly set combinations. All filter logic remains identical to original implementation.")

# Small controls
controls_col1, controls_col2 = st.columns([1, 2])
with controls_col1:
    st.info(f"Universe: **{total_stocks_count}** NSE stocks")
with controls_col2:
    st.write("")  # spacing
    st.write("")  # spacing

# --- Scan button card ---
scan_col1, scan_col2, scan_col3 = st.columns([1,2,1])
with scan_col1:
    st.empty()
with scan_col2:
    run_scan = st.button("🚀 Run Scan on All NSE Stocks", key="run_scan", help="This will download data and run your filters across the NSE universe.")
with scan_col3:
    st.empty()

# Space for KPIs while scanning / after scan
kpi_row = st.container()
kpi_cols = kpi_row.columns(4)
kpi_cols[0].metric("Scanned", "0", delta=None)
kpi_cols[1].metric("Matches", "0", delta=None)
kpi_cols[2].metric("Top Mover", "—", delta=None)
kpi_cols[3].metric("Avg Volume (matched)", "—", delta=None)

# Placeholder for results and charts
results_container = st.container()
insights_container = st.container()

# --- Main scan logic: keep exactly the same mechanics as original but enhance progress display ---
results = []
scan_timestamp = None

if run_scan:
    start_time = datetime.now(ist)
    st.toast(status_message, icon="✅")
    with st.spinner(f"Downloading historical data for {total_stocks_count} stocks..."):
        tickers = [f"{symbol}.NS" for symbol in all_stocks.keys()]
        data = download_all_data(tickers)
    with st.spinner("Downloading current market data (batch)..."):
        current_data = download_current_data(tickers)
        retries = 3
        while current_data.empty and retries > 0:
            st.warning("Rate limit hit on current data download. Retrying after delay...")
            time.sleep(60)
            current_data = download_current_data(tickers)
            retries -= 1

    # Progress UI
    status_text = st.empty()
    progress_bar = st.progress(0)
    matches_count = 0
    scanned_count = 0

    # Iterate through tickers (same flow)
    for i, (symbol, name) in enumerate(all_stocks.items()):
        scanned_count += 1
        status_text.text(f"Scanning {i+1}/{total_stocks_count}: {symbol}")
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
            
            if passes_filters(stock_df, active_filters, rsi_daily_above_threshold, rsi_daily_crossed_threshold, rsi_weekly_above_threshold, rsi_weekly_crossed_threshold):
                matches_count += 1
                latest = stock_df.iloc[-1]
                pct_change = ((latest['Close'] - latest['Open']) / latest['Open'] * 100) if latest['Open'] != 0 else 0
                d_rsi = ta.RSI(stock_df['Close'], timeperiod=14).iloc[-1] if len(stock_df) > 14 else np.nan
                weekly_data = stock_df.resample('W-MON').agg({'Close': 'last'}).dropna()
                w_rsi = ta.RSI(weekly_data['Close'], timeperiod=14).iloc[-1] if len(weekly_data) > 14 else np.nan
                results.append({
                    "Symbol": symbol,
                    "Name": name,
                    "Close Price": float(latest['Close']),
                    "% Change": float(pct_change),
                    "Volume": int(latest['Volume']) if not pd.isna(latest['Volume']) else 0,
                    "Daily RSI": float(d_rsi) if not pd.isna(d_rsi) else np.nan,
                    "Weekly RSI": float(w_rsi) if not pd.isna(w_rsi) else np.nan
                })
        except KeyError:
            continue

    end_time = datetime.now(ist)
    scan_duration = end_time - start_time
    status_text.empty()
    progress_bar.empty()
    scan_timestamp = f"Scan completed on {end_time.strftime('%Y-%m-%d %I:%M:%S %p')} (Duration: {str(scan_duration).split('.')[0]})"

    # Update KPIs
    kpi_cols = kpi_row.columns(4)
    kpi_cols[0].metric("Scanned", f"{scanned_count}/{total_stocks_count}")
    kpi_cols[1].metric("Matches", f"{matches_count}")
    if results:
        sorted_by_change = sorted(results, key=lambda r: abs(r["% Change"]), reverse=True)
        top_mover = sorted_by_change[0]["Symbol"]
        avg_vol = int(np.mean([r["Volume"] for r in results])) if results else 0
        kpi_cols[2].metric("Top Mover", top_mover)
        kpi_cols[3].metric("Avg Volume (matched)", f"{avg_vol:,}")
    else:
        kpi_cols[2].metric("Top Mover", "—")
        kpi_cols[3].metric("Avg Volume (matched)", "—")

    # --- Display results: table + insights ---
    with results_container:
        st.markdown(f"### ✅ Scan Results — {len(results)} matches.  _{scan_timestamp}_")
        if results:
            results_df = pd.DataFrame(results)
            results_df['Scan Date & Time'] = end_time.strftime('%Y-%m-%d %H:%M')
            # Format numbers for display
            display_df = results_df.copy()
            display_df["Close Price"] = display_df["Close Price"].map(lambda x: f"₹{x:,.2f}")
            display_df["% Change"] = display_df["% Change"].map(lambda x: f"{x:+.2f}%")
            display_df["Volume"] = display_df["Volume"].map(lambda x: f"{x:,}")
            st.dataframe(display_df.sort_values(by="% Change", ascending=False), use_container_width=True, hide_index=True)

            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"stock_scan_{end_time.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

            # Additional visuals in tabs
            tab1, tab2 = st.tabs(["📈 Price Sparklines", "🔎 Insights"])
            with tab1:
                st.write("Small recent-close sparklines for matched stocks (last 60 trading days where available).")
                # For each result, show mini sparkline + key metrics
                # We'll fetch 'data' for each symbol if present (it was downloaded earlier)
                cards_per_row = 4
                rows = (len(results) + cards_per_row - 1) // cards_per_row
                for r in range(rows):
                    cols = st.columns(cards_per_row)
                    for c in range(cards_per_row):
                        idx = r * cards_per_row + c
                        if idx >= len(results): break
                        item = results[idx]
                        symbol = item["Symbol"]
                        with cols[c]:
                            st.markdown(f'<div class="result-card"><b>{symbol}</b> — {item["Name"] if "Name" in item else ""}', unsafe_allow_html=True)
                            # try to get close series for sparkline
                            try:
                                df_sym = data.get(f"{symbol}.NS", pd.DataFrame()).dropna(how='all').copy()
                                if not df_sym.empty:
                                    df_sym.index = pd.to_datetime(df_sym.index)
                                    closes = df_sym['Close'].tail(60)
                                    if not closes.empty:
                                        st.line_chart(closes, height=120)
                            except Exception:
                                pass
                            st.write(f"Price: ₹{item['Close Price']:,}" if isinstance(item['Close Price'], (int,float)) else f"Price: {item['Close Price']}")
                            st.write(f"% Change: {item['% Change']:+.2f}%" if isinstance(item['% Change'], float) else f"% Change: {item['% Change']}")
                            st.markdown("</div>", unsafe_allow_html=True)

            with tab2:
                st.write("Top movers and volume leaders among matches.")
                if results:
                    df_matches = pd.DataFrame(results)
                    top_gainers = df_matches.sort_values(by="% Change", ascending=False).head(5)
                    top_volume = df_matches.sort_values(by="Volume", ascending=False).head(5)
                    col_a, col_b = st.columns(2)
                    col_a.metric("Top Gain %", ", ".join([f"{r['Symbol']} ({r['% Change']:+.2f}%)" for _, r in top_gainers.iterrows()]))
                    col_b.metric("Top Volume", ", ".join([f"{r['Symbol']} ({r['Volume']:,})" for _, r in top_volume.iterrows()]))
                else:
                    st.write("No matches to show insights for.")
        else:
            st.warning(f"⚠️ No stocks met all your criteria. {scan_timestamp}")
            st.info("Try relaxing some filters or use the 'Relaxed' preset on the sidebar.")

# If user hasn't run scan, show a friendly sample view
if not run_scan:
    with results_container:
        st.markdown("### 📌 Quick Preview")
        st.write("Click **Run Scan** to download data and run your filters. KPIs and insights will appear here.")
        sample_cols = st.columns(3)
        sample_cols[0].metric("Universe", f"{total_stocks_count}")
        sample_cols[1].metric("Filters Active", f"{sum(1 for v in active_filters.values() if v)} / {len(active_filters)}")
        sample_cols[2].metric("Last Load", status_message)

# End
st.markdown("---")
st.caption("UI upgraded — filter logic & functions preserved exactly. If you'd like an Ag-Grid result table, conditional coloring of rows, or export to Excel/XLSX, I can add that next.")
