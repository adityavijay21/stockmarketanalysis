import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import json
import time
import pytz
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict

# -------------------------------------------------------
# Timezone & Page Config
# -------------------------------------------------------
ist = pytz.timezone("Asia/Kolkata")
st.set_page_config(
    page_title="📈 NSE Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/nse-screener',
        'Report a bug': 'https://github.com/yourusername/nse-screener/issues',
        'About': "# NSE Stock Screener\nSimple technical analysis tool for Indian stock market"
    }
)

# -------------------------------------------------------
# Minimal Theme & Styles
# -------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* General Styling */
    .stApp {
        background: #ffffff;
        color: #333333;
    }
    
    /* Dark Mode Support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: #1e1e1e;
            color: #fafafa;
        }
        .metric-card, .sidebar-header, .results-header {
            background: #2a2a2a;
            color: #fafafa;
        }
        .stButton > button {
            background: #3498db;
            border-color: #3498db;
            color: white;
        }
        .stButton > button:hover {
            background: #2a80b9;
            border-color: #2a80b9;
        }
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: #f5f5f5;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Main Header */
    .main-header {
        background: #0074D9;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border: 2px solid #0074D9;
        background: #0074D9;
        color: white;
    }
    
    .stButton > button:hover {
        background: #005bb5;
        border-color: #005bb5;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        color: #666666;
        font-weight: 500;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
        color: #0074D9;
    }
    
    /* Results Header */
    .results-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333333;
        margin: 1.5rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #0074D9;
    }
    
    /* Sidebar Header */
    .sidebar-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0074D9;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Toggle Switch for Checkboxes */
    .stCheckbox > label {
        display: flex;
        align-items: center;
        font-weight: 500;
        color: #333333;
    }
    
    .stCheckbox > label > input {
        appearance: none;
        width: 40px;
        height: 20px;
        background: #ccc;
        border-radius: 10px;
        position: relative;
        margin-right: 10px;
        cursor: pointer;
    }
    
    .stCheckbox > label > input:checked {
        background: #0074D9;
    }
    
    .stCheckbox > label > input::before {
        content: '';
        position: absolute;
        width: 16px;
        height: 16px;
        background: white;
        border-radius: 50%;
        top: 2px;
        left: 2px;
        transition: transform 0.2s;
    }
    
    .stCheckbox > label > input:checked::before {
        transform: translateX(20px);
    }
    
    /* Number Input */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        padding: 0.5rem;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #0074D9;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f5f5f5;
        border-radius: 8px;
        font-weight: 500;
    }
    
    /* Dataframe */
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.4rem; }
        .metric-card { padding: 0.8rem; }
        .metric-card p { font-size: 1.2rem; }
        .results-header { font-size: 1.3rem; }
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown('''
<div class="main-header">
    <h1>📈 NSE Stock Screener</h1>
</div>
''', unsafe_allow_html=True)

# -------------------------------------------------------
# Data Utilities
# -------------------------------------------------------
@st.cache_data(ttl=43200, show_spinner=False)
def load_nse_stocks() -> Tuple[Dict, str]:
    """Load NSE stock list with enhanced error handling"""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url, timeout=30)
        df = df[df["SERIES"] == "EQ"].dropna(subset=["SYMBOL", "NAME OF COMPANY"])
        df = df.sort_values("SYMBOL")
        return dict(zip(df["SYMBOL"], df["NAME OF COMPANY"])), f"✅ Loaded {len(df):,} NSE stocks successfully"
    except Exception as e:
        # Fallback to local file
        p = Path("indian_stocks.json")
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                stocks_dict = {item["symbol"].replace(".NS", ""): item["name"] for item in data}
                return stocks_dict, f"📂 Loaded {len(stocks_dict):,} stocks from local file"
            except Exception:
                pass
        
        # Ultimate fallback
        fallback_stocks = {
            "RELIANCE": "Reliance Industries Limited",
            "TCS": "Tata Consultancy Services Limited",
            "HDFCBANK": "HDFC Bank Limited",
            "INFY": "Infosys Limited",
            "ICICIBANK": "ICICI Bank Limited",
            "HINDUNILVR": "Hindustan Unilever Limited",
            "ITC": "ITC Limited",
            "KOTAKBANK": "Kotak Mahindra Bank Limited",
            "LT": "Larsen & Toubro Limited",
            "AXISBANK": "Axis Bank Limited"
        }
        return fallback_stocks, "⚠️ Using fallback stock list (limited stocks available)"

@st.cache_data(ttl=1800, show_spinner=False)
def download_all_data(tickers):
    """Download historical data in batches with retry logic"""
    try:
        start = datetime.now(ist) - pd.DateOffset(years=2)
        batch_size = 50  # Reduced for stability
        all_data = {}
        errors = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            for attempt in range(3):  # Retry up to 3 times
                try:
                    data = yf.download(
                        batch_tickers,
                        start=start.strftime("%Y-%m-%d"),
                        group_by="ticker",
                        auto_adjust=True,
                        threads=False,
                        ignore_tz=True,
                        progress=False
                    )
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        for ticker in batch_tickers:
                            if ticker in data:
                                all_data[ticker] = data[ticker]
                    else:
                        errors.append(f"Empty data for batch {i//batch_size + 1}, attempt {attempt + 1}")
                    break  # Success, exit retry loop
                except Exception as e:
                    errors.append(f"Batch {i//batch_size + 1}, attempt {attempt + 1}: {str(e)}")
                    if attempt < 2:
                        time.sleep(2)  # Wait before retry
                    else:
                        errors.append(f"Failed batch {i//batch_size + 1} after 3 attempts")
            time.sleep(1)  # Delay between batches
        
        if errors:
            st.warning(f"⚠️ {len(errors)} errors during historical data download: {', '.join(errors[:3])}")
        
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading historical data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def download_current_data(tickers):
    """Download current market data in batches with retry logic"""
    try:
        today = datetime.now(ist)
        batch_size = 50
        all_data = {}
        errors = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            for attempt in range(3):
                try:
                    data = yf.download(
                        batch_tickers,
                        start=today.strftime("%Y-%m-%d"),
                        end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                        group_by="ticker",
                        auto_adjust=True,
                        threads=False,
                        ignore_tz=True,
                        progress=False
                    )
                    if isinstance(data, pd.DataFrame) and not data.empty:
                        for ticker in batch_tickers:
                            if ticker in data:
                                all_data[ticker] = data[ticker]
                    else:
                        errors.append(f"Empty data for batch {i//batch_size + 1}, attempt {attempt + 1}")
                    break
                except Exception as e:
                    errors.append(f"Batch {i//batch_size + 1}, attempt {attempt + 1}: {str(e)}")
                    if attempt < 2:
                        time.sleep(2)
                    else:
                        errors.append(f"Failed batch {i//batch_size + 1} after 3 attempts")
            time.sleep(1)
        
        if errors:
            st.warning(f"⚠️ {len(errors)} errors during current data download: {', '.join(errors[:3])}")
        
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading current data: {str(e)}")
        return pd.DataFrame()

def passes_filters(df, filters, volume_threshold, rsi_d, rsi_d_cross, rsi_w, rsi_w_cross, rsi_d_cross_below, rsi_w_cross_below, debug_mode=False):
    """Enhanced filter logic with debug logging"""
    try:
        if df is None or df.empty or len(df) < 30:
            return False, "Insufficient data (less than 30 days)"
        
        df = df.copy()
        latest = df.iloc[-1]
        
        # Volume filter
        if filters.get("Volume Filter") and latest["Volume"] < volume_threshold:
            return False, f"Volume ({latest['Volume']}) below threshold ({volume_threshold})"
        
        # Range filters
        if len(df) >= 5:
            df["Range"] = df["High"] - df["Low"]
            for i in range(1, 5):
                if filters.get(f"Range > {i}d") and df["Range"].iloc[-1] <= df["Range"].iloc[-(i+1)]:
                    return False, f"Range not greater than {i} day(s) ago"
        
        # Weekly and Monthly open filters
        if filters.get("Close > Weekly Open"):
            weekly_data = df.resample("W-MON").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if weekly_data.empty or latest["Close"] <= weekly_data["Open"].iloc[-1]:
                return False, "Close not above weekly open"
        
        if filters.get("Close > Monthly Open"):
            monthly_data = df.resample("MS").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if monthly_data.empty or latest["Close"] <= monthly_data["Open"].iloc[-1]:
                return False, "Close not above monthly open"
        
        # Daily RSI filters
        if any([filters.get("Daily RSI >"), filters.get("Daily RSI crossed above"), filters.get("Daily RSI crossed below")]):
            rsi_daily = ta.RSI(df["Close"], 14).dropna()
            if rsi_daily.empty:
                return False, "Empty daily RSI data"
            
            if debug_mode:
                st.write(f"Daily RSI: Current={rsi_daily.iloc[-1]:.2f}, Previous={rsi_daily.iloc[-2]:.2f if len(rsi_daily) >= 2 else 'N/A'}")
            
            if filters.get("Daily RSI >") and rsi_daily.iloc[-1] <= rsi_d:
                return False, f"Daily RSI ({rsi_daily.iloc[-1]:.2f}) <= threshold ({rsi_d})"
            
            if filters.get("Daily RSI crossed above"):
                if len(rsi_daily) < 2 or not (rsi_daily.iloc[-2] < rsi_d_cross < rsi_daily.iloc[-1]):
                    return False, f"Daily RSI did not cross above {rsi_d_cross}"
            
            if filters.get("Daily RSI crossed below"):
                if len(rsi_daily) < 2 or not (rsi_daily.iloc[-2] > rsi_d_cross_below > rsi_daily.iloc[-1]):
                    return False, f"Daily RSI did not cross below {rsi_d_cross_below}"
        
        # Weekly RSI filters
        weekly_close = df.resample("W-MON")["Close"].last().dropna()
        if any([filters.get("Weekly RSI >"), filters.get("Weekly RSI crossed above"), filters.get("Weekly RSI crossed below")]):
            if len(weekly_close) < 14:
                return False, f"Insufficient weekly data ({len(weekly_close)} weeks, need 14)"
            
            rsi_weekly = ta.RSI(weekly_close, 14).dropna()
            if rsi_weekly.empty:
                return False, "Empty weekly RSI data"
            
            if debug_mode:
                st.write(f"Weekly RSI: Current={rsi_weekly.iloc[-1]:.2f}, Previous={rsi_weekly.iloc[-2]:.2f if len(rsi_weekly) >= 2 else 'N/A'}")
            
            if filters.get("Weekly RSI >") and rsi_weekly.iloc[-1] <= rsi_w:
                return False, f"Weekly RSI ({rsi_weekly.iloc[-1]:.2f}) <= threshold ({rsi_w})"
            
            if filters.get("Weekly RSI crossed above"):
                if len(rsi_weekly) < 2 or not (rsi_weekly.iloc[-2] < rsi_w_cross < rsi_weekly.iloc[-1]):
                    return False, f"Weekly RSI did not cross above {rsi_w_cross}"
            
            if filters.get("Weekly RSI crossed below"):
                if len(rsi_weekly) < 2 or not (rsi_weekly.iloc[-2] > rsi_w_cross_below > rsi_weekly.iloc[-1]):
                    return False, f"Weekly RSI did not cross below {rsi_w_cross_below} (Previous: {rsi_weekly.iloc[-2]:.2f}, Current: {rsi_weekly.iloc[-1]:.2f})"
        
        return True, "Passed all filters"
        
    except Exception as e:
        return False, f"Exception in filters: {str(e)}"

# -------------------------------------------------------
# Sidebar Filters
# -------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">🎯 Filters</div>', unsafe_allow_html=True)

filters = {}

with st.sidebar.expander("📊 Range Filters", expanded=True):
    for i in range(1, 5):
        filters[f"Range > {i}d"] = st.checkbox(f"Range > {i} Day(s) Ago", True)

with st.sidebar.expander("🗓️ Timeframe Breakouts", expanded=True):
    filters["Close > Weekly Open"] = st.checkbox("Close > Weekly Open", True)
    filters["Close > Monthly Open"] = st.checkbox("Close > Monthly Open", True)

with st.sidebar.expander("💹 Volume & RSI Filters", expanded=True):
    # Volume Filter
    filters["Volume Filter"] = st.checkbox("Enable Volume Filter", True)
    volume_threshold = st.number_input(
        "Volume Threshold",
        min_value=10000,
        max_value=50000000,
        value=500000,
        step=50000,
        format="%d",
        help="Minimum volume required for stock selection"
    )
    st.caption(f"Threshold: {volume_threshold:,} shares")
    
    st.markdown("---")
    
    # Daily RSI Filters
    st.markdown("**Daily RSI Filters**")
    col1, col2 = st.columns(2)
    with col1:
        filters["Daily RSI >"] = st.checkbox("Daily RSI >", True)
        filters["Daily RSI crossed above"] = st.checkbox("RSI Crossed Above", True)
        filters["Daily RSI crossed below"] = st.checkbox("RSI Crossed Below", False)
    with col2:
        rsi_d = st.number_input("RSI Threshold", 0.0, 100.0, 50.0, 0.1, key="rsi_d_thresh")
        rsi_d_cross = st.number_input("Cross Above Level", 0.0, 100.0, 50.0, 0.1, key="rsi_d_cross_up")
        rsi_d_cross_below = st.number_input("Cross Below Level", 0.0, 100.0, 70.0, 0.1, key="rsi_d_cross_down")
    
    st.markdown("---")
    
    # Weekly RSI Filters
    st.markdown("**Weekly RSI Filters**")
    col1, col2 = st.columns(2)
    with col1:
        filters["Weekly RSI >"] = st.checkbox("Weekly RSI >", True)
        filters["Weekly RSI crossed above"] = st.checkbox("Weekly Crossed Above", True)
        filters["Weekly RSI crossed below"] = st.checkbox("Weekly Crossed Below", False)
    with col2:
        rsi_w = st.number_input("Weekly RSI Threshold", 0.0, 100.0, 45.0, 0.1, key="rsi_w_thresh")
        rsi_w_cross = st.number_input("Weekly Cross Above", 0.0, 100.0, 59.0, 0.1, key="rsi_w_cross_up")
        rsi_w_cross_below = st.number_input("Weekly Cross Below", 0.0, 100.0, 70.0, 0.1, key="rsi_w_cross_down")

# Debug Mode Toggle
debug_mode = st.sidebar.checkbox("Enable Debug Mode", False, help="Show detailed RSI and filter failure logs")

# Filter Summary
active_filters = sum(filters.values())
if active_filters > 0:
    st.sidebar.success(f"✅ {active_filters} filters active")
else:
    st.sidebar.warning("⚠️ No filters selected")

# -------------------------------------------------------
# Dashboard KPIs
# -------------------------------------------------------
with st.spinner("Loading NSE stock data..."):
    stocks, status_message = load_nse_stocks()

if "successfully" in status_message:
    st.success(status_message)
elif "fallback" in status_message:
    st.warning(status_message)
else:
    st.info(status_message)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f'''
    <div class="metric-card">
        <h3>Total Stocks</h3>
        <p>{len(stocks):,}</p>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown(f'''
    <div class="metric-card">
        <h3>Active Filters</h3>
        <p>{active_filters}</p>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown(f'''
    <div class="metric-card">
        <h3>Volume Threshold</h3>
        <p>{volume_threshold//1000}K</p>
    </div>
    ''', unsafe_allow_html=True)

with col4:
    current_time = datetime.now(ist)
    market_status = "🟢 Open" if 9 <= current_time.hour < 15 and current_time.weekday() < 5 else "🔴 Closed"
    st.markdown(f'''
    <div class="metric-card">
        <h3>Market Status</h3>
        <p>{market_status}</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Scan Logic
# -------------------------------------------------------
if st.button("🚀 Run Scan", use_container_width=True, help="Scan all NSE stocks with selected filters"):
    if active_filters == 0:
        st.warning("⚠️ Please select at least one filter before scanning!")
        st.stop()
    
    start_time = datetime.now(ist)
    tickers = [f"{symbol}.NS" for symbol in stocks.keys()]
    
    # Initialize live results table
    live_results_placeholder = st.empty()
    live_results = []
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown("**Initializing scan...**")
        
        # Download historical data
        with st.spinner("Downloading 2-year historical data..."):
            hist_data = download_all_data(tickers)
        
        if hist_data.empty:
            st.error("❌ Failed to download historical data. Please try again later.")
            st.stop()
        
        # Download current data
        with st.spinner("Fetching real-time market data..."):
            current_data = download_current_data(tickers)
    
    # Scanning process
    results = []
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    processed = 0
    errors = []
    
    for i, (symbol, company_name) in enumerate(stocks.items()):
        # Update progress
        progress_percentage = (i + 1) / len(stocks)
        progress_bar.progress(progress_percentage)
        
        status_placeholder.markdown(f'''
        <div style="text-align: center; padding: 0.5rem; background: #f5f5f5; border-radius: 8px;">
            <p style="margin: 0; font-weight: 500;">Scanning {i+1:,}/{len(stocks):,}: {symbol} - {company_name[:50]}{'...' if len(company_name) > 50 else ''}</p>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem;">Progress: {progress_percentage:.1%} • Found: {len(results)}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        try:
            # Get stock data
            ticker_symbol = f"{symbol}.NS"
            stock_df = hist_data.get(ticker_symbol, pd.DataFrame())
            
            if isinstance(stock_df, pd.DataFrame):
                stock_df = stock_df.dropna(how="all")
            else:
                stock_df = pd.DataFrame()
            
            if stock_df.empty:
                errors.append(f"{symbol}: Empty historical data")
                continue
            
            # Handle timezone
            if stock_df.index.tz is None:
                stock_df.index = stock_df.index.tz_localize("UTC")
            stock_df.index = stock_df.index.tz_convert("Asia/Kolkata")
            
            # Merge with current data
            current_stock = current_data.get(ticker_symbol, pd.DataFrame())
            if not current_stock.empty:
                try:
                    if current_stock.index.tz is None:
                        current_stock.index = current_stock.index.tz_localize("UTC")
                    current_stock.index = current_stock.index.tz_convert("Asia/Kolkata")
                    
                    latest = current_stock.iloc[-1]
                    current_date = pd.Timestamp(datetime.now(ist).date()).tz_localize("Asia/Kolkata")
                    
                    new_row = pd.DataFrame({
                        "Open": [latest["Open"]],
                        "High": [latest["High"]],
                        "Low": [latest["Low"]],
                        "Close": [latest["Close"]],
                        "Volume": [latest["Volume"]],
                    }, index=[current_date])
                    
                    if stock_df.index[-1].date() == current_date.date():
                        stock_df.iloc[-1] = new_row.iloc[0]
                    else:
                        stock_df = pd.concat([stock_df, new_row])
                except Exception as e:
                    errors.append(f"{symbol}: Error merging current data - {str(e)}")
            
            # Apply filters
            passed, reason = passes_filters(
                stock_df, filters, volume_threshold, 
                rsi_d, rsi_d_cross, rsi_w, rsi_w_cross, 
                rsi_d_cross_below, rsi_w_cross_below,
                debug_mode=debug_mode
            )
            
            if debug_mode:
                st.write(f"{symbol}: {reason}")
            
            if passed:
                latest_data = stock_df.iloc[-1]
                price_change = ((latest_data["Close"] - latest_data["Open"]) / latest_data["Open"] * 100) if latest_data["Open"] else 0
                daily_rsi = ta.RSI(stock_df["Close"], 14).iloc[-1] if len(stock_df) > 14 else np.nan
                weekly_close = stock_df.resample("W-MON")["Close"].last().dropna()
                weekly_rsi = ta.RSI(weekly_close, 14).iloc[-1] if len(weekly_close) > 14 else np.nan
                
                stock_data = {
                    "Symbol": symbol,
                    "Name": company_name,
                    "Close Price": f"₹{latest_data['Close']:.2f}",
                    "% Change": f"{price_change:+.2f}%",
                    "Volume": f"{int(latest_data['Volume']):,}",
                    "Daily RSI": f"{daily_rsi:.2f}" if pd.notna(daily_rsi) else "N/A",
                    "Weekly RSI": f"{weekly_rsi:.2f}" if pd.notna(weekly_rsi) else "N/A",
                }
                
                results.append(stock_data)
                live_results.append(stock_data)
                
                # Update live results table every 10 stocks to improve performance
                if len(live_results) % 10 == 0 or i == len(stocks) - 1:
                    live_results_df = pd.DataFrame(live_results)
                    live_results_placeholder.dataframe(
                        live_results_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                            "Name": st.column_config.TextColumn("Company Name", width="medium"),
                            "Close Price": st.column_config.TextColumn("Close Price", width="small"),
                            "% Change": st.column_config.TextColumn("% Change", width="small"),
                            "Volume": st.column_config.TextColumn("Volume", width="small"),
                            "Daily RSI": st.column_config.TextColumn("Daily RSI", width="small"),
                            "Weekly RSI": st.column_config.TextColumn("Weekly RSI", width="small"),
                        }
                    )
            
            else:
                errors.append(f"{symbol}: Failed filters - {reason}")
        
        except Exception as e:
            errors.append(f"{symbol}: Exception - {str(e)}")
            continue
        
        processed += 1
    
    # Display final results
    progress_bar.empty()
    status_placeholder.empty()
    end_time = datetime.now(ist)
    st.markdown('<p class="results-header">Scan Results</p>', unsafe_allow_html=True)
    stamp = f"Scan finished **{end_time.strftime('%Y-%m-%d %I:%M %p')}** (Duration: {str(end_time - start_time).split('.')[0]})"
    
    if results:
        st.success(f"✅ {len(results)} stocks matched all filters. {stamp}")
        df_out = pd.DataFrame(results)
        df_out["Scan Time"] = end_time.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            df_out,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                "Name": st.column_config.TextColumn("Company Name", width="medium"),
                "Close Price": st.column_config.TextColumn("Close Price", width="small"),
                "% Change": st.column_config.TextColumn("% Change", width="small"),
                "Volume": st.column_config.TextColumn("Volume", width="small"),
                "Daily RSI": st.column_config.TextColumn("Daily RSI", width="small"),
                "Weekly RSI": st.column_config.TextColumn("Weekly RSI", width="small"),
                "Scan Time": st.column_config.TextColumn("Scan Time", width="small"),
            }
        )
        st.download_button(
            "📥 Download CSV",
            df_out.to_csv(index=False).encode("utf-8"),
            f"scan_{end_time.strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    else:
        st.warning(f"⚠️ No stocks matched. {stamp}")
    
    # Scan statistics
    scan_duration = (end_time - start_time).total_seconds()
    st.markdown(f'''
    <div class="metric-card">
        <h3>Scan Statistics</h3>
        <p>Processed: {processed:,} stocks</p>
        <p>Errors: {len(errors):,}</p>
        <p>Duration: {scan_duration:.2f} seconds</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Display error log
    if errors:
        with st.expander("View Error Log"):
            st.write("**Error Log** (First 10 errors):")
            for error in errors[:10]:
                st.write(f"- {error}")
            if len(errors) > 10:
                st.write(f"... and {len(errors) - 10} more errors")
    
    # Debug mode summary
    if debug_mode:
        with st.expander("Debug Summary"):
            st.write(f"Processed {processed:,} stocks")
            st.write(f"Found {len(results):,} matches")
            st.write(f"Encountered {len(errors):,} errors")
            st.write("Check error log for details on filter failures")
