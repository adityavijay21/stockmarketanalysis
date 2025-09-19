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
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------------------------------
# Timezone & Page Config
# -------------------------------------------------------
ist = pytz.timezone("Asia/Kolkata")
st.set_page_config(
    page_title="📈 NSE Elite Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/nse-screener',
        'Report a bug': 'https://github.com/yourusername/nse-screener/issues',
        'About': "# Elite NSE Stock Screener\nAdvanced technical analysis tool for Indian stock market"
    }
)

# -------------------------------------------------------
# Enhanced Theme Support & Global Styles
# -------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Mobile Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.5rem !important; }
        .metric-card { margin-bottom: 0.8rem; padding: 0.8rem; }
        .stColumns > div { padding: 0 0.25rem; }
        .results-header { font-size: 1.3rem !important; }
    }
    
    /* Dark Mode Support */
    .stApp {
        background: var(--background-color);
        color: var(--text-color);
    }
    
    /* Light Theme Variables */
    :root {
        --primary-color: #0074D9;
        --secondary-color: #001f3f;
        --accent-color: #2ECC71;
        --danger-color: #E74C3C;
        --warning-color: #F39C12;
        --background-color: #ffffff;
        --surface-color: #f8f9fa;
        --text-color: #333333;
        --text-secondary: #666666;
        --border-color: #dee2e6;
        --shadow: 0 4px 14px rgba(0,0,0,.15);
        --shadow-light: 0 2px 8px rgba(0,0,0,.08);
    }
    
    /* Dark Theme Variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #3498DB;
            --secondary-color: #2C3E50;
            --accent-color: #27AE60;
            --danger-color: #E67E22;
            --warning-color: #F39C12;
            --background-color: #0E1117;
            --surface-color: #1E1E1E;
            --text-color: #FAFAFA;
            --text-secondary: #B0B0B0;
            --border-color: #404040;
            --shadow: 0 4px 14px rgba(0,0,0,.3);
            --shadow-light: 0 2px 8px rgba(0,0,0,.2);
        }
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: var(--surface-color) !important;
        border-right: 1px solid var(--border-color);
    }
    
    section[data-testid="stSidebar"] > div {
        background: var(--surface-color) !important;
    }
    
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #fff, #e3f2fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 1.5rem;
        border: 2px solid var(--primary-color);
        background: var(--primary-color);
        color: white;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-light);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        background: white;
        color: var(--primary-color);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,116,217,0.3);
    }
    
    /* Enhanced Metric Cards */
    .metric-card {
        background: var(--surface-color);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-light);
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow);
    }
    
    .metric-card h3 {
        margin: 0 0 0.5rem 0;
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card p {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Results Header */
    .results-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-color);
        position: relative;
    }
    
    .results-header::after {
        content: '';
        position: absolute;
        bottom: -3px;
        left: 0;
        width: 60px;
        height: 3px;
        background: var(--accent-color);
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: var(--surface-color) !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Status Messages */
    .status-success {
        background: linear-gradient(135deg, var(--accent-color), #2ECC71);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .status-warning {
        background: linear-gradient(135deg, var(--warning-color), #F39C12);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Enhanced Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow-light);
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading-text {
        animation: pulse 2s infinite;
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Mobile Optimizations */
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin-bottom: 0.8rem;
        }
        
        .metric-card p {
            font-size: 1.5rem;
        }
    }
    
    /* Sidebar Enhancements */
    .sidebar-header {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* Custom Checkbox Styling */
    .stCheckbox > label {
        font-weight: 500;
        color: var(--text-color);
    }
    
    /* Enhanced Number Input */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid var(--border-color);
        transition: border-color 0.3s;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(0, 116, 217, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# -------------------------------------------------------
# Enhanced Header with Status
# -------------------------------------------------------
st.markdown('''
<div class="main-header">
    <h1>🚀 Elite NSE Stock Screener</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem;">
        Advanced Technical Analysis • Real-time Filtering • Professional Grade
    </p>
</div>
''', unsafe_allow_html=True)

# -------------------------------------------------------
# Enhanced Data Utilities
# -------------------------------------------------------
@st.cache_data(ttl=43200, show_spinner=False)
def load_nse_stocks() -> Tuple[Dict, str]:
    """Load NSE stock list with enhanced error handling"""
    try:
        url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        df = pd.read_csv(url, timeout=30)
        df = df[df["SERIES"] == "EQ"].dropna(subset=["SYMBOL", "NAME OF COMPANY"])
        df = df.sort_values("SYMBOL")  # Sort alphabetically
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
    """Download historical data in batches to avoid threading issues"""
    try:
        start = datetime.now(ist) - pd.DateOffset(years=2)
        batch_size = 100  # Process tickers in batches to avoid overloading
        all_data = {}
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            data = yf.download(
                batch_tickers,
                start=start.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                threads=False,  # Disable threading to avoid 'can't start new thread' error
                ignore_tz=True,
                progress=False
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                for ticker in batch_tickers:
                    if ticker in data:
                        all_data[ticker] = data[ticker]
            time.sleep(0.5)  # Small delay to avoid rate-limiting
            
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading historical data: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300, show_spinner=False)
def download_current_data(tickers):
    """Download current market data in batches"""
    try:
        today = datetime.now(ist)
        batch_size = 100
        all_data = {}
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i + batch_size]
            data = yf.download(
                batch_tickers,
                start=today.strftime("%Y-%m-%d"),
                end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                threads=False,  # Disable threading
                ignore_tz=True,
                progress=False
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                for ticker in batch_tickers:
                    if ticker in data:
                        all_data[ticker] = data[ticker]
            time.sleep(0.5)  # Small delay to avoid rate-limiting
            
        return pd.concat(all_data, axis=1) if all_data else pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading current data: {str(e)}")
        return pd.DataFrame()

def passes_filters(df, filters, volume_threshold, rsi_d, rsi_d_cross, rsi_w, rsi_w_cross, rsi_d_cross_below, rsi_w_cross_below):
    """Enhanced filter logic with cross-below support"""
    try:
        if df is None or df.empty or len(df) < 30:
            return False
        
        df = df.copy()
        latest = df.iloc[-1]
        
        # Basic filters
        if filters.get("Close > Open") and latest["Close"] <= latest["Open"]:
            return False
        
        # Enhanced volume filter with custom threshold
        if filters.get("Volume Filter") and latest["Volume"] < volume_threshold:
            return False
        
        # Range filters
        if len(df) >= 5:
            df["Range"] = df["High"] - df["Low"]
            for i in range(1, 5):
                if filters.get(f"Range > {i}d") and df["Range"].iloc[-1] <= df["Range"].iloc[-(i+1)]:
                    return False
        
        # Weekly and Monthly open filters
        if filters.get("Close > Weekly Open"):
            weekly_data = df.resample("W-MON").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if weekly_data.empty or latest["Close"] <= weekly_data["Open"].iloc[-1]:
                return False
        
        if filters.get("Close > Monthly Open"):
            monthly_data = df.resample("MS").agg({
                "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
            }).dropna()
            if monthly_data.empty or latest["Close"] <= monthly_data["Open"].iloc[-1]:
                return False
        
        # Enhanced Daily RSI filters
        if any([filters.get("Daily RSI >"), filters.get("Daily RSI crossed above"), filters.get("Daily RSI crossed below")]):
            rsi_daily = ta.RSI(df["Close"], 14).dropna()
            if rsi_daily.empty:
                return False
            
            if filters.get("Daily RSI >") and rsi_daily.iloc[-1] <= rsi_d:
                return False
            
            if filters.get("Daily RSI crossed above"):
                if len(rsi_daily) < 2 or not (rsi_daily.iloc[-2] < rsi_d_cross < rsi_daily.iloc[-1]):
                    return False
            
            if filters.get("Daily RSI crossed below"):
                if len(rsi_daily) < 2 or not (rsi_daily.iloc[-2] > rsi_d_cross_below > rsi_daily.iloc[-1]):
                    return False
        
        # Enhanced Weekly RSI filters
        weekly_close = df.resample("W-MON")["Close"].last().dropna()
        if any([filters.get("Weekly RSI >"), filters.get("Weekly RSI crossed above"), filters.get("Weekly RSI crossed below")]):
            if len(weekly_close) < 14:
                return False
            
            rsi_weekly = ta.RSI(weekly_close, 14).dropna()
            if rsi_weekly.empty:
                return False
            
            if filters.get("Weekly RSI >") and rsi_weekly.iloc[-1] <= rsi_w:
                return False
            
            if filters.get("Weekly RSI crossed above"):
                if len(rsi_weekly) < 2 or not (rsi_weekly.iloc[-2] < rsi_w_cross < rsi_weekly.iloc[-1]):
                    return False
            
            if filters.get("Weekly RSI crossed below"):
                if len(rsi_weekly) < 2 or not (rsi_weekly.iloc[-2] > rsi_w_cross_below > rsi_weekly.iloc[-1]):
                    return False
        
        return True
        
    except Exception as e:
        return False

# -------------------------------------------------------
# Enhanced Sidebar Filters
# -------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">🎯 Advanced Filters</div>', unsafe_allow_html=True)

filters = {}

with st.sidebar.expander("📈 Daily & Range Filters", expanded=True):
    filters["Close > Open"] = st.checkbox("✅ Close > Open (Bullish Candle)", True)
    
    st.markdown("**📊 Range Expansion Filters**")
    for i in range(1, 5):
        filters[f"Range > {i}d"] = st.checkbox(f"📏 Range > {i} Day(s) Ago", True)

with st.sidebar.expander("🗓️ Timeframe Breakouts", expanded=True):
    filters["Close > Weekly Open"] = st.checkbox("📅 Close > Weekly Open", True)
    filters["Close > Monthly Open"] = st.checkbox("🗓️ Close > Monthly Open", True)

with st.sidebar.expander("💹 Volume & RSI Analysis", expanded=True):
    # Enhanced Volume Filter
    st.markdown("**🔊 Volume Analysis**")
    filters["Volume Filter"] = st.checkbox("📊 Enable Volume Filter", True)
    volume_threshold = st.number_input(
        "💧 Volume Threshold", 
        min_value=10000, 
        max_value=50000000, 
        value=500000, 
        step=50000,
        format="%d",
        help="Minimum volume required for stock selection"
    )
    st.caption(f"Current threshold: {volume_threshold:,} shares")
    
    st.markdown("---")
    
    # Enhanced Daily RSI Filters
    st.markdown("**📈 Daily RSI Filters**")
    col1, col2 = st.columns(2)
    
    with col1:
        filters["Daily RSI >"] = st.checkbox("📊 Daily RSI >", True)
        filters["Daily RSI crossed above"] = st.checkbox("⬆️ RSI Crossed Above", True)
        filters["Daily RSI crossed below"] = st.checkbox("⬇️ RSI Crossed Below", False)
    
    with col2:
        rsi_d = st.number_input("RSI Threshold", 0.0, 100.0, 50.0, 0.1, key="rsi_d_thresh")
        rsi_d_cross = st.number_input("Cross Above Level", 0.0, 100.0, 50.0, 0.1, key="rsi_d_cross_up")
        rsi_d_cross_below = st.number_input("Cross Below Level", 0.0, 100.0, 70.0, 0.1, key="rsi_d_cross_down")
    
    st.markdown("---")
    
    # Enhanced Weekly RSI Filters
    st.markdown("**📅 Weekly RSI Filters**")
    col1, col2 = st.columns(2)
    
    with col1:
        filters["Weekly RSI >"] = st.checkbox("📊 Weekly RSI >", True)
        filters["Weekly RSI crossed above"] = st.checkbox("⬆️ Weekly Crossed Above", True)
        filters["Weekly RSI crossed below"] = st.checkbox("⬇️ Weekly Crossed Below", False)
    
    with col2:
        rsi_w = st.number_input("Weekly RSI Threshold", 0.0, 100.0, 45.0, 0.1, key="rsi_w_thresh")
        rsi_w_cross = st.number_input("Weekly Cross Above", 0.0, 100.0, 59.0, 0.1, key="rsi_w_cross_up")
        rsi_w_cross_below = st.number_input("Weekly Cross Below", 0.0, 100.0, 70.0, 0.1, key="rsi_w_cross_down")

# Filter Summary
active_filters = sum(filters.values())
if active_filters > 0:
    st.sidebar.success(f"✅ {active_filters} filters active")
else:
    st.sidebar.warning("⚠️ No filters selected")

# -------------------------------------------------------
# Enhanced Dashboard KPIs
# -------------------------------------------------------
with st.spinner("Loading NSE stock data..."):
    stocks, status_message = load_nse_stocks()

# Display status message as toast
if "successfully" in status_message:
    st.success(status_message)
elif "fallback" in status_message:
    st.warning(status_message)
else:
    st.info(status_message)

# Enhanced KPI Cards
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
        <p style="font-size: 1.2rem;">{market_status}</p>
    </div>
    ''', unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------------
# Enhanced Scan Logic
# -------------------------------------------------------
if st.button("🚀 Launch Elite Stock Scan", use_container_width=True, help="Scan all NSE stocks with your selected filters"):
    if active_filters == 0:
        st.warning("⚠️ Please select at least one filter before scanning!")
        st.stop()
    
    start_time = datetime.now(ist)
    tickers = [f"{symbol}.NS" for symbol in stocks.keys()]
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown('<p class="loading-text">🔄 Initializing scan engine...</p>', unsafe_allow_html=True)
        
        # Download historical data
        with st.spinner("📊 Downloading 2-year historical data..."):
            hist_data = download_all_data(tickers)
        
        if hist_data.empty:
            st.error("❌ Failed to download historical data. Please try again later.")
            st.stop()
        
        # Download current data
        with st.spinner("🔄 Fetching real-time market data..."):
            current_data = download_current_data(tickers)
    
    # Scanning process
    results = []
    status_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    # Stats tracking
    processed = 0
    errors = 0
    
    for i, (symbol, company_name) in enumerate(stocks.items()):
        # Update progress
        progress_percentage = (i + 1) / len(stocks)
        progress_bar.progress(progress_percentage)
        
        status_placeholder.markdown(f'''
        <div style="text-align: center; padding: 1rem; background: var(--surface-color); border-radius: 8px; margin: 1rem 0;">
            <h4 style="margin: 0; color: var(--primary-color);">🔍 Analyzing Stock {i+1:,} of {len(stocks):,}</h4>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-weight: 600;">{symbol} - {company_name[:50]}{'...' if len(company_name) > 50 else ''}</p>
            <p style="margin: 0.2rem 0 0 0; font-size: 0.9rem; color: var(--text-secondary);">Progress: {progress_percentage:.1%} • Found: {len(results)} matches</p>
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
                errors += 1
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
                except Exception:
                    pass
            
            # Apply filters
            if passes_filters(
                stock_df, filters, volume_threshold, 
                rsi_d, rsi_d_cross, rsi_w, rsi_w_cross, 
                rsi_d_cross_below, rsi_w_cross_below
            ):
                latest_data = stock_df.iloc[-1]
                
                # Calculate metrics
                price_change = ((latest_data["Close"] - latest_data["Open"]) / latest_data["Open"] * 100) if latest_data["Open"] else 0
                
                # Calculate RSI values
                daily_rsi = ta.RSI(stock_df["Close"], 14).iloc[-1] if len(stock_df) > 14 else np.nan
                
                weekly_close = stock_df.resample("W-MON")["Close"].last().dropna()
                weekly_rsi = ta.RSI(weekly_close, 14).iloc[-1] if len(weekly_close) > 14 else np.nan
                
                results.append({
                    "Symbol": symbol,
                    "Name": company_name,
                    "Close Price": f"₹{latest_data['Close']:.2f}",
                    "% Change": f"{price_change:+.2f}%",
                    "Volume": f"{int(latest_data['Volume']):,}",
                    "Daily RSI": f"{daily_rsi:.2f}" if pd.notna(daily_rsi) else "N/A",
                    "Weekly RSI": f"{weekly_rsi:.2f}" if pd.notna(weekly_rsi) else "N/A",
                })
        
        except Exception as e:
            errors += 1
            continue
        
        processed += 1
    
    # Display results (using previous code's display logic)
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
        <p>Errors: {errors:,}</p>
        <p>Duration: {scan_duration:.2f} seconds</p>
    </div>
    ''', unsafe_allow_html=True)
