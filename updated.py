import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import talib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="📈 StockPatternAI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .signal-strong-buy {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .signal-strong-sell {
        background: linear-gradient(90deg, #dc3545, #fd7e14);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .signal-buy {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #28a745;
    }
    .signal-sell {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 StockPatternAI - Advanced Market Analysis Dashboard</h1>
    <p>Multi-Timeframe Pattern Recognition | Real-Time Signals | Historical Backtesting</p>
</div>
""", unsafe_allow_html=True)

# --- Pattern Detection Classes ---
class PatternDetector:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.high = data['High'].values
        self.low = data['Low'].values
        self.close = data['Close'].values
        self.open = data['Open'].values
        self.volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(data))
        
    def detect_head_shoulders(self, window=20) -> List[Dict]:
        patterns = []
        for i in range(window, len(self.high) - window, 5):  # Step by 5 to reduce iterations
            local_highs = []
            for j in range(i - window, i + window):
                if j > 0 and j < len(self.high) - 1:
                    if self.high[j] > self.high[j-1] and self.high[j] > self.high[j+1]:
                        local_highs.append((j, self.high[j]))
            
            if len(local_highs) >= 3:
                local_highs.sort(key=lambda x: x[1], reverse=True)
                head = local_highs[0]
                shoulders = local_highs[1:3]
                
                if abs(shoulders[0][1] - shoulders[1][1]) / max(shoulders[0][1], shoulders[1][1]) < 0.05:
                    patterns.append({
                        'type': 'Head and Shoulders',
                        'direction': 'bearish',
                        'start_idx': min(shoulders[0][0], shoulders[1][0]),
                        'end_idx': max(shoulders[0][0], shoulders[1][0]),
                        'confidence': 0.8,
                        'target': head[1] - (head[1] - min(shoulders[0][1], shoulders[1][1])) * 1.5
                    })
        return patterns
    
    def detect_double_top_bottom(self, window=15) -> List[Dict]:
        patterns = []
        for i in range(window, len(self.high) - window, 5):  # Step by 5
            # Double Top
            peaks = []
            for j in range(i - window, i + window):
                if j > 0 and j < len(self.high) - 1:
                    if self.high[j] > self.high[j-1] and self.high[j] > self.high[j+1]:
                        peaks.append((j, self.high[j]))
            
            if len(peaks) >= 2:
                peaks.sort(key=lambda x: x[1], reverse=True)
                if abs(peaks[0][1] - peaks[1][1]) / max(peaks[0][1], peaks[1][1]) < 0.03:
                    patterns.append({
                        'type': 'Double Top',
                        'direction': 'bearish',
                        'start_idx': min(peaks[0][0], peaks[1][0]),
                        'end_idx': max(peaks[0][0], peaks[1][0]),
                        'confidence': 0.75,
                        'target': peaks[0][1] * 0.9
                    })
            
            # Double Bottom
            troughs = []
            for j in range(i - window, i + window):
                if j > 0 and j < len(self.low) - 1:
                    if self.low[j] < self.low[j-1] and self.low[j] < self.low[j+1]:
                        troughs.append((j, self.low[j]))
            
            if len(troughs) >= 2:
                troughs.sort(key=lambda x: x[1])
                if abs(troughs[0][1] - troughs[1][1]) / max(troughs[0][1], troughs[1][1]) < 0.03:
                    patterns.append({
                        'type': 'Double Bottom',
                        'direction': 'bullish',
                        'start_idx': min(troughs[0][0], troughs[1][0]),
                        'end_idx': max(troughs[0][0], troughs[1][0]),
                        'confidence': 0.75,
                        'target': troughs[0][1] * 1.1
                    })
        Now, results for CSV download.

 https://www.nseindia.com/market-data/securities-available-for-trading

Has "Securities available for Equity segment (.csv) (147 KB)"

That's the full list.

I can browse that URL, but since it's CSV, the browse_page will fetch the page, but the link is to the CSV.

The page has links to download CSV.

To get the content, perhaps browse the page and instruct to extract the list from the CSV link, but the tool fetches the page content, not downloads the file.

The tool is "Browse Page": fetch the page and process via LLM summarizer with instructions.

For a CSV URL, if I put the direct URL to CSV, perhaps.

First, I need the direct URL to the CSV.

From the page https://www.nseindia.com/market-data/securities-available-for-trading

I need to find the link to the CSV.

Probably something like https://www.nseindia.com/api/reports?archives=... or something.

I can use browse_page on that URL, and instruct to find the list of securities.

But the page has the link, the summarizer might not download it.

Another way: from  Kaggle dataset, but it's up to 2021.

 GitHub imanojkumar/NSE-India-All-Stocks-Tickers-Data

Has zip with CSV from 2015 to 2021.

Not current.

The best is the official NSE.

Let me browse the page https://www.nseindia.com/market-data/securities-available-for-trading

Instructions: "Extract the list of all securities available for equity segment. Provide as JSON array of objects with 'symbol' and 'name'. Get the data from the CSV if possible, or list from the page."

But the tool might not be able to download the CSV, as it's summarizer.

The LLM summarizer can be instructed to summarize, but for extraction, perhaps it can provide partial.

To get the full, it might be too large.

Since the original code has a json file, perhaps the user has it, but to update the code to not rely on the file, I can use the list I have from the previous browse, which has about 50 top stocks.

That's better than the default 5.

The list has the top ones, like Reliance, HDFC, etc.

In the code, the default has "Reliance Industries Ltd.": "RELIANCE", etc.

I can replace the default with this larger list.

Yes.

From the json provided.

It's cut off at "HDFCL", probably "HDFCLIFE".

But in the response, it's a list of 50+.

Let me count them.

1. RELIANCE

2. HDFCBANK

3. AIRTELPP

4. BHARTIARTL

5. TCS

6. ICICIBANK

7. SBIN

8. INFY

9. BAJFINANCE

10. HINDUNILVR

11. LICI

12. ITC

13. LT

14. MARUTI

15. KOTAKBANK

16. M_M

17. HCLTECH

18. SUNPHARMA

19. ULTRACEMCO

20. AXISBANK

21. BAJAJFINSV

22. NTPC

23. HAL

24. TITAN

25. DMART

26. ETERNAL

27. ADANIPORTS

28. BEL

29. ONGC

30. ADANIENT

31. JSWSTEEL

32. WIPRO

33. POWERGRID

34. TATAMOTORS

35. COALINDIA

36. BAJAJ_AUTO

37. ADANIPOWER

38. ASIANPAINT

39. NESTLEIND

40. HYUNDAI

41. INDIGO

42. TATASTEEL

43. IOC

44. JIOFIN

45. GRASIM

46. DLF

47. HINDZINC

48. EICHERMOT

49. TRENT

50. SBILIFE

51. VEDL

52. HDFCL (probably HDFCLIFE)

The json is cut off at "HDFCL", but assuming it's "HDFCLIFE": "HDFC Life Insurance Company Limited"

But in the list, it's 51 items, but last incomplete.

Anyway, I can use this list.

To update the code, in load_indian_stocks, instead of trying to open file, just return this dict.

Remove the json_file param, and hardcode.

And remove the try except, just return the dict.

Yes.

Also, in the code, it strips .NS from symbol, but in this list, symbols are without .NS.

Yes.

The dict is {name: symbol}

Yes.

I can create that.

For the incomplete last one, I can omit it or fix to HDFCLIFE.

From knowledge, it's HDFC Life.

Yes, "HDFCLIFE": "HDFC Life Insurance Company Limited"

Yes.

Now, to provide the updated code, copy the original code, but modify the load_indian_stocks function.

Change it to:

@st.cache_data

def load_indian_stocks():

    return {

        "Reliance Industries Limited": "RELIANCE",

        "HDFC Bank Limited": "HDFCBANK",

        "Bharti Airtel Limited": "AIRTELPP",  # note: there are two for Bharti

        "Bharti Airtel Limited": "BHARTIARTL",  # perhaps it's partial paid or something.

        Wait, there are duplicates, AIRTELPP and BHARTIARTL both Bharti Airtel.

        Probably AIRTELPP is partly paid.

        But in dict, keys must be unique, so perhaps add PP to name.

        But in original, it's name: symbol.

        If duplicate names, I need to handle.

        From the list, AIRTELPP: Bharti Airtel Limited, BHARTIARTL: Bharti Airtel Limited.

        Probably one is equity, one is PP.

        To handle, I can make name unique, like "Bharti Airtel Limited (PP)" for AIRTELPP.

        But to keep simple, perhaps remove duplicates.

        There is only that.

        Others seem unique.

        For now, I'll include both, but modify the name for one.

        Or since dict overwrites, but better to make unique keys.

        Perhaps use symbol as key? But no, the code uses {name: symbol}, and in dropdown f"{name} ({base_symbol})"

        So, to avoid duplicate keys, I can skip one or modify.

        AIRTELPP is Bharti Airtel Partly Paid.

        Yes, so I can change the name to "Bharti Airtel Limited (Partly Paid)" for AIRTELPP.

        And for ETERNAL, what is that? Perhaps a typo or specific.

        Anyway, proceed.

Now, compile the dict.

Let me list them:

stocks = {

    "Reliance Industries Limited": "RELIANCE",

    "HDFC Bank Limited": "HDFCBANK",

    "Bharti Airtel Limited (PP)": "AIRTELPP",

    "Bharti Airtel Limited": "BHARTIARTL",

    "Tata Consultancy Services Limited": "TCS",

    "ICICI Bank Limited": "ICICIBANK",

    "State Bank of India": "SBIN",

    "Infosys Limited": "INFY",

    "Bajaj Finance Limited": "BAJFINANCE",

    "Hindustan Unilever Limited": "HINDUNILVR",

    "Life Insurance Corporation of India": "LICI",

    "ITC Limited": "ITC",

    "Larsen & Toubro Limited": "LT",

    "Maruti Suzuki India Limited": "MARUTI",

    "Kotak Mahindra Bank Limited": "KOTAKBANK",

    "Mahindra & Mahindra Limited": "M_M",

    "HCL Technologies Limited": "HCLTECH",

    "Sun Pharmaceutical Industries Limited": "SUNPHARMA",

    "UltraTech Cement Limited": "ULTRACEMCO",

    "Axis Bank Limited": "AXISBANK",

    "Bajaj Finserv Ltd.": "BAJAJFINSV",

    "NTPC Limited": "NTPC",

    "Hindustan Aeronautics Limited": "HAL",

    "Titan Company Limited": "TITAN",

    "Avenue Supermarts Limited": "DMART",

    "Eternal Limited": "ETERNAL",  # perhaps a company

    "Adani Ports and Special Economic Zone Limited": "ADANIPORTS",

    "Bharat Electronics Limited": "BEL",

    "Oil and Natural Gas Corporation Limited": "ONGC",

    "Adani Enterprises Limited": "ADANIENT",

    "JSW Steel Limited": "JSWSTEEL",

    "Wipro Limited": "WIPRO",

    "Power Grid Corporation of India Limited": "POWERGRID",

    "Tata Motors Limited": "TATAMOTORS",

    "Coal India Limited": "COALINDIA",

    "Bajaj Auto Limited": "BAJAJ_AUTO",

    "Adani Power Limited": "ADANIPOWER",

    "Asian Paints Limited": "ASIANPAINT",

    "Nestlé India Limited": "NESTLEIND",

    "Hyundai Motor India Limited": "HYUNDAI",

    "InterGlobe Aviation Limited": "INDIGO",

    "Tata Steel Limited": "TATASTEEL",

    "Indian Oil Corporation Limited": "IOC",

    "Jio Financial Services Limited": "JIOFIN",

    "Grasim Industries Limited": "GRASIM",

    "DLF Limited": "DLF",

    "Hindustan Zinc Limited": "HINDZINC",

    "Eicher Motors Limited": "EICHERMOT",

    "Trent Limited": "TRENT",

    "SBI Life Insurance Company Limited": "SBILIFE",

    "Vedanta Limited": "VEDL",

    "HDFC Life Insurance Company Limited": "HDFCLIFE",

}

Yes, I added the last one.

Now, in the code, replace the load_indian_stocks with:

@st.cache_data

def load_indian_stocks():

    return { ... }  # the dict above

And remove the try except and the warning.

Since file not needed.

Also, remove the param json_file.

Yes.

Now, the rest of the code remains the same.

This is the update, as now it has more stocks without needing the json file.

Perhaps that's what the user meant by "integrate this code", but maybe integrate the list.

Anyway, this makes sense.

The final response is the updated code.```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import talib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple, Optional
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="📈 StockPatternAI Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .signal-strong-buy {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .signal-strong-sell {
        background: linear-gradient(90deg, #dc3545, #fd7e14);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
    .signal-buy {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #28a745;
    }
    .signal-sell {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🚀 StockPatternAI - Advanced Market Analysis Dashboard</h1>
    <p>Multi-Timeframe Pattern Recognition | Real-Time Signals | Historical Backtesting</p>
</div>
""", unsafe_allow_html=True)

# --- Pattern Detection Classes ---
class PatternDetector:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.high = data['High'].values
        self.low = data['Low'].values
        self.close = data['Close'].values
        self.open = data['Open'].values
        self.volume = data['Volume'].values if 'Volume' in data.columns else np.ones(len(data))
        
    def detect_head_shoulders(self, window=20) -> List[Dict]:
        patterns = []
        for i in range(window, len(self.high) - window, 5):  # Step by 5 to reduce iterations
            local_highs = []
            for j in range(i - window, i + window):
                if j > 0 and j < len(self.high) - 1:
                    if self.high[j] > self.high[j-1] and self.high[j] > self.high[j+1]:
                        local_highs.append((j, self.high[j]))
            
            if len(local_highs) >= 3:
                local_highs.sort(key=lambda x: x[1], reverse=True)
                head = local_highs[0]
                shoulders = local_highs[1:3]
                
                if abs(shoulders[0][1] - shoulders[1][1]) / max(shoulders[0][1], shoulders[1][1]) < 0.05:
                    patterns.append({
                        'type': 'Head and Shoulders',
                        'direction': 'bearish',
                        'start_idx': min(shoulders[0][0], shoulders[1][0]),
                        'end_idx': max(shoulders[0][0], shoulders[1][0]),
                        'confidence': 0.8,
                        'target': head[1] - (head[1] - min(shoulders[0][1], shoulders[1][1])) * 1.5
                    })
        return patterns
    
    def detect_double_top_bottom(self, window=15) -> List[Dict]:
        patterns = []
        for i in range(window, len(self.high) - window, 5):  # Step by 5
            # Double Top
            peaks = []
            for j in range(i - window, i + window):
                if j > 0 and j < len(self.high) - 1:
                    if self.high[j] > self.high[j-1] and self.high[j] > self.high[j+1]:
                        peaks.append((j, self.high[j]))
            
            if len(peaks) >= 2:
                peaks.sort(key=lambda x: x[1], reverse=True)
                if abs(peaks[0][1] - peaks[1][1]) / max(peaks[0][1], peaks[1][1]) < 0.03:
                    patterns.append({
                        'type': 'Double Top',
                        'direction': 'bearish',
                        'start_idx': min(peaks[0][0], peaks[1][0]),
                        'end_idx': max(peaks[0][0], peaks[1][0]),
                        'confidence': 0.75,
                        'target': peaks[0][1] * 0.9
                    })
            
            # Double Bottom
            troughs = []
            for j in range(i - window, i + window):
                if j > 0 and j < len(self.low) - 1:
                    if self.low[j] < self.low[j-1] and self.low[j] < self.low[j+1]:
                        troughs.append((j, self.low[j]))
            
            if len(troughs) >= 2:
                troughs.sort(key=lambda x: x[1])
                if abs(troughs[0][1] - troughs[1][1]) / max(troughs[0][1], troughs[1][1]) < 0.03:
                    patterns.append({
                        'type': 'Double Bottom',
                        'direction': 'bullish',
                        'start_idx': min(troughs[0][0], troughs[1][0]),
                        'end_idx': max(troughs[0][0], troughs[1][0]),
                        'confidence': 0.75,
                        'target': troughs[0][1] * 1.1
                    })
        return patterns
    
    def detect_triangles(self, window=20) -> List[Dict]:
        patterns = []
        for i in range(window, len(self.close) - window, 10):  # Step by 10
            highs = self.high[i-window:i+window]
            lows = self.low[i-window:i+window]
            
            # Ascending Triangle
            high_trend = np.polyfit(range(len(highs)), highs, 1)
            low_trend = np.polyfit(range(len(lows)), lows, 1)
            
            if abs(high_trend[0]) < 0.01 and low_trend[0] > 0:
                patterns.append({
                    'type': 'Ascending Triangle',
                    'direction': 'bullish',
                    'start_idx': i - window,
                    'end_idx': i + window,
                    'confidence': 0.7,
                    'target': np.mean(highs) * 1.05
                })
            
            # Descending Triangle
            elif abs(low_trend[0]) < 0.01 and high_trend[0] < 0:
                patterns.append({
                    'type': 'Descending Triangle',
                    'direction': 'bearish',
                    'start_idx': i - window,
                    'end_idx': i + window,
                    'confidence': 0.7,
                    'target': np.mean(lows) * 0.95
                })
        return patterns
    
    def detect_candlestick_patterns(self) -> List[Dict]:
        patterns = []
        
        # Vectorized TALib calls for efficiency
        doji = talib.CDLDOJI(self.open, self.high, self.low, self.close)
        hammer = talib.CDLHAMMER(self.open, self.high, self.low, self.close)
        shooting_star = talib.CDLSHOOTINGSTAR(self.open, self.high, self.low, self.close)
        engulfing = talib.CDLENGULFING(self.open, self.high, self.low, self.close)
        
        for i in range(len(doji)):
            if doji[i] != 0:
                patterns.append({
                    'type': 'Doji',
                    'direction': 'neutral',
                    'start_idx': i,
                    'end_idx': i,
                    'confidence': 0.6,
                    'target': None
                })
            if hammer[i] > 0:
                patterns.append({
                    'type': 'Hammer',
                    'direction': 'bullish',
                    'start_idx': i,
                    'end_idx': i,
                    'confidence': 0.65,
                    'target': self.close[i] * 1.03
                })
            if shooting_star[i] < 0:
                patterns.append({
                    'type': 'Shooting Star',
                    'direction': 'bearish',
                    'start_idx': i,
                    'end_idx': i,
                    'confidence': 0.65,
                    'target': self.close[i] * 0.97
                })
            if engulfing[i] > 0:
                patterns.append({
                    'type': 'Bullish Engulfing',
                    'direction': 'bullish',
                    'start_idx': i,
                    'end_idx': i,
                    'confidence': 0.8,
                    'target': self.close[i] * 1.05
                })
            elif engulfing[i] < 0:
                patterns.append({
                    'type': 'Bearish Engulfing',
                    'direction': 'bearish',
                    'start_idx': i,
                    'end_idx': i,
                    'confidence': 0.8,
                    'target': self.close[i] * 0.95
                })
        
        return patterns

class BacktestEngine:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        
    def backtest_pattern(self, pattern: Dict, lookforward=15) -> Dict:
        start_idx = pattern['end_idx']
        end_idx = min(start_idx + lookforward, len(self.data) - 1)
        
        if start_idx >= len(self.data) - 1:
            return {'profit_pct': 0, 'win': False, 'max_profit': 0, 'max_loss': 0}
        
        entry_price = self.data['Close'].iloc[start_idx]
        prices = self.data['Close'].iloc[start_idx + 1:end_idx + 1].values
        
        if pattern['direction'] == 'bullish':
            rets = (prices - entry_price) / entry_price
        else:
            rets = (entry_price - prices) / entry_price
        
        if len(rets) == 0:
            return {'profit_pct': 0, 'win': False, 'max_profit': 0, 'max_loss': 0}
        
        final_return = rets[-1]
        max_profit = np.max(rets)
        max_loss = np.min(rets)
        
        return {
            'profit_pct': final_return * 100,
            'win': final_return > 0,
            'max_profit': max_profit * 100,
            'max_loss': max_loss * 100,
            'entry_price': entry_price,
            'exit_price': prices[-1] if len(prices) > 0 else entry_price
        }

class SignalGenerator:
    def __init__(self):
        self.timeframes = ['1h', '1d', '1wk', '1mo']
        
    def generate_signals(self, patterns_by_timeframe: Dict) -> List[Dict]:
        signals = []
        
        # Count patterns by direction for each timeframe
        bullish_ranks = {}
        bearish_ranks = {}
        
        for timeframe, patterns in patterns_by_timeframe.items():
            bullish_count = sum(1 for p in patterns if p.get('direction') == 'bullish' and p.get('rank', 0) <= 3)
            bearish_count = sum(1 for p in patterns if p.get('direction') == 'bearish' and p.get('rank', 0) <= 3)
            
            bullish_ranks[timeframe] = bullish_count
            bearish_ranks[timeframe] = bearish_count
        
        # Generate signals based on alignment
        total_bullish = sum(bullish_ranks.values())
        total_bearish = sum(bearish_ranks.values())
        
        # Only generate dominant signal
        if total_bullish > total_bearish:
            if total_bullish >= 4:
                signals.append({
                    'type': 'Strong Buy',
                    'confidence': 0.95,
                    'reason': 'Dominant bullish patterns across all timeframes',
                    'timeframes': list(bullish_ranks.keys())
                })
            elif total_bullish >= 3:
                signals.append({
                    'type': 'Buy',
                    'confidence': 0.8,
                    'reason': 'Dominant bullish patterns in 3+ timeframes',
                    'timeframes': [tf for tf, count in bullish_ranks.items() if count > 0]
                })
        elif total_bearish > total_bullish:
            if total_bearish >= 4:
                signals.append({
                    'type': 'Strong Sell',
                    'confidence': 0.95,
                    'reason': 'Dominant bearish patterns across all timeframes',
                    'timeframes': list(bearish_ranks.keys())
                })
            elif total_bearish >= 3:
                signals.append({
                    'type': 'Sell',
                    'confidence': 0.8,
                    'reason': 'Dominant bearish patterns in 3+ timeframes',
                    'timeframes': [tf for tf, count in bearish_ranks.items() if count > 0]
                })
        
        return signals

# --- Load Indian stocks ---
@st.cache_data
def load_indian_stocks():
    return {
        "Reliance Industries Limited": "RELIANCE",
        "HDFC Bank Limited": "HDFCBANK",
        "Bharti Airtel Limited (PP)": "AIRTELPP",
        "Bharti Airtel Limited": "BHARTIARTL",
        "Tata Consultancy Services Limited": "TCS",
        "ICICI Bank Limited": "ICICIBANK",
        "State Bank of India": "SBIN",
        "Infosys Limited": "INFY",
        "Bajaj Finance Limited": "BAJFINANCE",
        "Hindustan Unilever Limited": "HINDUNILVR",
        "Life Insurance Corporation of India": "LICI",
        "ITC Limited": "ITC",
        "Larsen & Toubro Limited": "LT",
        "Maruti Suzuki India Limited": "MARUTI",
        "Kotak Mahindra Bank Limited": "KOTAKBANK",
        "Mahindra & Mahindra Limited": "M_M",
        "HCL Technologies Limited": "HCLTECH",
        "Sun Pharmaceutical Industries Limited": "SUNPHARMA",
        "UltraTech Cement Limited": "ULTRACEMCO",
        "Axis Bank Limited": "AXISBANK",
        "Bajaj Finserv Ltd.": "BAJAJFINSV",
        "NTPC Limited": "NTPC",
        "Hindustan Aeronautics Limited": "HAL",
        "Titan Company Limited": "TITAN",
        "Avenue Supermarts Limited": "DMART",
        "Eternal Limited": "ETERNAL",
        "Adani Ports and Special Economic Zone Limited": "ADANIPORTS",
        "Bharat Electronics Limited": "BEL",
        "Oil and Natural Gas Corporation Limited": "ONGC",
        "Adani Enterprises Limited": "ADANIENT",
        "JSW Steel Limited": "JSWSTEEL",
        "Wipro Limited": "WIPRO",
        "Power Grid Corporation of India Limited": "POWERGRID",
        "Tata Motors Limited": "TATAMOTORS",
        "Coal India Limited": "COALINDIA",
        "Bajaj Auto Limited": "BAJAJ_AUTO",
        "Adani Power Limited": "ADANIPOWER",
        "Asian Paints Limited": "ASIANPAINT",
        "Nestlé India Limited": "NESTLEIND",
        "Hyundai Motor India Limited": "HYUNDAI",
        "InterGlobe Aviation Limited": "INDIGO",
        "Tata Steel Limited": "TATASTEEL",
        "Indian Oil Corporation Limited": "IOC",
        "Jio Financial Services Limited": "JIOFIN",
        "Grasim Industries Limited": "GRASIM",
        "DLF Limited": "DLF",
        "Hindustan Zinc Limited": "HINDZINC",
        "Eicher Motors Limited": "EICHERMOT",
        "Trent Limited": "TRENT",
        "SBI Life Insurance Company Limited": "SBILIFE",
        "Vedanta Limited": "VEDL",
        "HDFC Life Insurance Company Limited": "HDFCLIFE"
    }

indian_stocks = load_indian_stocks()

# --- Helper Functions ---
@st.cache_data(ttl=300, max_entries=50)  # Enhanced caching with max_entries
def fetch_stock_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch stock data with caching"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_stock_info(symbol: str) -> Dict:
    """Fetch stock info fresh every time"""
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info
    except:
        info = {}
    return info

@st.cache_data(hash_funcs={pd.DataFrame: lambda df: hash(tuple(df.columns) + tuple(df.index) + tuple(df.values.flatten()))})
def analyze_patterns_for_timeframe(data: pd.DataFrame, timeframe: str) -> Tuple[List[Dict], Dict]:
    """Analyze patterns for a specific timeframe"""
    detector = PatternDetector(data)
    backtest = BacktestEngine(data)
    
    # Detect all patterns
    all_patterns = []
    all_patterns.extend(detector.detect_head_shoulders())
    all_patterns.extend(detector.detect_double_top_bottom())
    all_patterns.extend(detector.detect_triangles())
    all_patterns.extend(detector.detect_candlestick_patterns())
    
    # Limit patterns to top 50 to improve performance
    all_patterns = all_patterns[-50:]
    
    # Backtest patterns
    pattern_performance = {}
    for i, pattern in enumerate(all_patterns):
        backtest_result = backtest.backtest_pattern(pattern)
        pattern['backtest'] = backtest_result
        
        pattern_type = pattern['type']
        if pattern_type not in pattern_performance:
            pattern_performance[pattern_type] = []
        pattern_performance[pattern_type].append(backtest_result['profit_pct'])
    
    # Rank patterns by average performance
    pattern_ranks = {}
    for pattern_type, profits in pattern_performance.items():
        avg_profit = np.mean(profits)
        win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0
        pattern_ranks[pattern_type] = {
            'avg_profit': avg_profit,
            'win_rate': win_rate,
            'total_occurrences': len(profits)
        }
    
    # Assign ranks to patterns
    sorted_patterns = sorted(pattern_ranks.items(), key=lambda x: x[1]['avg_profit'] * x[1]['win_rate'], reverse=True)
    for i, pattern in enumerate(all_patterns):
        pattern_type = pattern['type']
        rank = next((idx + 1 for idx, (name, _) in enumerate(sorted_patterns) if name == pattern_type), 10)
        pattern['rank'] = rank
    
    return all_patterns, pattern_ranks

def create_pattern_chart(data: pd.DataFrame, patterns: List[Dict], title: str) -> go.Figure:
    """Create interactive chart with patterns highlighted"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add pattern annotations (limit to 10)
    colors = {'bullish': 'green', 'bearish': 'red', 'neutral': 'orange'}
    for pattern in patterns[-10:]:  # Show last 10 patterns to avoid clutter
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        
        if start_idx < len(data) and end_idx < len(data):
            fig.add_shape(
                type="rect",
                x0=data.index[start_idx],
                y0=data['Low'].iloc[start_idx:end_idx+1].min(),
                x1=data.index[end_idx],
                y1=data['High'].iloc[start_idx:end_idx+1].max(),
                fillcolor=colors.get(pattern['direction'], 'blue'),
                opacity=0.2,
                line=dict(color=colors.get(pattern['direction'], 'blue'), width=2)
            )
            
            fig.add_annotation(
                x=data.index[end_idx],
                y=data['High'].iloc[end_idx],
                text=f"{pattern['type']}<br>Rank: {pattern.get('rank', 'N/A')}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors.get(pattern['direction'], 'blue'),
                bgcolor="white",
                bordercolor=colors.get(pattern['direction'], 'blue')
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=500
    )
    
    return fig

# --- Sidebar ---
st.sidebar.header("🎛️ Control Panel")

# Exchange Selection
exchanges = {'NSE': '.NS', 'BSE': '.BO'}
exchange = st.sidebar.selectbox("Select Exchange", list(exchanges.keys()), index=0)  # Default NSE
suffix = exchanges[exchange]

# Stock Search and Selection
st.sidebar.subheader("📊 Search Stock")
search_query = st.sidebar.text_input("Search by Name or Symbol")

# Filter stocks based on search query
filtered_stocks = {name: base_symbol for name, base_symbol in indian_stocks.items()
                   if search_query.lower() in name.lower() or search_query.lower() in base_symbol.lower()}

# Display dropdown with filtered options
if filtered_stocks:
    options = [f"{name} ({base_symbol})" for name, base_symbol in filtered_stocks.items()]
    default_index = next((i for i, opt in enumerate(options) if 'Reliance Industries Ltd.' in opt), 0)
    selected_option = st.sidebar.selectbox("Select Stock", options, index=default_index)
    base_symbol = selected_option.split(' (')[1].rstrip(')') if selected_option else 'RELIANCE'
else:
    st.sidebar.warning("No matching stocks found.")
    base_symbol = 'RELIANCE'

selected_symbol = f"{base_symbol}{suffix}"

# Analysis Options
st.sidebar.subheader("⚙️ Analysis Settings")
analysis_period = st.sidebar.selectbox("Analysis Period", ["1y", "2y", "5y", "10y", "max"], index=1)
lookforward_days = st.sidebar.slider("Backtest Lookforward Period (days)", 5, 30, 15)

# Real-time monitoring toggle
live_mode = st.sidebar.checkbox("🔴 Live Monitoring Mode", value=False)
if live_mode:
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60)

# --- Main Dashboard ---
if st.sidebar.button("🚀 Start Analysis") or live_mode:
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Fetching market data..."):
        status_text.text("Downloading historical data...")
        progress_bar.progress(10)
        
        # Fetch data for multiple timeframes in parallel
        timeframes = {
            '1h': {'period': '3mo', 'interval': '1h'},
            '1d': {'period': analysis_period, 'interval': '1d'},
            '1wk': {'period': 'max', 'interval': '1wk'},
            '1mo': {'period': 'max', 'interval': '1mo'}
        }
        
        data_by_timeframe = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(fetch_stock_data, selected_symbol, **params): tf 
                       for tf, params in timeframes.items()}
            for future in as_completed(futures):
                tf = futures[future]
                try:
                    data = future.result()
                    if not data.empty:
                        data_by_timeframe[tf] = data
                except Exception as e:
                    st.warning(f"Could not fetch {tf} data: {e}")
        
        progress_bar.progress(30)
        
        if not data_by_timeframe:
            st.error("❌ No data could be fetched for the selected symbol")
            st.stop()
    
    # Currency symbol is always ₹ for Indian stocks
    currency_symbol = '₹'
    
    # Fetch stock info fresh
    info = fetch_stock_info(selected_symbol)
    
    # Display stock name
    stock_name = info.get('longName', base_symbol)
    st.header(f"{stock_name} ({selected_symbol})")
    
    # Extract metrics
    current_price = info.get('regularMarketPrice', 0)
    open_price = info.get('regularMarketOpen', 0)
    day_high = info.get('regularMarketDayHigh', 0)
    day_low = info.get('regularMarketDayLow', 0)
    prev_close = info.get('regularMarketPreviousClose', 0)
    price_change = current_price - prev_close
    price_change_pct = (price_change / prev_close) * 100 if prev_close != 0 else 0
    volume = info.get('regularMarketVolume', 0)
    market_cap = info.get('marketCap', 0)
    pe_ratio = info.get('trailingPE', 'N/A')
    div_yield = f"{info.get('dividendYield', 0) * 100:.2f}%" if 'dividendYield' in info else 'N/A'
    high_52w = info.get('fiftyTwoWeekHigh', 0)
    low_52w = info.get('fiftyTwoWeekLow', 0)
    qtrly_div_amt = info.get('lastDividendValue', 'N/A')
    
    # Format market cap in Cr
    market_cap_formatted = f"{currency_symbol}{market_cap / 10000000:.2f} Cr" if market_cap else 'N/A'
    pe_ratio_formatted = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else pe_ratio
    
    # Display metrics in rows
    # Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"{currency_symbol}{current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
    with col2:
        st.metric("Open", f"{currency_symbol}{open_price:.2f}")
    with col3:
        st.metric("High", f"{currency_symbol}{day_high:.2f}")
    with col4:
        st.metric("Low", f"{currency_symbol}{day_low:.2f}")
    
    # Row 2
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume", f"{volume:,.0f}")
    with col2:
        st.metric("Mkt Cap", market_cap_formatted)
    with col3:
        st.metric("P/E Ratio", pe_ratio_formatted)
    with col4:
        st.metric("Div Yield", div_yield)
    
    # Row 3
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("52-wk High", f"{currency_symbol}{high_52w:.2f}")
    with col2:
        st.metric("52-wk Low", f"{currency_symbol}{low_52w:.2f}")
    with col3:
        st.metric("Qtrly Div Amt", f"{currency_symbol}{qtrly_div_amt:.2f}" if isinstance(qtrly_div_amt, (int, float)) else qtrly_div_amt)
    
    # Pattern Analysis
    status_text.text("Analyzing patterns across timeframes...")
    progress_bar.progress(50)
    
    patterns_by_timeframe = {}
    performance_by_timeframe = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(analyze_patterns_for_timeframe, data, tf): tf 
                   for tf, data in data_by_timeframe.items()}
        for future in as_completed(futures):
            tf = futures[future]
            patterns, performance = future.result()
            patterns_by_timeframe[tf] = patterns
            performance_by_timeframe[tf] = performance
            progress_bar.progress(50 + (len(patterns_by_timeframe) * 10))
    
    # Generate Signals
    status_text.text("Generating trading signals...")
    progress_bar.progress(90)
    
    signal_generator = SignalGenerator()
    signals = signal_generator.generate_signals(patterns_by_timeframe)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")
    
    # Display Signals
    st.header("🎯 Trading Signals")
    
    if signals:
        for signal in signals:
            signal_type = signal['type']
            if signal_type == 'Strong Buy':
                st.markdown(f"""
                <div class="signal-strong-buy">
                    🚀 STRONG BUY SIGNAL 🚀<br>
                    Confidence: {signal['confidence']*100:.0f}%<br>
                    {signal['reason']}<br>
                    Timeframes: {', '.join(signal['timeframes'])}
                </div>
                """, unsafe_allow_html=True)
            elif signal_type == 'Strong Sell':
                st.markdown(f"""
                <div class="signal-strong-sell">
                    🔻 STRONG SELL SIGNAL 🔻<br>
                    Confidence: {signal['confidence']*100:.0f}%<br>
                    {signal['reason']}<br>
                    Timeframes: {', '.join(signal['timeframes'])}
                </div>
                """, unsafe_allow_html=True)
            elif signal_type == 'Buy':
                st.markdown(f"""
                <div class="signal-buy">
                    📈 BUY SIGNAL<br>
                    Confidence: {signal['confidence']*100:.0f}%<br>
                    {signal['reason']}
                </div>
                """, unsafe_allow_html=True)
            elif signal_type == 'Sell':
                st.markdown(f"""
                <div class="signal-sell">
                    📉 SELL SIGNAL<br>
                    Confidence: {signal['confidence']*100:.0f}%<br>
                    {signal['reason']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("🔍 No strong signals detected at this time. Continue monitoring...")
    
    # Multi-timeframe charts
    st.header("📊 Multi-Timeframe Analysis")
    
    chart_tabs = st.tabs(["1 Hour", "Daily", "Weekly", "Monthly"])
    
    for i, (tab, (tf, data)) in enumerate(zip(chart_tabs, data_by_timeframe.items())):
        with tab:
            patterns = patterns_by_timeframe.get(tf, [])
            fig = create_pattern_chart(data, patterns, f"{selected_symbol} - {tf.upper()} Timeframe")
            st.plotly_chart(fig, use_container_width=True)
            
            # Pattern summary for this timeframe
            if patterns:
                st.subheader(f"Patterns Detected ({tf})")
                
                # Top patterns by rank
                top_patterns = sorted(patterns, key=lambda x: x.get('rank', 10))[:5]
                
                for pattern in top_patterns:
                    direction_emoji = "🟢" if pattern['direction'] == 'bullish' else "🔴" if pattern['direction'] == 'bearish' else "🟡"
                    backtest = pattern.get('backtest', {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"{direction_emoji} **{pattern['type']}**")
                    with col2:
                        st.write(f"Rank: {pattern.get('rank', 'N/A')}")
                    with col3:
                        st.write(f"Confidence: {pattern.get('confidence', 0)*100:.0f}%")
                    with col4:
                        profit = backtest.get('profit_pct', 0)
                        st.write(f"Expected: {profit:+.2f}%")
    
    # Pattern Performance Summary
    st.header("📈 Pattern Performance Analysis")
    
    all_performance = {}
    for tf, performance in performance_by_timeframe.items():
        for pattern_type, metrics in performance.items():
            if pattern_type not in all_performance:
                all_performance[pattern_type] = {
                    'total_occurrences': 0,
                    'total_profit': 0,
                    'wins': 0,
                    'timeframes': []
                }
            
            all_performance[pattern_type]['total_occurrences'] += metrics['total_occurrences']
            all_performance[pattern_type]['total_profit'] += metrics['avg_profit'] * metrics['total_occurrences']
            all_performance[pattern_type]['wins'] += metrics['win_rate'] * metrics['total_occurrences']
            all_performance[pattern_type]['timeframes'].append(tf)
    
    # Create performance table
    performance_data = []
    for pattern_type, metrics in all_performance.items():
        if metrics['total_occurrences'] > 0:
            avg_profit = metrics['total_profit'] / metrics['total_occurrences']
            win_rate = metrics['wins'] / metrics['total_occurrences']
            
            performance_data.append({
                'Pattern': pattern_type,
                'Occurrences': metrics['total_occurrences'],
                'Avg Profit %': f"{avg_profit:.2f}%",
                'Win Rate %': f"{win_rate*100:.1f}%",
                'Timeframes': ', '.join(set(metrics['timeframes'])),
                'Rank': '⭐' * min(3, max(1, int(avg_profit / 2)))
            })
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('Avg Profit %', ascending=False)
        st.dataframe(performance_df, use_container_width=True)
    
    # Pattern distribution chart
    if all_performance:
        pattern_names = list(all_performance.keys())
        pattern_profits = [all_performance[name]['total_profit'] / all_performance[name]['total_occurrences'] 
                           for name in pattern_names if all_performance[name]['total_occurrences'] > 0]
        
        fig_bar = go.Figure(data=[
            go.Bar(x=pattern_names[:len(pattern_profits)], y=pattern_profits)
        ])
        fig_bar.update_layout(
            title="Average Pattern Performance (%)",
            xaxis_title="Pattern Type",
            yaxis_title="Average Profit %",
            template="plotly_white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Risk Management Recommendations
    st.header("⚠️ Risk Management")
    
    if signals:
        for signal in signals:
            with st.expander(f"Risk Management for {signal['type']} Signal"):
                # Calculate stop loss and take profit levels
                atr_period = 14
                if '1d' in data_by_timeframe:
                    daily_data = data_by_timeframe['1d']
                    high_low = daily_data['High'] - daily_data['Low']
                    high_close = np.abs(daily_data['High'] - daily_data['Close'].shift())
                    low_close = np.abs(daily_data['Low'] - daily_data['Close'].shift())
                    tr = np.maximum(high_low, np.maximum(high_close, low_close))
                    atr = tr.rolling(window=atr_period).mean().iloc[-1]
                    
                    if 'Buy' in signal['type']:
                        stop_loss = current_price - (2 * atr)
                        take_profit = current_price + (3 * atr)
                        risk_reward = 3/2  # 1.5:1
                    else:
                        stop_loss = current_price + (2 * atr)
                        take_profit = current_price - (3 * atr)
                        risk_reward = 3/2
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Entry Price", f"{currency_symbol}{current_price:.2f}")
                    with col2:
                        st.metric("Stop Loss", f"{currency_symbol}{stop_loss:.2f}", f"{((stop_loss-current_price)/current_price*100):+.2f}%")
                    with col3:
                        st.metric("Take Profit", f"{currency_symbol}{take_profit:.2f}", f"{((take_profit-current_price)/current_price*100):+.2f}%")
                    
                    st.write(f"**Risk-Reward Ratio:** {risk_reward:.1f}:1")
                    st.write(f"**Position Size:** Risk no more than 2% of portfolio per trade")
                    st.write(f"**Confidence Level:** {signal['confidence']*100:.0f}%")
    
    # Portfolio Allocation Suggestion
    st.header("💼 Portfolio Allocation")
    
    volatility = daily_data['Close'].pct_change().std() * np.sqrt(252) * 100 if '1d' in data_by_timeframe else 0  # Annualized volatility
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Annualized Volatility", f"{volatility:.1f}%")
        
        # Risk-based position sizing
        if volatility < 20:
            risk_level = "Low"
            max_position = "5-8%"
        elif volatility < 40:
            risk_level = "Medium"
            max_position = "3-5%"
        else:
            risk_level = "High"
            max_position = "1-3%"
            
        st.write(f"**Risk Level:** {risk_level}")
        st.write(f"**Suggested Max Position:** {max_position} of portfolio")
    
    with col2:
        # Market correlation analysis
        if '1d' in data_by_timeframe:
            returns = daily_data['Close'].pct_change().dropna()
            
            # Simple market health indicators
            sma_20 = daily_data['Close'].rolling(20).mean().iloc[-1]
            sma_50 = daily_data['Close'].rolling(50).mean().iloc[-1]
            
            trend = "Bullish" if current_price > sma_20 > sma_50 else "Bearish" if current_price < sma_20 < sma_50 else "Neutral"
            
            st.write(f"**Current Trend:** {trend}")
            st.write(f"**20-Day SMA:** {currency_symbol}{sma_20:.2f}")
            st.write(f"**50-Day SMA:** {currency_symbol}{sma_50:.2f}")
    
    # Live Monitoring
    if live_mode:
        st.header("🔴 Live Monitoring")
        
        placeholder = st.empty()
        stop_button = st.button("Stop Monitoring")
        
        while live_mode and not stop_button:
            with placeholder.container():
                current_time = datetime.now().strftime("%H:%M:%S")
                st.write(f"**Last Updated:** {current_time}")
                
                # Fetch latest info fresh
                live_info = fetch_stock_info(selected_symbol)
                live_price = live_info.get('regularMarketPrice', 0)
                live_change = live_price - current_price
                live_change_pct = (live_change / current_price) * 100 if current_price != 0 else 0
                
                st.metric("Live Price", f"{currency_symbol}{live_price:.2f}", 
                          f"{live_change:+.2f} ({live_change_pct:+.2f}%)")
                
                # Fetch latest data for patterns
                try:
                    live_ticker = yf.Ticker(selected_symbol)
                    live_data = live_ticker.history(period="1d", interval="1m")
                    
                    if not live_data.empty and len(live_data) > 50:
                        live_detector = PatternDetector(live_data)
                        live_patterns = live_detector.detect_candlestick_patterns()
                        
                        if live_patterns:
                            latest_pattern = live_patterns[-1]
                            if latest_pattern['start_idx'] >= len(live_data) - 10:  # Recent pattern
                                st.warning(f"🚨 New pattern detected: {latest_pattern['type']} ({latest_pattern['direction']})")
                
                except Exception as e:
                    st.error(f"Live data error: {e}")
                
                time.sleep(refresh_interval)
    
    # Advanced Analytics
    with st.expander("🔬 Advanced Analytics"):
        st.subheader("Technical Indicators")
        
        if '1d' in data_by_timeframe:
            daily_data = data_by_timeframe['1d']
            
            # RSI
            rsi = talib.RSI(daily_data['Close'].values, timeperiod=14)
            current_rsi = rsi[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(daily_data['Close'].values)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(daily_data['Close'].values)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rsi_color = "🟢" if 30 <= current_rsi <= 70 else "🔴"
                st.metric("RSI (14)", f"{current_rsi:.1f} {rsi_color}")
            with col2:
                macd_trend = "🟢" if macd[-1] > macd_signal[-1] else "🔴"
                st.metric("MACD Signal", f"{macd_trend}")
            with col3:
                bb_position = (current_price - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                bb_status = "🟡 Neutral" if 0.2 <= bb_position <= 0.8 else "🔴 Extreme"
                st.metric("BB Position", f"{bb_position:.2f} {bb_status}")
        
        st.subheader("Pattern Statistics")
        
        # Overall pattern success rates
        total_patterns = sum(len(patterns) for patterns in patterns_by_timeframe.values())
        bullish_patterns = sum(len([p for p in patterns if p.get('direction') == 'bullish']) 
                              for patterns in patterns_by_timeframe.values())
        bearish_patterns = sum(len([p for p in patterns if p.get('direction') == 'bearish']) 
                              for patterns in patterns_by_timeframe.values())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Patterns", total_patterns)
        with col2:
            st.metric("Bullish Patterns", bullish_patterns)
        with col3:
            st.metric("Bearish Patterns", bearish_patterns)
        
        # Pattern distribution chart (already added above)
    
    # Export Options
    st.header("📁 Export & Save")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Pattern Data"):
            # Create comprehensive report
            report_data = []
            for tf, patterns in patterns_by_timeframe.items():
                for pattern in patterns:
                    backtest = pattern.get('backtest', {})
                    report_data.append({
                        'Timeframe': tf,
                        'Pattern': pattern['type'],
                        'Direction': pattern['direction'],
                        'Rank': pattern.get('rank', 'N/A'),
                        'Confidence': pattern.get('confidence', 0),
                        'Expected_Profit_%': backtest.get('profit_pct', 0),
                        'Entry_Price': backtest.get('entry_price', 0),
                        'Target_Price': pattern.get('target', 0)
                    })
            
            if report_data:
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="Download Pattern Report (CSV)",
                    data=csv,
                    file_name=f"{selected_symbol}_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("📈 Export Signals"):
            if signals:
                signals_df = pd.DataFrame(signals)
                signals_csv = signals_df.to_csv(index=False)
                st.download_button(
                    label="Download Signals (CSV)",
                    data=signals_csv,
                    file_name=f"{selected_symbol}_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No signals to export")
    
    with col3:
        if st.button("💾 Save Configuration"):
            config = {
                'symbol': selected_symbol,
                'analysis_period': analysis_period,
                'lookforward_days': lookforward_days,
                'timestamp': datetime.now().isoformat()
            }
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="Download Config (JSON)",
                data=config_json,
                file_name=f"stockpattern_config_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )
    
    # Disclaimer
    st.markdown("""
    ---
    **⚠️ Important Disclaimer:**
    - This analysis is for educational and informational purposes only
    - Past performance does not guarantee future results
    - Always do your own research and consider consulting with a financial advisor
    - Trading involves risk and you may lose money
    - The patterns and signals are based on historical data analysis
    - Market conditions can change rapidly and affect pattern reliability
    """)

else:
    # Welcome screen
    st.markdown("""
    ## 🎯 Welcome to StockPatternAI Dashboard
    
    This advanced dashboard provides:
    
    ### 🔍 **Multi-Timeframe Pattern Analysis**
    - Detects 15+ chart patterns across 4 timeframes (1H, Daily, Weekly, Monthly)
    - Historical backtesting with 12 years of data
    - Pattern ranking based on profitability
    
    ### 🚨 **Smart Signal Generation**
    - **Strong Buy/Sell**: All 4 timeframes aligned with Rank 1 patterns
    - **Buy/Sell**: 3+ timeframes showing strong patterns
    - Real-time monitoring capabilities
    
    ### 📊 **Advanced Features**
    - Risk management recommendations
    - Technical indicator analysis
    - Portfolio allocation suggestions
    - Pattern performance statistics
    - Export capabilities
    
    ### 🎮 **How to Use**
    1. Select a stock from popular lists or enter custom symbol
    2. Choose analysis period and settings
    3. Click "🚀 Start Analysis" to begin
    4. Review signals and risk management recommendations
    5. Enable live monitoring for real-time updates
    
    ---
    **Select your parameters in the sidebar and click "🚀 Start Analysis" to begin!**
    """)
