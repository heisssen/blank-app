import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

# =========================
# 1. CONFIG & STYLES
# =========================
st.set_page_config(page_title="KuCoin Sniper Pro", layout="wide", page_icon="⚡")

# Custom CSS залишаємо для мобільної оптимізації
st.markdown("""
<style>
    .stDataFrame {font-size: 14px;}
    div[data-testid="stMetricValue"] {font-size: 16px !important;}
    .mobile-card {
        background-color: #262730;
        border: 1px solid #464b5f;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .signal-long { color: #00ff00; font-weight: bold; }
    .signal-short { color: #ff4b4b; font-weight: bold; }
    .card-header { display: flex; justify-content: space-between; align-items: center; }
    .stButton button { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ KuCoin Sniper Pro: Streamlit Edition")
st.markdown("✅ Оптимізовано для роботи на серверах у США (KuCoin API)")

# =========================
# 2. CORE FUNCTIONS (KUCOIN VERSION)
# =========================
@st.cache_resource
def get_exchange():
    # Використовуємо KuCoin, який менш схильний до блокування IP США для публічних даних
    return ccxt.kucoin({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}, # Вказуємо на ф'ючерси
    })

def fmt_price(price: float) -> str:
    """Розумне форматування ціни"""
    if price >= 1000: return f"{price:.1f}"
    if price >= 10: return f"{price:.2f}"
    if price >= 1: return f"{price:.4f}"
    return f"{price:.6f}"

# =========================
# 3. DATA ENGINE (KUCOIN ADAPTED)
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def get_top_usdt_perp_symbols(top_n: int):
    ex = get_exchange()
    # KuCoin використовує формат 'BTC-USDT' або 'XBTUSDTM' для ф'ючерсів
    fallback = ["BTC/USDT", "ETH/USDT", "SOL/USDT"] 
    
    try:
        markets = ex.load_markets()
        
        # Фільтруємо для USDT Perpetual (Futures) на KuCoin
        active_perps = [
            s for s, m in markets.items() 
            if m.get('type') == 'future' 
            and m.get('quote') == 'USDT' 
            and m.get('active')
        ]
        
        if not active_perps:
            return fallback, {}

        tickers = ex.fetch_tickers(active_perps)
        scored = []
        for s, t in tickers.items():
            # KuCoin використовує 'baseVolume', 'quoteVolume' або 'volume'
            vol = t.get('quoteVolume', 0) or t.get('volume', 0)
            change_24h = t.get('percentage', 0) or 0
            scored.append((s, vol, change_24h))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        top_coins = [x[0] for x in scored[:top_n]]
        changes_dict = {x[0]: x[2] for x in scored[:top_n]}
        return top_coins, changes_dict
    except Exception as e:
        st.error(f"Error fetching symbols from KuCoin: {e}")
        return fallback, {}

def fetch_single_coin(args):
    """Worker function for threading"""
    symbol, tf, lim, ex_config = args
    # Для ф'ючерсів KuCoin потрібен окремий синтаксис, ccxt це обробляє через defaultType: 'future'
    ex = ccxt.kucoin(ex_config) 
    
    try:
        bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
        if not bars:
            return symbol, None, "No data"
            
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return symbol, df, None
    except Exception as e:
        return symbol, None, str(e)

# =========================
# 4. LOGIC (БЕЗ ЗМІН)
# =========================
def calculate_indicators(df, rsi_per=14, atr_per=14, ema_per=200):
    if df is None or len(df) < ema_per: return df
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/rsi_per, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_per, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/atr_per, adjust=False).mean()

    # EMA
    df["ema"] = df["close"].ewm(span=ema_per, adjust=False).mean()
    return df

def get_signal(row, oversold, overbought):
    rsi = row["rsi"]
    price = row["close"]
    ema = row["ema"]
    
    signal = None
    if rsi < oversold: signal = "LONG"
    elif rsi > overbought: signal = "SHORT"
    
    trend = "NEUTRAL"
    if price > ema * 1.001: trend = "BULLISH 🟢"
    elif price < ema * 0.999: trend = "BEARISH 🔴"
    
    warning = ""
    if (signal == "SHORT" and "BULLISH" in trend) or (signal == "LONG" and "BEARISH" in trend):
        warning = "Counter-Trend ⚠️"
        
    return signal, trend, warning

def generate_telegram_post(coin, price, atr, side, lev_range, offset_pct, sl_mult, tp_mults, tp_percents):
    base = coin.split("/")[0]
    
    if side == "SHORT":
        limit_entry = price * (1 + offset_pct)
        emoji = "🔴"
        sl_price = ((price + limit_entry)/2) + (atr * sl_mult)
        tps = [((price + limit_entry)/2) - (atr * m) for m in tp_mults]
    else:
        limit_entry = price * (1 - offset_pct)
        emoji = "🟢"
        sl_price = ((price + limit_entry)/2) - (atr * sl_mult)
        tps = [((price + limit_entry)/2) + (atr * m) for m in tp_mults]

    entry_avg = (price + limit_entry) / 2
    risk = abs(entry_avg - sl_price)
    reward_max = abs(entry_avg - tps[-1])
    rr = reward_max / risk if risk > 0 else 0

    txt = f"#{base} {emoji} {side} (Lev: x{lev_range[0]}-{lev_range[1]})\n\n"
    txt += f"💰 Market: {fmt_price(price)}\n"
    txt += f"⏳ Limit: {fmt_price(limit_entry)}\n\n"
    
    for i, tp in enumerate(tps):
        p = tp_percents[i] if i < len(tp_percents) else 0
        txt += f"🎯 TP{i+1}: {fmt_price(tp)} ({p}%)\n"
        
    txt += f"\n🛑 SL: {fmt_price(sl_price)}\n"
    txt += f"⚖️ RR: 1:{rr:.1f}"
    
    return txt

# =========================
# 5. SIDEBAR
# =========================
st.sidebar.header("⚙️ KuCoin Scanner Config")

with st.sidebar.expander("🌍 Coins & Mode", expanded=False):
    scan_mode = st.radio("Mode:", ["Auto Top-Volume", "Manual"], index=0)
    n_coins = st.slider("Coins count", 10, 50, 20)
    manual_coins = st.multiselect("Manual list", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], default=["BTC/USDT"])

with st.sidebar.expander("📊 Strategy", expanded=False):
    tf = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=1)
    rsi_len = st.number_input("RSI Length", 7, 21, 14)
    ob_level = st.slider("Short >", 60, 90, 70)
    os_level = st.slider("Long <", 10, 40, 30)
    ema_len = st.number_input("EMA Trend", 50, 200, 200)

with st.sidebar.expander("💰 Risk Manager", expanded=False):
    lev_range = (10, 20) 
    limit_offset = st.slider("Limit Offset %", 0.0, 3.0, 1.0) / 100
    sl_mult = st.slider("SL xATR", 1.0, 4.0, 2.0)
    tp_setup = [1.0, 2.5, 4.0] 
    tp_pcts = [50, 30, 20]

# =========================
# 6. MAIN APP
# =========================
col_act1, col_act2 = st.columns([3, 1])
with col_act1:
    st.info("💡 Використовуйте горизонтальну прокрутку для таблиць або вкладку 'Сигнали' для карток.")
with col_act2:
    start_btn = st.button("🚀 SCAN KUCOIN", type="primary")

if start_btn:
    coins = []
    changes = {}
    
    with st.spinner("Fetching KuCoin markets..."):
        if scan_mode.startswith("Auto"):
            coins, changes = get_top_usdt_perp_symbols(n_coins)
        else:
            coins = manual_coins

    status_bar = st.progress(0)
    results = []
    
    # Threading setup
    ex_conf = {"enableRateLimit": True, "options": {"defaultType": "future"}}
    tasks = [(c, tf, ema_len+50, ex_conf) for c in coins]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        processed_count = 0
        for symbol, df, err in executor.map(fetch_single_coin, tasks):
            processed_count += 1
            status_bar.progress(processed_count / len(coins))
            
            if df is not None and not df.empty:
                df = calculate_indicators(df, rsi_len, 14, ema_len)
                last = df.iloc[-1]
                
                sig, trnd, warn = get_signal(last, os_level, ob_level)
                
                post_txt = ""
                if sig:
                    post_txt = generate_telegram_post(
                        symbol, last["close"], last["atr"], sig, 
                        lev_range, limit_offset, sl_mult, tp_setup, tp_pcts
                    )

                results.append({
                    "Coin": symbol,
                    "Price": last["close"],
                    "RSI": last["rsi"],
                    "Trend": trnd,
                    "Signal": sig,
                    "Warning": warn,
                    "Post": post_txt,
                    "24h%": changes.get(symbol, 0)
                })

    status_bar.empty()
    
    df_res = pd.DataFrame(results)
    
    if not df_res.empty:
        df_res["_sort"] = df_res["Signal"].apply(lambda x: 0 if x else 1)
        df_res = df_res.sort_values(["_sort", "RSI"], ascending=True)

        tab1, tab2 = st.tabs(["📱 Сигнали (Mobile)", "📊 Таблиця (Desktop)"])
        
        with tab1:
            signals_only = df_res[df_res["Signal"].notna()]
            
            if signals_only.empty:
                st.warning("No active signals found right now.")
            else:
                for _, row in signals_only.iterrows():
                    border_color = "#00ff00" if row["Signal"] == "LONG" else "#ff4b4b"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="mobile-card" style="border-left: 5px solid {border_color};">
                            <div class="card-header">
                                <h3 style="margin:0">{row['Coin']}</h3>
                                <span class="{'signal-long' if row['Signal']=='LONG' else 'signal-short'}">{row['Signal']}</span>
                            </div>
                            <div style="display:flex; justify-content:space-between; margin-top:10px;">
                                <span>Price: <b>{fmt_price(row['Price'])}</b></span>
                                <span>RSI: <b>{row['RSI']:.1f}</b></span>
                            </div>
                            <div style="margin-top:5px; color: #888;">{row['Trend']} {row['Warning']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.text("👇 Copy Signal:")
                        st.code(row["Post"], language="text")
                        st.divider()

        with tab2:
            st.dataframe(
                df_res.style.apply(lambda x: ['background-color: #1e3a2f' if x.Signal == 'LONG' else ('background-color: #3a1e1e' if x.Signal == 'SHORT' else '') for i in x], axis=1),
                column_config={
                    "RSI": st.column_config.ProgressColumn("RSI", format="%.1f", min_value=0, max_value=100),
                    "Price": st.column_config.NumberColumn(format="%.4f"),
                    "24h%": st.column_config.NumberColumn(format="%.2f%%"),
                },
                use_container_width=True,
                height=600,
                hide_index=True,
                column_order=["Coin", "Price", "24h%", "RSI", "Signal", "Trend", "Warning"]
            )
    else:
        st.error("Дані не отримані. Перевірте підключення до KuCoin API.")
