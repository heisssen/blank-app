import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================
# 1. CONFIG & STYLES
# =========================
st.set_page_config(page_title="Crypto Sniper Pro V2", layout="wide", page_icon="⚡")

# Custom CSS for better tables
st.markdown("""
<style>
    .stDataFrame {font-size: 14px;}
    div[data-testid="stMetricValue"] {font-size: 18px;}
</style>
""", unsafe_allow_html=True)

st.title("⚡ Crypto Sniper Pro V2: Risk Manager")
st.markdown("Сканер RSI + Trend Filter + Генератор Money Management.")

# =========================
# 2. CORE FUNCTIONS
# =========================
@st.cache_resource
def get_exchange():
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    return ex

def normalize_symbol(symbol: str) -> list[str]:
    candidates = [symbol]
    if ":USDT" not in symbol and symbol.endswith("/USDT"):
        candidates.append(symbol.replace("/USDT", "/USDT:USDT"))
    return candidates

def fmt_price(symbol_used: str, price: float) -> str:
    """Розумне форматування ціни"""
    if price >= 1000: return f"{price:.1f}"
    if price >= 10: return f"{price:.2f}"
    if price >= 1: return f"{price:.4f}"
    return f"{price:.5f}" # Для монет типу 0.00023

# =========================
# 3. DATA ENGINE
# =========================
@st.cache_data(ttl=300, show_spinner=False)
def get_top_usdt_perp_symbols(top_n: int):
    ex = get_exchange()
    fallback = ["BTC/USDT","ETH/USDT","SOL/USDT","XRP/USDT","BNB/USDT","DOGE/USDT","PEPE/USDT","ARB/USDT"]
    try:
        # Завантажуємо ринки, щоб відфільтрувати тільки USDT Swap
        markets = ex.load_markets()
        active_perps = [
            s for s, m in markets.items() 
            if m.get('swap') and m.get('linear') and m.get('active') and m.get('quote') == 'USDT'
        ]
        
        # Отримуємо тікери для сортування за об'ємом
        tickers = ex.fetch_tickers(active_perps)
        scored = []
        for s, t in tickers.items():
            vol = t.get('quoteVolume', 0) or 0
            change_24h = t.get('percentage', 0) or 0
            scored.append((s, vol, change_24h))
        
        # Сортуємо за об'ємом
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Повертаємо список і словник зі зміною 24h (для відображення)
        top_coins = [x[0] for x in scored[:top_n]]
        changes_dict = {x[0]: x[2] for x in scored[:top_n]}
        return top_coins, changes_dict
    except:
        return fallback, {}

@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, tf: str, lim: int):
    ex = get_exchange()
    for s in normalize_symbol(symbol):
        try:
            bars = ex.fetch_ohlcv(s, timeframe=tf, limit=lim)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol_used"] = s
            return df, None
        except Exception as e:
            pass
    return None, "Error fetching data"

# =========================
# 4. INDICATORS & LOGIC
# =========================
def calculate_indicators(df, rsi_per=14, atr_per=14, ema_per=200):
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

    # EMA Trend
    df["ema"] = df["close"].ewm(span=ema_per, adjust=False).mean()
    
    return df

def get_signal(row, oversold, overbought):
    rsi = row["rsi"]
    price = row["close"]
    ema = row["ema"]
    
    # Визначаємо базовий сигнал RSI
    signal = None
    if rsi < oversold:
        signal = "LONG"
    elif rsi > overbought:
        signal = "SHORT"
    
    # Аналіз тренду (фільтр)
    trend = "NEUTRAL"
    if price > ema * 1.001: trend = "BULLISH 🟢"
    elif price < ema * 0.999: trend = "BEARISH 🔴"
    
    # Попередження (якщо шортимо проти бичачого тренду)
    warning = ""
    if signal == "SHORT" and "BULLISH" in trend:
        warning = "⚠️ Counter-Trend"
    if signal == "LONG" and "BEARISH" in trend:
        warning = "⚠️ Counter-Trend"
        
    return signal, trend, warning

# =========================
# 5. TEXT GENERATOR (FORMATTED)
# =========================
def generate_telegram_post(
    coin, symbol_used, price, atr, side, 
    lev_range, offset_pct, sl_mult, tp_mults, tp_percents
):
    base = coin.split("/")[0].split(":")[0]
    
    # Розрахунок входу
    if side == "SHORT":
        limit_entry = price * (1 + offset_pct)
        emoji = "📈"
        sl_price = ((price + limit_entry)/2) + (atr * sl_mult)
        tps = [((price + limit_entry)/2) - (atr * m) for m in tp_mults]
    else:
        limit_entry = price * (1 - offset_pct)
        emoji = "📉"
        sl_price = ((price + limit_entry)/2) - (atr * sl_mult)
        tps = [((price + limit_entry)/2) + (atr * m) for m in tp_mults]

    entry_avg = (price + limit_entry) / 2
    
    # Розрахунок RR (Risk Reward) для останнього тейка
    risk = abs(entry_avg - sl_price)
    reward_max = abs(entry_avg - tps[-1])
    rr_ratio = reward_max / risk if risk > 0 else 0

    # Текст
    lines = [
        f"{base} {emoji} {side} x{lev_range[0]}-{lev_range[1]}",
        "",
        "✅ Вход: два ордера",
        f"Рынок {fmt_price(symbol_used, price)}",
        f"Лимит {fmt_price(symbol_used, limit_entry)}",
        "",
        "💸 Take-Profit:",
    ]
    
    for i, tp in enumerate(tps):
        pct = tp_percents[i] if i < len(tp_percents) else 0
        lines.append(f"{i+1}) {fmt_price(symbol_used, tp)} (Fix {pct}%)")
        
    lines.append("")
    lines.append(f"❌ Stop-loss: {fmt_price(symbol_used, sl_price)}")
    lines.append(f"⚖️ RR: 1:{rr_ratio:.1f}")

    return "\n".join(lines)

# =========================
# 6. SIDEBAR UI
# =========================
st.sidebar.header("⚙️ Налаштування Сканера")

# A. Universe
with st.sidebar.expander("🌍 Вибір монет", expanded=False):
    scan_mode = st.radio("Режим:", ["Auto Top-Volume", "Ручний"], index=0)
    n_coins = st.slider("К-сть монет (Top Volume)", 10, 100, 40)
    manual_coins = st.multiselect("Монети", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "PEPE/USDT"], default=["BTC/USDT"])

# B. Strategy
with st.sidebar.expander("📊 Стратегія (RSI & Trend)", expanded=True):
    tf = st.selectbox("Таймфрейм", ["5m", "15m", "1h", "4h"], index=1)
    rsi_len = st.number_input("RSI Period", 7, 21, 14)
    ob_level = st.slider("Overbought (Short)", 60, 90, 70)
    os_level = st.slider("Oversold (Long)", 10, 40, 30)
    ema_len = st.number_input("EMA Trend Filter", 50, 200, 200)

# C. Risk Management
with st.sidebar.expander("💰 Ризик Менеджмент", expanded=True):
    lev_min = st.number_input("Плече Min", 10, 125, 20)
    lev_max = st.number_input("Плече Max", 10, 125, 25)
    limit_offset = st.slider("Відступ лімітки (%)", 0.0, 3.0, 1.5, step=0.1) / 100
    sl_mult = st.slider("SL (x ATR)", 1.0, 5.0, 1.5, step=0.1)
    
    st.write("---")
    st.write("**Налаштування Тейків (ATR Multiplier & % Exit)**")
    c1, c2 = st.columns(2)
    tp1_m = c1.number_input("TP1 xATR", 0.5, 5.0, 1.0)
    tp1_p = c2.number_input("TP1 Закрити %", 0, 100, 50)
    
    c3, c4 = st.columns(2)
    tp2_m = c3.number_input("TP2 xATR", 1.0, 10.0, 2.0)
    tp2_p = c4.number_input("TP2 Закрити %", 0, 100, 30)
    
    c5, c6 = st.columns(2)
    tp3_m = c5.number_input("TP3 xATR", 2.0, 20.0, 4.0)
    tp3_p = c6.number_input("TP3 Закрити %", 0, 100, 20)

    tp_mults = [tp1_m, tp2_m, tp3_m]
    tp_percents = [tp1_p, tp2_p, tp3_p]

# =========================
# 7. MAIN LOGIC
# =========================
if st.button("🚀 СКАНУВАТИ РИНОК", type="primary"):
    
    # 1. Get Coins
    with st.spinner("Завантаження списку монет та об'ємів..."):
        if scan_mode.startswith("Auto"):
            coins, changes_dict = get_top_usdt_perp_symbols(n_coins)
        else:
            coins = manual_coins
            changes_dict = {}

    # 2. Analyze
    results = []
    posts = []
    errors = []
    
    prog_bar = st.progress(0)
    status_text = st.empty()
    
    for i, coin in enumerate(coins):
        status_text.text(f"Аналіз {coin}...")
        # Більший ліміт свічок для коректної EMA 200
        df, err = fetch_ohlcv_cached(coin, tf, lim=ema_len + 100) 
        
        if df is None:
            errors.append(f"{coin}: {err}")
            continue
            
        df = calculate_indicators(df, rsi_len, 14, ema_len)
        
        last = df.iloc[-1]
        side, trend, warning = get_signal(last, os_level, ob_level)
        
        chg_24h = changes_dict.get(coin, 0)
        
        # Записуємо результат
        res_row = {
            "Coin": coin,
            "Price": last["close"],
            "24h%": chg_24h,
            "RSI": last["rsi"],
            "Trend": trend,
            "Signal": side if side else "-",
            "Warning": warning,
            "ATR": last["atr"],
            "SymbolUsed": last["symbol_used"]
        }
        results.append(res_row)
        
        # Якщо є сигнал - робимо пост
        if side:
            post = generate_telegram_post(
                coin, res_row["SymbolUsed"], last["close"], last["atr"], side,
                (lev_min, lev_max), limit_offset, sl_mult, tp_mults, tp_percents
            )
            posts.append(post)
            
        prog_bar.progress((i+1)/len(coins))
    
    prog_bar.empty()
    status_text.empty()
    
    # 3. Show Results
    df_res = pd.DataFrame(results)
    
    if not df_res.empty:
        # Sort: Signals first, then by RSI deviation from 50
        df_res["_sort"] = df_res["Signal"].apply(lambda x: 0 if x in ["LONG", "SHORT"] else 1)
        df_res["_rsi_dev"] = abs(df_res["RSI"] - 50)
        df_res = df_res.sort_values(["_sort", "_rsi_dev"], ascending=[True, False]).drop(columns=["_sort", "_rsi_dev"])

        # Tabs
        t1, t2, t3 = st.tabs(["📋 Аналіз ринку", "📢 Готові сигнали", "📉 Графік"])
        
        with t1:
            st.subheader("Зведена таблиця")
            
            def color_rows(val):
                if val == "LONG": return "color: #00ff00; font-weight: bold"
                if val == "SHORT": return "color: #ff0000; font-weight: bold"
                return ""
                
            st.dataframe(
                df_res.style.map(color_rows, subset=["Signal"])
                .format({"Price": "{:.4f}", "24h%": "{:+.2f}%", "RSI": "{:.1f}", "ATR": "{:.5f}"}),
                use_container_width=True,
                height=600
            )
            
        with t2:
            st.subheader(f"Знайдено сигналів: {len(posts)}")
            if posts:
                cols = st.columns(2)
                for idx, p in enumerate(posts):
                    with cols[idx % 2]:
                        st.text_area(f"Signal {idx+1}", p, height=350)
            else:
                st.info("Сигналів немає. Ринок у флеті або RSI в нормі.")

        with t3:
            coin_sel = st.selectbox("Перевірити графік:", df_res["Coin"].unique())
            if coin_sel:
                row = df_res[df_res["Coin"] == coin_sel].iloc[0]
                # Re-fetch for clean plotting
                df_p, _ = fetch_ohlcv_cached(coin_sel, tf, ema_len+100)
                df_p = calculate_indicators(df_p, rsi_len, 14, ema_len)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df_p["timestamp"], open=df_p["open"], high=df_p["high"],
                    low=df_p["low"], close=df_p["close"], name="Price"
                ), row=1, col=1)
                
                # EMA
                fig.add_trace(go.Scatter(x=df_p["timestamp"], y=df_p["ema"], name=f"EMA {ema_len}", line=dict(color='orange')), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=df_p["timestamp"], y=df_p["rsi"], name="RSI", line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=ob_level, line_color="red", row=2, col=1)
                fig.add_hline(y=os_level, line_color="green", row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", title=f"{coin_sel} ({tf}) Analysis")
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Налаштуйте параметри зліва та натисніть 'Сканувати'")

