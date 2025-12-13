import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# 1. CONFIG & STYLES
# =========================
st.set_page_config(page_title="Crypto 5-Exchange Sniper Pro", layout="wide", page_icon="🌐")

st.markdown("""
<style>
    /* ... (CSS стилі без змін для мобільної адаптації) ... */
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

st.title("🌐 Crypto 5-Exchange Sniper Pro V4")
st.markdown("Сканер RSI + Trend Filter для **Binance, Bybit, KuCoin, OKX, Kraken**.")

# =========================
# 2. CORE FUNCTIONS
# =========================
def fmt_price(price: float) -> str:
    if price >= 1000: return f"{price:.1f}"
    if price >= 10: return f"{price:.2f}"
    if price >= 1: return f"{price:.4f}"
    return f"{price:.6f}"

# =========================
# 3. DATA ENGINE (UNIVERSAL)
# =========================
EXCHANGE_CLASSES = {
    'binance': ccxt.binance,
    'bybit': ccxt.bybit,
    'kucoin': ccxt.kucoin,
    'okx': ccxt.okx,      # Додано OKX
    'kraken': ccxt.kraken, # Додано Kraken
}

@st.cache_resource
def get_exchange(exchange_id: str):
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass:
        raise ValueError(f"Exchange {exchange_id} not supported.")
    
    config = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"}, # Default для ф'ючерсів
    }
    
    if exchange_id == 'kraken':
        # Kraken Futures може мати інший endpoint
        config["options"]["defaultType"] = "future"
        # CCXT може потребувати окремий ID для Kraken Futures, але спробуємо так
    
    return ExClass(config)

@st.cache_data(ttl=300, show_spinner=False)
def get_top_usdt_perp_symbols(exchange_id: str, top_n: int):
    ex = get_exchange(exchange_id)
    fallback = ["BTC/USDT", "ETH/USDT"] 
    
    try:
        markets = ex.load_markets()
        active_perps = []
        
        # УНІВЕРСАЛЬНА ЛОГІКА ФІЛЬТРУ
        for s, m in markets.items():
            # Базовий фільтр
            if not m.get('active') or m.get('quote') != 'USDT':
                continue
                
            # Специфічні фільтри для ф'ючерсів / безстрокових контрактів
            if exchange_id == 'binance' and m.get('swap') and m.get('linear'):
                active_perps.append(s)
            elif exchange_id == 'bybit' and m.get('linear') is True and 'PERP' in s:
                active_perps.append(s)
            elif exchange_id == 'kucoin' and m.get('type') == 'future':
                active_perps.append(s)
            elif exchange_id == 'okx' and m.get('swap') and 'SWAP' in s: # OKX використовує 'SWAP'
                 active_perps.append(s)
            elif exchange_id == 'kraken' and m.get('type') == 'future':
                 active_perps.append(s)
            # Якщо біржа не підтримує USDT ф'ючерси, тут може бути порожньо
        
        if not active_perps:
            # Спробуємо ще раз, використовуючи лише Spot, якщо ф'ючерсів немає (наприклад, для Kraken)
            active_perps = [s for s, m in markets.items() if m.get('active') and m.get('quote') == 'USDT' and m.get('type') == 'spot']
            if not active_perps:
                st.warning(f"No perpetual or spot USDT markets found on {exchange_id}. Using fallback.")
                return fallback, {}
            
        tickers = ex.fetch_tickers(active_perps)
        scored = []
        for s, t in tickers.items():
            vol = t.get('quoteVolume', 0) or t.get('volume', 0) 
            change_24h = t.get('percentage', 0) or 0
            scored.append((s, vol, change_24h))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        top_coins = [x[0] for x in scored[:top_n]]
        changes_dict = {x[0]: x[2] for x in scored[:top_n]}
        return top_coins, changes_dict
    except Exception as e:
        # Kraken може видавати помилку, якщо не налаштовано environment
        st.error(f"Error fetching symbols from {exchange_id}: {e}")
        return fallback, {}

def fetch_single_coin(args):
    """Worker function for threading"""
    symbol, tf, lim, exchange_id, ex_config = args
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass: return symbol, None, "Invalid exchange ID"
    
    ex = ExClass(ex_config)
    
    try:
        bars = ex.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
        if not bars: return symbol, None, "No data"
            
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return symbol, df, None
    except Exception as e:
        return symbol, None, str(e)

# =========================
# 4. INDICATORS & LOGIC (БЕЗ ЗМІН)
# =========================
def calculate_indicators(df, rsi_per=14, atr_per=14, ema_per=200):
    if df is None or len(df) < ema_per: return df
    
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/rsi_per, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_per, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/atr_per, adjust=False).mean()

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

def generate_telegram_post(coin, price, atr, side, lev_range, offset_pct, sl_mult, tp_mults, tp_percents, exchange_id):
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

    txt = f"#{base} {emoji} {side} ({exchange_id.upper()} | Lev: x{lev_range[0]}-{lev_range[1]})\n\n"
    txt += f"💰 Market: {fmt_price(price)}\n"
    txt += f"⏳ Limit: {fmt_price(limit_entry)}\n\n"
    
    for i, tp in enumerate(tps):
        p = tp_percents[i] if i < len(tp_percents) else 0
        txt += f"🎯 TP{i+1}: {fmt_price(tp)} ({p}%)\n"
        
    txt += f"\n🛑 SL: {fmt_price(sl_price)}\n"
    txt += f"⚖️ RR: 1:{rr:.1f}"
    
    return txt

# =========================
# 5. SIDEBAR UI (UNIVERSAL)
# =========================
st.sidebar.header("⚙️ Scanner Config")

# A. Exchange Selection (UPDATED)
with st.sidebar.expander("🌐 Вибір Біржі", expanded=True):
    exchange_id = st.selectbox(
        "Біржа:",
        options=["kucoin", "okx", "bybit", "binance", "kraken"],
        index=0,
        format_func=lambda x: x.upper()
    )
    st.markdown("> **KuCoin/OKX:** Найкраще для Streamlit Cloud (США).")
    
# B. Universe
with st.sidebar.expander("🌍 Вибір монет", expanded=False):
    scan_mode = st.radio("Режим:", ["Auto Top-Volume", "Ручний"], index=0)
    n_coins = st.slider("К-сть монет (Top Volume)", 10, 50, 20)
    manual_coins = st.multiselect("Монети", ["BTC/USDT", "ETH/USDT", "SOL/USDT"], default=["BTC/USDT"])

# C. Strategy
with st.sidebar.expander("📊 Стратегія (RSI & Trend)", expanded=False):
    tf = st.selectbox("Таймфрейм", ["5m", "15m", "1h", "4h"], index=1)
    rsi_len = st.number_input("RSI Length", 7, 21, 14)
    ob_level = st.slider("Overbought (Short) >", 60, 90, 70)
    os_level = st.slider("Oversold (Long) <", 10, 40, 30)
    ema_len = st.number_input("EMA Trend Filter", 50, 200, 200)

# D. Risk Management
with st.sidebar.expander("💰 Ризик Менеджмент", expanded=False):
    lev_range = (10, 20)
    limit_offset = st.slider("Відступ лімітки (%)", 0.0, 3.0, 1.0, step=0.1) / 100
    sl_mult = st.slider("SL (x ATR)", 1.0, 4.0, 2.0, step=0.1)
    tp_setup = [1.0, 2.5, 4.0] 
    tp_pcts = [50, 30, 20]

# =========================
# 6. MAIN LOGIC
# =========================
col_act1, col_act2 = st.columns([3, 1])
with col_act1:
    st.info(f"💡 Сканування ринку: **{exchange_id.upper()}**")
with col_act2:
    start_btn = st.button(f"🚀 SCAN {exchange_id.upper()}", type="primary")


if start_btn:
    coins = []
    changes = {}
    
    with st.spinner(f"Завантаження списку монет з {exchange_id.upper()}..."):
        coins, changes = get_top_usdt_perp_symbols(exchange_id, n_coins)
        if scan_mode == "Ручний" and manual_coins:
            coins = manual_coins

    status_bar = st.progress(0)
    results = []
    
    # Конфігурація біржі для потоків
    # Ми беремо конфіг з кешованого об'єкта
    base_ex = get_exchange(exchange_id)
    ex_conf = {"enableRateLimit": True, "options": base_ex.options}
    
    tasks = [(c, tf, ema_len+50, exchange_id, ex_conf) for c in coins]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        processed_count = 0
        for symbol, df, err in executor.map(fetch_single_coin, tasks):
            processed_count += 1
            status_bar.progress(processed_count / len(coins))
            
            # Якщо виникла помилка під час fetch, записуємо нульові дані
            if df is None or df.empty:
                results.append({
                    "Coin": symbol,
                    "Price": 0.0,
                    "RSI": 50.0,
                    "Trend": "N/A",
                    "Signal": "Error",
                    "Warning": f"Data Error ({err})",
                    "Post": "",
                    "24h%": 0.0
                })
                continue
                
            df = calculate_indicators(df, rsi_len, 14, ema_len)
            last = df.iloc[-1]
            
            sig, trnd, warn = get_signal(last, os_level, ob_level)
            
            post_txt = ""
            if sig:
                post_txt = generate_telegram_post(
                    symbol, last["close"], last["atr"], sig, 
                    lev_range, limit_offset, sl_mult, tp_setup, tp_pcts, exchange_id
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
        # Сортування: Сигнали спочатку, потім за відхиленням RSI від 50
        df_res["_sort"] = df_res["Signal"].apply(lambda x: 0 if x else 1)
        df_res["_rsi_dev"] = abs(df_res["RSI"] - 50)
        df_res = df_res.sort_values(["_sort", "_rsi_dev"], ascending=[True, False]).drop(columns=["_sort", "_rsi_dev"])

        tab1, tab2, tab3 = st.tabs(["📱 Сигнали", "📊 Зведена таблиця", "📈 Графік"])
        
        # --- TAB 1: MOBILE CARDS (Тільки сигнали) ---
        with tab1:
            signals_only = df_res[df_res["Signal"].notna() & (df_res["Signal"] != "Error")]
            
            if signals_only.empty:
                st.warning(f"No active signals found on {exchange_id.upper()} right now.")
            else:
                for _, row in signals_only.iterrows():
                    border_color = "#00ff00" if row["Signal"] == "LONG" else "#ff4b4b"
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="mobile-card" style="border-left: 5px solid {border_color};">
                            <div class="card-header">
                                <h3 style="margin:0">{row['Coin']} ({exchange_id.upper()})</h3>
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

        # --- TAB 2: ADVANCED TABLE (ВСІ монети) ---
        with tab2:
            st.subheader(f"Результати сканування: {len(df_res)} монет")
            
            # Функція для фарбування рядків таблиці
            def color_rows(val):
                if val == "LONG": return "background-color: #1e3a2f; color: white; font-weight: bold"
                if val == "SHORT": return "background-color: #3a1e1e; color: white; font-weight: bold"
                if val == "Error": return "background-color: #58411d; color: yellow; font-weight: bold"
                return ""
            
            st.dataframe(
                df_res.style.applymap(color_rows, subset=["Signal"]) # Фарбуємо стовпець Signal
                .format({"Price": "{:.4f}", "24h%": "{:+.2f}%", "RSI": "{:.1f}"}),
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

        # --- TAB 3: GRAPH (Графік) ---
        with tab3:
            # Фільтруємо монети без помилок для вибору графіку
            valid_coins = df_res[df_res["Signal"] != "Error"]["Coin"].unique()
            if valid_coins.size > 0:
                coin_sel = st.selectbox("Перевірити графік:", valid_coins)
                
                # Повторний fetch для clean plotting
                df_p, _ = fetch_single_coin((coin_sel, tf, ema_len+100, exchange_id, ex_conf))
                if df_p is not None:
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
                    
                    fig.update_layout(height=600, template="plotly_dark", title=f"{coin_sel} ({tf}) Analysis on {exchange_id.upper()}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Неможливо завантажити дані для графіку.")
            else:
                st.warning("Немає доступних монет для графіку.")
    else:
        st.error(f"Не вдалося отримати дані. Перевірте підключення до {exchange_id.upper()}.")
