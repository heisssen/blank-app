import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go # Залишаю, хоча не використовується
from concurrent.futures import ThreadPoolExecutor

# =========================
# 1. CONFIG & STYLES
# =========================
st.set_page_config(page_title="Crypto Multi-Exchange Sniper Pro", layout="wide", page_icon="🌐")

st.markdown("""
<style>
    /* Базові стилі */
    .stDataFrame {font-size: 14px;}
    div[data-testid="stMetricValue"] {font-size: 16px !important;}
    .stButton button { width: 100%; border-radius: 8px; }

    /* Покращений мобільний дизайн картки */
    .mobile-card {
        background-color: #1e1f26; /* Темніший фон */
        border: 1px solid #3d3e47;
        border-radius: 12px; /* Більш округлий */
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .signal-long { color: #00ff00; font-weight: bold; font-size: 1.2em; }
    .signal-short { color: #ff4b4b; font-weight: bold; font-size: 1.2em; }
    .card-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        margin-bottom: 8px;
    }
    .card-data-row {
        display: flex; 
        justify-content: space-between; 
        margin-top: 5px;
        font-size: 0.95em;
    }
    .card-data-row span:first-child { color: #aaaaaa; }
    .card-data-row span:last-child { font-weight: 500; }
    .warning-text { color: orange; font-weight: bold; }
    .trend-text { color: #aaaaaa; font-size: 0.9em; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

st.title("🌐 Multi-Exchange Sniper Pro V3")
st.markdown("Сканер RSI + Trend Filter для **Binance, Bybit, KuCoin**.")

# =========================
# 2. CORE FUNCTIONS
# =========================
def fmt_price(price: float) -> str:
    """Розумне форматування ціни"""
    if not isinstance(price, (int, float)): return "N/A" # Додано перевірку
    if price >= 1000: return f"{price:.1f}"
    if price >= 10: return f"{price:.2f}"
    if price >= 1: return f"{price:.4f}"
    # Використовуємо strip для видалення зайвих нулів наприкінці
    return f"{price:.8f}".rstrip('0').rstrip('.')

# =========================
# 3. DATA ENGINE (UNIVERSAL)
# =========================
EXCHANGE_CLASSES = {
    'binance': ccxt.binance,
    'bybit': ccxt.bybit,
    'kucoin': ccxt.kucoin,
}

# Краще кешувати лише конфігурацію, а не об'єкт біржі, який не є потоково-безпечним
# Ми використовуємо цю функцію для отримання конфігурації, щоб *потім* створювати об'єкти в потоках
@st.cache_resource
def get_exchange_config(exchange_id: str):
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass:
        raise ValueError(f"Exchange {exchange_id} not supported.")

    config = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }

    if exchange_id == 'binance':
        config["options"]["defaultType"] = "future"
    elif exchange_id == 'bybit':
        # Bybit: іноді потрібно явно вказати для уніфікації
        config["options"]["defaultType"] = "future"
    elif exchange_id == 'kucoin':
        config["options"]["defaultType"] = "future"

    return config

# Для fetch_tickers ми створюємо один тимчасовий об'єкт, який не є потоково-безпечним, але працює для цієї єдиної операції
@st.cache_data(ttl=300, show_spinner=False)
def get_top_usdt_perp_symbols(exchange_id: str, top_n: int):
    # Створюємо тимчасовий інстанс для fetch_tickers (поза потоками)
    config = get_exchange_config(exchange_id)
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass: return ["BTC/USDT", "ETH/USDT"], {} # Fallback
    ex = ExClass(config)
    fallback = ["BTC/USDT", "ETH/USDT"] 

    try:
        markets = ex.load_markets()
        active_perps = []

        # УНІВЕРСАЛЬНА ЛОГІКА ФІЛЬТРУ
        for s, m in markets.items():
            # Базові фільтри
            if not m.get('active') or m.get('quote') != 'USDT':
                continue
            
            # Фільтри за біржею (для Perpetual Futures)
            if exchange_id == 'binance' and m.get('swap') and m.get('linear'):
                active_perps.append(s)
            elif exchange_id == 'bybit' and m.get('linear') is True and 'PERP' in s:
                # Bybit часто має суфікс PERP або використовує type='swap'
                active_perps.append(s)
            elif exchange_id == 'kucoin' and m.get('type') == 'future':
                active_perps.append(s)

        if not active_perps:
            st.warning(f"No active perpetual USDT markets found on {exchange_id}. Using fallback.")
            return fallback, {}

        # Обмежуємо кількість символів для tickers, щоб не перевищити rate limit
        tickers = ex.fetch_tickers(active_perps[:100])
        scored = []
        for s, t in tickers.items():
            # ccxt уніфікація volume
            vol = t.get('quoteVolume', 0) or t.get('volume', 0) 
            change_24h = t.get('percentage', 0) or 0
            scored.append((s, vol, change_24h))

        scored.sort(key=lambda x: x[1], reverse=True)

        top_coins = [x[0] for x in scored[:top_n]]
        changes_dict = {x[0]: x[2] for x in scored[:top_n]}
        return top_coins, changes_dict
    except Exception as e:
        # st.error(f"Error fetching symbols from {exchange_id}: {e}")
        return fallback, {}


def fetch_single_coin(args):
    """Worker function for threading"""
    symbol, tf, lim, exchange_id, ex_config = args
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass: return symbol, None, "Invalid exchange ID"

    # Створюємо інстанс біржі для потоку з кешованою конфігурацією
    # Це єдиний спосіб уникнути race conditions та використовувати rate limit
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
# 4. INDICATORS & LOGIC (Без змін, але з коментарем про візуалізацію)
# =========================
def calculate_indicators(df, rsi_per=14, atr_per=14, ema_per=200):
    # ... (функція calculate_indicators без змін) ...
    if df is None or len(df) < ema_per: return df

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # Змінюємо на ewm (exp_mean) для коректного RSI
    avg_gain = gain.ewm(span=rsi_per, adjust=False).mean() 
    avg_loss = loss.ewm(span=rsi_per, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=atr_per, adjust=False).mean() # Змінюємо на span
    
    df["ema"] = df["close"].ewm(span=ema_per, adjust=False).mean()
    return df

def get_signal(row, oversold, overbought):
    # ... (функція get_signal без змін) ...
    rsi = row["rsi"]
    price = row["close"]
    ema = row["ema"]

    signal = None
    if rsi < oversold: signal = "LONG"
    elif rsi > overbought: signal = "SHORT"

    trend = "NEUTRAL"
    if price > ema * 1.001: trend = "BULLISH 🟢"
    elif price < ema * 0.999: trend = "BEARISH 🔴"
    
    # Використовуємо .format для чистоти рядка
    trend_raw = trend.split(' ')[0] # BULLISH / BEARISH / NEUTRAL

    warning = ""
    if (signal == "SHORT" and "BULLISH" in trend) or (signal == "LONG" and "BEARISH" in trend):
        warning = "Counter-Trend ⚠️"

    return signal, trend, warning

def generate_telegram_post(coin, price, atr, side, lev_range, offset_pct, sl_mult, tp_mults, tp_percents, exchange_id):
    # *** ВИПРАВЛЕНО: entry_avg використовує тільки limit_entry ***
    base = coin.split("/")[0]

    if side == "SHORT":
        limit_entry = price * (1 + offset_pct)
        entry_avg = limit_entry # Цільова ціна входу
        emoji = "🔴"
        sl_price = entry_avg + (atr * sl_mult)
        tps = [entry_avg - (atr * m) for m in tp_mults]
    else:
        limit_entry = price * (1 - offset_pct)
        entry_avg = limit_entry # Цільова ціна входу
        emoji = "🟢"
        sl_price = entry_avg - (atr * sl_mult)
        tps = [entry_avg + (atr * m) for m in tp_mults]

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

# A. Exchange Selection (NEW!)
with st.sidebar.expander("🌐 Вибір Біржі", expanded=True):
    exchange_id = st.selectbox(
        "Біржа:",
        options=["kucoin", "bybit", "binance"],
        index=0,
        format_func=lambda x: x.upper(),
        key="exchange_select" # Додано key
    )
    st.markdown(f"> **KuCoin:** Рекомендовано для Streamlit Cloud (США) / **Binance, Bybit:** Краще з VPN/EU/UA IP.")

# B. Universe
with st.sidebar.expander("🌍 Вибір монет", expanded=False):
    scan_mode = st.radio("Режим:", ["Auto Top-Volume", "Ручний"], index=0, key="scan_mode")
    n_coins = st.slider("К-сть монет (Top Volume)", 10, 50, 20, key="n_coins")
    manual_coins = st.multiselect("Монети", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "DOGE/USDT"], default=["BTC/USDT"], key="manual_coins")

# C. Strategy
with st.sidebar.expander("📊 Стратегія (RSI & Trend)", expanded=False):
    tf = st.selectbox("Таймфрейм", ["5m", "15m", "1h", "4h"], index=1, key="timeframe")
    rsi_len = st.number_input("RSI Length", 7, 21, 14, key="rsi_len")
    ob_level = st.slider("Overbought (Short) >", 60, 90, 70, key="ob_level")
    os_level = st.slider("Oversold (Long) <", 10, 40, 30, key="os_level")
    ema_len = st.number_input("EMA Trend Filter", 50, 200, 200, key="ema_len")

# D. Risk Management
with st.sidebar.expander("💰 Ризик Менеджмент", expanded=False):
    lev_range = (10, 20)
    limit_offset = st.slider("Відступ лімітки (%)", 0.0, 3.0, 1.0, step=0.1, key="limit_offset") / 100
    sl_mult = st.slider("SL (x ATR)", 1.0, 4.0, 2.0, step=0.1, key="sl_mult")
    tp_setup = [1.0, 2.5, 4.0]  
    tp_pcts = [50, 30, 20]

# =========================
# 6. MAIN LOGIC
# =========================
col_act1, col_act2 = st.columns([3, 1])
with col_act1:
    st.info(f"💡 Сканування ринку: **{exchange_id.upper()}**")
with col_act2:
    # Кнопка скидає кеш для Symbol Fetch, якщо змінили біржу
    start_btn = st.button(f"🚀 SCAN {exchange_id.upper()}", type="primary", key="start_scan")
    if start_btn:
        # Примусове скидання кешу, якщо змінюється exchange_id, щоб отримати нові маркети
        get_top_usdt_perp_symbols.clear() 

if start_btn:
    coins = []
    changes = {}

    with st.spinner(f"Завантаження списку монет з {exchange_id.upper()}..."):
        if scan_mode.startswith("Auto"):
            coins, changes = get_top_usdt_perp_symbols(exchange_id, n_coins)
        else:
            coins = manual_coins

    status_bar = st.progress(0)
    results = []

    # Кешована конфігурація біржі для потоків
    ex_conf = get_exchange_config(exchange_id)

    # Створюємо завдання для потоків: (символ, ТФ, ліміт, ID біржі, конфіг)
    tasks = [(c, tf, ema_len + 50, exchange_id, ex_conf) for c in coins]

    # Використовуємо max_workers=8 або 10 - це зазвичай безпечно для Streamlit Cloud
    with ThreadPoolExecutor(max_workers=8) as executor: 
        processed_count = 0
        
        # Обробка результатів у тому порядку, в якому вони повертаються
        for symbol, df, err in executor.map(fetch_single_coin, tasks):
            processed_count += 1
            status_bar.progress(processed_count / len(coins))

            if df is not None and not df.empty:
                df = calculate_indicators(df, rsi_len, 14, ema_len)
                last = df.iloc[-1]

                # Перевірка на NaNs після розрахунку індикаторів
                if pd.isna(last["rsi"]):
                     results.append({
                        "Coin": symbol, "Price": last["close"], "RSI": np.nan, 
                        "Trend": "N/A", "Signal": None, "Warning": "Not Enough Data",
                        "Post": "", "24h%": changes.get(symbol, 0)
                    })
                     continue

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
            else:
                 results.append({
                    "Coin": symbol, "Price": np.nan, "RSI": np.nan, 
                    "Trend": "N/A", "Signal": None, "Warning": f"Data Error: {err}",
                    "Post": "", "24h%": changes.get(symbol, 0)
                })


    status_bar.empty()

    df_res = pd.DataFrame(results)

    if not df_res.empty:
        # Сортування: 1. Сигнал (нагору) 2. RSI (ближче до краю)
        df_res["_sort"] = df_res["Signal"].apply(lambda x: 0 if x else 1)
        # Сортуємо LONG за зростанням RSI, SHORT за спаданням RSI
        df_res["_rsi_sort"] = df_res.apply(
            lambda row: row["RSI"] if row["Signal"] == "LONG" else (100 - row["RSI"]) if row["Signal"] == "SHORT" else 50, axis=1
        )
        df_res = df_res.sort_values(["_sort", "_rsi_sort"], ascending=True).drop(columns=["_sort", "_rsi_sort"])


        tab1, tab2 = st.tabs(["📱 Сигнали (Mobile)", "📊 Таблиця (Desktop)"])

        # --- TAB 1: MOBILE CARDS ---
        with tab1:
            signals_only = df_res[df_res["Signal"].notna()]

            if signals_only.empty:
                st.info(f"🟢 Наразі немає активних сигналів RSI/Trend на **{exchange_id.upper()}**.")
            else:
                for _, row in signals_only.iterrows():
                    border_color = "#00ff00" if row["Signal"] == "LONG" else "#ff4b4b"
                    warning_html = f'<div class="warning-text">{row["Warning"]}</div>' if row["Warning"] else ''

                    with st.container():
                        st.markdown(f"""
                        <div class="mobile-card" style="border-left: 5px solid {border_color};">
                            <div class="card-header">
                                <h3 style="margin:0; font-size:1.4em;">{row['Coin']}</h3>
                                <span class="{'signal-long' if row['Signal']=='LONG' else 'signal-short'}">{row['Signal']}</span>
                            </div>
                            <div class="card-data-row">
                                <span>Price:</span> 
                                <span><b>{fmt_price(row['Price'])}</b> ({row['24h%']:.2f}%)</span>
                            </div>
                            <div class="card-data-row">
                                <span>RSI:</span> 
                                <span><b>{row['RSI']:.1f}</b></span>
                            </div>
                            <div class="trend-text">{row['Trend']} {warning_html}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.text("👇 Telegram Post:")
                        st.code(row["Post"], language="text")
                        st.divider()

        # --- TAB 2: ADVANCED TABLE ---
        with tab2:
            st.dataframe(
                df_res.style.apply(lambda x: ['background-color: #1e3a2f' if x.Signal == 'LONG' else ('background-color: #3a1e1e' if x.Signal == 'SHORT' else '') for i in x], axis=1),
                column_config={
                    "RSI": st.column_config.ProgressColumn("RSI", format="%.1f", min_value=0, max_value=100),
                    "Price": st.column_config.NumberColumn(format="%.4f"),
                    "24h%": st.column_config.NumberColumn(format="%.2f%%"),
                    "Post": st.column_config.TextColumn(label="Post (Copy)", width="large"),
                },
                use_container_width=True,
                height=600,
                hide_index=True,
                column_order=["Coin", "Price", "24h%", "RSI", "Signal", "Trend", "Warning", "Post"]
            )
    else:
        st.error(f"❌ Не вдалося отримати або обробити дані. Перевірте підключення до {exchange_id.upper()}, обмеження IP-адреси або наявність активних ф'ючерсних ринків.")