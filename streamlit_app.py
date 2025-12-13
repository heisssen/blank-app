import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# =========================
# 1. CONFIG & STYLES
# =========================
st.set_page_config(page_title="Crypto Sniper Pro V5", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    /* Глобальні стилі */
    .stApp { background-color: #0e1117; }
    .stDataFrame { font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 16px !important; }
    
    /* Картка сигналу */
    .mobile-card {
        background-color: #1a1c24;
        border: 1px solid #2b2d35;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .card-header { 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        margin-bottom: 12px;
        border-bottom: 1px solid #2b2d35;
        padding-bottom: 8px;
    }
    .coin-title { font-size: 1.3em; font-weight: 700; color: #fff; }
    .signal-badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }
    .badge-long { background-color: #1e3a2f; color: #00ff00; border: 1px solid #00ff00; }
    .badge-short { background-color: #3a1e1e; color: #ff4b4b; border: 1px solid #ff4b4b; }
    
    /* Рядки даних */
    .data-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.95em; }
    .label { color: #8b92a6; }
    .value { color: #e0e0e0; font-weight: 500; font-family: 'Roboto Mono', monospace; }
    
    /* Текст тренду */
    .trend-info { margin-top: 10px; font-size: 0.85em; color: #8b92a6; font-style: italic; }
    .warning { color: #ffa726; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Multi-Exchange Sniper Pro V5")
st.markdown("RSI + Trend Scanner: **Binance, Bybit, KuCoin, OKX, Kraken**")

# =========================
# 2. CORE UTILS
# =========================
def fmt_price(price):
    if not isinstance(price, (int, float)): return "N/A"
    if price >= 1000: return f"{price:.1f}"
    if price >= 10: return f"{price:.2f}"
    if price >= 0.1: return f"{price:.4f}"
    return f"{price:.8f}".rstrip('0').rstrip('.')

# =========================
# 3. DATA ENGINE
# =========================
EXCHANGE_CLASSES = {
    'binance': ccxt.binance,
    'bybit': ccxt.bybit,
    'kucoin': ccxt.kucoin,
    'okx': ccxt.okx,
    'kraken': ccxt.kraken,
}

@st.cache_resource
def get_exchange_config(exchange_id: str):
    """Повертає базову конфігурацію для біржі (без створення важкого об'єкта)"""
    config = {
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    }
    if exchange_id == 'okx':
        config["options"]["defaultType"] = "swap"
    elif exchange_id == 'kraken':
        config["options"]["defaultType"] = "future" 
    
    return config

@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(exchange_id: str, scan_mode: str, top_n: int, manual_list: list):
    """Отримує список монет і їх 24h зміни"""
    config = get_exchange_config(exchange_id)
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass: return [], {}
    
    # Тимчасовий інстанс для отримання тікерів
    ex = ExClass(config)
    
    try:
        markets = ex.load_markets()
        target_symbols = []

        # 1. Визначаємо пул монет (всі активні ф'ючерси)
        if scan_mode.startswith("Auto"):
            for s, m in markets.items():
                if not m.get('active') or m.get('quote') != 'USDT': continue
                
                # Фільтрація по біржах
                is_target = False
                if exchange_id == 'binance' and m.get('linear') and m.get('swap'): is_target = True
                elif exchange_id == 'bybit' and m.get('linear') and 'PERP' in s: is_target = True
                elif exchange_id == 'kucoin' and m.get('type') == 'future': is_target = True
                elif exchange_id == 'okx' and m.get('swap') and m.get('linear'): is_target = True
                elif exchange_id == 'kraken' and m.get('linear'): is_target = True # Kraken linear futures

                if is_target:
                    target_symbols.append(s)
        else:
            # Ручний режим: перевіряємо, чи існують введені монети на біржі
            target_symbols = [s for s in manual_list if s in markets]

        if not target_symbols:
            return [], {}

        # 2. Отримуємо дані (ціна, об'єм, зміна)
        # Беремо топ-N за об'ємом або всі для ручного режиму
        limit = top_n if scan_mode.startswith("Auto") else len(target_symbols)
        
        # Щоб не перевантажити API, fetch_tickers іноді краще брати пакетами, 
        # але тут спростимо:
        if len(target_symbols) > 100 and scan_mode.startswith("Auto"):
             # Це "брудна" евристика, але fetch_tickers без аргументів повертає все, що часто швидше ніж список
             tickers = ex.fetch_tickers()
             # Фільтруємо отримане
             tickers = {k: v for k, v in tickers.items() if k in target_symbols}
        else:
             tickers = ex.fetch_tickers(target_symbols)

        scored = []
        for s, t in tickers.items():
            if s not in target_symbols: continue
            vol = t.get('quoteVolume', 0) or t.get('volume', 0) or 0
            change = t.get('percentage', 0) or 0
            scored.append((s, vol, change))

        # Сортуємо за об'ємом
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Обрізаємо до ліміту
        final_list = scored[:limit]
        
        coins = [x[0] for x in final_list]
        changes = {x[0]: x[2] for x in final_list}
        
        return coins, changes

    except Exception as e:
        st.error(f"API Error ({exchange_id}): {e}")
        return [], {}

def fetch_candle_data(args):
    """Потокова функція завантаження свічок"""
    symbol, tf, limit, exchange_id, config = args
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    ex = ExClass(config)
    
    try:
        # Невелика пауза для запобігання rate-limit в потоках
        time.sleep(0.1) 
        
        # OKX вимагає ініціалізації ринків для коректного парсингу
        if exchange_id == 'okx': ex.load_markets()
            
        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not ohlcv: return symbol, None, "Empty Data"
        
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return symbol, df, None
    except Exception as e:
        return symbol, None, str(e)

# =========================
# 4. ANALYSIS LOGIC
# =========================
def analyze_market(df, rsi_len, ema_len, os_level, ob_level):
    if df is None or len(df) < ema_len: return None

    # RSI Calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR Calculation (Simple)
    high_low = df["high"] - df["low"]
    df["atr"] = high_low.ewm(span=14, adjust=False).mean()

    # Trend EMA
    df["ema"] = df["close"].ewm(span=ema_len, adjust=False).mean()

    last = df.iloc[-1]
    
    # Signals
    sig = None
    if last["rsi"] < os_level: sig = "LONG"
    elif last["rsi"] > ob_level: sig = "SHORT"

    # Trend Status
    trend = "NEUTRAL"
    if last["close"] > last["ema"]: trend = "BULLISH 🟢"
    elif last["close"] < last["ema"]: trend = "BEARISH 🔴"

    # Warning
    warn = ""
    if (sig == "LONG" and "BEARISH" in trend) or (sig == "SHORT" and "BULLISH" in trend):
        warn = "⚠️ Counter-Trend"

    return {
        "price": last["close"],
        "rsi": last["rsi"],
        "atr": last["atr"],
        "trend": trend,
        "signal": sig,
        "warning": warn
    }

def create_telegram_post(coin, data, params, exchange_id):
    side = data["signal"]
    price = data["price"]
    atr = data["atr"]
    
    # Unpack params
    lev = params['lev']
    offset = params['offset']
    sl_mult = params['sl']
    tps = params['tps'] # list of multipliers
    
    emoji = "🟢" if side == "LONG" else "🔴"
    
    # Logic: Limit Entry
    limit_price = price * (1 - offset) if side == "LONG" else price * (1 + offset)
    
    # Logic: SL & TP based on Entry
    if side == "LONG":
        sl_price = limit_price - (atr * sl_mult)
        tp_prices = [limit_price + (atr * m) for m in tps]
    else:
        sl_price = limit_price + (atr * sl_mult)
        tp_prices = [limit_price - (atr * m) for m in tps]
        
    risk = abs(limit_price - sl_price)
    reward = abs(limit_price - tp_prices[-1])
    rr = reward / risk if risk else 0

    txt = f"#{coin.split('/')[0]} {emoji} {side} SETUP\n"
    txt += f"🏦 Ex: {exchange_id.upper()} | Lev: x{lev[0]}-{lev[1]}\n"
    txt += "------------------\n"
    txt += f"🎯 Entry (Limit): {fmt_price(limit_price)}\n"
    txt += f"🛡️ Stop-Loss: {fmt_price(sl_price)}\n"
    for i, tp in enumerate(tp_prices):
        txt += f"💰 TP{i+1}: {fmt_price(tp)}\n"
    txt += "------------------\n"
    txt += f"⚖️ RR: 1:{rr:.1f} | Market: {fmt_price(price)}"
    
    return txt

# =========================
# 5. UI SIDEBAR
# =========================
st.sidebar.header("🛠️ Налаштування")

with st.sidebar.expander("🌐 Біржа та Активи", expanded=True):
    exch = st.selectbox("Біржа", ["binance", "bybit", "kucoin", "okx", "kraken"], format_func=str.upper)
    mode = st.radio("Режим пошуку", ["Auto (Top Volume)", "Manual List"])
    
    if "Auto" in mode:
        n_coins = st.slider("Кількість монет", 10, 100, 30)
        manual_coins = []
    else:
        n_coins = 0
        default_list = "BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, DOGE/USDT, XRP/USDT, LTC/USDT"
        raw_manual = st.text_area("Список монет (через кому)", default_list)
        manual_coins = [x.strip().upper() for x in raw_manual.split(",")]

with st.sidebar.expander("📊 Стратегія", expanded=False):
    tf = st.selectbox("Timeframe", ["5m", "15m", "1h", "4h"], index=1)
    rsi_len = st.number_input("RSI Length", 7, 21, 14)
    ob = st.slider("Overbought (>)", 60, 95, 70)
    os = st.slider("Oversold (<)", 5, 40, 30)
    ema_len = st.number_input("Trend EMA", 50, 300, 200)

with st.sidebar.expander("💰 Ризик-менеджмент (для поста)", expanded=False):
    p_lev = st.slider("Leverage", 1, 50, (10, 20))
    p_off = st.slider("Entry Offset (%)", 0.0, 5.0, 0.5, step=0.1) / 100
    p_sl = st.slider("Stop Loss (xATR)", 1.0, 5.0, 2.0)
    p_tps = [1.0, 2.5, 4.0] # Multipliers for TP1, TP2, TP3

# =========================
# 6. MAIN APP
# =========================
c1, c2 = st.columns([3, 1])
c1.subheader(f"📡 Сканер: {exch.upper()} [{tf}]")
run = c2.button("🚀 ЗАПУСТИТИ СКАНЕР", type="primary", use_container_width=True)

if run:
    # 1. Fetch Symbols
    with st.spinner("Отримання ринкових даних..."):
        # Очистка кешу даних ринку при новому запуску для актуальності
        get_market_data.clear()
        coins, changes_dict = get_market_data(exch, mode, n_coins, manual_coins)
    
    if not coins:
        st.error("Не знайдено монет. Перевірте список або налаштування.")
        st.stop()
        
    # 2. Scanning
    progress = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    # Threading Config
    ex_conf = get_exchange_config(exch)
    tasks = [(c, tf, ema_len + 50, exch, ex_conf) for c in coins]
    
    # Зменшуємо кількість воркерів для стабільності на Cloud
    MAX_WORKERS = 5 
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        completed = 0
        total = len(coins)
        
        for symbol, df, err in executor.map(fetch_candle_data, tasks):
            completed += 1
            progress.progress(completed / total)
            status_text.caption(f"Аналіз: {symbol} ({completed}/{total})")
            
            if err:
                continue
                
            analysis = analyze_market(df, rsi_len, ema_len, os, ob)
            if not analysis: continue
            
            # Post Gen
            post_txt = ""
            if analysis["signal"]:
                post_params = {'lev': p_lev, 'offset': p_off, 'sl': p_sl, 'tps': p_tps}
                post_txt = create_telegram_post(symbol, analysis, post_params, exch)

            results.append({
                "Coin": symbol,
                "Price": analysis["price"],
                "24h%": changes_dict.get(symbol, 0),
                "RSI": analysis["rsi"],
                "Signal": analysis["signal"],
                "Trend": analysis["trend"],
                "Warning": analysis["warning"],
                "Post": post_txt
            })
            
    progress.empty()
    status_text.empty()
    
    # 3. Visualization
    df_res = pd.DataFrame(results)
    
    if df_res.empty:
        st.warning("Дані не отримано.")
    else:
        # Sorting
        df_res["_sort_sig"] = df_res["Signal"].apply(lambda x: 0 if x else 1)
        df_res["_sort_rsi"] = df_res.apply(lambda r: r["RSI"] if r["Signal"]=="LONG" else (100-r["RSI"] if r["Signal"]=="SHORT" else 50), axis=1)
        df_res = df_res.sort_values(by=["_sort_sig", "_sort_rsi"])
        
        tab_sig, tab_all = st.tabs(["📱 Сигнали", "📋 Всі Результати"])
        
        # --- MOBILE VIEW ---
        with tab_sig:
            signals = df_res[df_res["Signal"].notna()]
            if signals.empty:
                st.info("🟢 Немає активних сигналів за вказаними параметрами.")
            else:
                for _, row in signals.iterrows():
                    sig_class = "badge-long" if row["Signal"] == "LONG" else "badge-short"
                    warn_html = f'<span class="warning">{row["Warning"]}</span>' if row["Warning"] else ""
                    
                    st.markdown(f"""
                    <div class="mobile-card">
                        <div class="card-header">
                            <span class="coin-title">{row['Coin']}</span>
                            <span class="signal-badge {sig_class}">{row['Signal']}</span>
                        </div>
                        <div class="data-row">
                            <span class="label">Ціна (24h%)</span>
                            <span class="value">{fmt_price(row['Price'])} ({row['24h%']:.2f}%)</span>
                        </div>
                        <div class="data-row">
                            <span class="label">RSI</span>
                            <span class="value">{row['RSI']:.1f}</span>
                        </div>
                        <div class="trend-info">
                            Trend: {row['Trend']} &nbsp; {warn_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📋 Копіювати сигнал"):
                        st.code(row["Post"], language="text")
        
        # --- DESKTOP TABLE ---
        with tab_all:
            # Форматуємо для Data Editor
            st.data_editor(
                df_res[["Coin", "Price", "24h%", "RSI", "Signal", "Trend", "Warning"]],
                column_config={
                    "RSI": st.column_config.ProgressColumn("RSI", min_value=0, max_value=100, format="%.1f"),
                    "Price": st.column_config.NumberColumn(format="%.4f"),
                    "24h%": st.column_config.NumberColumn(format="%.2f%%"),
                },
                use_container_width=True,
                height=600,
                hide_index=True
            )