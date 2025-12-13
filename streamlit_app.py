import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================
# 1. PAGE CONFIG
# =========================
st.set_page_config(page_title="Crypto Sniper Pro", layout="wide", page_icon="📈")
st.title("⚡ Crypto Futures Sniper")
st.markdown("Сканер Binance Futures за RSI + генератор плану угоди (формат під Telegram).")

# =========================
# 2. EXCHANGE & UTILS
# =========================
@st.cache_resource
def get_exchange():
    """Ініціалізація CCXT для Binance Futures"""
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })
    try:
        ex.load_markets()
    except Exception:
        pass
    return ex

def normalize_symbol(symbol: str) -> list[str]:
    """Додає варіанти символів для пошуку (наприклад, BTC/USDT:USDT)"""
    candidates = [symbol]
    if ":USDT" not in symbol and symbol.endswith("/USDT"):
        candidates.append(symbol.replace("/USDT", "/USDT:USDT"))
    return candidates

def fmt_price(symbol_used: str, price: float) -> str:
    """Форматує ціну згідно з точністю біржі"""
    ex = get_exchange()
    try:
        # Спробуємо використати точність пари з біржі
        return ex.price_to_precision(symbol_used, price)
    except Exception:
        # Fallback, якщо API не віддало точність
        if price >= 1000:
            return f"{price:.1f}"
        elif price >= 1:
            return f"{price:.4f}"
        else:
            return f"{price:.5f}"  # Як у прикладі 0.23802

# =========================
# 3. DATA FETCHING
# =========================
@st.cache_data(ttl=30, show_spinner=False)
def fetch_ohlcv_cached(symbol: str, tf: str, lim: int):
    ex = get_exchange()
    last_error = None

    for s in normalize_symbol(symbol):
        try:
            bars = ex.fetch_ohlcv(s, timeframe=tf, limit=lim)
            df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["symbol_used"] = s
            return df, None
        except Exception as e:
            last_error = str(e)

    return None, last_error

@st.cache_data(ttl=180, show_spinner=False)
def get_top_usdt_perp_symbols(top_n: int):
    """Отримує ТОП монет за об'ємом торгів (USDT Perps)"""
    ex = get_exchange()
    
    # Список монет на випадок помилки API
    fallback = [
        "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT","SHIB/USDT",
        "AVAX/USDT","LINK/USDT","DOT/USDT","TRX/USDT","LTC/USDT","BCH/USDT","ATOM/USDT","NEAR/USDT",
        "OP/USDT","ARB/USDT","APT/USDT","SUI/USDT","FIL/USDT","INJ/USDT","RNDR/USDT","RUNE/USDT",
        "PEPE/USDT","FLOKI/USDT","BONK/USDT","WIF/USDT","SEI/USDT","TON/USDT"
    ]

    try:
        markets = ex.markets if hasattr(ex, "markets") and ex.markets else ex.load_markets()
        allowed = set()
        
        # Фільтруємо тільки активні USDT ф'ючерси
        for sym, m in markets.items():
            if not m.get("active", True): continue
            if not m.get("swap", False): continue     # Тільки перпетуал
            if not m.get("linear", False): continue   # Тільки лінійні (USDT)
            if m.get("quote") != "USDT": continue
            allowed.add(sym)

        tickers = ex.fetch_tickers()
        scored = []
        for sym, t in tickers.items():
            if sym not in allowed: continue
            qv = t.get("quoteVolume", 0) or 0
            scored.append((sym, float(qv)))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s, _ in scored[:top_n]]
        
        return top if top else fallback[:top_n]
    except Exception:
        return fallback[:top_n]

# =========================
# 4. INDICATORS
# =========================
def rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# =========================
# 5. SIGNAL & FORMATTING (UPDATED)
# =========================
def side_from_rsi(last_rsi: float, oversold: float, overbought: float):
    if last_rsi < oversold: return "LONG"
    if last_rsi > overbought: return "SHORT"
    return None

def build_trade_plan(
    coin: str,
    symbol_used: str,
    last_price: float,
    atr_value: float,
    side: str,
    lev_min: int,
    lev_max: int,
    limit_offset_pct: float,
    sl_atr_mult: float,
    tp_multipliers: list[float],
):
    """
    Генерує текст сигналу, ідентичний до прикладу.
    """
    # Очищаємо тікер (XLM/USDT -> XLM)
    base = coin.split("/")[0].split(":")[0]

    # Розрахунок цін входу
    market_entry = last_price
    if side == "SHORT":
        limit_entry = last_price * (1 + limit_offset_pct) # Ліміт вище ринку для шорта
    else:
        limit_entry = last_price * (1 - limit_offset_pct) # Ліміт нижче ринку для лонга

    entry_avg = (market_entry + limit_entry) / 2.0
    R = atr_value * sl_atr_mult

    # Розрахунок SL та TP
    if side == "SHORT":
        stop = entry_avg + R
        tps = [entry_avg - (R * m) for m in tp_multipliers]
        header = f"{base} 📈 SHORT x{lev_min}-{lev_max}"
    else:
        stop = entry_avg - R
        tps = [entry_avg + (R * m) for m in tp_multipliers]
        header = f"{base} 📉 LONG x{lev_min}-{lev_max}"

    # === ФОРМУВАННЯ ТЕКСТУ (СУВОРИЙ ФОРМАТ) ===
    lines = []
    lines.append(header)
    lines.append("") # Порожній рядок
    lines.append("✅ Вход: два ордера")
    lines.append(f"Рынок {fmt_price(symbol_used, market_entry)}")
    lines.append(f"Лимит {fmt_price(symbol_used, limit_entry)}")
    lines.append("")
    lines.append("💸Take-Profit:") # Використовую Take-Profit як у стандарті
    for i, tp in enumerate(tps, start=1):
        lines.append(f"{i}) {fmt_price(symbol_used, tp)}")
    lines.append("")
    lines.append(f"❌Stop-loss: {fmt_price(symbol_used, stop)}")

    return "\n".join(lines)

def style_side(val: str) -> str:
    if val == "LONG": return "color: #00ff7f; font-weight: bold"
    if val == "SHORT": return "color: #ff4d4d; font-weight: bold"
    return "color: #cfcfcf"

# =========================
# 6. SIDEBAR UI
# =========================
st.sidebar.header("🔍 Налаштування")
universe_mode = st.sidebar.radio(
    "Джерело монет:",
    ["Auto: Top Volume USDT", "Ручний список"],
    index=0
)

st.sidebar.subheader("Логіка RSI")
timeframe = st.sidebar.selectbox("Таймфрейм", ['5m', '15m', '1h', '4h'], index=1)
rsi_period = st.sidebar.slider("Період RSI", 7, 21, 14)
overbought = st.sidebar.number_input("RSI для SHORT (>)", value=70.0)
oversold = st.sidebar.number_input("RSI для LONG (<)", value=30.0)
limit_candles = 200

st.sidebar.subheader("Параметри Угоди")
col_lev1, col_lev2 = st.sidebar.columns(2)
lev_min = int(col_lev1.number_input("Плече мін", value=20))
lev_max = int(col_lev2.number_input("Плече макс", value=25))
limit_offset_pct = st.sidebar.slider("Відступ лімітки (%)", 0.1, 5.0, 2.0, 0.1) / 100.0

atr_period = st.sidebar.slider("Період ATR", 7, 21, 14)
sl_atr_mult = st.sidebar.slider("SL множник ATR", 0.5, 4.0, 1.5, 0.1)
tp_mult_str = st.sidebar.text_input("TP множники (через кому)", value="1, 2, 3")

# Парсинг множників TP
try:
    tp_multipliers = [float(x.strip()) for x in tp_mult_str.split(",") if x.strip()]
except:
    tp_multipliers = [1, 2, 3]

if universe_mode.startswith("Auto"):
    top_n = st.sidebar.slider("Скільки монет сканувати?", 10, 100, 30)
    coins_universe = get_top_usdt_perp_symbols(top_n)
    st.sidebar.success(f"Завантажено {len(coins_universe)} активних монет.")
else:
    coins_universe = st.sidebar.multiselect(
        "Оберіть монети:",
        ['BTC/USDT','ETH/USDT','SOL/USDT','XRP/USDT','BNB/USDT','DOGE/USDT','PEPE/USDT','APT/USDT'],
        default=['BTC/USDT','ETH/USDT','SOL/USDT']
    )

# =========================
# 7. STATE & EXECUTION
# =========================
if "scan_data" not in st.session_state:
    st.session_state.scan_data = None
    st.session_state.trade_posts = []
    st.session_state.scan_errors = []

def analyze_market(coins):
    rows = []
    errors = []
    
    progress_bar = st.progress(0)
    
    for idx, coin in enumerate(coins):
        df, err = fetch_ohlcv_cached(coin, timeframe, limit_candles)
        if df is None or df.empty:
            errors.append((coin, err))
            continue
            
        df["rsi"] = rsi_series(df["close"], rsi_period)
        df["atr"] = atr_series(df, atr_period)
        
        last_price = float(df["close"].iloc[-1])
        last_rsi = float(df["rsi"].iloc[-1])
        last_atr = float(df["atr"].iloc[-1])
        sym_used = df["symbol_used"].iloc[-1]
        
        side = side_from_rsi(last_rsi, oversold, overbought)
        
        rows.append({
            "Coin": coin,
            "Price": last_price,
            "RSI": last_rsi,
            "ATR": last_atr,
            "Side": side if side else "-",
            "SymbolUsed": sym_used
        })
        progress_bar.progress((idx + 1) / len(coins))
        
    progress_bar.empty()
    
    out = pd.DataFrame(rows)
    if not out.empty:
        # Сортуємо: спочатку сигнали, потім решта
        out["_sort"] = out["Side"].apply(lambda x: 0 if x in ["LONG", "SHORT"] else 1)
        out = out.sort_values(["_sort", "Coin"]).drop(columns=["_sort"])
    return out, errors

# Кнопка запуску
if st.button("🚀 СКАНУВАТИ РИНОК", type="primary"):
    with st.spinner("Аналізую ринок..."):
        data, errs = analyze_market(coins_universe)
        
        # Генерація постів
        posts = []
        if not data.empty:
            signals = data[data["Side"].isin(["LONG", "SHORT"])]
            for _, row in signals.iterrows():
                post_text = build_trade_plan(
                    coin=row["Coin"],
                    symbol_used=row["SymbolUsed"],
                    last_price=row["Price"],
                    atr_value=row["ATR"],
                    side=row["Side"],
                    lev_min=lev_min,
                    lev_max=lev_max,
                    limit_offset_pct=limit_offset_pct,
                    sl_atr_mult=sl_atr_mult,
                    tp_multipliers=tp_multipliers
                )
                posts.append(post_text)

        st.session_state.scan_data = data
        st.session_state.scan_errors = errs
        st.session_state.trade_posts = posts

# =========================
# 8. OUTPUT DISPLAY
# =========================
data = st.session_state.scan_data

if data is not None:
    tab1, tab2, tab3 = st.tabs(["📋 Таблиця", "📢 Сигнали (Copy-Paste)", "📈 Графік"])
    
    with tab1:
        st.subheader(f"Результати ({len(data)} монет)")
        
        # Виділяємо кольором LONG/SHORT
        styled_df = data.style.map(style_side, subset=["Side"]).format({
            "Price": "{:.5f}", 
            "RSI": "{:.1f}", 
            "ATR": "{:.5f}"
        })
        st.dataframe(styled_df, use_container_width=True, height=600)
        
        if st.session_state.scan_errors:
            with st.expander("Помилки завантаження"):
                st.write(st.session_state.scan_errors)

    with tab2:
        st.subheader("Готові пости для Telegram")
        if not st.session_state.trade_posts:
            st.info("Наразі сигналів немає (RSI в межах норми). Спробуйте змінити налаштування RSI.")
        else:
            cols = st.columns(2)
            for i, post in enumerate(st.session_state.trade_posts):
                with cols[i % 2]:
                    st.text_area(f"Сигнал #{i+1}", post, height=300)
                    st.button(f"Копіювати #{i+1}", disabled=True, help="Виділіть текст вище та скопіюйте")

    with tab3:
        st.subheader("Перевірка на графіку")
        coin_list = data["Coin"].tolist()
        coin_sel = st.selectbox("Оберіть монету", coin_list)
        
        if coin_sel:
            df_chart, _ = fetch_ohlcv_cached(coin_sel, timeframe, limit_candles)
            if df_chart is not None:
                df_chart["rsi"] = rsi_series(df_chart["close"], rsi_period)
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                fig.add_trace(go.Candlestick(x=df_chart["timestamp"], open=df_chart["open"], high=df_chart["high"],
                                             low=df_chart["low"], close=df_chart["close"], name="Price"), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart["timestamp"], y=df_chart["rsi"], name="RSI", line=dict(color='purple')), row=2, col=1)
                
                fig.add_hline(y=overbought, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=oversold, line_dash="dash", line_color="green", row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Натисніть кнопку **СКАНУВАТИ РИНОК** зліва або зверху.")