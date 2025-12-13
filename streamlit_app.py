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

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .stDataFrame { font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 16px !important; }

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

    .data-row { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.95em; }
    .label { color: #8b92a6; }
    .value { color: #e0e0e0; font-weight: 500; font-family: 'Roboto Mono', monospace; }

    .trend-info { margin-top: 10px; font-size: 0.85em; color: #8b92a6; font-style: italic; }
    .warning { color: #ffa726; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🎯 Multi-Exchange Sniper Pro V5")
st.markdown("RSI + Trend Scanner: **Binance, Bybit, KuCoin Futures, OKX, Kraken Futures**")

# =========================
# 2. CORE UTILS
# =========================
def fmt_price(price):
    if not isinstance(price, (int, float, np.floating)) or pd.isna(price):
        return "N/A"
    price = float(price)
    if price >= 1000:
        return f"{price:.1f}"
    if price >= 10:
        return f"{price:.2f}"
    if price >= 0.1:
        return f"{price:.4f}"
    return f"{price:.8f}".rstrip("0").rstrip(".")


def normalize_manual_item(x: str) -> tuple[str, str]:
    """Приймає 'BTC/USDT', 'BTCUSDT', 'btc/usdt' і т.п. Повертає (base, quote)."""
    x = (x or "").strip().upper().replace(" ", "")
    if not x:
        return ("", "")

    if "/" in x:
        base, quote = x.split("/", 1)
        return base, quote

    # евристика: BTCUSDT -> BTC/USDT
    if x.endswith("USDT"):
        return x[:-4], "USDT"
    if x.endswith("USD"):
        return x[:-3], "USD"

    # якщо незрозуміло — вважаємо USDT
    return x, "USDT"


# =========================
# 3. DATA ENGINE
# =========================
# ✅ Важливо: KuCoin та Kraken futures мають окремі класи в CCXT
EXCHANGE_CLASSES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "kucoin": ccxt.kucoinfutures,
    "okx": ccxt.okx,
    "kraken": ccxt.krakenfutures,
}

SUPPORTED_EXCHANGES = list(EXCHANGE_CLASSES.keys())


@st.cache_resource
def get_exchange_config(exchange_id: str):
    """
    Базова конфігурація. Не створюємо тут інстанс біржі.
    defaultType відрізняється між біржами.
    """
    config = {"enableRateLimit": True, "options": {}}

    # практично корисні дефолти для деривативів
    if exchange_id == "binance":
        config["options"]["defaultType"] = "future"  # USDⓈ-M futures
    elif exchange_id == "bybit":
        config["options"]["defaultType"] = "swap"    # perp
    elif exchange_id == "kucoin":
        config["options"]["defaultType"] = "swap"    # kucoin futures = swap/perp в ccxt
    elif exchange_id == "okx":
        config["options"]["defaultType"] = "swap"    # OKX swap
    elif exchange_id == "kraken":
        config["options"]["defaultType"] = "future"  # Kraken Futures
    else:
        config["options"]["defaultType"] = "swap"

    return config


def is_target_derivative_market(exchange_id: str, m: dict) -> bool:
    """
    Надійний фільтр деривативів:
    - active
    - contract (swap/future)
    - quote (USDT для більшості; для Kraken часто USD)
    - для USDT-маржинальних бірж: linear True
    """
    if not m or not m.get("active"):
        return False

    allowed_quotes = {"USDT"} if exchange_id != "kraken" else {"USD", "USDT"}
    if m.get("quote") not in allowed_quotes:
        return False

    # contract=True відділяє деривативи від spot
    if not m.get("contract"):
        return False

    # маємо бути swap або future
    if not (m.get("swap") or m.get("future")):
        return False

    # Для більшості “USDT perpetual” хочемо linear (USDT-margined)
    # Kraken futures може мати інші поля/структуру — не душимо.
    if exchange_id != "kraken":
        if not m.get("linear", False):
            return False

    return True


@st.cache_data(ttl=300, show_spinner=False)
def get_market_data(exchange_id: str, scan_mode: str, top_n: int, manual_list: list):
    """
    Отримує список символів та 24h%:
    - Auto: беремо всі деривативи, ранжуємо по об’єму, обрізаємо топ-N
    - Manual: матчимо по base/quote, але повертаємо реальні symbol біржі
    """
    config = get_exchange_config(exchange_id)
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass:
        return [], {}

    ex = ExClass(config)

    try:
        markets = ex.load_markets()

        # --- 1) Формуємо пул символів
        if scan_mode.startswith("Auto"):
            target_symbols = [s for s, m in markets.items() if is_target_derivative_market(exchange_id, m)]
        else:
            wanted = []
            for it in manual_list:
                base, quote = normalize_manual_item(it)
                if base and quote:
                    wanted.append((base, quote))

            target_symbols = []
            for base, quote in wanted:
                found = None
                for s, m in markets.items():
                    if not is_target_derivative_market(exchange_id, m):
                        continue
                    if (m.get("base") == base) and (m.get("quote") == quote):
                        found = s
                        break
                if found:
                    target_symbols.append(found)

        # прибираємо дублікати
        target_symbols = list(dict.fromkeys(target_symbols))
        if not target_symbols:
            return [], {}

        # --- 2) Тікери: пробуємо fetch_tickers(list), якщо падає — беремо всі і фільтруємо
        try:
            tickers = ex.fetch_tickers(target_symbols)
        except Exception:
            tickers_all = ex.fetch_tickers()
            tickers = {k: v for k, v in tickers_all.items() if k in target_symbols}

        scored = []
        for s in target_symbols:
            t = tickers.get(s) or {}
            vol = t.get("quoteVolume") or t.get("baseVolume") or t.get("volume") or 0
            chg = t.get("percentage") or 0

            try:
                vol = float(vol) if vol else 0.0
            except Exception:
                vol = 0.0

            try:
                chg = float(chg) if chg is not None else 0.0
            except Exception:
                chg = 0.0

            scored.append((s, vol, chg))

        scored.sort(key=lambda x: x[1], reverse=True)
        final = scored[: (top_n if scan_mode.startswith("Auto") else len(scored))]

        coins = [x[0] for x in final]
        changes = {x[0]: x[2] for x in final}
        return coins, changes

    except Exception as e:
        st.error(f"API Error ({exchange_id}): {e}")
        return [], {}


def fetch_candle_data(args):
    """Потокове завантаження OHLCV"""
    symbol, tf, limit, exchange_id, config = args
    ExClass = EXCHANGE_CLASSES.get(exchange_id)
    if not ExClass:
        return symbol, None, "Unknown exchange class"

    ex = ExClass(config)

    try:
        time.sleep(0.1)  # м’яка пауза проти rate-limit у потоках

        # OKX іноді потребує load_markets для коректного парсингу символа
        if exchange_id == "okx":
            ex.load_markets()

        ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not ohlcv:
            return symbol, None, "Empty Data"

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return symbol, df, None
    except Exception as e:
        return symbol, None, str(e)


# =========================
# 4. ANALYSIS LOGIC
# =========================
def analyze_market(df: pd.DataFrame, rsi_len: int, ema_len: int, os_level: float, ob_level: float):
    if df is None or len(df) < max(ema_len, rsi_len, 20):
        return None

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # RSI (Wilder-like via ewm)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_len, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / rsi_len, adjust=False).mean()

    # якщо loss==0 -> RSI=100; якщо gain==0 -> RSI=0
    rs = pd.Series(np.where(loss.values == 0, np.inf, (gain / loss).values), index=df.index)
    rsi = 100 - (100 / (1 + rs))
    df["rsi"] = rsi

    # ATR (True Range, ewm)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()

    # Trend EMA
    df["ema"] = close.ewm(span=ema_len, adjust=False).mean()

    last = df.iloc[-1]
    if pd.isna(last["rsi"]) or pd.isna(last["ema"]) or pd.isna(last["atr"]):
        return None

    sig = None
    if last["rsi"] < os_level:
        sig = "LONG"
    elif last["rsi"] > ob_level:
        sig = "SHORT"

    trend = "NEUTRAL"
    if last["close"] > last["ema"]:
        trend = "BULLISH 🟢"
    elif last["close"] < last["ema"]:
        trend = "BEARISH 🔴"

    warn = ""
    if (sig == "LONG" and "BEARISH" in trend) or (sig == "SHORT" and "BULLISH" in trend):
        warn = "⚠️ Counter-Trend"

    return {
        "price": float(last["close"]),
        "rsi": float(last["rsi"]),
        "atr": float(last["atr"]),
        "trend": trend,
        "signal": sig,
        "warning": warn,
    }


def create_telegram_post(coin, data, params, exchange_id):
    side = data["signal"]
    price = data["price"]
    atr = data["atr"]

    lev = params["lev"]
    offset = params["offset"]
    sl_mult = params["sl"]
    tps = params["tps"]

    emoji = "🟢" if side == "LONG" else "🔴"

    # Limit entry
    limit_price = price * (1 - offset) if side == "LONG" else price * (1 + offset)

    # SL/TP from limit
    if side == "LONG":
        sl_price = limit_price - (atr * sl_mult)
        tp_prices = [limit_price + (atr * m) for m in tps]
    else:
        sl_price = limit_price + (atr * sl_mult)
        tp_prices = [limit_price - (atr * m) for m in tps]

    risk = abs(limit_price - sl_price)
    reward = abs(limit_price - tp_prices[-1])
    rr = reward / risk if risk else 0.0

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
    exch = st.selectbox("Біржа", SUPPORTED_EXCHANGES, format_func=str.upper)
    mode = st.radio("Режим пошуку", ["Auto (Top Volume)", "Manual List"])

    if "Auto" in mode:
        n_coins = st.slider("Кількість монет", 10, 150, 30)
        manual_coins = []
    else:
        n_coins = 0
        default_list = "BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, DOGE/USDT, XRP/USDT, LTC/USDT"
        raw_manual = st.text_area("Список монет (через кому)", default_list)
        manual_coins = [x.strip() for x in raw_manual.split(",") if x.strip()]

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
    p_tps = [1.0, 2.5, 4.0]

# =========================
# 6. MAIN APP
# =========================
c1, c2 = st.columns([3, 1])
c1.subheader(f"📡 Сканер: {exch.upper()} [{tf}]")
run = c2.button("🚀 ЗАПУСТИТИ СКАНЕР", type="primary", use_container_width=True)

if run:
    with st.spinner("Отримання ринкових даних..."):
        # важливо: чистимо саме cache_data, щоб взяти свіжий топ/список
        get_market_data.clear()
        coins, changes_dict = get_market_data(exch, mode, n_coins, manual_coins)

    if not coins:
        st.error("Не знайдено монет. Перевірте список або налаштування (особливо Manual).")
        st.stop()

    progress = st.progress(0)
    status_text = st.empty()

    results = []

    ex_conf = get_exchange_config(exch)
    candle_limit = max(ema_len + 50, 250)  # запас під EMA/RSI/ATR
    tasks = [(c, tf, candle_limit, exch, ex_conf) for c in coins]

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
            if not analysis:
                continue

            post_txt = ""
            if analysis["signal"] in ("LONG", "SHORT"):
                post_params = {"lev": p_lev, "offset": p_off, "sl": p_sl, "tps": p_tps}
                post_txt = create_telegram_post(symbol, analysis, post_params, exch)

            results.append(
                {
                    "Coin": symbol,
                    "Price": analysis["price"],
                    "24h%": float(changes_dict.get(symbol, 0) or 0),
                    "RSI": analysis["rsi"],
                    "Signal": analysis["signal"],
                    "Trend": analysis["trend"],
                    "Warning": analysis["warning"],
                    "Post": post_txt,
                }
            )

    progress.empty()
    status_text.empty()

    df_res = pd.DataFrame(results)

    if df_res.empty:
        st.warning("Дані не отримано (або все відвалилось на API/rate-limit).")
        st.stop()

    # =========================
    # Sorting (✅ фікс NaN/None)
    # =========================
    df_res["_sort_sig"] = df_res["Signal"].apply(lambda x: 1 if pd.isna(x) else 0)
    df_res["_sort_rsi"] = df_res.apply(
        lambda r: r["RSI"]
        if r["Signal"] == "LONG"
        else (100 - r["RSI"] if r["Signal"] == "SHORT" else 50),
        axis=1,
    )
    df_res = df_res.sort_values(by=["_sort_sig", "_sort_rsi"], ascending=[True, True])

    tab_sig, tab_all = st.tabs(["📱 Сигнали", "📋 Всі Результати"])

    # --- MOBILE VIEW ---
    with tab_sig:
        signals = df_res[df_res["Signal"].isin(["LONG", "SHORT"])]
        if signals.empty:
            st.info("🟢 Немає активних сигналів за вказаними параметрами.")
        else:
            for _, row in signals.iterrows():
                sig_class = "badge-long" if row["Signal"] == "LONG" else "badge-short"
                warn_html = f'<span class="warning">{row["Warning"]}</span>' if row["Warning"] else ""

                st.markdown(
                    f"""
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
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("📋 Копіювати сигнал"):
                    st.code(row["Post"], language="text")

    # --- DESKTOP TABLE ---
    with tab_all:
        view = df_res[["Coin", "Price", "24h%", "RSI", "Signal", "Trend", "Warning"]].copy()

        st.data_editor(
            view,
            column_config={
                "RSI": st.column_config.ProgressColumn("RSI", min_value=0, max_value=100, format="%.1f"),
                "Price": st.column_config.NumberColumn(format="%.8f"),
                "24h%": st.column_config.NumberColumn(format="%.2f%%"),
            },
            use_container_width=True,
            height=600,
            hide_index=True,
        )