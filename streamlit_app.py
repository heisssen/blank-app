import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =========================================================
# 0) PAGE
# =========================================================
st.set_page_config(page_title="Arbitrage Radar Pro", layout="wide", page_icon="🔁")

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .card {
        background-color:#151822; border:1px solid #2b2d35;
        border-radius:12px; padding:14px; margin:10px 0;
        box-shadow:0 4px 10px rgba(0,0,0,0.25);
    }
    .row { display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap; }
    .pill { padding:4px 10px; border-radius:999px; font-weight:700; font-size:12px; }
    .pill-ok { background:#123a2a; color:#40ff9a; border:1px solid #40ff9a; }
    .pill-warn { background:#3a2b12; color:#ffcc66; border:1px solid #ffcc66; }
    .pill-bad { background:#3a1212; color:#ff6666; border:1px solid #ff6666; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .muted { color:#8b92a6; }
    .big { font-size:18px; font-weight:800; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔁 Arbitrage Radar Pro")
st.caption("Cross-Exchange Spot Arbitrage Scanner (USDT pairs) — fees + slippage + orderbook depth analytics")

# =========================================================
# 1) EXCHANGES
# =========================================================
EXCHANGE_CLASSES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "okx": ccxt.okx,
    "kucoin": ccxt.kucoin,
    "kraken": ccxt.kraken,
}

DEFAULT_TAKER = {
    "binance": 0.0010,  # 0.10%
    "bybit":   0.0010,
    "okx":     0.0010,
    "kucoin":  0.0010,
    "kraken":  0.0026,  # часто вище
}

def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def fmt_price(p):
    if p is None or not np.isfinite(p):
        return "N/A"
    p = float(p)
    if p >= 1000: return f"{p:.2f}"
    if p >= 10: return f"{p:.4f}"
    if p >= 0.1: return f"{p:.6f}"
    return f"{p:.10f}".rstrip("0").rstrip(".")

def fmt_pct(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.2f}%"

@st.cache_resource
def get_exchange(ex_id: str):
    Ex = EXCHANGE_CLASSES[ex_id]
    ex = Ex({"enableRateLimit": True})
    return ex

@st.cache_data(ttl=3600, show_spinner=False)
def load_markets_cached(ex_id: str):
    ex = get_exchange(ex_id)
    markets = ex.load_markets()
    return markets

def is_good_usdt_spot_market(m: dict):
    # spot, active, quote USDT, без leveraged токенів та екзотики
    if not m or not m.get("active"):
        return False
    if not m.get("spot", True):
        return False
    if m.get("quote") != "USDT":
        return False
    sym = m.get("symbol", "")
    bad = ["UP/", "DOWN/", "BULL/", "BEAR/", "3L/", "3S/", "5L/", "5S/"]
    if any(b in sym for b in bad):
        return False
    return True

def get_taker_fee(ex_id: str, markets: dict, symbol: str, fallback: float):
    m = markets.get(symbol) or {}
    fee = m.get("taker", None)
    if fee is None:
        return fallback
    fee = safe_float(fee, fallback)
    if not np.isfinite(fee) or fee <= 0:
        return fallback
    return fee

# =========================================================
# 2) DATA FETCH
# =========================================================
def fetch_tickers_all(ex_id: str):
    ex = get_exchange(ex_id)
    try:
        t = ex.fetch_tickers()
        return ex_id, t, None
    except Exception as e:
        return ex_id, None, str(e)

def fetch_orderbook(ex_id: str, symbol: str, limit: int = 50):
    ex = get_exchange(ex_id)
    try:
        ob = ex.fetch_order_book(symbol, limit=limit)
        return ex_id, symbol, ob, None
    except Exception as e:
        return ex_id, symbol, None, str(e)

def orderbook_depth_usdt(ob: dict, side: str, top_price: float, band_pct: float):
    """
    Рахуємо сумарний notional (USDT) в межах band_pct від top_price.
    side='asks' для покупки, 'bids' для продажу.
    """
    if not ob or side not in ob or not ob[side]:
        return 0.0, 0.0

    band = band_pct / 100.0
    levels = ob[side]
    notional = 0.0
    qty = 0.0

    if side == "asks":
        max_price = top_price * (1 + band)
        for price, amount in levels:
            price = safe_float(price, np.nan)
            amount = safe_float(amount, 0.0)
            if not np.isfinite(price) or amount <= 0:
                continue
            if price > max_price:
                break
            notional += price * amount
            qty += amount
    else:
        min_price = top_price * (1 - band)
        for price, amount in levels:
            price = safe_float(price, np.nan)
            amount = safe_float(amount, 0.0)
            if not np.isfinite(price) or amount <= 0:
                continue
            if price < min_price:
                break
            notional += price * amount
            qty += amount

    return float(notional), float(qty)

# =========================================================
# 3) ARB ENGINE
# =========================================================
def build_symbol_universe(selected_exchanges, mode, top_n, manual_syms, ref_exchange="binance"):
    """
    Повертає список символів для скану.
    Auto: беремо топ по об'єму на ref_exchange (USDT spot), і залишаємо ті, що є хоча б на 2 біржах.
    Manual: перетворюємо на стандартний формат 'AAA/USDT', і фільтруємо по доступності на >=2 біржах.
    """
    markets_by_ex = {}
    sym_sets = []
    for ex_id in selected_exchanges:
        mk = load_markets_cached(ex_id)
        markets_by_ex[ex_id] = mk
        syms = {s for s, m in mk.items() if is_good_usdt_spot_market(m)}
        sym_sets.append(syms)

    # символи, що зустрічаються мінімум на 2 біржах
    union = set().union(*sym_sets) if sym_sets else set()
    common2 = [s for s in union if sum(1 for ss in sym_sets if s in ss) >= 2]

    if mode.startswith("Manual"):
        cleaned = []
        for x in manual_syms:
            x = (x or "").strip().upper().replace(" ", "")
            if not x:
                continue
            if "/" not in x:
                x = f"{x}/USDT"
            cleaned.append(x)
        cleaned = list(dict.fromkeys(cleaned))
        final = [s for s in cleaned if s in common2]
        return final, markets_by_ex

    # Auto: топ по об'єму на ref_exchange
    if ref_exchange not in selected_exchanges:
        ref_exchange = selected_exchanges[0]

    ref_markets = markets_by_ex[ref_exchange]
    ref_symbols = [s for s in common2 if s in ref_markets]

    # для швидкості: беремо tickers ref_exchange і сортуємо по quoteVolume
    ex = get_exchange(ref_exchange)
    try:
        tickers = ex.fetch_tickers()
    except Exception:
        tickers = {}

    scored = []
    for s in ref_symbols:
        t = tickers.get(s) or {}
        vol = safe_float(t.get("quoteVolume") or t.get("baseVolume") or t.get("volume"), 0.0)
        last = safe_float(t.get("last"), np.nan)
        if not np.isfinite(last) or last <= 0:
            continue
        scored.append((s, vol))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_n]], markets_by_ex

def compute_arb_for_symbol(symbol, selected_exchanges, tickers_by_ex, markets_by_ex, default_fee, slippage_pct):
    """
    Для символу:
    - знаходимо мінімальний ask (buy) та максимальний bid (sell)
    - рахуємо gross/net, враховуючи taker fees і slippage buffer
    """
    quotes = []
    for ex_id in selected_exchanges:
        t = (tickers_by_ex.get(ex_id) or {}).get(symbol) or {}
        bid = safe_float(t.get("bid"), np.nan)
        ask = safe_float(t.get("ask"), np.nan)
        last = safe_float(t.get("last"), np.nan)

        # якщо bid/ask відсутні — пробуємо з last і пропускаємо (бо арб без bid/ask не чесний)
        if not np.isfinite(bid) or not np.isfinite(ask) or bid <= 0 or ask <= 0:
            continue

        mk = markets_by_ex[ex_id]
        taker = get_taker_fee(ex_id, mk, symbol, fallback=default_fee.get(ex_id, 0.001))

        quotes.append({
            "ex": ex_id,
            "bid": bid,
            "ask": ask,
            "last": last,
            "taker": taker
        })

    if len(quotes) < 2:
        return None

    buy = min(quotes, key=lambda x: x["ask"])
    sell = max(quotes, key=lambda x: x["bid"])

    if sell["bid"] <= buy["ask"]:
        return None

    buy_price = buy["ask"]
    sell_price = sell["bid"]

    gross = (sell_price - buy_price) / buy_price * 100.0

    # fees (%)
    fee_pct = (buy["taker"] + sell["taker"]) * 100.0

    # slippage buffer (%): застосуємо двічі (buy гірше, sell гірше)
    slip = slippage_pct * 2.0

    net = gross - fee_pct - slip

    return {
        "symbol": symbol,
        "buy_ex": buy["ex"],
        "sell_ex": sell["ex"],
        "buy_price": buy_price,
        "sell_price": sell_price,
        "gross_pct": gross,
        "net_pct": net,
        "fee_pct": fee_pct,
        "slip_pct": slip,
        "buy_taker": buy["taker"],
        "sell_taker": sell["taker"],
    }

def make_telegram_text(row, notional, depth_band, buy_depth, sell_depth):
    sym = row["symbol"].split("/")[0]
    txt = f"#{sym} 🔁 ARB SPOT\n"
    txt += f"🟢 BUY: {row['buy_ex'].upper()} @ {fmt_price(row['buy_price'])}\n"
    txt += f"🔴 SELL: {row['sell_ex'].upper()} @ {fmt_price(row['sell_price'])}\n"
    txt += "------------------\n"
    txt += f"📈 Gross: {row['gross_pct']:.2f}%\n"
    txt += f"🧾 Fees: {row['fee_pct']:.2f}% | Slippage buf: {row['slip_pct']:.2f}%\n"
    txt += f"✅ Net: {row['net_pct']:.2f}%\n"
    txt += "------------------\n"
    txt += f"💧 Depth ±{depth_band:.2f}%: BUY≈{buy_depth:,.0f} USDT | SELL≈{sell_depth:,.0f} USDT\n"
    txt += f"💰 Est. PnL on {notional:,.0f} USDT: {(notional*row['net_pct']/100.0):,.2f} USDT\n"
    txt += f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"
    return txt

# =========================================================
# 4) SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Налаштування")

with st.sidebar.expander("🏦 Біржі", expanded=True):
    selected = st.multiselect(
        "Обери біржі (мінімум 2)",
        list(EXCHANGE_CLASSES.keys()),
        default=["binance", "bybit", "okx"],
        format_func=str.upper,
    )

    ref_exchange = st.selectbox(
        "Reference біржа (для Auto топ-об’єму)",
        options=selected if selected else list(EXCHANGE_CLASSES.keys()),
        index=0 if selected else 0,
        format_func=str.upper,
    )

with st.sidebar.expander("🎛️ Скан-режим", expanded=True):
    mode = st.radio("Режим", ["Auto (Top Volume)", "Manual List"])

    if mode.startswith("Auto"):
        top_n = st.slider("Скільки монет сканити", 10, 200, 60)
        manual_syms = []
    else:
        top_n = 0
        raw = st.text_area("Монети (через кому): BTC, ETH, SOL ...", "BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, LINK")
        manual_syms = [x.strip() for x in raw.split(",") if x.strip()]

with st.sidebar.expander("🧾 Фі/фільтри", expanded=True):
    # дефолтні комісії (taker) — якщо біржа не поверне market.taker
    df_fee = st.number_input("Default taker fee (якщо біржа не віддає)", min_value=0.0, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    # можна перезаписати DEFAULT_TAKER під це
    fee_override = {k: DEFAULT_TAKER.get(k, df_fee) for k in EXCHANGE_CLASSES.keys()}
    for k in fee_override:
        fee_override[k] = df_fee  # глобальний дефолт (простий варіант)

    slippage_pct = st.slider("Slippage buffer (в %)", 0.0, 1.0, 0.15, step=0.05)

    min_net = st.slider("Мінімальний Net % (показувати)", 0.0, 5.0, 0.40, step=0.05)
    max_results = st.slider("Скільки топ-угод показувати", 5, 100, 25)

with st.sidebar.expander("💧 Ліквідність (orderbook)", expanded=True):
    depth_band = st.slider("Depth band ±% від best", 0.05, 1.00, 0.30, step=0.05)
    min_depth_usdt = st.number_input("Мін. depth (USDT) на buy і sell", min_value=0.0, value=20000.0, step=5000.0)
    ob_limit = st.selectbox("Orderbook levels", [20, 50, 100], index=1)

with st.sidebar.expander("💰 PnL калькулятор", expanded=False):
    notional = st.number_input("Номінал (USDT) для оцінки прибутку", min_value=50.0, value=1000.0, step=50.0)

# =========================================================
# 5) RUN
# =========================================================
c1, c2 = st.columns([3, 1])
c1.subheader("📡 Сканер можливостей")
run = c2.button("🚀 СКАН", type="primary", use_container_width=True)

if not selected or len(selected) < 2:
    st.warning("Обери мінімум 2 біржі.")
    st.stop()

if run:
    t0 = time.time()

    # 1) Universe
    with st.spinner("Збираю спільні ринки..."):
        symbols, markets_by_ex = build_symbol_universe(
            selected_exchanges=selected,
            mode=mode,
            top_n=top_n if top_n else 0,
            manual_syms=manual_syms,
            ref_exchange=ref_exchange,
        )

    if not symbols:
        st.error("Немає символів для скану (нема спільних USDT spot пар на >=2 біржах).")
        st.stop()

    # 2) Fetch tickers for each exchange (all → filter)
    with st.spinner("Тягну тікери з бірж..."):
        tickers_by_ex = {}
        errors = []

        with ThreadPoolExecutor(max_workers=min(8, len(selected))) as exr:
            futs = [exr.submit(fetch_tickers_all, ex_id) for ex_id in selected]
            for f in as_completed(futs):
                ex_id, data, err = f.result()
                if err:
                    errors.append((ex_id, err))
                    tickers_by_ex[ex_id] = {}
                else:
                    tickers_by_ex[ex_id] = data or {}

    if errors:
        st.info("Частина бірж могла не віддати тікери (rate-limit/ban/мережа). Я продовжив з тим, що є.")
        with st.expander("⚠️ Помилки бірж"):
            for ex_id, err in errors:
                st.write(f"{ex_id.upper()}: {err}")

    # 3) Compute candidates
    with st.spinner("Рахую арбітражні спреди..."):
        rows = []
        for sym in symbols:
            r = compute_arb_for_symbol(
                symbol=sym,
                selected_exchanges=selected,
                tickers_by_ex=tickers_by_ex,
                markets_by_ex=markets_by_ex,
                default_fee=fee_override,
                slippage_pct=slippage_pct,
            )
            if not r:
                continue
            if r["net_pct"] >= min_net:
                rows.append(r)

        if not rows:
            st.warning("Нічого не знайшов за твоїми фільтрами (net%/fees/slippage).")
            st.stop()

        df = pd.DataFrame(rows)
        df = df.sort_values("net_pct", ascending=False).head(max_results).reset_index(drop=True)

    # 4) For top candidates, pull orderbooks for buy/sell and compute depth
    with st.spinner("Підтягую orderbook і рахую depth..."):
        depth_buy = []
        depth_sell = []
        depth_ok = []
        tg_texts = []

        # сформуємо задачі на orderbook лише для топу
        tasks = []
        for _, r in df.iterrows():
            tasks.append((r["buy_ex"], r["symbol"]))
            tasks.append((r["sell_ex"], r["symbol"]))

        ob_map = {}  # (ex,sym) -> ob
        with ThreadPoolExecutor(max_workers=10) as exr:
            futs = [exr.submit(fetch_orderbook, ex_id, sym, ob_limit) for ex_id, sym in tasks]
            for f in as_completed(futs):
                ex_id, sym, ob, err = f.result()
                if err or not ob:
                    ob_map[(ex_id, sym)] = None
                else:
                    ob_map[(ex_id, sym)] = ob

        for _, r in df.iterrows():
            buy_ob = ob_map.get((r["buy_ex"], r["symbol"]))
            sell_ob = ob_map.get((r["sell_ex"], r["symbol"]))

            # top from ob (якщо є), інакше з тікера
            buy_top = None
            sell_top = None
            if buy_ob and buy_ob.get("asks"):
                buy_top = safe_float(buy_ob["asks"][0][0], r["buy_price"])
            else:
                buy_top = r["buy_price"]

            if sell_ob and sell_ob.get("bids"):
                sell_top = safe_float(sell_ob["bids"][0][0], r["sell_price"])
            else:
                sell_top = r["sell_price"]

            b_notional, _ = orderbook_depth_usdt(buy_ob, "asks", buy_top, depth_band)
            s_notional, _ = orderbook_depth_usdt(sell_ob, "bids", sell_top, depth_band)

            ok = (b_notional >= min_depth_usdt) and (s_notional >= min_depth_usdt)

            depth_buy.append(b_notional)
            depth_sell.append(s_notional)
            depth_ok.append(ok)

            tg_texts.append(make_telegram_text(r, notional, depth_band, b_notional, s_notional))

        df["buy_depth_usdt"] = depth_buy
        df["sell_depth_usdt"] = depth_sell
        df["depth_ok"] = depth_ok
        df["telegram"] = tg_texts

    # 5) Output
    dt = time.time() - t0
    st.success(f"Готово. Символів у скані: {len(symbols)} | Кандидатів (net≥{min_net}%): {len(df)} | {dt:.1f}s")

    good = df[df["depth_ok"] == True].copy()
    meh = df[df["depth_ok"] == False].copy()

    st.subheader("✅ Найкращі (net + достатня ліквідність)")
    if good.empty:
        st.info("Немає варіантів, що проходять по depth. Зменш min_depth_usdt або збільш depth_band.")
    else:
        for _, r in good.iterrows():
            badge = "pill-ok"
            st.markdown(
                f"""
<div class="card">
  <div class="row">
    <div class="big">{r['symbol']}</div>
    <div class="pill {badge}">NET {fmt_pct(r['net_pct'])}</div>
  </div>
  <div class="row muted">
    <div>BUY: <span class="mono">{r['buy_ex'].upper()}</span> @ <span class="mono">{fmt_price(r['buy_price'])}</span></div>
    <div>SELL: <span class="mono">{r['sell_ex'].upper()}</span> @ <span class="mono">{fmt_price(r['sell_price'])}</span></div>
  </div>
  <div class="row muted">
    <div>Gross: <span class="mono">{fmt_pct(r['gross_pct'])}</span></div>
    <div>Fees: <span class="mono">{fmt_pct(r['fee_pct'])}</span> | Slippage: <span class="mono">{fmt_pct(r['slip_pct'])}</span></div>
  </div>
  <div class="row muted">
    <div>Depth ±{depth_band:.2f}%: BUY≈<span class="mono">{r['buy_depth_usdt']:,.0f}</span> USDT</div>
    <div>Depth ±{depth_band:.2f}%: SELL≈<span class="mono">{r['sell_depth_usdt']:,.0f}</span> USDT</div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            with st.expander("📋 Telegram"):
                st.code(r["telegram"], language="text")

    st.subheader("⚠️ Є спред, але depth слабкий")
    if meh.empty:
        st.caption("Порожньо.")
    else:
        with st.expander("Показати"):
            st.dataframe(
                meh[[
                    "symbol","buy_ex","sell_ex","buy_price","sell_price",
                    "gross_pct","fee_pct","slip_pct","net_pct","buy_depth_usdt","sell_depth_usdt"
                ]],
                use_container_width=True,
                height=420
            )

    st.subheader("📋 Таблиця (все)")
    st.dataframe(
        df[[
            "symbol","buy_ex","sell_ex","buy_price","sell_price",
            "gross_pct","fee_pct","slip_pct","net_pct","buy_depth_usdt","sell_depth_usdt","depth_ok"
        ]],
        use_container_width=True,
        height=520
    )