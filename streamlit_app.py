import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# =========================================================
# 0) PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Arbitrage Radar Fix", layout="wide", page_icon="🔁")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .card {
        background-color:#151822; border:1px solid #2b2d35;
        border-radius:12px; padding:14px; margin:10px 0;
        box-shadow:0 4px 10px rgba(0,0,0,0.25);
    }
    .row { display:flex; justify-content:space-between; gap:14px; flex-wrap:wrap; }
    .pill { padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px; }
    .pill-ok { background:#123a2a; color:#40ff9a; border:1px solid #40ff9a; }
    .mono { font-family: monospace; color: #e0e0e0; }
    .muted { color:#8b92a6; font-size: 14px; }
    .big { font-size:18px; font-weight:800; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🔁 Arbitrage Radar Pro (Fixed)")
st.caption("Cross-Exchange Spot Arbitrage Scanner (USDT pairs)")

# =========================================================
# 1) EXCHANGES SETUP
# =========================================================
EXCHANGE_CLASSES = {
    "binance": ccxt.binance,
    "bybit": ccxt.bybit,
    "okx": ccxt.okx,
    "kucoin": ccxt.kucoin,
    "kraken": ccxt.kraken,
    "gateio": ccxt.gateio,
    "mexc": ccxt.mexc,
}

# Дефолтні комісії, якщо API не повертає
DEFAULT_FEES = {
    "binance": 0.001,
    "bybit": 0.001,
    "okx": 0.001,
    "kucoin": 0.001,
    "kraken": 0.0026, # Kraken дорожчий
    "gateio": 0.002,
    "mexc": 0.001, 
}

def safe_float(x, default=0.0):
    """Надійна конвертація в float, навіть якщо приходить рядок"""
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def fmt_price(p):
    p = safe_float(p)
    if p >= 1000: return f"{p:.2f}"
    if p >= 10: return f"{p:.3f}"
    if p >= 0.1: return f"{p:.5f}"
    return f"{p:.8f}".rstrip("0").rstrip(".")

def fmt_pct(x):
    return f"{safe_float(x):.2f}%"

@st.cache_resource
def get_exchange(ex_id: str):
    """Ініціалізація біржі з rateLimit"""
    Ex = EXCHANGE_CLASSES.get(ex_id)
    if not Ex: return None
    return Ex({
        "enableRateLimit": True, # ВАЖЛИВО для уникнення банів
        "options": {"defaultType": "spot"} # Примусово Spot
    })

@st.cache_data(ttl=1800, show_spinner=False)
def load_markets_cached(ex_id: str):
    ex = get_exchange(ex_id)
    if not ex: return {}
    try:
        return ex.load_markets()
    except Exception as e:
        print(f"Error loading markets for {ex_id}: {e}")
        return {}

def is_good_usdt_spot_market(m: dict):
    """Фільтр пар: Тільки активні SPOT USDT"""
    if not m: return False
    
    # Перевірка на активність (деякі біржі не мають поля active)
    if 'active' in m and not m['active']: return False
    
    # Перевірка Spot
    if m.get('spot') is False: return False # Явно не спот
    if m.get('future') is True: return False # Явно ф'ючерс
    if m.get('type') and m.get('type') != 'spot': return False
    
    # Перевірка Quote currency
    if m.get('quote') != 'USDT': return False
    
    # Фільтр сміттєвих токенів (UP/DOWN/BEAR/BULL)
    sym = m.get('symbol', "")
    bad_tokens = ["UP/", "DOWN/", "BULL/", "BEAR/", "3L", "3S", "5L", "5S"]
    if any(b in sym for b in bad_tokens): return False
    
    return True

# =========================================================
# 2) DATA FETCHING
# =========================================================
def fetch_tickers_safe(ex_id: str):
    ex = get_exchange(ex_id)
    try:
        # Невелика пауза перед запитом, щоб не перевантажити якщо багато потоків
        time.sleep(0.5) 
        t = ex.fetch_tickers()
        return ex_id, t, None
    except Exception as e:
        return ex_id, None, str(e)

def fetch_orderbook_safe(ex_id: str, symbol: str, limit: int = 50):
    ex = get_exchange(ex_id)
    try:
        time.sleep(0.2) # Rate limit protect
        ob = ex.fetch_order_book(symbol, limit=limit)
        return ex_id, symbol, ob, None
    except Exception as e:
        return ex_id, symbol, None, str(e)

def calculate_depth(ob: dict, side: str, price_level: float, band_pct: float):
    """Рахує об'єм (USDT) в межах % від ціни"""
    if not ob or side not in ob: return 0.0
    
    rows = ob[side] # bids or asks
    if not rows: return 0.0
    
    limit_price = price_level * (1 + band_pct/100) if side == 'asks' else price_level * (1 - band_pct/100)
    
    total_usdt = 0.0
    
    for row in rows:
        # row може бути [price, amount] або {'price':..., 'amount':...}
        p = safe_float(row[0]) if isinstance(row, list) else safe_float(row.get('price'))
        a = safe_float(row[1]) if isinstance(row, list) else safe_float(row.get('amount'))
        
        if side == 'asks':
            if p > limit_price: break
        else: # bids (sorted desc)
            if p < limit_price: break
            
        total_usdt += (p * a)
        
    return total_usdt

# =========================================================
# 3) CORE LOGIC
# =========================================================
def find_opportunities(selected_exs, mode, top_n, manual_coins, slippage, min_net):
    # 1. Load Markets & Find Common Symbols
    markets_db = {}
    all_tickers = {}
    
    # Завантажуємо маркети
    for ex_id in selected_exs:
        markets_db[ex_id] = load_markets_cached(ex_id)
        
    # Формуємо список символів
    common_symbols = set()
    sets = []
    
    for ex_id, mkts in markets_db.items():
        # Фільтруємо тікери для цієї біржі
        valid = {s for s, m in mkts.items() if is_good_usdt_spot_market(m)}
        sets.append(valid)
        
    if not sets: return [], [], "Не вдалося завантажити ринки."
    
    # Знаходимо перетин (символи, що є хоча б на 2 біржах)
    # Використовуємо "плоский" підхід: беремо всі унікальні, і перевіряємо count >= 2
    from collections import Counter
    all_syms_flat = [item for sublist in sets for item in sublist]
    counts = Counter(all_syms_flat)
    common_symbols = [s for s, c in counts.items() if c >= len(selected_exs) or c >= 2] # Хоча б на 2х
    
    if mode == "Manual List":
        # Фільтруємо вручну
        targets = [c.upper().strip() for c in manual_coins]
        # Додаємо /USDT якщо забули
        targets = [t if "/" in t else f"{t}/USDT" for t in targets]
        common_symbols = [s for s in common_symbols if s in targets]
        
    # Якщо Auto - обрізаємо по топу (беремо Binance як еталон об'єму)
    elif "Auto" in mode:
        ref_ex = "binance" if "binance" in selected_exs else selected_exs[0]
        # Потрібно завантажити тікери reference біржі для сортування
        _, t_ref, _ = fetch_tickers_safe(ref_ex)
        if t_ref:
            # Сортуємо common_symbols по об'єму на reference біржі
            def get_vol(s):
                if s in t_ref:
                    return safe_float(t_ref[s].get('quoteVolume'), 0)
                return 0
            common_symbols.sort(key=get_vol, reverse=True)
            common_symbols = common_symbols[:top_n]
    
    st.info(f"🔍 Аналізую {len(common_symbols)} спільних пар...")

    # 2. Завантажуємо тікери ВСІХ обраних бірж (паралельно)
    with ThreadPoolExecutor(max_workers=len(selected_exs)) as executor:
        futures = {executor.submit(fetch_tickers_safe, ex): ex for ex in selected_exs}
        for future in as_completed(futures):
            ex_id, data, err = future.result()
            if data:
                all_tickers[ex_id] = data
            elif err:
                st.error(f"Error fetching {ex_id}: {err}")

    # 3. Шукаємо спреди
    opps = []
    
    for sym in common_symbols:
        prices = []
        for ex_id in selected_exs:
            if ex_id not in all_tickers: continue
            t = all_tickers[ex_id].get(sym)
            if not t: continue
            
            bid = safe_float(t.get('bid'))
            ask = safe_float(t.get('ask'))
            
            if bid > 0 and ask > 0:
                # Беремо fee
                fee = safe_float(markets_db[ex_id][sym].get('taker'), DEFAULT_FEES.get(ex_id, 0.002))
                prices.append({'ex': ex_id, 'bid': bid, 'ask': ask, 'fee': fee})
        
        if len(prices) < 2: continue
        
        # Знаходимо найкращий BUY (min ask) і найкращий SELL (max bid)
        best_buy = min(prices, key=lambda x: x['ask'])
        best_sell = max(prices, key=lambda x: x['bid'])
        
        if best_sell['bid'] > best_buy['ask']:
            # Є "брудний" спред
            buy_price = best_buy['ask']
            sell_price = best_sell['bid']
            
            gross_pct = ((sell_price - buy_price) / buy_price) * 100
            
            # Рахуємо витрати
            total_fee_pct = (best_buy['fee'] + best_sell['fee']) * 100
            total_slip_pct = slippage * 2 # slip on buy + slip on sell
            
            net_pct = gross_pct - total_fee_pct - total_slip_pct
            
            if net_pct >= min_net:
                opps.append({
                    'symbol': sym,
                    'buy_ex': best_buy['ex'],
                    'sell_ex': best_sell['ex'],
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'gross': gross_pct,
                    'net': net_pct,
                    'fees': total_fee_pct
                })
                
    return opps, markets_db, None

# =========================================================
# 4) UI & EXECUTION
# =========================================================
with st.sidebar:
    st.header("⚙️ Налаштування")
    
    exs = st.multiselect("Біржі", list(EXCHANGE_CLASSES.keys()), default=["binance", "bybit", "okx"], format_func=str.upper)
    
    mode = st.radio("Режим", ["Auto (Top Volume)", "Manual List"])
    manual_list = []
    top_n = 50
    
    if "Manual" in mode:
        txt = st.text_area("Список (BTC, ETH...)", "BTC, ETH, SOL, LTC, XRP")
        manual_list = txt.split(",")
    else:
        top_n = st.slider("Топ монет", 10, 200, 50)
        
    st.divider()
    min_net = st.slider("Мін. профіт (Net %)", 0.0, 5.0, 0.3, step=0.1)
    slippage = st.slider("Закласти сліппедж (%)", 0.0, 1.0, 0.1)
    
    depth_check = st.checkbox("Перевіряти глибину стакану", value=True)
    min_depth = st.number_input("Мін. глибина ($)", 100, 50000, 1000)
    depth_band = st.slider("Ширина стакану (%)", 0.1, 2.0, 0.5)

start = st.button("🚀 POISK ARBITRAGE", type="primary", use_container_width=True)

if start:
    if len(exs) < 2:
        st.error("Вибери мінімум 2 біржі!")
        st.stop()
        
    opps, markets, err = find_opportunities(exs, mode, top_n, manual_list, slippage, min_net)
    
    if err: st.error(err)
    
    if not opps:
        st.warning("Арбітражу не знайдено з такими налаштуваннями.")
    else:
        # Сортуємо за профітом
        opps.sort(key=lambda x: x['net'], reverse=True)
        
        # Якщо треба перевірка глибини - робимо додатковий запит
        final_list = []
        
        progress = st.progress(0)
        status = st.empty()
        
        if depth_check:
            # Обмежуємо кількість запитів, щоб не чекати вічність
            check_list = opps[:20] 
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Формуємо таски: (ex, sym)
                tasks = []
                for op in check_list:
                    tasks.append((op['buy_ex'], op['symbol']))
                    tasks.append((op['sell_ex'], op['symbol']))
                
                # Виконуємо запити стаканів
                ob_results = {}
                futures = {executor.submit(fetch_orderbook_safe, ex, sym): (ex, sym) for ex, sym in tasks}
                
                completed = 0
                for f in as_completed(futures):
                    ex, sym, ob, err = f.result()
                    if ob: ob_results[(ex, sym)] = ob
                    completed += 1
                    progress.progress(completed / len(tasks))
                    status.text(f"Scanning depth: {sym} on {ex}")
            
            # Аналізуємо глибину
            for op in check_list:
                buy_ob = ob_results.get((op['buy_ex'], op['symbol']))
                sell_ob = ob_results.get((op['sell_ex'], op['symbol']))
                
                if buy_ob and sell_ob:
                    # Чи можемо купити на min_depth $ в межах спреду?
                    vol_buy = calculate_depth(buy_ob, 'asks', op['buy_price'], depth_band)
                    vol_sell = calculate_depth(sell_ob, 'bids', op['sell_price'], depth_band)
                    
                    op['depth_buy'] = vol_buy
                    op['depth_sell'] = vol_sell
                    
                    if vol_buy >= min_depth and vol_sell >= min_depth:
                        final_list.append(op)
        else:
            final_list = opps
            
        progress.empty()
        status.empty()
        
        st.success(f"Знайдено {len(final_list)} можливостей!")
        
        for item in final_list:
            # Гарна картка
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div class="card">
                    <div class="row">
                        <span class="big">{item['symbol']}</span>
                        <span class="pill pill-ok">NET: +{item['net']:.2f}%</span>
                    </div>
                    <div class="row muted" style="margin-top:5px;">
                        <span>🔵 BUY: <b>{item['buy_ex'].upper()}</b> ({fmt_price(item['buy_price'])})</span>
                        <span>🔴 SELL: <b>{item['sell_ex'].upper()}</b> ({fmt_price(item['sell_price'])})</span>
                    </div>
                     <div class="row muted">
                        <span>Gross: {item['gross']:.2f}% | Fees: {item['fees']:.2f}%</span>
                        <span>Depth: ${item.get('depth_buy', 0):.0f} / ${item.get('depth_sell', 0):.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Генерація тексту для копіювання
                txt = f"#{item['symbol'].split('/')[0]} ARBITRAGE\n"
                txt += f"Buy: {item['buy_ex'].upper()} @ {item['buy_price']}\n"
                txt += f"Sell: {item['sell_ex'].upper()} @ {item['sell_price']}\n"
                txt += f"Profit: {item['net']:.2f}% (Net)"
                st.text_area("Copy", txt, height=100, label_visibility="collapsed")