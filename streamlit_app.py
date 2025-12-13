import streamlit as st
import ccxt
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="Arbitrage Debugger", layout="wide", page_icon="🛠️")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .debug-box { background: #1c1c1c; padding: 10px; border-radius: 5px; font-family: monospace; color: #00ff00; font-size: 12px; margin-bottom: 5px; border-left: 3px solid #00ff00; }
    .error-box { background: #2b1111; padding: 10px; border-radius: 5px; font-family: monospace; color: #ff4b4b; font-size: 12px; margin-bottom: 5px; border-left: 3px solid #ff4b4b; }
    .success-card { background: #123a2a; padding: 15px; border-radius: 10px; border: 1px solid #40ff9a; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🛠️ Arbitrage Scanner: DEBUG MODE")
st.warning("Цей режим показує всі технічні деталі, щоб зрозуміти, чому не знаходяться пари.")

# ==========================================
# 2. EXCHANGE SETUP
# ==========================================
EXCHANGE_IDS = ['binance', 'bybit', 'okx', 'kraken', 'kucoin', 'gateio', 'huobi', 'mexc']

@st.cache_resource
def init_exchange(ex_id):
    try:
        exchange_class = getattr(ccxt, ex_id)
        return exchange_class({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'} 
        })
    except Exception as e:
        return None

# ==========================================
# 3. LOGIC WITH LOGGING
# ==========================================

def normalize_symbol(symbol):
    """Виправляє Kraken XBT та інші аномалії"""
    if not symbol: return ""
    # Kraken fix
    if "XBT" in symbol:
        symbol = symbol.replace("XBT", "BTC")
    return symbol

def get_tickers_safe(ex, ex_id):
    try:
        # Для деяких бірж краще явно вказати fetch_tickers() без аргументів
        tickers = ex.fetch_tickers()
        return tickers
    except Exception as e:
        st.markdown(f"<div class='error-box'>❌ {ex_id}: Помилка fetch_tickers: {e}</div>", unsafe_allow_html=True)
        return {}

def run_debug_scan(selected_exchanges, limit_top_n):
    
    # 1. ЗАВАНТАЖЕННЯ РИНКІВ
    st.subheader("1. Завантаження ринків (Load Markets)")
    
    market_sets = {} # ex_id -> set of symbols
    
    col_log = st.container()
    
    with col_log:
        for ex_id in selected_exchanges:
            ex = init_exchange(ex_id)
            if not ex:
                st.markdown(f"<div class='error-box'>Не вдалося ініціалізувати {ex_id}</div>", unsafe_allow_html=True)
                continue
            
            try:
                # Завантажуємо ринки
                markets = ex.load_markets()
                
                # Фільтруємо ТІЛЬКИ USDT SPOT
                valid_symbols = []
                for s, m in markets.items():
                    # Дуже м'який фільтр для тесту
                    if m.get('quote') == 'USDT' and m.get('spot', True) and m.get('active', True):
                        # Нормалізація назви (щоб Kraken XBT/USDT стало BTC/USDT)
                        norm_s = normalize_symbol(s)
                        valid_symbols.append(norm_s)
                
                valid_set = set(valid_symbols)
                market_sets[ex_id] = valid_set
                
                st.markdown(f"<div class='debug-box'>✅ <b>{ex_id.upper()}</b>: Завантажено {len(markets)} ринків -> З них {len(valid_set)} USDT Spot пар.</div>", unsafe_allow_html=True)
                
                # Показати приклад пар для перевірки
                sample = list(valid_set)[:5]
                st.caption(f"Приклади пар {ex_id}: {sample}")
                
            except Exception as e:
                st.markdown(f"<div class='error-box'>❌ {ex_id}: Критична помилка load_markets: {e}</div>", unsafe_allow_html=True)

    # 2. ПОШУК СПІЛЬНИХ ПАР
    st.subheader("2. Пошук перетинів (Common Pairs)")
    
    if len(market_sets) < 2:
        st.error("Потрібно успішно завантажити дані мінімум з 2 бірж.")
        return

    # Знаходимо спільні елементи у всіх вибраних сетах
    # Використовуємо intersection всіх множин
    common_symbols = set.intersection(*market_sets.values())
    
    st.markdown(f"📊 **Знайдено спільних пар:** `{len(common_symbols)}`")
    
    if len(common_symbols) == 0:
        st.error("⚠️ Нуль спільних пар! Це означає, що назви монет не співпадають або фільтр занадто суворий.")
        st.info("Спробуємо знайти пари, які є хоча б на 2 біржах (а не на всіх зразу)...")
        
        # Fallback: пари, які є хоча б на 2 біржах
        all_syms = [item for sublist in market_sets.values() for item in sublist]
        from collections import Counter
        counts = Counter(all_syms)
        common_symbols = [s for s, c in counts.items() if c >= 2]
        st.success(f"🔎 Знайдено пар, які є хоча б на 2-х біржах: {len(common_symbols)}")

    # Перетворюємо в список і обрізаємо
    target_list = list(common_symbols)
    # Сортуємо просто за алфавітом, бо у нас немає поки об'ємів
    target_list.sort()
    
    if limit_top_n > 0:
        target_list = target_list[:limit_top_n]
        st.caption(f"Взято перші {limit_top_n} для тесту.")

    st.text_area("Список монет для скану:", ", ".join(target_list), height=60)

    # 3. ОТРИМАННЯ ЦІН (FETCH TICKERS)
    st.subheader("3. Отримання цін (Fetch Tickers)")
    
    final_opportunities = []
    
    progress = st.progress(0)
    
    # Словник для зберігання цін: prices[symbol][ex_id] = {'bid': ..., 'ask': ...}
    prices_db = {} 

    # Тягнемо тікери паралельно
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ex = {executor.submit(get_tickers_safe, init_exchange(ex), ex): ex for ex in selected_exchanges}
        
        completed = 0
        for future in as_completed(future_to_ex):
            ex_id = future_to_ex[future]
            tickers = future.result()
            completed += 1
            progress.progress(completed / len(selected_exchanges))
            
            if not tickers:
                continue
                
            count_matched = 0
            for s, t in tickers.items():
                norm_s = normalize_symbol(s)
                if norm_s in target_list:
                    if norm_s not in prices_db: prices_db[norm_s] = {}
                    
                    bid = t.get('bid')
                    ask = t.get('ask')
                    
                    if bid and ask:
                        prices_db[norm_s][ex_id] = {'bid': bid, 'ask': ask}
                        count_matched += 1
            
            st.markdown(f"<div class='debug-box'>📥 <b>{ex_id.upper()}</b>: Отримано ціни для {count_matched} цільових монет.</div>", unsafe_allow_html=True)

    # 4. РОЗРАХУНОК СПРЕДІВ
    st.subheader("4. Результати (Calculation)")
    
    for symbol, ex_data in prices_db.items():
        if len(ex_data) < 2: continue
        
        # Знаходимо макс бід і мін аск
        # ex_data = {'binance': {'bid': 100, 'ask': 101}, 'bybit': ...}
        
        best_buy = min(ex_data.items(), key=lambda x: x[1]['ask']) # (ex, {data})
        best_sell = max(ex_data.items(), key=lambda x: x[1]['bid'])
        
        buy_ex = best_buy[0]
        buy_price = best_buy[1]['ask']
        
        sell_ex = best_sell[0]
        sell_price = best_sell[1]['bid']
        
        if sell_price > buy_price:
            diff_pct = ((sell_price - buy_price) / buy_price) * 100
            
            # Груба оцінка комісій (0.1% + 0.1% = 0.2%)
            est_fees = 0.2 
            net_profit = diff_pct - est_fees
            
            if net_profit > 0.1: # Показуємо все, що більше 0.1% для тесту
                final_opportunities.append({
                    'symbol': symbol,
                    'buy': f"{buy_ex} ({buy_price})",
                    'sell': f"{sell_ex} ({sell_price})",
                    'gross%': round(diff_pct, 2),
                    'net%': round(net_profit, 2)
                })

    if not final_opportunities:
        st.warning("☹️ Ціни отримані, але арбітражних ситуацій > 0.1% не знайдено.")
    else:
        df = pd.DataFrame(final_opportunities)
        df = df.sort_values('net%', ascending=False)
        
        for index, row in df.iterrows():
            st.markdown(f"""
            <div class="success-card">
                <h3 style="margin:0; color:#fff">{row['symbol']} <span style="float:right; color:#40ff9a">NET: {row['net%']}%</span></h3>
                <div style="color:#aaa; margin-top:5px;">
                    🔵 BUY: <b>{row['buy']}</b> <br>
                    🔴 SELL: <b>{row['sell']}</b> <br>
                    Gross: {row['gross%']}% (Fees approx 0.2%)
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# 5. SIDEBAR & RUN
# ==========================================
with st.sidebar:
    st.header("Налаштування")
    selected_exs = st.multiselect("Біржі", EXCHANGE_IDS, default=['binance', 'bybit', 'kucoin'])
    limit = st.slider("Ліміт монет для тесту", 10, 100, 20)
    st.info("Якщо ви виберете занадто багато бірж, процес може зайняти час.")

if st.button("🚀 ЗАПУСТИТИ ДІАГНОСТИКУ", type="primary"):
    run_debug_scan(selected_exs, limit)