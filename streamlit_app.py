import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import textwrap  # <--- ВАЖЛИВО: Цей модуль виправляє відображення HTML

# =========================
# 1. НАЛАШТУВАННЯ ТА ДИЗАЙН (TAILWIND + CSS)
# =========================
st.set_page_config(
    page_title="Crypto Sniper V6: UA Edition",
    layout="wide",
    page_icon="🦅",
    initial_sidebar_state="expanded"
)

# Інтеграція Tailwind CSS та кастомних стилів для Streamlit
st.markdown("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        /* Глобальні налаштування */
        @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;600&family=Inter:wght@400;700&display=swap');
        
        body {
            background-color: #0f172a; /* slate-900 */
            color: #e2e8f0;
            font-family: 'Inter', sans-serif;
        }
        
        /* Прибираємо стандартні відступи Streamlit */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        /* Стилізація сайдбару */
        section[data-testid="stSidebar"] {
            background-color: #1e293b; /* slate-800 */
            border-right: 1px solid #334155;
        }
        
        /* Стилізація кнопок */
        div.stButton > button {
            background: linear-gradient(to right, #2563eb, #3b82f6);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        div.stButton > button:hover {
            background: linear-gradient(to right, #1d4ed8, #2563eb);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
            transform: translateY(-1px);
        }

        /* Стилізація метрик */
        div[data-testid="stMetricValue"] {
            color: #38bdf8; /* sky-400 */
            font-family: 'Roboto Mono', monospace;
        }
        
        /* Inputs */
        .stTextInput > div > div > input, .stSelectbox > div > div > div {
            background-color: #0f172a;
            color: white;
            border-radius: 6px;
            border: 1px solid #334155;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0f172a; 
        }
        ::-webkit-scrollbar-thumb {
            background: #334155; 
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569; 
        }
    </style>
""", unsafe_allow_html=True)

# Ініціалізація стану (Session State)
if "processed_signals" not in st.session_state:
    st.session_state.processed_signals = {}
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

# =========================
# 2. МАТЕМАТИЧНЕ ЯДРО (UKRAINIAN LOGIC)
# =========================

def format_price(price):
    """Розумне форматування ціни"""
    if price < 0.01: return f"{price:.6f}"
    if price < 1: return f"{price:.4f}"
    if price < 100: return f"{price:.2f}"
    return f"{price:.1f}"

def calculate_advanced_score(df):
    """
    Багатофакторний скоринг сигналу (0-100)
    Враховує: RSI, Bollinger, Volume, Momentum
    """
    last = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0
    side = None
    
    # 1. RSI Strength (0-40 points)
    if last['rsi'] < 25:
        score += 40
        side = "LONG"
    elif last['rsi'] > 75:
        score += 40
        side = "SHORT"
    elif last['rsi'] < 35:
        score += 25
        side = "LONG"
    elif last['rsi'] > 65:
        score += 25
        side = "SHORT"
    
    if side:
        # 2. Bollinger Penetration (0-25 points)
        if side == "LONG":
            bb_penetration = (last['bb_low'] - last['close']) / last['atr']
            if bb_penetration > 0.5: score += 25
            elif bb_penetration > 0.2: score += 15
        else:
            bb_penetration = (last['close'] - last['bb_up']) / last['atr']
            if bb_penetration > 0.5: score += 25
            elif bb_penetration > 0.2: score += 15
        
        # 3. Volume Confirmation (0-20 points)
        vol_ratio = last['vol'] / df['vol'].rolling(20).mean().iloc[-1]
        if vol_ratio > 1.5: score += 20
        elif vol_ratio > 1.2: score += 10
        
        # 4. Momentum Alignment (0-15 points)
        if side == "LONG" and last['close'] > prev['close']:
            score += 15
        elif side == "SHORT" and last['close'] < prev['close']:
            score += 15
    
    return side, min(score, 100)

def calculate_setup(row, side, atr_multiplier=1.5, limit_offset_pct=0.015, lev_range="20-25"):
    """Генерація сетапу з українським текстом"""
    price = row['close']
    atr = row['atr']
    symbol = row['symbol'].replace("/USDT", "")
    
    if side == "SHORT":
        emoji = "🟥"
        limit_entry = price * (1 + limit_offset_pct)
        avg_entry = (price + limit_entry) / 2
        sl_price = avg_entry + (atr * atr_multiplier)
        tp1 = avg_entry - (atr * 1.0)
        tp2 = avg_entry - (atr * 2.0)
        tp3 = avg_entry - (atr * 4.0)
    else:
        emoji = "🟩"
        limit_entry = price * (1 - limit_offset_pct)
        avg_entry = (price + limit_entry) / 2
        sl_price = avg_entry - (atr * atr_multiplier)
        tp1 = avg_entry + (atr * 1.0)
        tp2 = avg_entry + (atr * 2.0)
        tp3 = avg_entry + (atr * 4.0)

    # Форматування повідомлення для Telegram (в стилі Sniper)
    text_msg = f"""
🎯 <b>СИГНАЛ: {symbol}</b>
{emoji} Напрям: <b>{side}</b>
⚡ Плече: x{lev_range}

📊 <b>ВХІД У ПОЗИЦІЮ:</b>
• По ринку: {format_price(price)}
• Лімітний ордер: {format_price(limit_entry)}
<i>(Середня точка входу: {format_price(avg_entry)})</i>

💰 <b>ФІКСАЦІЯ (Take-Profit):</b>
1️⃣ {format_price(tp1)}
2️⃣ {format_price(tp2)}
3️⃣ 🚀 {format_price(tp3)}

🛡️ <b>СТОП-ЛОС:</b> {format_price(sl_price)}
    """
    
    return {
        "symbol": symbol,
        "side": side,
        "msg": text_msg.strip(),
        "price": price,
        "score": row['score'],
        "avg_entry": avg_entry,
        "sl": sl_price,
        "tp1": tp1,
        "atr": atr
    }

def analyze_single_symbol(exchange, symbol, tf, min_score):
    """Аналіз одного символу"""
    try:
        bars = exchange.fetch_ohlcv(symbol, tf, limit=100)
        if not bars or len(bars) < 50: return None
        
        df = pd.DataFrame(bars, columns=['time', 'open', 'high', 'low', 'close', 'vol'])
        if df['close'].isna().any() or len(df) < 50: return None
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger
        df['sma'] = df['close'].rolling(20).mean()
        df['std'] = df['close'].rolling(20).std()
        df['bb_up'] = df['sma'] + (df['std'] * 2.0)
        df['bb_low'] = df['sma'] - (df['std'] * 2.0)
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        df['tr'] = ranges.max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        last = df.iloc[-1]
        if pd.isna(last['rsi']) or pd.isna(last['atr']) or last['atr'] == 0: return None
        
        side, score = calculate_advanced_score(df)
        
        if side and score >= min_score:
            last = df.iloc[-1].copy()
            last['symbol'] = symbol
            last['score'] = score
            return (last, side, score)
        return None
    except:
        return None

def normalize_exchange_name(exchange_name):
    mapping = {
        "HTX (Huobi)": "htx", "Gate.io": "gateio", "Binance": "binance",
        "Bybit": "bybit", "OKX": "okx", "Bitget": "bitget",
        "MEXC": "mexc", "Kraken": "kraken", "KuCoin": "kucoinfutures"
    }
    return mapping.get(exchange_name, exchange_name.lower())

def analyze_market_parallel(exchange_name, symbols, tf, min_score=60, max_workers=10):
    ex_name = normalize_exchange_name(exchange_name)
    exchange = getattr(ccxt, ex_name)({
        'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'swap'}
    })
    signals = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(analyze_single_symbol, exchange, sym, tf, min_score): sym 
            for sym in symbols
        }
        for future in as_completed(future_to_symbol):
            result = future.result()
            if result: signals.append(result)
    signals.sort(key=lambda x: x[2], reverse=True)
    return signals

def get_top_symbols(exchange_name, limit=100):
    """Отримання списку монет"""
    try:
        ex_name = normalize_exchange_name(exchange_name)
        if not hasattr(ccxt, ex_name): return []
        exchange = getattr(ccxt, ex_name)({
            'enableRateLimit': True, 'timeout': 30000, 'options': {'defaultType': 'swap'}
        })
        
        markets = exchange.load_markets()
        valid_symbols = []
        
        try:
            tickers = exchange.fetch_tickers()
            for symbol in markets:
                if ("/USDT" in symbol or ":USDT" in symbol):
                    market = markets[symbol]
                    if market.get('swap') or market.get('future') or market.get('type') in ['swap', 'future']:
                        ticker = tickers.get(symbol, {})
                        volume = ticker.get('quoteVolume') or ticker.get('volume') or 0
                        if volume > 0: valid_symbols.append((symbol, volume))
            
            if valid_symbols:
                valid_symbols.sort(key=lambda x: x[1], reverse=True)
                return [s[0] for s in valid_symbols[:limit]]
        except:
            # Fallback
            for symbol in markets:
                if ("/USDT" in symbol or ":USDT" in symbol):
                    valid_symbols.append(symbol)
            return valid_symbols[:limit]
            
        return []
    except:
        return []

def send_to_tg(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=5)
        return True
    except:
        return False

# =========================
# 3. ІНТЕРФЕЙС (TAILWIND COMPONENTS)
# =========================

# Header with custom styling
st.markdown("""
    <div class="flex flex-col items-center justify-center mb-8">
        <h1 class="text-4xl md:text-6xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600 tracking-tighter">
            EAGLE EYE <span class="text-white">V6</span>
        </h1>
        <p class="text-slate-400 mt-2 font-mono text-sm tracking-widest uppercase">Система автоматичного сканування ринку</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="text-xl font-bold text-white mb-4 px-2 border-l-4 border-blue-500">⚙️ Конфігурація</div>', unsafe_allow_html=True)
    
    ex_sel = st.selectbox("Оберіть Біржу", [
        "Binance", "Bybit", "OKX", "Gate.io", "Bitget", "MEXC", "Kraken", "KuCoin"
    ], index=1)
    
    tf = st.selectbox("Таймфрейм", ["5m", "15m", "1h", "4h"], index=1)
    
    st.markdown('<div class="h-px bg-slate-700 my-4"></div>', unsafe_allow_html=True)
    st.markdown('<div class="text-sm font-bold text-slate-300 mb-2 uppercase tracking-wide">Налаштування сканера</div>', unsafe_allow_html=True)
    
    num_symbols = st.slider("Кількість монет", 20, 300, 100, step=10)
    min_score = st.slider("Мін. точність входу (%)", 50, 95, 65, step=5)
    max_workers = st.slider("Потоки (Швидкість)", 5, 30, 15, step=1)
    
    st.markdown('<div class="h-px bg-slate-700 my-4"></div>', unsafe_allow_html=True)
    st.markdown('<div class="text-sm font-bold text-slate-300 mb-2 uppercase tracking-wide">Параметри угоди</div>', unsafe_allow_html=True)
    
    lev_range = st.text_input("Кредитне плече", "20-50")
    limit_pct = st.slider("Відступ лімітки (%)", 0.1, 4.0, 1.2, step=0.1) / 100
    atr_mult = st.slider("Множник стоп-лосу (ATR)", 1.0, 4.0, 1.5, step=0.1)
    
    st.markdown('<div class="bg-slate-900 p-4 rounded-lg border border-slate-700 mt-4"><div class="text-xs text-slate-400">🤖 <b>Telegram Бот</b></div>', unsafe_allow_html=True)
    auto_send = st.checkbox("Авто-відправка", value=False)
    tg_token = st.text_input("Token бота", type="password", placeholder="123:ABC...")
    tg_chat = st.text_input("Chat ID", placeholder="-100...")
    st.markdown('</div>', unsafe_allow_html=True)

# Metrics Dashboard
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
            <div class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Історія сканувань</div>
            <div class="text-2xl font-mono text-white">{len(st.session_state.scan_history)}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
            <div class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Знайдено сигналів</div>
            <div class="text-2xl font-mono text-green-400">{len(st.session_state.processed_signals)}</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    last_scan = st.session_state.scan_history[-1].strftime("%H:%M:%S") if st.session_state.scan_history else "--:--"
    st.markdown(f"""
        <div class="bg-slate-800 p-4 rounded-xl border border-slate-700 shadow-lg">
            <div class="text-slate-400 text-xs uppercase font-bold tracking-wider mb-1">Останній запуск</div>
            <div class="text-2xl font-mono text-blue-400">{last_scan}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("") # Spacer

# =========================
# 4. ГОЛОВНА ЛОГІКА
# =========================

if st.button("🚀 ЗАПУСТИТИ СКАНЕР РИНКУ", type="primary"):
    
    start_time = time.time()
    
    # Progress UI
    progress_text = st.markdown('<p class="text-blue-400 animate-pulse text-center">⏳ Підключення до біржі...</p>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    try:
        # 1. Fetch
        symbols = get_top_symbols(ex_sel, num_symbols)
        if not symbols:
            st.error("❌ Не вдалося отримати дані. Перевірте з'єднання або змініть біржу.")
            st.stop()
            
        progress_text.markdown(f'<p class="text-blue-400 text-center">🔭 Аналіз {len(symbols)} активів...</p>', unsafe_allow_html=True)
        progress_bar.progress(25)
        
        # 2. Analyze
        raw_signals = analyze_market_parallel(ex_sel, symbols, tf, min_score, max_workers)
        progress_bar.progress(100)
        elapsed = time.time() - start_time
        
        # 3. Results
        progress_text.empty()
        st.session_state.scan_history.append(datetime.now())
        
        if raw_signals:
            st.markdown(f"""
                <div class="flex items-center justify-between bg-green-900/30 border border-green-500/50 p-4 rounded-lg mb-6">
                    <div class="flex items-center gap-3">
                        <span class="text-2xl">✅</span>
                        <div>
                            <div class="font-bold text-green-400">Сканування завершено!</div>
                            <div class="text-sm text-green-200/70">Знайдено {len(raw_signals)} потенційних угод за {elapsed:.1f} сек.</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # ВІДОБРАЖЕННЯ КАРТОК СИГНАЛІВ
            for i, (row_data, side, score) in enumerate(raw_signals):
                setup = calculate_setup(row_data, side, atr_mult, limit_pct, lev_range)
                
                # Кольори та стилі для картки
                border_color = "border-green-500" if side == "LONG" else "border-red-500"
                bg_gradient = "from-green-900/20 to-slate-800" if side == "LONG" else "from-red-900/20 to-slate-800"
                text_color = "text-green-400" if side == "LONG" else "text-red-400"
                badge_bg = "bg-green-600" if side == "LONG" else "bg-red-600"
                
                # HTML структура картки (Tailwind)
                # ВИПРАВЛЕНО: textwrap.dedent для коректного рендеру HTML
                card_html = textwrap.dedent(f"""
                    <div class="mb-6 p-1 rounded-2xl bg-gradient-to-r {bg_gradient} p-[1px]">
                        <div class="bg-slate-900 rounded-2xl p-6 border border-slate-700 relative overflow-hidden">
                            <div class="flex justify-between items-start mb-4">
                                <div>
                                    <div class="flex items-center gap-2">
                                        <h2 class="text-2xl font-bold text-white">{setup['symbol']}</h2>
                                        <span class="{badge_bg} text-white text-xs font-bold px-2 py-1 rounded-md uppercase">{side}</span>
                                    </div>
                                    <div class="text-slate-400 text-sm font-mono mt-1">Ціна: ${format_price(setup['price'])}</div>
                                </div>
                                <div class="text-right">
                                    <div class="text-3xl font-black {text_color}">{score}</div>
                                    <div class="text-xs text-slate-500 uppercase font-bold tracking-wider">Рейтинг</div>
                                </div>
                            </div>
                            
                            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
                                <div>
                                    <div class="text-slate-500 text-xs">Вхід (Ліміт)</div>
                                    <div class="text-white font-mono font-bold">${format_price(setup['avg_entry'])}</div>
                                </div>
                                 <div>
                                    <div class="text-slate-500 text-xs">Стоп-лос</div>
                                    <div class="text-red-400 font-mono font-bold">${format_price(setup['sl'])}</div>
                                </div>
                                 <div>
                                    <div class="text-slate-500 text-xs">Тейк-профіт 1</div>
                                    <div class="text-green-400 font-mono font-bold">${format_price(setup['tp1'])}</div>
                                </div>
                                 <div>
                                    <div class="text-slate-500 text-xs">Ризик/Прибуток</div>
                                    <div class="text-blue-300 font-mono font-bold">1:{((setup["tp1"] - setup["avg_entry"]) / (setup["avg_entry"] - setup["sl"])):.1f}</div>
                                </div>
                            </div>
                            
                            <div class="bg-slate-950 p-4 rounded-lg font-mono text-xs text-slate-300 whitespace-pre-wrap border-l-4 {border_color}">
{setup['msg']}
                            </div>
                        </div>
                    </div>
                """)
                
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Send Logic
                if auto_send and tg_token and tg_chat:
                    sig_id = f"{setup['symbol']}_{side}_{datetime.now().strftime('%Y%m%d_%H')}"
                    if sig_id not in st.session_state.processed_signals:
                        success = send_to_tg(tg_token, tg_chat, setup['msg']) 
                        if success:
                            st.session_state.processed_signals[sig_id] = True
                            st.toast(f"📨 Відправлено в Telegram: {setup['symbol']}")

        else:
            st.markdown("""
                <div class="bg-slate-800 p-6 rounded-xl border border-slate-700 text-center">
                    <div class="text-4xl mb-2">😴</div>
                    <h3 class="text-xl font-bold text-white mb-2">Ринок спить</h3>
                    <p class="text-slate-400">Сильні сигнали відсутні. Спробуйте змінити таймфрейм або знизити поріг рейтингу.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"⚠️ Помилка виконання: {e}")
