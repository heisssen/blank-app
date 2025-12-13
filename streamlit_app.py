import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# =========================
# 1. CONFIG & STYLES
# =========================
st.set_page_config(page_title="Signal Post Generator", layout="centered", page_icon="📝")

st.markdown("""
<style>
    /* Головний контейнер */
    .stApp { background-color: #0e1117; }
    /* Картка з результатом */
    .signal-card {
        background-color: #1a1c24;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }
    /* Заголовок (LONG/SHORT) */
    .direction-title {
        font-size: 1.8em;
        font-weight: 800;
        margin-bottom: 10px;
        color: white;
    }
    /* Блок з R:R */
    .rr-box {
        background-color: #2b2d35;
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        margin-top: 15px;
    }
    .rr-value {
        font-size: 1.5em;
        font-weight: 900;
        color: #40ff9a; /* GREEN */
    }
    .rr-label {
        font-size: 0.8em;
        color: #8b92a6;
    }
    /* Код для Telegram */
    .stCodeBlock {
        background-color: #121418;
        border-radius: 8px;
        border: 1px solid #2b2d35;
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 Генератор Торгових Сигналів")
st.caption("Автоматичний розрахунок R:R та створення поста для Telegram")
st.divider()

# =========================
# 2. HELPER FUNCTIONS
# =========================

def safe_float(x):
    """Конвертація в float з обробкою помилок"""
    try:
        return float(x)
    except:
        return np.nan

def fmt_price(p):
    """Форматування ціни згідно її величини"""
    if not np.isfinite(p): return "N/A"
    if p >= 10: return f"{p:.4f}"
    if p >= 0.1: return f"{p:.6f}"
    return f"{p:.8f}"

def calculate_metrics(entry, sl, tps, direction):
    """Розрахунок R:R та %-змін"""
    entry = safe_float(entry)
    sl = safe_float(sl)
    tps = [safe_float(tp) for tp in tps if safe_float(tp) > 0]
    
    if not np.isfinite(entry) or not np.isfinite(sl) or not tps:
        return None
    
    # 1. Визначення базового Ризику
    risk = abs(entry - sl)
    if risk == 0: return None
    
    # Перевірка на валідність напрямку
    if (direction == "LONG" and sl >= entry) or (direction == "SHORT" and sl <= entry):
        st.error(f"Помилка: Для {direction} SL має бути {'нижче' if direction == 'LONG' else 'вище'} ціни входу.")
        return None

    results = {
        "risk_abs": risk,
        "risk_pct": risk / entry * 100,
        "entry": entry,
        "sl": sl,
        "tps": []
    }

    # 2. Розрахунок Прибутку та R:R для кожного TP
    for i, tp in enumerate(tps):
        if (direction == "LONG" and tp <= entry) or (direction == "SHORT" and tp >= entry):
            continue # Пропускаємо невалідні TP

        profit_abs = abs(tp - entry)
        rr = profit_abs / risk
        
        results["tps"].append({
            "tp": tp,
            "profit_abs": profit_abs,
            "profit_pct": profit_abs / entry * 100,
            "rr": rr
        })
        
    if not results["tps"]:
        st.error("Помилка: Усі TP знаходяться на невірній стороні або дорівнюють точці входу.")
        return None

    # Додаємо загальну метрику (R:R беремо від останнього TP)
    results["max_rr"] = results["tps"][-1]["rr"]
    results["max_profit_pct"] = results["tps"][-1]["profit_pct"]
    
    return results

def generate_telegram_post(coin, direction, leverage, market_entry, limit_entry, sl, tps, metrics):
    """Генерація тексту для Телеграм"""
    emoji = "🟢" if direction == "LONG" else "🔴"
    
    # Формування тіла
    txt = f"#{coin.upper().split('/')[0]} {emoji} {direction} x{leverage}\n"
    txt += "\n"
    
    # Входи
    if market_entry > 0 and limit_entry > 0:
        txt += f"✅ Вхід: два ордери\n"
        txt += f"Рынок {fmt_price(market_entry)}\n"
        txt += f"Лимит {fmt_price(limit_entry)}\n"
        avg_entry = (market_entry + limit_entry) / 2
        txt += f"> Сер. ціна: {fmt_price(avg_entry)}\n"
    elif market_entry > 0:
        txt += f"✅ Вхід (Market): {fmt_price(market_entry)}\n"
        avg_entry = market_entry
    else:
        txt += f"✅ Вхід (Limit): {fmt_price(limit_entry)}\n"
        avg_entry = limit_entry

    txt += "\n"
    
    # Take-Profit (використовуємо дані з розрахунками)
    txt += "💸 Take-Profit:\n"
    for i, tp_data in enumerate(metrics['tps']):
        rr_txt = f" (R:R {tp_data['rr']:.1f})"
        txt += f"{i+1}) {fmt_price(tp_data['tp'])} | +{tp_data['profit_pct']:.2f}%{rr_txt}\n"
        
    txt += "\n"
    
    # Stop-Loss
    risk_pct = metrics['risk_pct']
    txt += f"❌ Stop-loss: {fmt_price(sl)} | -{risk_pct:.2f}%\n"
    
    txt += "\n"
    # Метрики
    txt += f"💎 Макс R:R: 1:{metrics['max_rr']:.1f}\n"
    txt += f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    
    return txt

# =========================
# 3. UI INPUTS (Sidebar)
# =========================

with st.sidebar:
    st.header("Вхідні дані сигналу")

    # Основні параметри
    coin = st.text_input("Тікер монети", "XLM/USDT").upper()
    direction = st.radio("Напрямок", ["SHORT", "LONG"], index=0)
    leverage = st.text_input("Кредитне плече", "x20-25")
    
    st.divider()
    
    # Ціни входу
    st.subheader("Ціни Входу (USD)")
    entry_market = st.number_input("1. Market (Рынок)", value=0.23802, format="%.8f")
    entry_limit = st.number_input("2. Limit (Ліміт)", value=0.243, format="%.8f")
    
    # SL
    st.subheader("Stop-Loss (USD)")
    sl_price = st.number_input("Stop-loss", value=0.2484, format="%.8f")
    
    # TP (до 5)
    st.subheader("Take-Profit (USD)")
    tp1 = st.number_input("TP 1", value=0.2351, format="%.8f")
    tp2 = st.number_input("TP 2", value=0.2284, format="%.8f")
    tp3 = st.number_input("TP 3", value=0.1988, format="%.8f")
    tp4 = st.number_input("TP 4", value=0.0, format="%.8f")
    tp5 = st.number_input("TP 5", value=0.0, format="%.8f")
    
    tps_input = [tp1, tp2, tp3, tp4, tp5]
    
    st.divider()
    if st.button("Згенерувати Пост"):
        st.session_state['run_calc'] = True
    else:
        st.session_state['run_calc'] = False

# =========================
# 4. MAIN OUTPUT
# =========================

if st.session_state.get('run_calc', False) or st.button("Показати результати", key='main_btn'):
    
    # 1. Визначення ціни для розрахунків (використовуємо середню, якщо обидві вказані)
    if entry_market > 0 and entry_limit > 0:
        calc_entry = (entry_market + entry_limit) / 2
        entry_description = f"Сер. Вхід: {fmt_price(calc_entry)}"
    elif entry_market > 0:
        calc_entry = entry_market
        entry_description = f"Вхід: {fmt_price(calc_entry)} (Market)"
    elif entry_limit > 0:
        calc_entry = entry_limit
        entry_description = f"Вхід: {fmt_price(calc_entry)} (Limit)"
    else:
        st.error("Введіть хоча б одну ціну входу (Market або Limit).")
        st.stop()
        
    # 2. Розрахунок метрик
    metrics = calculate_metrics(calc_entry, sl_price, tps_input, direction)
    
    if metrics is None:
        st.error("Неможливо провести розрахунки. Перевірте вхідні дані та коректність SL/TP відносно ціни входу.")
        st.stop()
        
    # 3. Генерація тексту
    telegram_post = generate_telegram_post(
        coin, direction, leverage, entry_market, entry_limit, sl_price, tps_input, metrics
    )

    # 4. Відображення результатів
    
    st.markdown(f"""
    <div class="signal-card">
        <div class="direction-title" style="color: {'#40ff9a' if direction == 'LONG' else '#ff4b4b'}">
            #{coin.upper().split('/')[0]} | {direction}
        </div>
        
        <div class="rr-box">
            <div class="rr-label">Максимальне співвідношення R:R</div>
            <div class="rr-value">1:{metrics['max_rr']:.1f}</div>
            <div class="rr-label">Профіт до останнього TP: +{metrics['max_profit_pct']:.2f}%</div>
        </div>
        
        <h4 style="margin-top:20px; color:#ccc;">📝 Розрахунок</h4>
    """, unsafe_allow_html=True)

    # Таблиця з TP
    data = []
    for tp_data in metrics['tps']:
        data.append({
            "TP Price": fmt_price(tp_data['tp']),
            "Profit %": f"+{tp_data['profit_pct']:.2f}%",
            "R:R": f"1:{tp_data['rr']:.1f}"
        })
        
    col_tps, col_risk = st.columns(2)
    
    with col_tps:
        st.subheader("🎯 Take-Profit рівні")
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    
    with col_risk:
        st.subheader("🛡️ Ризик")
        st.metric(label="Розрахункова ціна входу", value=entry_description)
        st.metric(label="Stop-Loss", value=fmt_price(sl_price))
        st.metric(label="Ризик до SL", value=f"-{metrics['risk_pct']:.2f}%", delta_color="inverse")
    
    st.divider()

    # 5. Генератор поста
    st.subheader("📩 Готовий пост для Telegram (Копіювати)")
    st.code(telegram_post, language="text")

# Інакше показуємо інструкцію
else:
    st.info("Введіть параметри сигналу в бічній панелі та натисніть 'Показати результати'.")