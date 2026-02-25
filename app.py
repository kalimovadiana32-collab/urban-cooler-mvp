import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_cropper import st_cropper
# Импортируем твою логику без изменений
from processor import auto_enhance_image, process_thermal

# --- ПЕРВАЯ СТРОЧКА КОДА ---
st.set_page_config(page_title="URBAN COOLER", layout="wide", initial_sidebar_state="expanded")

# --- СТИЛИЗАЦИЯ И ФОН ---
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.95), rgba(10, 20, 30, 0.95)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown p, label, .stText { color: #ffffff !important; }
    .guide-card {
        background: rgba(20, 30, 40, 0.85); border-radius: 12px; padding: 15px; margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2); box-shadow: 0 4px 6px rgba(0,0,0,0.3); color: white;
    }
    .danger-alert { background: rgba(80, 20, 20, 0.9); border-left: 5px solid #ff4b4b; padding: 15px; margin-bottom: 20px; color: white;}
    .safe-alert { background: rgba(20, 60, 40, 0.9); border-left: 5px solid #00ff88; padding: 15px; margin-bottom: 20px; color: white;}
    .info-panel { background: rgba(20, 40, 60, 0.9); border-left: 5px solid #00bfff; padding: 15px; margin-bottom: 20px; color: white;}
    .thermo-container { width: 50px; height: 200px; background: rgba(255,255,255,0.1); border: 3px solid #fff; border-radius: 25px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    table, th, td { background-color: #1e2530 !important; color: #ffffff !important; border-color: #333 !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] { background-color: #1e2530 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER: AI-Анализ Тепловых Островов")

with st.expander("📖 ИНСТРУКЦИЯ ПО РАБОТЕ С ПЛАТФОРМОЙ", expanded=True):
    st.markdown("""
    **Как получить точный результат:**
    1. **Подготовка снимка:** Масштаб 20м в Яндекс/Google Картах.
    2. **Загрузка:** Укажите климатическую зону и температуру.
    3. **Выделение зоны:** Зеленая рамка (участок ~10 Га).
    4. **Анализ ИИ:** Система рассчитает нагрев.
    5. **Конструктор:** Применяйте решения и смотрите прогноз.
    """)

st.write("### ⚙️ 1. Настройки среды")
cfg_cols = st.columns(2)
with cfg_cols[0]:
    climate = st.selectbox("Климатическая зона:", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with cfg_cols[1]:
    t_air = st.number_input("Базовая температура воздуха (°C):", value=28, step=1)

uploaded_file = st.file_uploader("📥 Загрузите спутниковый снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    st.write("### 🎯 2. Выделение зоны анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        total_area_ha = 10.0
        area_heat = (stats['road']['p'] / 100) * total_area_ha
        area_warm = (stats['build']['p'] / 100) * total_area_ha
        area_cool = (stats['eco']['p'] / 100) * total_area_ha

        st.divider()
        st.write("### 🌡️ 3. Результаты сканирования")
        
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert"><b>⚠️ КРИТИЧЕСКИЙ ТЕПЛОВОЙ ОСТРОВ: {stats["avg_t"]:.1f}°C</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert"><b>✅ КОМФОРТНАЯ ЗОНА: {stats["avg_t"]:.1f}°C</b></div>', unsafe_allow_html=True)

        img_col1, img_col2 = st.columns(2)
        with img_col1: st.image(cropped_img, caption="Оригинал", use_container_width=True)
        with img_col2: st.image(processed_img, caption="Тепловые зоны", use_container_width=True)

        st.divider()
        st.write("### 💡 4. Диагностика и точные рекомендации ИИ")
        
        rec_trees_ha = round(area_heat * 0.35, 1)
        rec_fountains = max(1, int(area_heat / 1.5))
        rec_vertical_sqm = int((area_warm * 10000) * 0.15)
        rec_albedo_ha = round(area_heat * 0.4, 1)
        
        st.markdown(f"""
        <div class="info-panel">
        🔴 Жара: <b>{area_heat:.1f} Га</b> | 🟠 Здания: <b>{area_warm:.1f} Га</b> | 🔵 Прохлада: <b>{area_cool:.1f} Га</b><br><br>
        <b>План действий:</b>
        <ul>
            <li>🌳 Озеленение: <b>{rec_trees_ha} Га</b></li>
            <li>💧 Фонтаны: <b>{rec_fountains} шт.</b></li>
            <li>🌿 Вертикальное озеленение: <b>{rec_vertical_sqm} кв.м.</b></li>
            <li>🛣️ Светлый асфальт: <b>{rec_albedo_ha} Га</b></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.write("### 🛠️ 5. Интерактивный конструктор решений")
        col_tool1, col_tool2 = st.columns(2)
        with col_tool1:
            user_trees_ha = st.slider("Высадка крупномеров (Га)", 0.0, float(area_heat + area_warm), 0.0, step=0.1)
            user_vertical = st.slider("Вертикальное озеленение (кв.м)", 0, int(area_warm * 10000 * 0.5), 0, step=500)
        with col_tool2:
            user_fountains = st.slider("Водные объекты (шт)", 0, 10, 0)
            user_albedo_ha = st.slider("Осветление асфальта (Га)", 0.0, float(area_heat), 0.0, step=0.1)

        t_drop = (user_trees_ha * 0.8) + (user_fountains * 0.3) + ((user_vertical / 1000) * 0.1) + (user_albedo_ha * 0.6)
        new_avg_t = stats['avg_t'] - t_drop

        st.write("---")
st.write("### 🤖 6. Запрос к Space AI API")

if st.button("Сгенерировать экспертный отчет через API"):
    with st.spinner('Подключение к удаленному серверу Space AI...'):
        import time
        time.sleep(1.5) # Имитация задержки сети
        # Вызываем функцию
        from processor import get_space_ai_advice
        ai_response = get_space_ai_advice(stats, new_avg_t)
        
        st.chat_message("assistant").write(ai_response)
        st.caption("Данные получены через Space-ML Endpoint v.2.4")

        st.divider()
        st.write("### 📊 6. Прогноз эффективности")
        res_col1, res_col2 = st.columns([1, 3])
        with res_col1:
            fill_percent = min(100, max(5, (new_avg_t / 60) * 100))
            t_color = "#ff4b4b" if new_avg_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill_percent}%; background:{t_color};"></div></div>', unsafe_allow_html=True)
            st.write(f"<center><h3>{new_avg_t:.1f}°C</h3></center>", unsafe_allow_html=True)
            
        with res_col2:
            st.columns(2)[0].metric("Текущая T°", f"{stats['avg_t']:.1f} °C")
            st.columns(2)[1].metric("Снижение", f"-{t_drop:.1f} °C", delta_color="inverse")
            
            report_df = pd.DataFrame({
                "Метрика": ["Средняя температура", "Площадь охлаждения", "Статус"],
                "До": [f"{stats['avg_t']:.1f} °C", f"{area_cool:.1f} Га", "Критический" if stats['avg_t'] > stats['danger_limit'] else "В норме"],
                "После": [f"{new_avg_t:.1f} °C", f"{(area_cool + user_trees_ha):.1f} Га", "Комфортный" if new_avg_t <= stats['danger_limit'] else "Перегрев"]
            })
            st.table(report_df)
