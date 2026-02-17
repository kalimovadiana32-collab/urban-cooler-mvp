import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# 1. ФУНКЦИЯ ОБРАБОТКИ
def process_thermal(img, ambient_temp, climate_type):
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    offsets = {
        "Умеренный": {"heat": 8.5, "warm": 2.3, "cool": -10.2},
        "Тропики / Пустыня": {"heat": 15.0, "warm": 5.0, "cool": -5.0},
        "Арктический / Зима": {"heat": 3.5, "warm": 1.0, "cool": -15.0}
    }
    selected_offset = offsets[climate_type]

    # Маски
    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([33, 10, 10]), np.array([95, 255, 255])), cv2.inRange(gray, 0, 75))
    mask_warm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, -30)
    mask_heat = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 0, 45]), np.array([180, 85, 185])), cv2.bitwise_not(mask_cool))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # Синий
    overlay[mask_warm > 0] = [0, 140, 255]  # Оранжевый
    overlay[mask_heat > 0] = [10, 10, 230]  # Красный
    
    res = cv2.addWeighted(img_bgr, 0.3, overlay, 0.7, 0)
    
    total = img.shape[0] * img.shape[1]
    stats = {
        "heat": (np.sum(mask_heat > 0) / total * 100, ambient_temp + selected_offset["heat"]),
        "warm": (np.sum(mask_warm > 0) / total * 100, ambient_temp + selected_offset["warm"]),
        "cool": (np.sum(mask_cool > 0) / total * 100, ambient_temp + selected_offset["cool"])
    }
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), stats

# 2. ИНТЕРФЕЙС
st.set_page_config(page_title="Thermal AI MVP", layout="wide")
st.title("🛰️ THERMAL VISION SYSTEM v3.2 Pro")

# Единая детальная инструкция
with st.expander("📖 ИНСТРУКЦИЯ И ИСТОЧНИКИ (Масштаб 20м)"):
    col_t, col_l = st.columns([2, 1])
    with col_t:
        st.markdown("""
        **Как подготовить снимок:**
        1. **Масштаб:** Максимальное приближение (**20-50 метров**). 
        2. **Угол:** Нажмите **'U'** в Google Maps для вида строго сверху.
        3. **Объекты:** Должны быть видны отдельные машины и тени.
        """)
    with col_l:
        st.markdown("**🔗 Ссылки:**")
        st.markdown("[Google Maps Satellite](https://www.google.com/maps?t=k)")
        st.markdown("[Yandex Maps Satellite](https://yandex.ru/maps/?l=sat)")

# Боковая панель
with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики / Пустыня", "Арктический / Зима"])
    t_air = st.slider("🌡️ Температура воздуха (°C)", -20, 55, 25)
    uploaded_file = st.file_uploader("📥 Загрузить снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_input = Image.open(uploaded_file)
    processed_img, metrics = process_thermal(img_input, t_air, climate)
    
    # 3. ЛОГИКА АЛЕРТОВ (Твоя "фишка" с исправленной логикой)
    heat_area = metrics['heat'][0]
    road_temp = metrics['heat'][1]
    
    if t_air >= 25 and heat_area > 30:
        st.error(f"⚠️ **ВНИМАНИЕ: ТЕПЛОВОЙ ОСТРОВ!** Обнаружен критический перегрев поверхностей ({road_temp:.1f}°C) на {heat_area:.1f}% площади.")
    elif t_air < 20:
        st.success(f"✅ **БЕЗОПАСНО:** При температуре {t_air}°C эффект теплового острова не зафиксирован.")
    else:
        st.info("📊 Распределение температур в пределах нормы.")

    # 4. СРАВНЕНИЕ (До и После)
    st.markdown("### 🔍 Сравнение: Оригинал vs Анализ")
    c1, c2 = st.columns(2)
    with c1:
        st.image(img_input, caption="Спутниковый снимок (RGB)", use_container_width=True)
    with c2:
        st.image(processed_img, caption="Тепловая реконструкция", use_container_width=True)

    # 5. ПОЛНЫЙ ОТЧЕТ (Таблица и Скачивание)
    st.markdown("### 📝 Полный аналитический отчет")
    report_data = {
        "Зона": ["Жара (Асфальт)", "Тепло (Застройка)", "Прохлада (Зелень/Тени)"],
        "Площадь (%)": [f"{metrics['heat'][0]:.2f}", f"{metrics['warm'][0]:.2f}", f"{metrics['cool'][0]:.2f}"],
        "Расчетная Temp (°C)": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"]
    }
    df = pd.DataFrame(report_data)
    st.table(df)

    # Кнопка скачивания CSV-отчета
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать полный отчет (CSV)",
        data=csv,
        file_name='thermal_report.csv',
        mime='text/csv',
    )
