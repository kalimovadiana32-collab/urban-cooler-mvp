import streamlit as st
import cv2
import numpy as np
from PIL import Image

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

st.title("🛰️ THERMAL VISION SYSTEM v3.1 Pro")

# --- БЛОК ИНСТРУКЦИИ И ССЫЛОК ---
with st.expander("📖 ИНСТРУКЦИЯ И ИСТОЧНИКИ (Масштаб: Детальный)"):
    col_text, col_links = st.columns([2, 1])
    
    with col_text:
        st.markdown("""
        ### Как подготовить снимок:
        1. **Масштаб:** Максимальное приближение (**20-50 метров**). Должны быть видны отдельные машины и разметка.
        2. **Выравнивание:** Нажмите клавишу **'U'** в Google Maps, чтобы камера смотрела строго вниз.
        3. **Качество:** Сделайте скриншот без лишних интерфейсных кнопок карт.
        4. **Объекты:** Идеально, если в кадре есть сочетание: *асфальт + дерево + тень*.
        """)
        
    with col_links:
        st.markdown("### 🔗 Спутниковые карты:")
        st.markdown("- [Google Maps](https://www.google.com/maps?t=k)")
        st.markdown("- [Yandex Maps](https://yandex.ru/maps/?l=sat)")

# Боковая панель
with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики / Пустыня", "Арктический / Зима"])
    t_air = st.slider("🌡️ Температура (°C)", -20, 55, 25)
    uploaded_file = st.file_uploader("📥 Загрузить скриншот", type=['jpg', 'png', 'jpeg'])
    st.markdown("---")
    st.caption("Aura Thermal Engine v3.1")

# 3. ЛОГИКА ВЫВОДА
if uploaded_file:
    img_input = Image.open(uploaded_file)
    processed_img, metrics = process_thermal(img_input, t_air, climate)
    
    # Фикс логики тревоги
    heat_area = metrics['heat'][0]
    road_temp = metrics['heat'][1]
    
    if t_air >= 25 and heat_area > 35:
        st.error(f"⚠️ **ТЕПЛОВОЙ ОСТРОВ:** Критический нагрев ({road_temp:.1f}°C) при высокой температуре воздуха!")
    elif t_air < 20:
        st.success(f"✅ **КОМФОРТНЫЙ ФОН:** При {t_air}°C перегрев поверхностей не опасен.")
    else:
        st.info("📊 Сканирование завершено. Данные выведены в таблицу.")

    # Метрики и результат
    c1, c2, c3 = st.columns(3)
    c1.metric("🔥 ЖАРА (Дороги)", f"{metrics['heat'][1]:.1f} °C", f"{metrics['heat'][0]:.1f}%")
    c2.metric("🏠 ТЕПЛО (Дома)", f"{metrics['warm'][1]:.1f} °C", f"{metrics['warm'][0]:.1f}%")
    c3.metric("❄️ ПРОХЛАДА", f"{metrics['cool'][1]:.1f} °C", f"{metrics['cool'][0]:.1f}%")

    st.image(processed_img, use_container_width=True)
