import streamlit as st
import cv2
import numpy as np
from PIL import Image
st.title("🛰️ THERMAL VISION SYSTEM v2.0")
st.markdown("---")

# --- ВОТ ЭТОТ БЛОК НУЖНО ВСТАВИТЬ ---
with st.expander("📖 ИНСТРУКЦИЯ И ТРЕБОВАНИЯ К СНИМКАМ"):
    st.write("Для корректной работы алгоритма следуйте рекомендациям:")
    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        st.markdown("""
        **✅ Рекомендуется:**
        - **Высота:** 300-800 метров (масштаб квартала).
        - **Угол:** Строго вертикально (Надир).
        - **Солнце:** Ясный полдень (максимальный контраст).
        """)
    with col_inf2:
        st.markdown("""
        **❌ Избегать:**
        - Снимков под углом (искажает площадь).
        - Сильной облачности и тумана.
        - Мелкого масштаба (весь город в кадре).
        """)
# --- КОНЕЦ БЛОКА ИНСТРУКЦИИ ---

def process_thermal(img, ambient_temp, climate_type):
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Логика коэффициентов в зависимости от климата
    # В пустыне асфальт жарит сильнее (+15), в Арктике - меньше (+3)
    offsets = {
        "Умеренный": {"heat": 8.5, "warm": 2.3, "cool": -10.2},
        "Тропики / Пустыня": {"heat": 15.0, "warm": 5.0, "cool": -5.0},
        "Арктический / Зима": {"heat": 3.5, "warm": 1.0, "cool": -15.0}
    }
    
    selected_offset = offsets[climate_type]

    # Маски (твоя проверенная база)
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

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="Thermal AI MVP", layout="wide")

st.title("🛰️ THERMAL VISION SYSTEM v2.5")

# Блок инструкции
with st.expander("📖 ИНСТРУКЦИЯ"):
    st.write("Загружайте снимки в надире (вид сверху), масштаб 300-800м.")

# Боковая панель (Пункт 2: Пресеты климата)
with st.sidebar:
    st.header("⚙️ Настройки сканера")
    climate = st.selectbox("🌍 Тип климата", ["Умеренный", "Тропики / Пустыня", "Арктический / Зима"])
    t_air = st.slider("🌡️ Температура воздуха (°C)", -20, 55, 25)
    uploaded_file = st.file_uploader("📥 Загрузить снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_input = Image.open(uploaded_file)
    processed_img, metrics = process_thermal(img_input, t_air, climate)
    
    # Пункт 3: Система алертов
    heat_area = metrics['heat'][0]
    
    if heat_area > 35:
        st.error(f"⚠️ **КРИТИЧЕСКИЙ УРОВЕНЬ ТЕПЛА:** Дороги и постройки перегреты ({heat_area:.1f}% площади). Возможен эффект теплового острова!")
    elif heat_area > 20:
        st.warning(f"🔔 **ПОВЫШЕННЫЙ НАГРЕВ:** Зоны жары составляют {heat_area:.1f}%. Рекомендуется озеленение.")
    else:
        st.success(f"✅ **БЕЗОПАСНАЯ СРЕДА:** Зона жары всего {heat_area:.1f}%. Температурный баланс в норме.")

    # Метрики
    c1, c2, c3 = st.columns(3)
    c1.metric("🔥 ЖАРА (Дороги)", f"{metrics['heat'][1]:.1f} °C", f"{metrics['heat'][0]:.1f}%")
    c2.metric("🏠 ТЕПЛО (Здания)", f"{metrics['warm'][1]:.1f} °C", f"{metrics['warm'][0]:.1f}%")
    c3.metric("❄️ ПРОХЛАДА", f"{metrics['cool'][1]:.1f} °C", f"{metrics['cool'][0]:.1f}%")

    st.image(processed_img, use_container_width=True)


