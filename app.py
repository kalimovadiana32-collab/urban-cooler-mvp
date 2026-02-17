import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_cropper import st_cropper # Библиотека для выделения зоны

# --- ФУНКЦИЯ ОБРАБОТКИ ---
def process_thermal(img, ambient_temp, climate_type):
    # Превращаем фото в массив для обработки
    img = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Настройки климата
    offsets = {
        "Умеренный": {"heat": 8.0, "warm": 2.0, "cool": -10.0, "danger": 30.0},
        "Тропики (Влажно)": {"heat": 10.0, "warm": 4.0, "cool": -4.0, "danger": 35.0},
        "Пустыня (Сухо)": {"heat": 18.0, "warm": 7.0, "cool": -3.0, "danger": 45.0},
        "Арктика / Зима": {"heat": 4.0, "warm": 15.0, "cool": -5.0, "danger": 5.0}
    }
    
    conf = offsets[climate_type]
    
    # Маски поиска зон
    if climate_type == "Арктика / Зима":
        mask_cool = cv2.inRange(gray, 200, 255) # Снег
    else:
        mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([33, 10, 10]), np.array([95, 255, 255])), cv2.inRange(gray, 0, 75))

    mask_warm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, -30)
    mask_heat = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 0, 45]), np.array([180, 85, 185])), cv2.bitwise_not(mask_cool))

    # Красим результат
    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0] # Синий
    overlay[mask_warm > 0] = [0, 140, 255] # Оранжевый
    overlay[mask_heat > 0] = [10, 10, 230] # Красный
    
    res = cv2.addWeighted(img_bgr, 0.3, overlay, 0.7, 0)
    total = img.shape[0] * img.shape[1]
    
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), {
        "heat": (np.sum(mask_heat > 0) / total * 100, ambient_temp + conf["heat"]),
        "warm": (np.sum(mask_warm > 0) / total * 100, ambient_temp + conf["warm"]),
        "cool": (np.sum(mask_cool > 0) / total * 100, ambient_temp + conf["cool"]),
        "danger_limit": conf["danger"]
    }

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="Thermal AI Pro", layout="wide")
st.title("🛰️ THERMAL VISION v4.0 (Интерактивный выбор)")

# Загрузка файла (теперь в центре, чтобы было видно на мобильном)
st.subheader("1. Загрузите снимок (со ссылок в инструкции выше)")
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики (Влажно)", "Пустыня (Сухо)", "Арктика / Зима"])
    t_air = st.slider("🌡️ Температура (°C)", -30, 55, 20)

if uploaded_file:
    img_raw = Image.open(uploaded_file)
    
    st.markdown("### 🎯 Выделите участок для детального анализа")
    st.caption("Перетаскивайте края рамки, чтобы выбрать конкретный объект (дом, дорогу или парк).")
    
    # ЭТОТ БЛОК ДЕЛАЕТ КАРТУ ИНТЕРАКТИВНОЙ
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        
        # Сравнение
        c1, c2 = st.columns(2)
        with c1:
            st.image(cropped_img, caption="Выбранная зона", use_container_width=True)
        with c2:
            st.image(processed_img, caption="Результат сканирования", use_container_width=True)

        # СОВЕТЫ ПО УЛУЧШЕНИЮ (Твой запрос)
        st.markdown("---")
        st.subheader("💡 Рекомендации эксперта по этой зоне")
        
        heat_area = metrics['heat'][0]
        
        col_advice = st.columns(2)
        with col_advice[0]:
            if heat_area > 30 and t_air > 25:
                st.error("🚨 **ОБНАРУЖЕН ПЕРЕГРЕВ!**")
                st.write("- **Совет:** Замените темный асфальт на светлую плитку или 'холодное покрытие'.")
                st.write("- **Совет:** Установите здесь 'зеленые' остановки или теневые навесы.")
            else:
                st.success("✅ В этой зоне температурный баланс соблюден.")
                
        with col_advice[1]:
            if metrics['cool'][0] < 15:
                st.warning("🌵 **ДЕФИЦИТ ЗЕЛЕНИ**")
                st.write("- **Совет:** Посадите деревья с плотной кроной в этом квадрате.")
                st.write("- **Совет:** Рассмотрите создание 'живой стены' на фасаде здания.")
        
        # Данные
        df = pd.DataFrame({
            "Зона": ["Жара", "Тепло", "Прохлада"],
            "Площадь (%)": [f"{metrics['heat'][0]:.1f}", f"{metrics['warm'][0]:.1f}", f"{metrics['cool'][0]:.1f}"],
            "Темп. (°C)": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"]
        })
        st.table(df)
