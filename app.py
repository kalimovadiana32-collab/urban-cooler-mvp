import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_cropper import st_cropper # Новый модуль для выделения зон

# --- ФУНКЦИЯ ОБРАБОТКИ ---
def process_thermal(img, ambient_temp, climate_type):
    img = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    offsets = {
        "Умеренный": {"heat": 8.0, "warm": 2.0, "cool": -10.0, "danger": 30.0},
        "Тропики (Влажно)": {"heat": 10.0, "warm": 4.0, "cool": -4.0, "danger": 35.0},
        "Пустыня (Сухо)": {"heat": 18.0, "warm": 7.0, "cool": -3.0, "danger": 45.0},
        "Арктика / Зима": {"heat": 4.0, "warm": 15.0, "cool": -5.0, "danger": 5.0}
    }
    
    conf = offsets[climate_type]
    if climate_type == "Арктика / Зима":
        mask_cool = cv2.inRange(gray, 200, 255)
    else:
        mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([33, 10, 10]), np.array([95, 255, 255])), cv2.inRange(gray, 0, 75))

    mask_warm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, -30)
    mask_heat = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 0, 45]), np.array([180, 85, 185])), cv2.bitwise_not(mask_cool))

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
st.title("🛰️ THERMAL VISION v4.0 (Interactive)")

with st.sidebar:
    st.header("⚙️ ПАРАМЕТРЫ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики (Влажно)", "Пустыня (Сухо)", "Арктика / Зима"])
    t_air = st.slider("🌡️ Температура воздуха (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Загрузите снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file)
    
    st.subheader("🎯 Шаг 1: Выделите зону для анализа")
    st.info("Используйте рамку ниже, чтобы выбрать конкретный участок (двор, перекресток, крышу).")
    
    # ИНТЕРАКТИВНОЕ ВЫДЕЛЕНИЕ (Кроппер)
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
    
    if cropped_img:
        st.subheader("🌡️ Шаг 2: Тепловой анализ выделенной зоны")
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cropped_img, caption="Выбранный участок", use_container_width=True)
        with col2:
            st.image(processed_img, caption="Результат сканирования", use_container_width=True)

        # ЛОГИКА СОВЕТОВ
        st.markdown("---")
        st.subheader("💡 Рекомендации по улучшению среды")
        
        heat_area = metrics['heat'][0]
        
        if climate == "Арктика / Зима":
            st.info("**Совет эксперта:** Основная задача зимой — энергоэффективность. Если здания 'светятся' оранжевым, проверьте теплоизоляцию фасадов и целостность теплотрасс.")
        else:
            advice_cols = st.columns(2)
            with advice_cols[0]:
                if heat_area > 25:
                    st.error(f"⚠️ **Проблема:** Слишком много асфальта ({heat_area:.1f}%).")
                    st.write("- **Решение:** Используйте 'светлый' асфальт или бетон (у них выше альбедо).")
                    st.write("- **Решение:** Установите перголы или навесы над парковками.")
                else:
                    st.success("✅ Застройка сбалансирована.")
            
            with advice_cols[1]:
                if metrics['cool'][0] < 15:
                    st.warning("🍃 **Мало зелени!**")
                    st.write("- **Решение:** Посадите деревья с широкой кроной для создания тени.")
                    st.write("- **Решение:** Вертикальное озеленение стен зданий снизит их нагрев на 5-10°C.")
                else:
                    st.success("🌳 Достаточное количество затененных зон.")

        # Таблица данных
        df = pd.DataFrame({
            "Зона": ["Жара", "Тепло", "Прохлада"],
            "Площадь (%)": [f"{metrics['heat'][0]:.1f}", f"{metrics['warm'][0]:.1f}", f"{metrics['cool'][0]:.1f}"],
            "Температура (°C)": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"]
        })
        st.table(df)
