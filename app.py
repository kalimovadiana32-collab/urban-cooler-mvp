import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from streamlit_cropper import st_cropper

# --- ЯДРО ОБРАБОТКИ ---
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
    overlay[mask_cool > 0] = [240, 80, 0] 
    overlay[mask_warm > 0] = [0, 140, 255] 
    overlay[mask_heat > 0] = [10, 10, 230] 
    
    res = cv2.addWeighted(img_bgr, 0.3, overlay, 0.7, 0)
    total = img.shape[0] * img.shape[1]
    
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), {
        "heat": [np.sum(mask_heat > 0) / total * 100, ambient_temp + conf["heat"]],
        "warm": [np.sum(mask_warm > 0) / total * 100, ambient_temp + conf["warm"]],
        "cool": [np.sum(mask_cool > 0) / total * 100, ambient_temp + conf["cool"]],
        "danger_limit": conf["danger"]
    }

# --- ИНТЕРФЕЙС ---
st.set_page_config(page_title="Thermal AI Expert", layout="wide")
st.title("🛰️ THERMAL VISION v4.1 Expert System")

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики (Влажно)", "Пустыня (Сухо)", "Арктика / Зима"])
    t_air = st.slider("🌡️ Температура воздуха (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Загрузите снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = Image.open(uploaded_file)
    st.info("🎯 Выделите зону интереса на карте")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        
        # --- БЛОК 1: ТЕКУЩИЕ ПОКАЗАТЕЛИ ---
        st.subheader("📊 Анализ текущего состояния")
        c1, c2, c3 = st.columns(3)
        c1.metric("🔥 Темп. асфальта", f"{metrics['heat'][1]:.1f} °C")
        c2.metric("🏠 Темп. зданий", f"{metrics['warm'][1]:.1f} °C")
        c3.metric("🌳 Зона прохлады", f"{metrics['cool'][0]:.1f}%")

        st.image(processed_img, caption="Распределение тепла в выбранной зоне", use_container_width=True)

        # --- БЛОК 2: ИНТЕРАКТИВНЫЕ СОВЕТЫ ---
        st.markdown("---")
        st.subheader("💡 Симулятор улучшений")
        st.write("Выберите меры по снижению температуры, чтобы увидеть прогноз:")
        
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            fix_trees = st.checkbox("🌳 Посадить деревья (-3°C в тени)")
            fix_roofs = st.checkbox("🏠 'Холодные крыши' / Озеленение кровли (-5°C)")
        with col_adv2:
            fix_pavement = st.checkbox("🚜 Светлое покрытие дорог (-4°C)")
            fix_water = st.checkbox("⛲ Установка фонтанов/водоемов (-2°C зонально)")

        # Расчет прогноза
        predicted_temp = metrics['heat'][1]
        if fix_trees: predicted_temp -= 3
        if fix_roofs: predicted_temp -= 5
        if fix_pavement: predicted_temp -= 4
        if fix_water: predicted_temp -= 2

        # --- БЛОК 3: ПРОГНОЗ ---
        st.markdown("### 📉 Прогноз после модернизации")
        delta = predicted_temp - metrics['heat'][1]
        st.metric("🌡️ Новая средняя температура зоны", f"{predicted_temp:.1f} °C", f"{delta:.1f} °C")
        
        if predicted_temp < metrics['danger_limit']:
            st.success("🎉 Среда станет комфортной для жителей!")
        else:
            st.warning("⚠️ Даже этих мер недостаточно, нужно больше озеленения.")

        # Таблица и скачивание
        df = pd.DataFrame({
            "Параметр": ["Асфальт", "Здания", "Прохлада"],
            "Площадь (%)": [f"{metrics['heat'][0]:.1f}", f"{metrics['warm'][0]:.1f}", f"{metrics['cool'][0]:.1f}"],
            "Текущая T": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"],
            "Прогноз T": [f"{predicted_temp:.1f}", f"{metrics['warm'][1]-2 if fix_roofs else metrics['warm'][1]:.1f}", "—"]
        })
        st.table(df)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Скачать полный экспертный отчет", data=csv, file_name='thermal_analysis.csv')
