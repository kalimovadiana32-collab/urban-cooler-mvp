import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# --- 1. ФУНКЦИИ УЛУЧШЕНИЯ КАЧЕСТВА ---
def enhance_image(img):
    img_array = np.array(img.convert('RGB'))
    # Фильтр резкости (Unsharp Mask)
    gaussian_3 = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img_array, 1.5, gaussian_3, -0.5, 0)
    enhanced_img = Image.fromarray(unsharp_image)
    enhancer = ImageEnhance.Contrast(enhanced_img)
    return enhancer.enhance(1.2)

def check_blur(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return score

# --- 2. ЯДРО ТЕПЛОВОГО АНАЛИЗА ---
def process_thermal(img, ambient_temp, climate_type):
    img_arr = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
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
    total = img_arr.shape[0] * img_arr.shape[1]
    
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), {
        "heat": [np.sum(mask_heat > 0) / total * 100, ambient_temp + conf["heat"]],
        "warm": [np.sum(mask_warm > 0) / total * 100, ambient_temp + conf["warm"]],
        "cool": [np.sum(mask_cool > 0) / total * 100, ambient_temp + conf["cool"]],
        "danger_limit": conf["danger"]
    }

# --- 3. ИНТЕРФЕЙС ---
st.set_page_config(page_title="Thermal AI Ultimate", layout="wide")
st.title("🛰️ THERMAL VISION v4.3 Global Expert")

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики (Влажно)", "Пустыня (Сухо)", "Арктика / Зима"])
    t_air = st.slider("🌡️ Температура воздуха (°C)", -30, 55, 25)
    uploaded_file = st.file_uploader("📥 Загрузите снимок", type=['jpg', 'png', 'jpeg'])
    st.markdown("---")
    auto_enhance = st.checkbox("🪄 Улучшить четкость снимка", value=True)

if uploaded_file:
    img_raw = Image.open(uploaded_file)
    
    # Проверка и улучшение качества
    b_score = check_blur(img_raw)
    if b_score < 100:
        st.warning(f"⚠️ Снимок размыт (Качество: {int(b_score)}). Рекомендуем масштаб 20-50м.")
    
    if auto_enhance:
        img_raw = enhance_image(img_raw)

    st.subheader("🎯 Выделите зону анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
    
    if cropped_img:
        # ЗАПУСК НЕЙРОСЕТЕВОЙ ОБРАБОТКИ
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        
        # Вывод показателей
        st.markdown("---")
        st.subheader("📊 Текущие показатели зоны")
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Асфальт", f"{metrics['heat'][1]:.1f} °C")
        m2.metric("🏠 Здания", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Озеленение", f"{metrics['cool'][0]:.1f}%")

        # КАРТИНКА ОТ НЕЙРОСЕТИ
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.image(cropped_img, caption="Оригинал (Zoom)", use_container_width=True)
        with col_res2:
            st.image(processed_img, caption="Тепловой анализ ИИ", use_container_width=True)

        # СИМУЛЯТОР СОВЕТОВ
        st.markdown("---")
        st.subheader("💡 Симулятор экологических решений")
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            fix_trees = st.checkbox("🌳 Добавить деревья (-3°C)")
            fix_pavement = st.checkbox("🚜 Светлое покрытие (-4°C)")
        with c_adv2:
            fix_roofs = st.checkbox("🏠 Зеленые крыши (-5°C)")
            fix_water = st.checkbox("⛲ Фонтаны/Водоемы (-2°C)")

        # Расчет прогноза
        pred_t = metrics['heat'][1]
        if fix_trees: pred_t -= 3
        if fix_roofs: pred_t -= 5
        if fix_pavement: pred_t -= 4
        if fix_water: pred_t -= 2

        st.metric("🌡️ Прогноз температуры после улучшений", f"{pred_t:.1f} °C", f"{pred_t - metrics['heat'][1]:.1f} °C")

        # Таблица и отчет
        report_df = pd.DataFrame({
            "Параметр": ["Асфальт", "Здания", "Зелень/Тени"],
            "Площадь (%)": [f"{metrics['heat'][0]:.1f}", f"{metrics['warm'][0]:.1f}", f"{metrics['cool'][0]:.1f}"],
            "Тек. Темп.": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"],
            "Прогноз": [f"{pred_t:.1f}", "—", "—"]
        })
        st.table(report_df)
        csv = report_df.to_csv(index=False).encode('utf-8-sig
