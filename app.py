import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# --- 1. АВТОМАТИЧЕСКОЕ УЛУЧШЕНИЕ КАЧЕСТВА ---
def auto_enhance_image(img):
    img_array = np.array(img.convert('RGB'))
    # Повышение резкости (Unsharp Mask)
    gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    unsharp = cv2.addWeighted(img_array, 1.6, gaussian, -0.6, 0)
    enhanced_img = Image.fromarray(unsharp)
    # Повышение контраста для четкости границ
    enhancer = ImageEnhance.Contrast(enhanced_img)
    return enhancer.enhance(1.25)

# --- 2. ЯДРО АНАЛИЗА ---
def process_thermal(img, ambient_temp, climate_type):
    img_arr = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    offsets = {
        "Умеренный": {"heat": 8.0, "warm": 2.0, "cool": -10.0, "danger": 30.0},
        "Тропики": {"heat": 10.0, "warm": 4.0, "cool": -4.0, "danger": 35.0},
        "Пустыня": {"heat": 18.0, "warm": 7.0, "cool": -3.0, "danger": 45.0},
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

# --- 3. ИНТЕРФЕЙС В ЭКО-СТИЛЕ ---
st.set_page_config(page_title="EcoThermal AI", layout="wide")

# Кастомный CSS для эко-стиля
st.markdown("""
    <style>
    .main { background-color: #f0f4f0; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 10px; }
    .status-box { padding: 20px; border-radius: 15px; text-align: center; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 AURA: Thermal Eco-Monitor v4.5")
st.markdown("##### *Интеллектуальный контроль теплового загрязнения городов*")

with st.sidebar:
    st.header("🌍 Климатический пост")
    climate = st.selectbox("Регион", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
    t_air = st.slider("Температура за бортом (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Загрузите снимок (20-50м)", type=['jpg', 'png', 'jpeg'])
    st.markdown("---")
    st.info("☘️ Мы автоматически улучшаем качество вашего снимка для точного анализа.")

if uploaded_file:
    # Авто-улучшение сразу при загрузке
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    
    st.subheader("🎯 Выберите область для эко-анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#2e7d32', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        
        # --- ТАБЛИЧКИ СТАТУСА (Logic) ---
        road_t = metrics['heat'][1]
        danger_t = metrics['danger_limit']
        
        st.markdown("### 📊 Статус экологической обстановки")
        if road_t > danger_t:
            st.error(f"🔴 КРИТИЧЕСКИЙ ЖАР: Температура поверхностей ({road_t:.1f}°C) выше нормы! Срочно требуются меры озеленения.")
        elif road_t > (danger_t - 5):
            st.warning(f"🟡 ПРЕДУПРЕЖДЕНИЕ: Наблюдается умеренный тепловой остров. Рекомендуется увеличить площадь тени.")
        else:
            st.success(f"🟢 ЭКО-НОРМА: Температурный баланс в норме для региона {climate}.")

        # Вывод данных
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Асфальт", f"{metrics['heat'][1]:.1f} °C")
        m2.metric("🏠 Здания", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Озеленение", f"{metrics['cool'][0]:.1f}%")

        c_img1, c_img2 = st.columns(2)
        with c_img1:
            st.image(cropped_img, caption="Снимок с улучшенной четкостью", use_container_width=True)
        with c_img2:
            st.image(processed_img, caption="Тепловая карта ИИ", use_container_width=True)

        # СИМУЛЯТОР РЕШЕНИЙ
        st.markdown("---")
        st.subheader("💡 Симулятор борьбы с потеплением")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            trees = st.checkbox("🌳 Массовая посадка деревьев (-3°C)")
            cool_p = st.checkbox("🚜 Светоотражающие дороги (-4°C)")
        with col_s2:
            roofs = st.checkbox("🌿 Озеленение крыш (-5°C)")
            water = st.checkbox("⛲ Городские фонтаны (-2°C)")

        res_t = road_t
        if trees: res_t -= 3
        if cool_p: res_t -= 4
        if roofs: res_t -= 5
        if water: res_t -= 2

        st.metric("🌡️ Прогноз после внедрения мер", f"{res_t:.1f} °C", f"{res_t - road_t:.1f} °C")

        # Отчет
        report_df = pd.DataFrame({
            "Зона": ["Асфальт", "Застройка", "Природа"],
            "Площадь %": [f"{metrics['heat'][0]:.1f}", f"{metrics['warm'][0]:.1f}", f"{metrics['cool'][0]:.1f}"],
            "Тек. Темп.": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"],
            "Прогноз": [f"{res_t:.1f}", "—", "—"]
        })
        st.table(report_df)
        csv = report_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Скачать экологический отчет", data=csv, file_name='eco_report.csv')
