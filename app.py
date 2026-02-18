import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper
import time

# --- 1. АВТО-УЛУЧШЕНИЕ КАЧЕСТВА ---
def auto_enhance_image(img):
    img_array = np.array(img.convert('RGB'))
    gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    unsharp = cv2.addWeighted(img_array, 1.6, gaussian, -0.6, 0)
    enhanced_img = Image.fromarray(unsharp)
    enhancer = ImageEnhance.Contrast(enhanced_img)
    return enhancer.enhance(1.25)

# --- 2. ЯДРО АНАЛИЗА ---
def process_thermal(img, ambient_temp, climate_type):
    img_arr = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    offsets = {
        "Умеренный": {"heat": 8.0, "warm": 2.0, "cool": -10.0, "danger": 35.0},
        "Тропики": {"heat": 10.0, "warm": 4.0, "cool": -4.0, "danger": 38.0},
        "Пустыня": {"heat": 18.0, "warm": 7.0, "cool": -3.0, "danger": 48.0},
        "Арктика / Зима": {"heat": 4.0, "warm": 15.0, "cool": -5.0, "danger": 10.0}
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
    total = img_arr.shape[0] * img_arr.shape[1]
    
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), {
        "heat": [np.sum(mask_heat > 0) / total * 100, ambient_temp + conf["heat"]],
        "warm": [np.sum(mask_warm > 0) / total * 100, ambient_temp + conf["warm"]],
        "cool": [np.sum(mask_cool > 0) / total * 100, ambient_temp + conf["cool"]],
        "danger_limit": conf["danger"]
    }

# --- 3. ИНТЕРФЕЙС И СТИЛИЗАЦИЯ ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.9), rgba(10, 20, 30, 0.9)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    h1, h2, h3, h4, h5, p, span, label { color: white !important; }
    
    /* Стили градусника */
    .thermo-container {
        width: 100px; height: 300px;
        background: rgba(255,255,255,0.1);
        border: 4px solid #fff;
        border-radius: 50px 50px 10px 10px;
        position: relative; margin: 20px auto;
        overflow: hidden;
    }
    .thermo-fill {
        position: absolute; bottom: 0; width: 100%;
        transition: all 0.5s ease-in-out;
    }
    .thermo-bulb {
        width: 60px; height: 60px;
        background: inherit; border: 4px solid #fff;
        border-radius: 50%; margin: -30px auto 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# --- ГЛАВНЫЕ НАСТРОЙКИ ---
st.markdown("### 🛠 Параметры системы")
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with c2:
    t_air = st.number_input("Температура воздуха (°C)", -30, 55, 25)
with c3:
    uploaded_file = st.file_uploader("📥 Загрузите скриншот карты", type=['jpg', 'png', 'jpeg'])

with st.expander("📖 ИНСТРУКЦИЯ"):
    st.markdown("Спутник • Масштаб 20-50м • Клавиша 'U' • [Google](http://maps.google.com) | [Yandex](https://yandex.ru/maps)")

if uploaded_file:
    with st.status("🛠 Оптимизация качества...", expanded=False):
        img_raw = auto_enhance_image(Image.open(uploaded_file))
        time.sleep(0.5)

    st.subheader("🎯 Зона анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        road_t = metrics['heat'][1]
        
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Поверхность", f"{road_t:.1f} °C")
        m2.metric("🏠 Здания", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Природа", f"{metrics['cool'][0]:.1f}%")

        c_img1, c_img2 = st.columns(2)
        with c_img1: st.image(cropped_img, use_container_width=True)
        with c_img2: st.image(processed_img, use_container_width=True)

        st.markdown("---")
        st.subheader("🧪 Симулятор охлаждения")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            trees = st.slider("🌳 Озеленение (%)", 0, 100, 0)
            pavement = st.slider("🚜 Светлое покрытие (%)", 0, 100, 0)
        with s_col2:
            water = st.slider("⛲ Водное охлаждение (%)", 0, 100, 0)
            white_arch = st.slider("🏙️ Отражающие фасады (%)", 0, 100, 0)

        reduction = (trees * 0.08) + (pavement * 0.05) + (water * 0.04) + (white_arch * 0.06)
        res_t = road_t - reduction

        # --- ВИЗУАЛЬНЫЙ ГРАДУСНИК ---
        st.markdown("### 🌡️ СОСТОЯНИЕ УЧАСТКА")
        
        # Расчет высоты и цвета (от 0 до 60 градусов для наглядности)
        fill_height = min(100, max(10, (res_t / 60) * 100))
        color = "#ff4b4b" if res_t > metrics['danger_limit'] else "#00ff88"
        
        t_col1, t_col2 = st.columns([1, 3])
        with t_col1:
            st.markdown(f"""
                <div class="thermo-container">
                    <div class="thermo-fill" style="height: {fill_height}%; background: {color};"></div>
                </div>
                <p style="text-align:center; font-weight:bold;">{res_t:.1f}°C</p>
            """, unsafe_allow_html=True)
            
        with t_col2:
            st.write("")
            st.write("")
            if res_t > metrics['danger_limit']:
                st.error(f"🚨 КРИТИЧЕСКИЙ ЖАР! Температура выше нормы.")
            else:
                st.success(f"✅ ЭКО-КОМФОРТ. Цель достигнута.")
            
            progress = min(1.0, max(0.0, reduction / 10))
            st.write(f"**Эффективность охлаждения:** {int(progress*100)}%")
            st.progress(progress)

        st.markdown("### 📝 Итоговый отчет")
        report_df = pd.DataFrame({
            "Параметр": ["Проект", "Зона", "Старт Т", "Итог Т", "Эффективность"],
            "Данные": ["URBAN COOLER", climate, f"{road_t:.1f}°C", f"{res_t:.1f}°C", f"{int(progress*100)}%"]
        })
        st.table(report_df)
        st.download_button("📥 Скачать .csv", data=report_df.to_csv(index=False).encode('utf-8-sig'), file_name='urban_cooler.csv')
