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

# --- 2. УМНОЕ ЯДРО АНАЛИЗА ---
def process_thermal(img, ambient_temp, climate_type):
    img_arr = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    offsets = {
        "Умеренный": {"heat": 12.0, "warm": 4.0, "cool": -6.0, "danger": 32.0},
        "Тропики": {"heat": 15.0, "warm": 6.0, "cool": -3.0, "danger": 38.0},
        "Пустыня": {"heat": 22.0, "warm": 10.0, "cool": -2.0, "danger": 48.0},
        "Арктика / Зима": {"heat": 5.0, "warm": 12.0, "cool": -8.0, "danger": 10.0}
    }
    conf = offsets[climate_type]

    # Маска растений (Синий на теплокарте)
    mask_cool = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255])),
        cv2.inRange(gray, 0, 60)
    )

    # Маска асфальта (Красный) - исключаем холодные зоны
    mask_heat = cv2.bitwise_and(cv2.inRange(gray, 100, 180), cv2.bitwise_not(mask_cool))

    # Маска зданий (Оранжевый) - исключаем холодные зоны
    mask_warm = cv2.bitwise_and(cv2.inRange(gray, 181, 255), cv2.bitwise_not(mask_cool))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # СИНИЙ
    overlay[mask_warm > 0] = [0, 140, 255]  # ОРАНЖЕВЫЙ
    overlay[mask_heat > 0] = [10, 10, 230]  # КРАСНЫЙ
    
    res = cv2.addWeighted(img_bgr, 0.4, overlay, 0.6, 0)
    total_px = max(1, img_arr.shape[0] * img_arr.shape[1])
    
    # Расчет средней температуры ТОЛЬКО для выделенной области
    p_cool = np.sum(mask_cool > 0) / total_px
    p_heat = np.sum(mask_heat > 0) / total_px
    p_warm = np.sum(mask_warm > 0) / total_px
    
    avg_zone_t = (p_cool * (ambient_temp + conf["cool"])) + \
                 (p_heat * (ambient_temp + conf["heat"])) + \
                 (p_warm * (ambient_temp + conf["warm"]))

    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), {
        "road": {"p": p_heat * 100, "t": ambient_temp + conf["heat"]},
        "build": {"p": p_warm * 100, "t": ambient_temp + conf["warm"]},
        "eco": {"p": p_cool * 100, "t": ambient_temp + conf["cool"]},
        "avg_t": avg_zone_t,
        "danger_limit": conf["danger"]
    }

# --- 3. ИНТЕРФЕЙС ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.9), rgba(10, 20, 30, 0.9)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1920&q=80");
        background-size: cover; background-attachment: fixed; color: white;
    }
    .danger-alert { background: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b; border-radius: 15px; padding: 20px; text-align: center; animation: pulse 2s infinite; }
    .safe-alert { background: rgba(0, 255, 136, 0.1); border: 2px solid #00ff88; border-radius: 15px; padding: 20px; text-align: center; }
    @keyframes pulse { 0% {opacity: 1;} 50% {opacity: 0.6;} 100% {opacity: 1;} }
    .thermo-container { width: 80px; height: 250px; background: rgba(255,255,255,0.1); border: 3px solid #fff; border-radius: 40px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# Настройки
c1, c2, c3 = st.columns([1, 1, 2])
with c1: climate = st.selectbox("Климат", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with c2: t_air = st.number_input("T воздуха (°C)", -30, 55, 25)
with c3: uploaded_file = st.file_uploader("📥 Спутниковый снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    st.subheader("🎯 Анализ выделенной зоны")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # ЛОГИЧЕСКИЙ ВЫВОД СТАТУСА (На основе ВЫДЕЛЕННОЙ зоны)
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ВНИМАНИЕ: Выделенная зона является ТЕПЛОВЫМ ОСТРОВОМ ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)
        elif stats['eco']['p'] > 50:
            st.markdown(f'<div class="safe-alert">🌿 ЭКО-ЩИТ: Выделенная зона эффективно поглощает тепло ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)
        else:
            st.success(f"✅ Состояние зоны стабильно: {stats['avg_t']:.1f}°C")

        # Метрики
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("🔥 Асфальт", f"{stats['road']['t']:.1f} °C", f"{stats['road']['p']:.1f}% зоны")
        col_m2.metric("🏠 Здания", f"{stats['build']['t']:.1f} °C", f"{stats['build']['p']:.1f}% зоны")
        col_m3.metric("🌳 Природа", f"{stats['eco']['t']:.1f} °C", f"{stats['eco']['p']:.1f}% зоны")

        st.image([cropped_img, processed_img], caption=["Выделенная область", "Тепловой сканер"], use_container_width=True)

        # Симулятор
        st.markdown("---")
        trees = st.slider("🌳 Добавить деревья (%)", 0, 100, 0)
        reduction = (trees * 0.1)
        res_t = stats['avg_t'] - reduction

        # Термометр
        t_col1, t_col2 = st.columns([1, 4])
        with t_col1:
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"**{res_t:.1f}°C**")
        with t_col2:
            st.write(f"**Прогноз температуры зоны:** {res_t:.1f} °C")
            st.progress(min(1.0, reduction/15))
