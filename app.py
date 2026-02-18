import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper
import time

# --- 1. ТЕХНИЧЕСКИЕ ФУНКЦИИ ---
def auto_enhance_image(img):
    img_array = np.array(img.convert('RGB'))
    gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    unsharp = cv2.addWeighted(img_array, 1.6, gaussian, -0.6, 0)
    enhanced_img = Image.fromarray(unsharp)
    return ImageEnhance.Contrast(enhanced_img).enhance(1.25)

def process_thermal(img, ambient_temp, climate_type):
    img_arr = np.array(img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    offsets = {
        "Умеренный": {"heat": 12.0, "warm": 4.0, "cool": -6.0, "danger": 30.0},
        "Тропики": {"heat": 15.0, "warm": 6.0, "cool": -3.0, "danger": 38.0},
        "Пустыня": {"heat": 22.0, "warm": 10.0, "cool": -2.0, "danger": 48.0},
        "Арктика / Зима": {"heat": 5.0, "warm": 12.0, "cool": -8.0, "danger": 10.0}
    }
    conf = offsets[climate_type]

    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255])), cv2.inRange(gray, 0, 75))
    mask_heat = cv2.bitwise_and(cv2.inRange(gray, 100, 185), cv2.bitwise_not(mask_cool))
    mask_warm = cv2.bitwise_and(cv2.inRange(gray, 186, 255), cv2.bitwise_not(mask_cool))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   
    overlay[mask_warm > 0] = [0, 140, 255]  
    overlay[mask_heat > 0] = [10, 10, 230]  
    
    res = cv2.addWeighted(img_bgr, 0.4, overlay, 0.6, 0)
    total_px = max(1, img_arr.shape[0] * img_arr.shape[1])
    
    p_cool, p_heat, p_warm = np.sum(mask_cool > 0)/total_px, np.sum(mask_heat > 0)/total_px, np.sum(mask_warm > 0)/total_px
    avg_t = (p_cool*(ambient_temp+conf["cool"])) + (p_heat*(ambient_temp+conf["heat"])) + (p_warm*(ambient_temp+conf["warm"]))

    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), {
        "road": {"p": p_heat*100, "t": ambient_temp+conf["heat"]},
        "build": {"p": p_warm*100, "t": ambient_temp+conf["warm"]},
        "eco": {"p": p_cool*100, "t": ambient_temp+conf["cool"]},
        "avg_t": avg_t, "danger_limit": conf["danger"]
    }

# --- 2. ИНТЕРФЕЙС ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    /* Глобальные стили для мобильных устройств */
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.95), rgba(10, 20, 30, 0.95)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: white;
    }
    
    /* Адаптивные контейнеры инструкции */
    .step-box {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        min-height: 100px;
    }
    .step-num { font-size: 18px; font-weight: bold; color: #00ff88; }
    .step-text { font-size: 12px; line-height: 1.3; }
    .step-link { color: #00ff88 !important; text-decoration: underline; }
    
    /* Улучшение читаемости на малых экранах */
    @media (max-width: 640px) {
        .stMetric { background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; }
        h1 { font-size: 24px !important; }
    }
    
    .danger-alert { background: rgba(255, 75, 75, 0.3); border: 1px solid #ff4b4b; border-radius: 10px; padding: 12px; text-align: center; }
    .safe-alert { background: rgba(0, 255, 136, 0.2); border: 1px solid #00ff88; border-radius: 10px; padding: 12px; text-align: center; }
    
    .thermo-container { width: 40px; height: 150px; background: rgba(255,255,255,0.1); border: 2px solid #fff; border-radius: 20px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# --- МОБИЛЬНАЯ ИНСТРУКЦИЯ ---
st.markdown("##### 📋 Быстрый старт")
inst_cols = st.columns([1,1,1,1]) # На ПК в ряд, на мобилках Streamlit их сам перестроит

with inst_cols[0]:
    st.markdown('<div class="step-box"><span class="step-num">1.</span><br><span class="step-text">Спутник в <a class="step-link" href="https://www.google.com/maps" target="_blank">Google</a> или <a class="step-link" href="https://yandex.ru/maps" target="_blank">Yandex</a>.</span></div>', unsafe_allow_html=True)
with inst_cols[1]:
    st.markdown('<div class="step-box"><span class="step-num">2.</span><br><span class="step-text">Вид 2D (клавиша <b>U</b>). Масштаб 20-50м. Скриншот.</span></div>', unsafe_allow_html=True)
with inst_cols[2]:
    st.markdown('<div class="step-box"><span class="step-num">3.</span><br><span class="step-text">Загрузите фото и <b>выделите зону</b> для анализа.</span></div>', unsafe_allow_html=True)
with inst_cols[3]:
    st.markdown('<div class="step-box"><span class="step-num">4.</span><br><span class="step-text">Используйте <b>слайдеры</b> для симуляции охлаждения.</span></div>', unsafe_allow_html=True)

# --- НАСТРОЙКИ ---
st.write("")
cfg_cols = st.columns([1,1]) # На мобильном будут 1x1
with cfg_cols[0]:
    climate = st.selectbox("Климат", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
    t_air = st.number_input("T воздуха (°C)", -30, 55, 25)
with cfg_cols[1]:
    uploaded_file = st.file_uploader("📥 Снимок карты", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    
    st.subheader("🎯 Выделение зоны")
    # Кроппер автоматически подстраивается под ширину контейнера
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # СТАТУС
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ТЕПЛОВОЙ ОСТРОВ: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ КОМФОРТНАЯ ЗОНА: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)

        # МЕТРИКИ (Горизонтально на ПК, вертикально на мобилках)
        st.write("")
        m_cols = st.columns(3)
        m_cols[0].metric("🔥 Асфальт", f"{stats['road']['t']:.1f}°C")
        m_cols[1].metric("🏠 Здания", f"{stats['build']['t']:.1f}°C")
        m_cols[2].metric("🌳 Природа", f"{stats['eco']['p']:.0f}%")

        # ИЗОБРАЖЕНИЯ
        st.image(processed_img, caption="Тепловизор ИИ", use_container_width=True)
