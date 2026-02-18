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
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.95), rgba(10, 20, 30, 0.95)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: white;
    }
    .step-box { background: rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 5px; }
    .step-num { font-size: 16px; font-weight: bold; color: #00ff88; }
    .step-text { font-size: 11px; }
    .step-link { color: #00ff88 !important; text-decoration: underline; }
    
    .danger-alert { background: rgba(255, 75, 75, 0.3); border: 1px solid #ff4b4b; border-radius: 10px; padding: 10px; text-align: center; }
    .safe-alert { background: rgba(0, 255, 136, 0.2); border: 1px solid #00ff88; border-radius: 10px; padding: 10px; text-align: center; }
    
    .thermo-container { width: 35px; height: 130px; background: rgba(255,255,255,0.1); border: 2px solid #fff; border-radius: 20px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# ГОР. ИНСТРУКЦИЯ (Компактно для мобилок)
st.markdown("##### 📋 Быстрый старт")
i1, i2, i3, i4 = st.columns(4)
with i1: st.markdown('<div class="step-box"><span class="step-num">1</span> <a class="step-link" href="https://www.google.com/maps" target="_blank">Google</a></div>', unsafe_allow_html=True)
with i2: st.markdown('<div class="step-box"><span class="step-num">2</span> 2D/50м</div>', unsafe_allow_html=True)
with i3: st.markdown('<div class="step-box"><span class="step-num">3</span> Upload</div>', unsafe_allow_html=True)
with i4: st.markdown('<div class="step-box"><span class="step-num">4</span> Cool IT</div>', unsafe_allow_html=True)

# НАСТРОЙКИ
st.write("")
cfg_cols = st.columns(2)
with cfg_cols[0]: climate = st.selectbox("Климат", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with cfg_cols[1]: t_air = st.number_input("T воздуха (°C)", value=25)
uploaded_file = st.file_uploader("📥 Выберите скриншот", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    st.subheader("🎯 Кроп-анализ")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # СТАТУС
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ГОРЯЧО: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ НОРМА: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)

        # СРАВНЕНИЕ КАРТИНОК (ДО И ПОСЛЕ)
        st.write("")
        img_col1, img_col2 = st.columns(2)
        with img_col1: st.image(cropped_img, caption="До", use_container_width=True)
        with img_col2: st.image(processed_img, caption="После", use_container_width=True)

        # МЕТРИКИ
        m_cols = st.columns(3)
        m_cols[0].metric("🔥 Асф.", f"{stats['road']['t']:.0f}°C")
        m_cols[1].metric("🏠 Здан.", f"{stats['build']['t']:.0f}°C")
        m_cols[2].metric("🌳 Эко", f"{stats['eco']['p']:.0f}%")

        # СИМУЛЯТОР С ГРАДУСНИКОМ
        st.markdown("---")
        st.subheader("🧪 Симулятор")
        
        trees = st.slider("🌳 Деревья (%)", 0, 100, 0)
        water = st.slider("⛲ Вода/Фасады (%)", 0, 100, 0)

        reduction = (trees * 0.1) + (water * 0.08)
        res_t = stats['avg_t'] - reduction

        sim_res_col1, sim_res_col2 = st.columns([1, 3])
        with sim_res_col1:
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"**{res_t:.1f}°C**")
        with sim_res_col2:
            st.write(f"**Эффект:** -{reduction:.1f}°C")
            st.progress(min(1.0, reduction/15))
            if res_t <= stats['danger_limit']: st.success("Охлаждение успешно!")

        # ОТЧЕТ
        st.markdown("### 📝 Отчет")
        report_df = pd.DataFrame({
            "Параметр": ["База", "Итог", "Площадь %"],
            "Данные": [f"{stats['avg_t']:.1f}°C", f"{res_t:.1f}°C", f"{stats['eco']['p']:.1f}%"]
        })
        st.table(report_df)
        st.download_button("📥 Сохранить CSV", data=report_df.to_csv(index=False).encode('utf-8-sig'), file_name='urban.csv')
