import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# Устанавливаем настройки страницы ПЕРВЫМ делом
st.set_page_config(page_title="URBAN COOLER", layout="wide")

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

# --- ИНТЕРФЕЙС ---
st.markdown("""
    <style>
    .stApp { background: #0e1117; color: white; }
    .guide-card { background: rgba(255, 255, 255, 0.05); border-left: 4px solid #00ff88; padding: 10px; margin-bottom: 10px; border-radius: 0 10px 10px 0; }
    .step-title { font-size: 14px; font-weight: bold; color: #00ff88; }
    .danger-alert { background: rgba(255, 75, 75, 0.2); border: 1px solid #ff4b4b; border-radius: 10px; padding: 10px; text-align: center; }
    .safe-alert { background: rgba(0, 255, 136, 0.1); border: 1px solid #00ff88; border-radius: 10px; padding: 10px; text-align: center; }
    .thermo-container { width: 30px; height: 100px; background: rgba(255,255,255,0.1); border: 1px solid #fff; border-radius: 15px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER MVP")

# ИНСТРУКЦИЯ В РЯД
st.markdown("### 📖 Инструкция")
inst_cols = st.columns(4)
with inst_cols[0]: st.markdown('<div class="guide-card"><div class="step-title">1. Скриншот</div>2D вид, масштаб 50м</div>', unsafe_allow_html=True)
with inst_cols[1]: st.markdown('<div class="guide-card"><div class="step-title">2. Загрузка</div>Выбери файл ниже</div>', unsafe_allow_html=True)
with inst_cols[2]: st.markdown('<div class="guide-card"><div class="step-title">3. Зона</div>Выдели рамкой участок</div>', unsafe_allow_html=True)
with inst_cols[3]: st.markdown('<div class="guide-card"><div class="step-title">4. Итог</div>Смотри отчет внизу</div>', unsafe_allow_html=True)

st.divider()

# ПАРАМЕТРЫ
c1, c2 = st.columns(2)
with c1: climate = st.selectbox("Климат", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with c2: t_air = st.number_input("T воздуха (°C)", value=25)

uploaded_file = st.file_uploader("📥 Выберите фото карты", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ПЕРЕГРЕВ: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ НОРМА: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)

        st.write("")
        img_col1, img_col2 = st.columns(2)
        with img_col1: st.image(cropped_img, caption="Оригинал", use_container_width=True)
        with img_col2: st.image(processed_img, caption="Тепловизор", use_container_width=True)

        st.divider()
        st.subheader("🧪 Симулятор и Отчет")
        
        trees = st.slider("🌳 Озеленение (%)", 0, 100, 0)
        reduction = (trees * 0.1)
        res_t = stats['avg_t'] - reduction

        res_col1, res_col2 = st.columns([1, 3])
        with res_col1:
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"**{res_t:.1f}°C**")
        with res_col2:
            st.write(f"**Эффект:** -{reduction:.1f}°C")
            report_df = pd.DataFrame({
                "Параметр": ["Текущая T", "После модернизации", "Эко-зона"],
                "Значение": [f"{stats['avg_t']:.1f}°C", f"{res_t:.1f}°C", f"{stats['eco']['p']:.1f}%"]
            })
            st.table(report_df)
