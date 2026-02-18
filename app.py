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

    # Параметры теплового отклика
    offsets = {
        "Умеренный": {"heat": 12.0, "warm": 4.0, "cool": -6.0, "danger": 30.0},
        "Тропики": {"heat": 15.0, "warm": 6.0, "cool": -3.0, "danger": 38.0},
        "Пустыня": {"heat": 22.0, "warm": 10.0, "cool": -2.0, "danger": 48.0},
        "Арктика / Зима": {"heat": 5.0, "warm": 12.0, "cool": -8.0, "danger": 10.0}
    }
    conf = offsets[climate_type]

    # Маскирование
    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255])), cv2.inRange(gray, 0, 75))
    mask_heat = cv2.bitwise_and(cv2.inRange(gray, 100, 185), cv2.bitwise_not(mask_cool))
    mask_warm = cv2.bitwise_and(cv2.inRange(gray, 186, 255), cv2.bitwise_not(mask_cool))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # Синий (Природа)
    overlay[mask_warm > 0] = [0, 140, 255]  # Оранжевый (Здания)
    overlay[mask_heat > 0] = [10, 10, 230]  # Красный (Асфальт)
    
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

# --- 2. ИНТЕРФЕЙС URBAN COOLER ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.9), rgba(10, 20, 30, 0.9)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: white;
    }
    .danger-alert { background: rgba(255, 75, 75, 0.25); border: 2px solid #ff4b4b; border-radius: 10px; padding: 15px; text-align: center; animation: pulse 2s infinite; }
    .safe-alert { background: rgba(0, 255, 136, 0.15); border: 2px solid #00ff88; border-radius: 10px; padding: 15px; text-align: center; }
    @keyframes pulse { 0%{opacity:1;} 50%{opacity:0.7;} 100%{opacity:1;} }
    .thermo-container { width: 50px; height: 200px; background: rgba(255,255,255,0.1); border: 2px solid #fff; border-radius: 25px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# РАСШИРЕННАЯ ИНСТРУКЦИЯ
with st.expander("📖 РАСШИРЕННЫЙ ПРОТОКОЛ АНАЛИЗА (ИНСТРУКЦИЯ)"):
    st.markdown("""
    **1. Подготовка снимка:** Откройте [Google Maps](http://maps.google.com) (Спутник). 
    Нажмите **'U'** для строго вертикального вида. Масштаб: **20-50м**.
    **2. Параметры:** Укажите климат и текущую T воздуха (оптимально 20-25°C).
    **3. Анализ:** Выделите рамкой участок. ИИ определит % асфальта, зданий и зелени.
    **4. Модернизация:** Используйте слайдеры, чтобы снизить T до безопасного уровня.
    """)

# ПАРАМЕТРЫ
st.markdown("### ⚙️ Ввод данных")
c1, c2, c3 = st.columns([1, 1, 1])
with c1: climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with c2: t_air = st.number_input("T воздуха (°C)", -30, 55, 25)
with c3: uploaded_file = st.file_uploader("📥 Загрузить карту", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.status("ИИ оптимизирует детализацию...", expanded=False):
        img_raw = auto_enhance_image(Image.open(uploaded_file))
    
    st.subheader("🎯 Зона анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # СТАТУС (ЛОГИКА ВЫДЕЛЕННОЙ ЗОНЫ)
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ВНИМАНИЕ: ОБНАРУЖЕН ТЕПЛОВОЙ ОСТРОВ ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ ТЕМПЕРАТУРНЫЙ ФОН В НОРМЕ ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)

        # МЕТРИКИ
        st.write("")
        col_metrics = st.columns(3)
        col_metrics[0].metric("🔥 Асфальт", f"{stats['road']['t']:.1f}°C", f"{stats['road']['p']:.1f}%")
        col_metrics[1].metric("🏠 Здания", f"{stats['build']['t']:.1f}°C", f"{stats['build']['p']:.1f}%")
        col_metrics[2].metric("🌳 Природа", f"{stats['eco']['t']:.1f}°C", f"{stats['eco']['p']:.1f}%")

        # КОМПАКТНЫЕ КАРТИНКИ
        ci1, ci2 = st.columns(2)
        with ci1: st.image(cropped_img, caption="Зум-оригинал", use_container_width=True)
        with ci2: st.image(processed_img, caption="Теплосканер", use_container_width=True)

        # СИМУЛЯТОР
        st.markdown("---")
        st.subheader("🧪 Симулятор модернизации инфраструктуры")
        sc1, sc2 = st.columns(2)
        with sc1:
            trees = st.slider("🌳 Озеленение участка (%)", 0, 100, 0)
            pavement = st.slider("🚜 Отражающие дороги (%)", 0, 100, 0)
        with sc2:
            water = st.slider("⛲ Водные системы (%)", 0, 100, 0)
            white_arch = st.slider("🏙️ Светлые фасады (%)", 0, 100, 0)

        reduction = (trees * 0.1) + (pavement * 0.05) + (water * 0.04) + (white_arch * 0.06)
        res_t = stats['avg_t'] - reduction

        # ГРАДУСНИК
        t_col1, t_col2 = st.columns([1, 4])
        with t_col1:
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"**{res_t:.1f}°C**")
        with t_col2:
            st.write(f"**Прогноз охлаждения:** -{reduction:.1f}°C")
            st.progress(min(1.0, reduction/15))
            if res_t <= stats['danger_limit']: st.balloons()

        # ПОЛНЫЙ ОТЧЕТ
        st.markdown("### 📝 Итоговый технический отчет")
        report_df = pd.DataFrame({
            "Параметр": ["Тип климата", "Общая T зоны", "Прогнозная T", "Эффективность"],
            "Значение": [climate, f"{stats['avg_t']:.1f}°C", f"{res_t:.1f}°C", f"{int((reduction/15)*100)}%"]
        })
        st.table(report_df)
        st.download_button("📥 Сохранить отчет .csv", data=report_df.to_csv(index=False).encode('utf-8-sig'), file_name='urban_cooler_report.csv')
