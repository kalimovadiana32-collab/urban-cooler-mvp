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
    .guide-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #00ff88;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 10px 10px 0;
    }
    .step-label { color: #00ff88; font-weight: bold; text-transform: uppercase; font-size: 12px; }
    .step-title { font-size: 16px; font-weight: bold; margin-bottom: 5px; }
    .step-link { color: #00ff88 !important; text-decoration: underline; }
    
    .danger-alert { background: rgba(255, 75, 75, 0.3); border: 1px solid #ff4b4b; border-radius: 10px; padding: 10px; text-align: center; }
    .safe-alert { background: rgba(0, 255, 136, 0.2); border: 1px solid #00ff88; border-radius: 10px; padding: 10px; text-align: center; }
    
    .thermo-container { width: 35px; height: 130px; background: rgba(255,255,255,0.1); border: 2px solid #fff; border-radius: 20px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# --- ОБШИРНАЯ ИНСТРУКЦИЯ ---
st.markdown("### 📖 ГИД ПО ИСПОЛЬЗОВАНИЮ")
guide_col1, guide_col2 = st.columns(2)

with guide_col1:
    st.markdown("""
    <div class="guide-card">
        <div class="step-label">Шаг 1: Подготовка</div>
        <div class="step-title">Получение карты</div>
        Откройте <a class="step-link" href="https://www.google.com/maps" target="_blank">Google Maps</a> или 
        <a class="step-link" href="https://yandex.ru/maps" target="_blank">Yandex</a>. Перейдите в режим <b>Спутник</b>. 
        Нажмите клавишу <b>'U'</b> для выравнивания камеры в 2D. Выберите масштаб 20-50м и сделайте скриншот.
    </div>
    <div class="guide-card">
        <div class="step-label">Шаг 2: Загрузка</div>
        <div class="step-title">Параметры среды</div>
        Загрузите файл ниже. Выберите вашу <b>Климатическую зону</b> и введите текущую <b>Температуру воздуха</b>. 
        Это необходимо для калибровки теплового сканера ИИ.
    </div>
    """, unsafe_allow_html=True)

with guide_col2:
    st.markdown("""
    <div class="guide-card">
        <div class="step-label">Шаг 3: Анализ</div>
        <div class="step-title">Выделение области</div>
        С помощью рамки выделите интересующий вас участок (дорогу, парк или квартал). 
        ИИ моментально покажет распределение тепла: <span style="color:#ff4b4b">Красный</span> — асфальт, 
        <span style="color:#ffa500">Оранжевый</span> — здания, <span style="color:#4db8ff">Синий</span> — зелень/тени.
    </div>
    <div class="guide-card">
        <div class="step-label">Шаг 4: Решение</div>
        <div class="step-title">Симуляция охлаждения</div>
        Используйте слайдеры внизу, чтобы добавить деревья или изменить покрытия. 
        Следите за <b>Градусником</b>: ваша цель — вывести температуру в зеленую зону.
    </div>
    """, unsafe_allow_html=True)

# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
st.markdown("---")
st.markdown("### ⚙️ ПАНЕЛЬ УПРАВЛЕНИЯ")
cfg_cols = st.columns(2)
with cfg_cols[0]:
    climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with cfg_cols[1]:
    t_air = st.number_input("T воздуха на улице (°C)", value=25)

uploaded_file = st.file_uploader("📥 Загрузить скриншот карты", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    st.subheader("🎯 Анализ участка")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # ALERT STATUS
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ВНИМАНИЕ: ЗОНА ПЕРЕГРЕТА ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ ТЕМПЕРАТУРА В НОРМЕ ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)

        # СРАВНЕНИЕ (ДО И ПОСЛЕ)
        st.write("")
        img_col1, img_col2 = st.columns(2)
        with img_col1: st.image(cropped_img, caption="Оригинал", use_container_width=True)
        with img_col2: st.image(processed_img, caption="Тепловизор ИИ", use_container_width=True)

        # МЕТРИКИ
        m_cols = st.columns(3)
        m_cols[0].metric("🔥 Асфальт", f"{stats['road']['t']:.1f}°C")
        m_cols[1].metric("🏠 Здания", f"{stats['build']['t']:.1f}°C")
        m_cols[2].metric("🌳 Растительность", f"{stats['eco']['p']:.0f}%")

        # СИМУЛЯТОР
        st.markdown("---")
        st.subheader("🧪 СИМУЛЯТОР МОДЕРНИЗАЦИИ")
        
        trees = st.slider("🌳 Процент озеленения", 0, 100, 0)
        water = st.slider("⛲ Отражающие покрытия и вода", 0, 100, 0)

        reduction = (trees * 0.1) + (water * 0.08)
        res_t = stats['avg_t'] - reduction

        sim_res_col1, sim_res_col2 = st.columns([1, 3])
        with sim_res_col1:
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"**{res_t:.1f}°C**")
        with sim_res_col2:
            st.write(f"**Ожидаемый эффект:** -{reduction:.1f}°C")
            st.progress(min(1.0, reduction/15))
            if res_t <= stats['danger_limit']: st.success("Проект модернизации эффективен!")

        # ОТЧЕТ
        st.markdown("### 📝 ТЕХНИЧЕСКИЙ ОТЧЕТ")
        report_df = pd.DataFrame({
            "Характеристика": ["Текущая T", "T после улучшений", "Доля эко-зоны", "Статус"],
            "Значение": [f"{stats['avg_t']:.1f}°C", f"{res_t:.1f}°C", f"{stats['eco']['p']:.1f}%", "Исправлено" if res_t <= stats['danger_limit'] else "Риск"]
        })
        st.table(report_df)
        st.download_button("📥 Сохранить отчет в CSV", data=report_df.to_csv(index=False).encode('utf-8-sig'), file_name='urban_report.csv')
