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

    # Маска зданий (Оранжевый) - исключаем холодные import streamlit as st
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

    # Маски по цветам (Синий, Оранжевый, Красный)
    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255])), cv2.inRange(gray, 0, 70))
    mask_heat = cv2.bitwise_and(cv2.inRange(gray, 100, 185), cv2.bitwise_not(mask_cool))
    mask_warm = cv2.bitwise_and(cv2.inRange(gray, 186, 255), cv2.bitwise_not(mask_cool))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # СИНИЙ (Природа)
    overlay[mask_warm > 0] = [0, 140, 255]  # ОРАНЖЕВЫЙ (Здания)
    overlay[mask_heat > 0] = [10, 10, 230]  # КРАСНЫЙ (Асфальт)
    
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

# --- 3. ИНТЕРФЕЙС ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.9), rgba(10, 20, 30, 0.9)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: white;
    }
    .danger-alert { background: rgba(255, 75, 75, 0.2); border: 2px solid #ff4b4b; border-radius: 10px; padding: 10px; text-align: center; font-size: 14px; animation: pulse 2s infinite; }
    .safe-alert { background: rgba(0, 255, 136, 0.1); border: 2px solid #00ff88; border-radius: 10px; padding: 10px; text-align: center; }
    @keyframes pulse { 0%{opacity:1;} 50%{opacity:0.6;} 100%{opacity:1;} }
    .thermo-container { width: 60px; height: 180px; background: rgba(255,255,255,0.1); border: 2px solid #fff; border-radius: 30px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# ИНСТРУКЦИЯ (Компактно)
with st.expander("📖 ИНСТРУКЦИЯ (КАК ПОЛЬЗОВАТЬСЯ)"):
    st.write("1. Спутник (20-50м) → 2. Клавиша 'U' (вид сверху) → 3. Скриншот → 4. Загрузка.")
    st.markdown("[Google Maps](http://googleusercontent.com/maps.google.com/4) | [Yandex Maps](https://yandex.ru/maps)")

# НАСТРОЙКИ
st.markdown("### ⚙️ Ввод данных")
c1, c2, c3 = st.columns([1, 1, 1])
with c1: climate = st.selectbox("Климат", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with c2: t_air = st.number_input("T воздуха", -30, 55, 25)
with c3: uploaded_file = st.file_uploader("📥 Фото", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.status("ИИ улучшает четкость...", expanded=False):
        img_raw = auto_enhance_image(Image.open(uploaded_file))
    
    st.subheader("🎯 Область анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # СТАТУС ТАБЛИЧКА
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ВНИМАНИЕ: ТЕПЛОВОЙ ОСТРОВ ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">🌿 ЭКО-КОМФОРТ ({stats["avg_t"]:.1f}°C)</div>', unsafe_allow_html=True)

        # МЕТРИКИ (Мобильный вид)
        st.write("")
        col_m = st.columns(3)
        col_m[0].metric("🔥 Асфальт", f"{stats['road']['t']:.0f}°C")
        col_m[1].metric("🏠 Здания", f"{stats['build']['t']:.0f}°C")
        col_m[2].metric("🌳 Природа", f"{stats['eco']['p']:.0f}%")

        # КАРТИНКИ (Рядом для экономии места)
        ci1, ci2 = st.columns(2)
        with ci1: st.image(cropped_img, caption="Оригинал", use_container_width=True)
        with ci2: st.image(processed_img, caption="Тепловизор", use_container_width=True)

        # СИМУЛЯТОР УЛУЧШЕНИЯ
        st.markdown("---")
        st.subheader("🧪 Симулятор модернизации")
        sc1, sc2 = st.columns(2)
        with sc1:
            trees = st.slider("🌳 Озеленение (%)", 0, 100, 0)
            pavement = st.slider("🚜 Дороги (%)", 0, 100, 0)
        with sc2:
            water = st.slider("⛲ Вода (%)", 0, 100, 0)
            white_arch = st.slider("🏙️ Фасады (%)", 0, 100, 0)

        reduction = (trees * 0.08) + (pavement * 0.05) + (water * 0.04) + (white_arch * 0.06)
        res_t = stats['avg_t'] - reduction

        # ГРАДУСНИК
        t_col1, t_col2 = st.columns([1, 3])
        with t_col1:
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"**{res_t:.1f}°C**")
        with t_col2:
            st.write(f"**Прогноз:** -{reduction:.1f}°C")
            st.progress(min(1.0, reduction/15))
            if res_t <= stats['danger_limit']: st.success("Цель достигнута!")

        # ОТЧЕТ (Вернул таблицу)
        st.markdown("### 📝 Итоговый отчет")
        report_df = pd.DataFrame({
            "Поверхность": ["Асфальт", "Застройка", "Природа"],
            "Площадь %": [f"{stats['road']['p']:.1f}", f"{stats['build']['p']:.1f}", f"{stats['eco']['p']:.1f}"],
            "Темп. °C": [f"{stats['road']['t']:.1f}", f"{stats['build']['t']:.1f}", f"{stats['eco']['t']:.1f}"]
        })
        st.table(report_df)
        st.download_button("📥 Отчет .csv", data=report_df.to_csv(index=False).encode('utf-8-sig'), file_name='report.csv')
