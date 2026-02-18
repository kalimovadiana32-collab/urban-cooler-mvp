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
        "Умеренный": {"heat": 8.0, "warm": 2.0, "cool": -10.0, "danger": 32.0},
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
    
    /* Анимация пульсации для таблички опасности */
    @keyframes pulse-red {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7); }
        70% { transform: scale(1.02); box-shadow: 0 0 0 15px rgba(255, 75, 75, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(255, 75, 75, 0); }
    }
    .danger-alert {
        background: rgba(255, 75, 75, 0.2);
        border: 2px solid #ff4b4b;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        animation: pulse-red 2s infinite;
        margin: 20px 0;
    }
    
    /* Градусник */
    .thermo-container {
        width: 80px; height: 250px;
        background: rgba(255,255,255,0.1);
        border: 3px solid #fff;
        border-radius: 40px;
        position: relative; margin: 10px auto;
        overflow: hidden;
    }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# --- ОБШИРНАЯ ИНСТРУКЦИЯ ---
with st.expander("📖 РАСШИРЕННАЯ ИНСТРУКЦИЯ И ПОДГОТОВКА ДАННЫХ"):
    st.markdown("""
    ### 🛠 Как получить точный результат:
    1. **Выбор источника:** Используйте [Google Maps](http://googleusercontent.com/maps.google.com/3) или [Yandex Maps](https://yandex.ru/maps). Переключитесь в режим **Спутник**.
    2. **Масштаб:** Оптимально — **20-50 метров**. Если масштаб больше, ИИ может пропустить мелкие тепловые объекты.
    3. **Ракурс:** Нажмите клавишу **'U'** (в Google) или убедитесь, что вид строго вертикальный (2D). Это исключит искажение площади зданий.
    4. **Время снимка:** Старайтесь выбирать снимки, сделанные в летнее время (по состоянию растительности), чтобы ИИ корректно определил зоны перегрева.
    5. **Загрузка:** Сделайте скриншот области, где есть и асфальт, и зелень — это даст лучший сравнительный анализ.
    """)

# --- НАСТРОЙКИ ---
st.markdown("### ⚙️ Ввод данных")
c1, c2, c3 = st.columns([1, 1, 2])
with c1: climate = st.selectbox("Регион", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with c2: t_air = st.number_input("Температура воздуха (°C)", -30, 55, 25)
with c3: uploaded_file = st.file_uploader("📥 Загрузите изображение", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    with st.status("🛠 ИИ: Авто-улучшение качества...", expanded=False):
        img_raw = auto_enhance_image(Image.open(uploaded_file))
        time.sleep(0.5)

    st.subheader("🎯 Выделите зону теплового анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        road_t = metrics['heat'][1]
        danger_t = metrics['danger_limit']
        
        # --- ТАБЛИЧКА ОПАСНОСТИ (ПУЛЬСИРУЮЩАЯ) ---
        if road_t > danger_t:
            st.markdown(f"""
                <div class="danger-alert">
                    <h2 style="margin:0;">⚠️ ОБНАРУЖЕН ТЕПЛОВОЙ ОСТРОВ</h2>
                    <p style="margin:5px 0 0 0;">Критический перегрев поверхностей: <b>{road_t:.1f}°C</b>. <br> 
                    Требуется немедленное внедрение систем охлаждения.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ Температурная обстановка в пределах нормы.")

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Поверхность", f"{road_t:.1f} °C")
        m2.metric("🏠 Здания", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Природа", f"{metrics['cool'][0]:.1f}%")

        c_img1, c_img2 = st.columns(2)
        with c_img1: st.image(cropped_img, caption="Улучшенный оригинал", use_container_width=True)
        with c_img2: st.image(processed_img, caption="Тепловой сканер ИИ", use_container_width=True)

        st.markdown("---")
        st.subheader("🧪 Симулятор охлаждения")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            trees = st.slider("🌳 Создание парковых зон (%)", 0, 100, 0)
            pavement = st.slider("🚜 Отражающее покрытие дорог (%)", 0, 100, 0)
        with s_col2:
            water = st.slider("⛲ Установка фонтанов/водных зон (%)", 0, 100, 0)
            white_arch = st.slider("🏙️ Светлые фасады и крыши (%)", 0, 100, 0)

        reduction = (trees * 0.08) + (pavement * 0.05) + (water * 0.04) + (white_arch * 0.06)
        res_t = road_t - reduction

        # --- ГРАДУСНИК И ИТОГ ---
        st.markdown("### 🌡️ МОНИТОРИНГ ИЗМЕНЕНИЙ")
        
        fill_height = min(100, max(10, (res_t / 60) * 100))
        color = "#ff4b4b" if res_t > danger_t else "#00ff88"
        
        t_col1, t_col2 = st.columns([1, 4])
        with t_col1:
            st.markdown(f"""
                <div class="thermo-container">
                    <div class="thermo-fill" style="height: {fill_height}%; background: {color};"></div>
                </div>
                <p style="text-align:center;"><b>{res_t:.1f}°C</b></p>
            """, unsafe_allow_html=True)
            
        with t_col2:
            st.write(f"**Эффективность охлаждения:** {int((reduction/10)*100)}%")
            st.progress(min(1.0, reduction/10))
            if res_t <= danger_t:
                st.balloons()
                st.success(f"🎊 Цель достигнута! Участок охлажден до безопасных {res_t:.1f}°C.")
            else:
                st.warning(f"Уровень нагрева снижен, но зона всё еще требует внимания.")

        # ОТЧЕТ
        st.markdown("### 📝 Итоговый отчет")
        report_df = pd.DataFrame({
            "Показатель": ["Проект", "Регион", "Базовая Т", "Прогноз Т", "Статус"],
            "Данные": ["URBAN COOLER", climate, f"{road_t:.1f}°C", f"{res_t:.1f}°C", "Безопасно" if res_t <= danger_t else "Риск"]
        })
        st.table(report_df)
