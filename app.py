import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper
import time

# --- 1. АВТО-УЛУЧШЕНИЕ И КАЧЕСТВО ---
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

# --- 3. ИНТЕРФЕЙС URBAN COOLER ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f4f7f4; }
    .eco-label { font-size: 14px; color: #2e7d32; font-weight: bold; }
    .stProgress > div > div > div > div { background-color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")
st.markdown("##### *Система мониторинга и снижения теплового стресса городов*")

# --- ВОЗВРАЩАЕМ ИНСТРУКЦИИ ---
with st.expander("📖 ИНСТРУКЦИЯ И КАРТЫ"):
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        st.markdown("""
        1. Перейдите на карты (ссылки справа) и выберите режим **Спутник**.
        2. Найдите нужный участок города. Масштаб: **20-50 метров**.
        3. Нажмите **'U'** (в Google Maps) для вида строго сверху.
        4. Сделайте скриншот и загрузите его в панель слева.
        """)
    with col_i2:
        st.markdown("**🔗 Ссылки на карты:**")
        st.markdown("- [Google Maps](https://www.google.com/maps)")
        st.markdown("- [Yandex Maps](https://yandex.ru/maps/?l=sat)")

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
    t_air = st.slider("Температура воздуха (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Загрузите снимок", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # НАДПИСЬ ОБ УЛУЧШЕНИИ
    with st.status("🛠 ИИ производит автоматическое улучшение качества снимка...", expanded=False):
        img_raw = auto_enhance_image(Image.open(uploaded_file))
        time.sleep(1) # Небольшая пауза для эффекта работы
        st.write("✨ Контуры объектов оптимизированы.")
        st.write("📈 Резкость повышена на 25%.")

    st.subheader("🎯 Выделите участок для анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#2e7d32', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        road_t = metrics['heat'][1]
        danger_t = metrics['danger_limit']
        
        # Индикаторы
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Темп. поверхностей", f"{road_t:.1f} °C")
        m2.metric("🏠 Темп. застройки", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Природный щит", f"{metrics['cool'][0]:.1f}%")

        c_img1, c_img2 = st.columns(2)
        with c_img1: st.image(cropped_img, caption="Улучшенный оригинал", use_container_width=True)
        with c_img2: st.image(processed_img, caption="Тепловой анализ", use_container_width=True)

        # --- 4. УСОВЕРШЕНСТВОВАННЫЙ СИМУЛЯТОР (Слайдеры) ---
        st.markdown("---")
        st.subheader("🧪 Симулятор экологической модернизации")
        st.write("Определите объем вложений в инфраструктуру участка:")
        
        col_sim1, col_sim2 = st.columns(2)
        with col_sim1:
            trees_vol = st.slider("🌳 Площадь новых парковых зон (%)", 0, 100, 0)
            pavement_vol = st.slider("🚜 Замена покрытия на светлое (%)", 0, 100, 0)
        with col_sim2:
            water_vol = st.slider("⛲ Установка систем увлажнения/фонтанов (%)", 0, 100, 0)
            white_roofs = st.slider("🏙️ Отражающие фасады и светлые крыши (%)", 0, 100, 0)

        # Расчет прогноза (более сложная формула)
        reduction = (trees_vol * 0.08) + (pavement_vol * 0.05) + (water_vol * 0.04) + (white_roofs * 0.06)
        res_t = road_t - reduction

        # --- 5. ВИЗУАЛЬНАЯ ШКАЛА И ИТОГ ---
        st.markdown("### 🏆 РЕЗУЛЬТАТ МОДЕРНИЗАЦИИ")
        
        # Вычисляем прогресс эффективности (0% - нет изменений, 100% - достигли идеала)
        target_t = t_air + 2 # Идеальная температура
        current_range = road_t - target_t
        if current_range <= 0: current_range = 1
        progress = min(1.0, max(0.0, reduction / current_range))
        
        st.write(f"**Эффективность охлаждения участка:** {int(progress*100)}%")
        st.progress(progress)
        
        delta = res_t - road_t
        
        col_res = st.columns([2, 1])
        with col_res[0]:
            if res_t <= danger_t:
                st.success(f"🎉 **ЦЕЛЬ ДОСТИГНУТА!** Температура снижена до **{res_t:.1f}°C**. Зона перешла в разряд безопасных.")
            else:
                st.warning(f"📉 **ЧАСТИЧНЫЙ УСПЕХ.** Температура снижена до **{res_t:.1f}°C**, но риск перегрева сохраняется. Увеличьте площадь парков.")
        
        with col_res[1]:
            st.metric("ПРОГНОЗ T", f"{res_t:.1f}°C", f"{delta:.1f}°C")

        # ТОЧНЫЙ ОТЧЕТ
        st.markdown("### 📝 Точный отчет анализа")
        report_df = pd.DataFrame({
            "Параметр анализа": ["Название проекта", "Климат", "Тек. Темп. поверхностей", "Прогноз после модернизации", "Эффективность охлаждения"],
            "Значение": ["URBAN COOLER", climate, f"{road_t:.1f} °C", f"{res_t:.1f} °C", f"{int(progress*100)}%"]
        })
        st.table(report_df)
        
        csv = report_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Скачать экспертное заключение .csv", data=csv, file_name='urban_cooler_report.csv')
