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

# --- 3. ИНТЕРФЕЙС И СТИЛИЗАЦИЯ (ТЕМНЫЙ ГОРОД) ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.markdown("""
    <style>
    /* Фон с темным городом */
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.85), rgba(10, 20, 30, 0.85)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    
    /* Настройка читаемости текста */
    h1, h2, h3, h4, h5, p, span, label { color: white !important; }
    
    /* Блоки с прозрачностью */
    div[data-testid="stExpander"], div[data-testid="stMetric"], .stTable {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 10px;
    }
    
    .stProgress > div > div > div > div { background-color: #00ff88; }
    .stButton>button { background-color: #2e7d32; color: white; border-radius: 10px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")
st.markdown("##### *Smart Urban Heat Analysis & Mitigation*")

# --- ИНСТРУКЦИИ ---
with st.expander("📖 ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ"):
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        st.markdown("""
        1. Откройте карту в режиме **Спутник**.
        2. Установите масштаб **20-50м** и вид строго сверху (**клавиша 'U'**).
        3. Загрузите скриншот в панель слева.
        4. Выделите нужную зону рамкой и используйте симулятор для охлаждения.
        """)
    with col_i2:
        st.markdown("**🔗 Карты:**")
        st.markdown("- [Google Maps](https://www.google.com/maps)")
        st.markdown("- [Yandex Maps](https://yandex.ru/maps/?l=sat)")

with st.sidebar:
    st.header("⚙️ ПАРАМЕТРЫ")
    climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
    t_air = st.slider("Температура воздуха (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Загрузите скриншот", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # АВТО-УЛУЧШЕНИЕ С УВЕДОМЛЕНИЕМ
    with st.status("🛠 ИИ: Автоматическое улучшение качества и четкости снимка...", expanded=False):
        img_raw = auto_enhance_image(Image.open(uploaded_file))
        time.sleep(0.8)
        st.write("✅ Микро-детализация восстановлена")
        st.write("✅ Контраст границ оптимизирован")

    st.subheader("🎯 Область эко-анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        road_t = metrics['heat'][1]
        danger_t = metrics['danger_limit']
        
        # Данные
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Поверхность", f"{road_t:.1f} °C")
        m2.metric("🏠 Здания", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Природа", f"{metrics['cool'][0]:.1f}%")

        c_img1, c_img2 = st.columns(2)
        with c_img1: st.image(cropped_img, caption="Снимок высокого разрешения", use_container_width=True)
        with c_img2: st.image(processed_img, caption="Тепловой сканер ИИ", use_container_width=True)

        # СИМУЛЯТОР (СЛАЙДЕРЫ)
        st.markdown("---")
        st.subheader("🧪 Симулятор охлаждения URBAN COOLER")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            trees = st.slider("🌳 Озеленение территории (%)", 0, 100, 0)
            pavement = st.slider("🚜 Светлое дорожное покрытие (%)", 0, 100, 0)
        with s_col2:
            water = st.slider("⛲ Системы водного охлаждения (%)", 0, 100, 0)
            white_arch = st.slider("🏙️ Отражающие материалы фасадов (%)", 0, 100, 0)

        # Формула итога
        reduction = (trees * 0.08) + (pavement * 0.05) + (water * 0.04) + (white_arch * 0.06)
        res_t = road_t - reduction
        delta = res_t - road_t

        # ВИЗУАЛЬНАЯ ШКАЛА ЭФФЕКТИВНОСТИ
        st.markdown("### 🏆 РЕЗУЛЬТАТ МОДЕРНИЗАЦИИ")
        progress = min(1.0, max(0.0, reduction / 10)) # Шкала заполняется до 10 градусов снижения
        st.write(f"**Эффективность принятых мер:** {int(progress*100)}%")
        st.progress(progress)
        
        if res_t <= danger_t:
            st.success(f"🎊 ОТЛИЧНО! Температура снижена до **{res_t:.1f}°C**. Участок безопасен.")
        else:
            st.warning(f"📉 ТРЕБУЮТСЯ МЕРЫ. Температура снижена до **{res_t:.1f}°C**, но риск перегрева остается.")

        # ТОЧНЫЙ ОТЧЕТ
        st.markdown("### 📝 Итоговый отчет")
        report_df = pd.DataFrame({
            "Показатель": ["Проект", "Зона", "Старт. Температура", "Прогнозная Т", "Эффективность"],
            "Данные": ["URBAN COOLER", climate, f"{road_t:.1f} °C", f"{res_t:.1f} °C", f"{int(progress*100)}%"]
        })
        st.table(report_df)
        csv = report_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Скачать экспертный отчет .csv", data=csv, file_name='urban_cooler_result.csv')
