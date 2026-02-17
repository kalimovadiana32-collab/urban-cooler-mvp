import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# --- 1. АВТОМАТИЧЕСКОЕ УЛУЧШЕНИЕ КАЧЕСТВА ---
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

# --- 3. ИНТЕРФЕЙС В ЭКО-СТИЛЕ ---
st.set_page_config(page_title="AURA Eco-Monitor", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f7faf7; }
    .eco-card { padding: 20px; border-radius: 20px; border: 2px solid #2e7d32; background-color: white; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 AURA: Система эко-мониторинга v4.6")
st.markdown("##### *Прогноз и борьба с тепловым загрязнением городов*")

with st.sidebar:
    st.header("🌍 Параметры среды")
    climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
    t_air = st.slider("Температура воздуха (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Скриншот карты (20-50м)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Авто-улучшение
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    
    st.subheader("🎯 Область эко-анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#2e7d32', aspect_ratio=None)
    
    if cropped_img:
        processed_img, metrics = process_thermal(cropped_img, t_air, climate)
        road_t = metrics['heat'][1]
        danger_t = metrics['danger_limit']
        
        # --- БЛОК 1: ТЕКУЩИЙ СТАТУС ---
        st.markdown("---")
        if road_t > danger_t:
            st.error(f"🔴 КРИТИЧЕСКИЙ УРОВЕНЬ: Зона перегрета до {road_t:.1f}°C")
        else:
            st.success(f"🟢 НОРМА: Температурный фон стабилен ({road_t:.1f}°C)")

        m1, m2, m3 = st.columns(3)
        m1.metric("🔥 Поверхность", f"{road_t:.1f} °C")
        m2.metric("🏠 Застройка", f"{metrics['warm'][1]:.1f} °C")
        m3.metric("🌳 Озеленение", f"{metrics['cool'][0]:.1f}%")

        c_img1, c_img2 = st.columns(2)
        with c_img1: st.image(cropped_img, caption="Оригинал (HD)", use_container_width=True)
        with c_img2: st.image(processed_img, caption="Тепловая карта ИИ", use_container_width=True)

        # --- БЛОК 2: СОВЕТЫ И СИМУЛЯТОР ---
        st.markdown("---")
        st.subheader("💡 Симулятор экологических улучшений")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            trees = st.checkbox("🌳 Посадка деревьев (-3.5°C)")
            cool_p = st.checkbox("🚜 Светоотражающее покрытие (-4.0°C)")
        with col_s2:
            roofs = st.checkbox("🌿 Озеленение крыш (-5.0°C)")
            water = st.checkbox("⛲ Охлаждение водой (-2.5°C)")

        # Расчет итога
        res_t = road_t
        if trees: res_t -= 3.5
        if cool_p: res_t -= 4.0
        if roofs: res_t -= 5.0
        if water: res_t -= 2.5

        # --- БЛОК 3: ВОЗМОЖНЫЙ ИТОГ (То, что ты просила) ---
        st.markdown("### 🏆 ВОЗМОЖНЫЙ ИТОГ РЕКОНСТРУКЦИИ")
        
        delta = res_t - road_t
        with st.container():
            st.markdown('<div class="eco-card">', unsafe_allow_html=True)
            col_res1, col_res2 = st.columns([2, 1])
            
            with col_res1:
                if delta == 0:
                    st.write("👉 *Выберите меры улучшения выше, чтобы увидеть результат прогноза.*")
                elif res_t <= danger_t:
                    st.markdown(f"#### 🎉 УСПЕХ! Температура снижена до **{res_t:.1f}°C**")
                    st.write(f"Ваши действия позволили снизить нагрев на **{abs(delta):.1f}°C**. Участок теперь соответствует экологическим нормам региона {climate}.")
                else:
                    st.markdown(f"#### 📉 ТЕМПЕРАТУРА СНИЖЕНА ДО **{res_t:.1f}°C**")
                    st.write(f"Нагрев снизился на **{abs(delta):.1f}°C**, но зона всё еще остается в зоне риска. Попробуйте скомбинировать больше методов озеленения.")
            
            with col_res2:
                st.metric("Новая Темп.", f"{res_t:.1f}°C", f"{delta:.1f}°C")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Таблица и экспорт
        report_df = pd.DataFrame({
            "Параметр": ["Асфальт (Текущий)", "Асфальт (Прогноз)", "Застройка", "Зелень"],
            "Значение": [f"{road_t:.1f} °C", f"{res_t:.1f} °C", f"{metrics['warm'][1]:.1f} °C", f"{metrics['cool'][0]:.1f} %"]
        })
        csv = report_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Сохранить экспертный отчет", data=csv, file_name='eco_result.csv')
