import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# --- ПЕРВАЯ СТРОЧКА КОДА ---
st.set_page_config(page_title="URBAN COOLER", layout="wide")

# --- 1. ТЕХНИЧЕСКИЕ ФУНКЦИИ (БЕЗ ИЗМЕНЕНИЙ) ---
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

# --- 2. СТИЛИЗАЦИЯ И ФОН (ВОЗВРАЩЕНО) ---
st.markdown("""
    <style>
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.9), rgba(10, 20, 30, 0.9)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: white;
    }
    .guide-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px; padding: 12px; margin-bottom: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .step-num { font-size: 18px; font-weight: bold; color: #00ff88; }
    .step-text { font-size: 12px; line-height: 1.3; }
    .danger-alert { background: rgba(255, 75, 75, 0.3); border: 1px solid #ff4b4b; border-radius: 10px; padding: 12px; text-align: center; }
    .safe-alert { background: rgba(0, 255, 136, 0.2); border: 1px solid #00ff88; border-radius: 10px; padding: 12px; text-align: center; }
    .thermo-container { width: 40px; height: 150px; background: rgba(255,255,255,0.1); border: 2px solid #fff; border-radius: 20px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER")

# --- ИНСТРУКЦИЯ В РЯД ---
st.markdown("##### 📋 Быстрый старт")
inst_cols = st.columns(4)
with inst_cols[0]:
    st.markdown('<div class="guide-card"><span class="step-num">1.</span><br><span class="step-text">Сделайте скриншот карты в 2D (клавиша U)</span></div>', unsafe_allow_html=True)
with inst_cols[1]:
    st.markdown('<div class="guide-card"><span class="step-num">2.</span><br><span class="step-text">Загрузите файл и укажите климат</span></div>', unsafe_allow_html=True)
with inst_cols[2]:
    st.markdown('<div class="guide-card"><span class="step-num">3.</span><br><span class="step-text">Выделите рамкой участок анализа</span></div>', unsafe_allow_html=True)
with inst_cols[3]:
    st.markdown('<div class="guide-card"><span class="step-num">4.</span><br><span class="step-text">Следуйте советам ИИ для охлаждения</span></div>', unsafe_allow_html=True)

# --- ВВОД ДАННЫХ ---
st.write("")
cfg_cols = st.columns(2)
with cfg_cols[0]:
    climate = st.selectbox("Климатическая зона", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with cfg_cols[1]:
    t_air = st.number_input("T воздуха на улице (°C)", value=25)

uploaded_file = st.file_uploader("📥 Загрузить карту", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    st.subheader("🎯 Зона анализа")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # Статус перегрева
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert">⚠️ ТЕПЛОВОЙ ОСТРОВ: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert">✅ КОМФОРТНАЯ ЗОНА: {stats["avg_t"]:.1f}°C</div>', unsafe_allow_html=True)

        # Сравнение фото
        st.write("")
        img_col1, img_col2 = st.columns(2)
        with img_col1: st.image(cropped_img, caption="Оригинал", use_container_width=True)
        with img_col2: st.image(processed_img, caption="Тепловизор ИИ", use_container_width=True)

       # --- НОВЫЙ КОНСТРУКТОР БЛАГОУСТРОЙСТВА ---
        st.divider()
        st.subheader("🛠️ КОНСТРУКТОР ОХЛАЖДЕНИЯ")

        # 1. Тот самый умный совет от ИИ
        if stats['road']['p'] > 45:
            st.warning(f"🚨 **Анализ:** Здесь слишком много раскаленного асфальта ({stats['road']['p']:.0f}%). Нужна тень!")
        elif stats['build']['p'] > 55:
            st.warning(f"🏢 **Анализ:** Плотные стены создают тепловой мешок ({stats['build']['p']:.0f}%). Нужно вертикальное озеленение.")
        else:
            st.info("📍 **Анализ:** Сбалансированный участок. Можно точечно улучшить микроклимат.")

        # 2. Инструменты (пользователь сам собирает решение)
        col_tool1, col_tool2 = st.columns(2)
        
        with col_tool1:
            st.write("🌿 **Зеленые решения**")
            trees_count = st.slider("Крупные деревья (шт)", 0, 50, 0, help="Создают глубокую тень")
            v_green = st.checkbox("Вертикальное озеленение", help="Зелень на стенах зданий")
            
        with col_tool2:
            st.write("💧 **Инженерные решения**")
            water_zone = st.checkbox("Фонтаны / Водоемы", help="Охлаждение за счет испарения")
            cool_pave = st.toggle("Светлое покрытие", help="Замена черного асфальта на светлую плитку")

        # 3. ЛОГИКА РАСЧЕТА (прозрачная и понятная)
        # Считаем суммарное снижение температуры
        t_reduction = (trees_count * 0.2)  # Каждое дерево -0.2 градуса
        if v_green: t_reduction += 1.5    # Фасады -1.5 градуса
        if water_zone: t_reduction += 2.0 # Вода -2.0 градуса
        if cool_pave: t_reduction += 2.5  # Плитка -2.5 градуса
        
        res_t = stats['avg_t'] - t_reduction

        # 4. Результат: Градусник и Сводка
        st.write("")
        res_col1, res_col2 = st.columns([1, 3])
        
        with res_col1:
            # Визуализация градусника
            fill = min(100, max(10, (res_t / 60) * 100))
            color = "#ff4b4b" if res_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'<div class="thermo-container"><div class="thermo-fill" style="height:{fill}%; background:{color};"></div></div>', unsafe_allow_html=True)
            st.write(f"<center><b>{res_t:.1f}°C</b></center>", unsafe_allow_html=True)
            
        with res_col2:
            st.metric("Итоговая температура", f"{res_t:.1f} °C", f"-{t_reduction:.1f} °C", delta_color="normal")
            
            # Динамический комментарий к результату
            if t_reduction > 5:
                st.success("🌟 Проект максимально эффективен! Вы создали оазис.")
            elif t_reduction > 0:
                st.info("📉 Температура начала снижаться. Добавьте еще элементов.")
            else:
                st.write("Выберите инструменты выше, чтобы начать охлаждение.")

        # Отчет
        report_df = pd.DataFrame({
            "Компонент": ["Текущая", "Эффект правок", "Прогноз"],
            "Значение": [f"{stats['avg_t']:.1f}°C", f"-{t_reduction:.1f}°C", f"{res_t:.1f}°C"]
        })
        st.table(report_df)
