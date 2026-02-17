import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# --- ФУНКЦИИ УЛУЧШЕНИЯ КАЧЕСТВА ---
def enhance_image(img):
    # Повышение резкости через OpenCV
    img_array = np.array(img.convert('RGB'))
    gaussian_3 = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img_array, 1.5, gaussian_3, -0.5, 0)
    
    # Повышение контрастности через PIL
    enhanced_img = Image.fromarray(unsharp_image)
    enhancer = ImageEnhance.Contrast(enhanced_img)
    return enhancer.enhance(1.2)

def check_blur(img):
    img_gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    return score # Чем ниже число, тем более "мыльное" фото

# --- ОСНОВНАЯ ЛОГИКА ---
st.set_page_config(page_title="Thermal AI Quality+", layout="wide")
st.title("🛰️ THERMAL VISION v4.2 (Quality Guard)")

with st.sidebar:
    st.header("⚙️ НАСТРОЙКИ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики (Влажно)", "Пустыня (Сухо)", "Арктика / Зима"])
    t_air = st.slider("🌡️ Температура (°C)", -30, 55, 20)
    uploaded_file = st.file_uploader("📥 Загрузите снимок", type=['jpg', 'png', 'jpeg'])
    
    st.markdown("---")
    st.markdown("**🛠 Инструменты улучшения:**")
    auto_enhance = st.checkbox("🪄 Авто-улучшение четкости")

if uploaded_file:
    img_raw = Image.open(uploaded_file)
    
    # Проверка качества
    blur_score = check_blur(img_raw)
    if blur_score < 100:
        st.warning(f"⚠️ **Низкое качество:** Снимок слишком размыт (Score: {blur_score:.1f}). Рекомендуется сделать скриншот в более высоком разрешении.")
    
    if auto_enhance:
        img_raw = enhance_image(img_raw)
        st.caption("✨ Применен фильтр повышения резкости границ")

    st.info("🎯 Выделите зону интереса (масштаб 20-50м)")
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#FF4B4B', aspect_ratio=None)
    
    if cropped_img:
        # Здесь идет вызов твоей функции process_thermal (оставляем старую из v4.1)
        # ... (код обработки из предыдущего шага) ...
        
        # Добавим визуальный индикатор качества в отчет
        st.write(f"🔍 **Индекс детализации участка:** {int(blur_score)}")
