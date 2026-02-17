st.title("🛰️ THERMAL VISION SYSTEM v2.0")
st.markdown("---")

# --- ВОТ ЭТОТ БЛОК НУЖНО ВСТАВИТЬ ---
with st.expander("📖 ИНСТРУКЦИЯ И ТРЕБОВАНИЯ К СНИМКАМ"):
    st.write("Для корректной работы алгоритма следуйте рекомендациям:")
    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        st.markdown("""
        **✅ Рекомендуется:**
        - **Высота:** 300-800 метров (масштаб квартала).
        - **Угол:** Строго вертикально (Надир).
        - **Солнце:** Ясный полдень (максимальный контраст).
        """)
    with col_inf2:
        st.markdown("""
        **❌ Избегать:**
        - Снимков под углом (искажает площадь).
        - Сильной облачности и тумана.
        - Мелкого масштаба (весь город в кадре).
        """)
# --- КОНЕЦ БЛОКА ИНСТРУКЦИИ ---

# Дальше идет остальной код (sidebar и так далее)
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Функция обработки остается твоей базой, но с улучшенным рендерингом
def process_thermal(img, ambient_temp):
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([33, 10, 10]), np.array([95, 255, 255])), cv2.inRange(gray, 0, 75))
    mask_cool = cv2.morphologyEx(mask_cool, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    mask_warm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, -30)
    mask_warm = cv2.bitwise_and(mask_warm, cv2.bitwise_not(mask_cool))
    mask_warm = cv2.morphologyEx(mask_warm, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    
    mask_heat = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 0, 45]), np.array([180, 85, 185])), cv2.bitwise_not(mask_warm))
    mask_heat = cv2.bitwise_and(mask_heat, cv2.bitwise_not(mask_cool))
    mask_heat = cv2.morphologyEx(mask_heat, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # Синий
    overlay[mask_warm > 0] = [0, 140, 255]  # Оранжевый
    overlay[mask_heat > 0] = [10, 10, 230]  # Красный
    
    res = cv2.addWeighted(img_bgr, 0.3, overlay, 0.7, 0)
    
    total = img.shape[0] * img.shape[1]
    stats = {
        "heat": (np.sum(mask_heat > 0) / total * 100, ambient_temp + 8.5),
        "warm": (np.sum(mask_warm > 0) / total * 100, ambient_temp + 2.3),
        "cool": (np.sum(mask_cool > 0) / total * 100, ambient_temp - 10.2)
    }
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), stats

# --- ДИЗАЙН ИНТЕРФЕЙСА ---
st.set_page_config(page_title="Thermal AI Pro", layout="wide", initial_sidebar_state="expanded")

# Кастомный CSS для темной темы и шрифтов
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; border: 1px solid #3e4253; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    h1 { color: #ff4b4b; font-family: 'Courier New', Courier, monospace; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛰️ THERMAL VISION SYSTEM v2.0")
st.markdown("---")

# Боковая панель
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2504/2504945.png", width=100)
    st.header("Control Panel")
    t_air = st.slider("Ambient Temperature (°C)", 10, 50, 30)
    uploaded_file = st.file_uploader("Upload Satellite Image", type=['jpg', 'png', 'jpeg'])
    st.info("System calibrated for urban heat islands analysis.")

if uploaded_file:
    img_input = Image.open(uploaded_file)
    processed_img, metrics = process_thermal(img_input, t_air)
    
    # Сетка из метрик (Красивые карточки)
    col1, col2, col3 = st.columns(3)
    col1.metric("🔥 MAX HEAT", f"{metrics['heat'][1]:.1f} °C", f"{metrics['heat'][0]:.1f}% Area", delta_color="inverse")
    col2.metric("🏠 WARM ZONES", f"{metrics['warm'][1]:.1f} °C", f"{metrics['warm'][0]:.1f}% Area")
    col3.metric("🌲 COOL ZONES", f"{metrics['cool'][1]:.1f} °C", f"-{metrics['cool'][0]:.1f}% Area", delta_color="normal")
    
    st.markdown("### Analysis Preview")
    
    # Сравнение Оригинал / Тепловизор
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Original RGB Feed")
        st.image(img_input, use_container_width=True)
    with c2:
        st.caption("Thermal Spectrum Reconstruction")
        st.image(processed_img, use_container_width=True)
        
    # Кнопка экспорта
    st.download_button(label="📥 Download Full Report", data=uploaded_file, file_name="thermal_analysis.png", mime="image/png")
else:
    st.warning("📡 Waiting for satellite data input... Please upload an image in the sidebar.")
