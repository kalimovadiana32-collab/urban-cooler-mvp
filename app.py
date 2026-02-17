import streamlit as st
import cv2
import numpy as np
from PIL import Image

def process_thermal(img, ambient_temp):
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Маски зон
    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([33, 10, 10]), np.array([95, 255, 255])), cv2.inRange(gray, 0, 75))
    mask_cool = cv2.morphologyEx(mask_cool, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    mask_warm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, -30)
    mask_warm = cv2.bitwise_and(mask_warm, cv2.bitwise_not(mask_cool))
    mask_warm = cv2.morphologyEx(mask_warm, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    
    mask_heat = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 0, 45]), np.array([180, 85, 185])), cv2.bitwise_not(mask_warm))
    mask_heat = cv2.bitwise_and(mask_heat, cv2.bitwise_not(mask_cool))
    mask_heat = cv2.morphologyEx(mask_heat, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8))

    # Визуализация
    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # Синий (BGR)
    overlay[mask_warm > 0] = [0, 140, 255]  # Оранжевый (BGR)
    overlay[mask_heat > 0] = [10, 10, 230]  # Красный (BGR)
    
    res = cv2.addWeighted(img_bgr, 0.35, overlay, 0.65, 0)
    
    # Статистика
    total = img.shape[0] * img.shape[1]
    p_heat, p_warm, p_cool = (np.sum(m > 0)/total*100 for m in [mask_heat, mask_warm, mask_cool])
    
    # Рисуем панель
    cv2.rectangle(res, (15, 15), (450, 180), (15, 15, 15), -1)
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(res, f"HEAT (Roads): {ambient_temp+8.5:.1f}C | {p_heat:.1f}%", (30, 60), f, 0.7, (30, 30, 255), 2)
    cv2.putText(res, f"WARM (Buildings): {ambient_temp+2.3:.1f}C | {p_warm:.1f}%", (30, 105), f, 0.7, (0, 150, 255), 2)
    cv2.putText(res, f"COOL (Nature): {ambient_temp-10.2:.1f}C | {p_cool:.1f}%", (30, 150), f, 0.7, (255, 120, 0), 2)

    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

st.set_page_config(page_title="Thermal AI MVP", layout="wide")
st.title("🛰️ Thermal Analysis MVP")
st.sidebar.header("Настройки")
t = st.sidebar.slider("Температура воздуха (°C)", 10, 50, 35)
f = st.sidebar.file_uploader("Загрузить снимок", type=['jpg', 'png', 'jpeg'])

if f:
    img = Image.open(f)
    st.image(process_thermal(img, t), use_container_width=True)
