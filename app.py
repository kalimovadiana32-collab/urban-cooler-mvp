import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# 1. ЯДРО СИСТЕМЫ: ФУНКЦИЯ ОБРАБОТКИ
def process_thermal(img, ambient_temp, climate_type):
    img = np.array(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ФИЗИЧЕСКИЕ ПАРАМЕТРЫ ДЛЯ РАЗНЫХ КЛИМАТОВ
    offsets = {
        "Умеренный": {
            "heat": 8.0, "warm": 2.0, "cool": -10.0, 
            "danger": 30.0, "labels": ["Жара (Асфальт)", "Тепло (Дома)", "Прохлада (Зелень)"]
        },
        "Тропики (Влажно)": {
            "heat": 10.0, "warm": 4.0, "cool": -4.0, 
            "danger": 35.0, "labels": ["Жара (Дороги)", "Тепло (Застройка)", "Влажные зоны"]
        },
        "Пустыня (Сухо)": {
            "heat": 18.0, "warm": 7.0, "cool": -3.0, 
            "danger": 45.0, "labels": ["Экстремальный жар", "Нагретый песок", "Редкая тень"]
        },
        "Арктика / Зима": {
            "heat": 4.0, "warm": 15.0, "cool": -5.0, 
            "danger": 5.0, "labels": ["Очищенный путь", "Теплопотери зданий", "Снег / Лед"]
        }
    }
    
    conf = offsets[climate_type]

    # Создание масок (Зимой снег ищем по яркости белого цвета)
    if climate_type == "Арктика / Зима":
        mask_cool = cv2.inRange(gray, 200, 255) # Снег белый
    else:
        mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([33, 10, 10]), np.array([95, 255, 255])), cv2.inRange(gray, 0, 75))

    mask_warm = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 145, -30)
    mask_heat = cv2.bitwise_and(cv2.inRange(hsv, np.array([0, 0, 45]), np.array([180, 85, 185])), cv2.bitwise_not(mask_cool))

    # Визуализация (Overlay)
    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # Синий
    overlay[mask_warm > 0] = [0, 140, 255]  # Оранжевый
    overlay[mask_heat > 0] = [10, 10, 230]  # Красный
    
    res = cv2.addWeighted(img_bgr, 0.3, overlay, 0.7, 0)
    
    total = img.shape[0] * img.shape[1]
    stats = {
        "heat": (np.sum(mask_heat > 0) / total * 100, ambient_temp + conf["heat"]),
        "warm": (np.sum(mask_warm > 0) / total * 100, ambient_temp + conf["warm"]),
        "cool": (np.sum(mask_cool > 0) / total * 100, ambient_temp + conf["cool"]),
        "danger_limit": conf["danger"],
        "labels": conf["labels"]
    }
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB), stats

# 2. ОФОРМЛЕНИЕ ИНТЕРФЕЙСА
st.set_page_config(page_title="Thermal AI Pro", layout="wide")

st.title("🛰️ THERMAL VISION v3.4 Global")

# Инструкция и ссылки
with st.expander("📖 ИНСТРУКЦИЯ И КАРТЫ (Масштаб 20м)"):
    col_i1, col_i2 = st.columns([2, 1])
    with col_i1:
        st.markdown("""
        1. **Масштаб:** 20-50 метров (видны машины/тени).
        2. **Угол:** Нажмите **'U'** в Google Maps для вида строго сверху.
        3. **Загрузка:** Выберите файл ниже.
        """)
    with col_i2:
        st.markdown("**🔗 Ссылки:**")
        st.markdown("- [Google Maps](https://www.google.com/maps/@?api=1&map_action=map&basemap=satellite)")
        st.markdown("- [Yandex Maps](https://yandex.ru/maps/?l=sat)")

# Основное поле загрузки (Mobile Friendly)
st.subheader("1. Выберите снимок города")
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

# Настройки в боковой панели
with st.sidebar:
    st.header("⚙️ ПАРАМЕТРЫ")
    climate = st.selectbox("🌍 Климат", ["Умеренный", "Тропики (Влажно)", "Пустыня (Сухо)", "Арктика / Зима"])
    t_air = st.slider("🌡️ Температура воздуха (°C)", -30, 55, 20)
    st.markdown("---")
    st.caption("Aura Thermal Engine v3.4")

# 3. ЛОГИКА ОБРАБОТКИ И ВЫВОДА
if uploaded_file:
    img_input = Image.open(uploaded_file)
    processed_img, metrics = process_thermal(img_input, t_air, climate)
    
    # Умные алерты в зависимости от климата
    main_temp = metrics['heat'][1] if climate != "Арктика / Зима" else metrics['warm'][1]
    danger_val = metrics['danger_limit']
    
    st.subheader("2. Результаты анализа")
    
    if climate == "Арктика / Зима":
        if main_temp > danger_val:
            st.warning(f"❄️ **ЗИМНИЙ АНАЛИЗ:** Здания имеют высокую температуру ({main_temp:.1f}°C). Вероятны теплопотери.")
        else:
            st.success(f"✅ **НОРМА:** Аномальных утечек тепла не обнаружено.")
    else:
        if main_temp > danger_val:
            st.error(f"⚠️ **ТЕПЛОВОЙ ОСТРОВ:** Критический нагрев ({main_temp:.1f}°C) для зоны '{climate}'!")
        elif main_temp > (danger_val - 5):
            st.warning("🔔 **ПРЕДУПРЕЖДЕНИЕ:** Температура близка к порогу опасности.")
        else:
            st.success("✅ **БЕЗОПАСНО:** Температурный фон в норме.")

    # Вывод изображений (Сравнение)
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(img_input, caption="Спутниковый оригинал", use_container_width=True)
    with col_img2:
        st.image(processed_img, caption="Тепловая карта", use_container_width=True)

    # Таблица отчета
    st.markdown("### 📝 Аналитический отчет")
    df = pd.DataFrame({
        "Зона": metrics['labels'],
        "Площадь (%)": [f"{metrics['heat'][0]:.1f}", f"{metrics['warm'][0]:.1f}", f"{metrics['cool'][0]:.1f}"],
        "Температура (°C)": [f"{metrics['heat'][1]:.1f}", f"{metrics['warm'][1]:.1f}", f"{metrics['cool'][1]:.1f}"]
    })
    st.table(df)

    # Скачивание (Фикс для Excel)
    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Скачать отчет для Excel (.csv)", data=csv, file_name='thermal_report.csv', mime='text/csv')
