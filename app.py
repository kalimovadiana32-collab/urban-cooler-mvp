import streamlit as st
from PIL import Image
from processor import process_thermal_image, get_ai_recommendations

# Настройка страницы (ДОЛЖНА БЫТЬ ПЕРВОЙ)
st.set_page_config(page_title="URBAN COOLER", layout="wide")

st.title("🏙️ URBAN COOLER: Space AI Analysis")
st.write("Командный проект: Калимова Диана и Умаржан Айлин (AeroSpace)")

# Боковая панель
st.sidebar.header("Настройки анализа")
city_type = st.sidebar.selectbox("Климатическая зона", ["Степной (Астана)", "Предгорный (Алматы)", "Пустынный (Шымкент)", "Умеренный"])
base_temp = st.sidebar.slider("Базовая температура воздуха (°C)", 10, 50, 25)

uploaded_file = st.file_uploader("Загрузите скриншот карты (спутник)", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Тепловой анализ ИИ")
        # Вызываем логику из файла processor.py
        heatmap, result_temp = process_thermal_image(image, base_temp, city_type)
        st.image(heatmap, use_container_width=True)
        st.metric("Прогноз температуры участка", f"{result_temp} °C", f"{round(result_temp - base_temp, 1)} °C")

    # Секция AI Ассистента
    st.divider()
    st.subheader("🤖 Space AI Advisor")
    recommendation = get_ai_recommendations(result_temp - base_temp)
    st.info(recommendation)

st.markdown("---")
st.caption("Разработано для конкурса AeroSpace (Space AI) 🚀")
