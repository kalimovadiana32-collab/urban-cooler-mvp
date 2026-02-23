import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pandas as pd
from streamlit_cropper import st_cropper

# --- ПЕРВАЯ СТРОЧКА КОДА ---
st.set_page_config(page_title="URBAN COOLER", layout="wide", initial_sidebar_state="expanded")

# --- 1. ТЕХНИЧЕСКИЕ ФУНКЦИИ ---
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

    # Анализ зон
    mask_cool = cv2.bitwise_or(cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255])), cv2.inRange(gray, 0, 75))
    mask_heat = cv2.bitwise_and(cv2.inRange(gray, 100, 185), cv2.bitwise_not(mask_cool))
    mask_warm = cv2.bitwise_and(cv2.inRange(gray, 186, 255), cv2.bitwise_not(mask_cool))

    overlay = img_bgr.copy()
    overlay[mask_cool > 0] = [240, 80, 0]   # Голубой (BGR)
    overlay[mask_warm > 0] = [0, 140, 255]  # Оранжевый
    overlay[mask_heat > 0] = [10, 10, 230]  # Красный
    
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

# --- 2. СТИЛИЗАЦИЯ И ФОН (БЕЛЫЙ ТЕКСТ) ---
st.markdown("""
    <style>
    /* Основной фон и глобальный белый цвет текста */
    .stApp {
        background-image: linear-gradient(rgba(10, 20, 30, 0.95), rgba(10, 20, 30, 0.95)), 
        url("https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80");
        background-size: cover; background-attachment: fixed; color: #ffffff !important;
    }
    
    /* Принудительно делаем заголовки и текст белыми */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown, .stText {
        color: #ffffff !important;
    }

    /* Карточки инструкций */
    .guide-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px; padding: 15px; margin-bottom: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .guide-title { font-size: 18px; font-weight: bold; color: #00ff88 !important; margin-bottom: 8px;}
    
    /* Алерт-панели */
    .danger-alert { background: rgba(255, 75, 75, 0.2); border-left: 5px solid #ff4b4b; padding: 15px; margin-bottom: 20px;}
    .safe-alert { background: rgba(0, 255, 136, 0.15); border-left: 5px solid #00ff88; padding: 15px; margin-bottom: 20px;}
    .info-panel { background: rgba(0, 191, 255, 0.15); border-left: 5px solid #00bfff; padding: 15px; margin-bottom: 20px;}
    
    /* Градусник */
    .thermo-container { width: 50px; height: 200px; background: rgba(255,255,255,0.1); border: 3px solid #fff; border-radius: 25px; position: relative; margin: 0 auto; overflow: hidden; }
    .thermo-fill { position: absolute; bottom: 0; width: 100%; transition: all 0.5s ease; }
    
    /* Таблицы */
    table { color: white !important; background-color: rgba(255,255,255,0.05) !important; }
    thead tr th { background-color: rgba(255,255,255,0.1) !important; color: #00ff88 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ URBAN COOLER: AI-Анализ Тепловых Островов")

# --- ОБШИРНАЯ ИНСТРУКЦИЯ ---
with st.expander("📖 ИНСТРУКЦИЯ ПО РАБОТЕ С ПЛАТФОРМОЙ (Нажмите, чтобы развернуть)", expanded=True):
    st.markdown("""
    Добро пожаловать в **Urban Cooler** — интеллектуальную систему оценки городского микроклимата и проектирования благоустройства.
    
    **Как получить точный результат:**
    1. **Подготовка снимка:** Откройте Яндекс/Google Карты, переключитесь в режим "Спутник" (строго 2D, вид сверху). Сделайте скриншот проблемного района.
    2. **Загрузка и Настройка:** Загрузите снимок ниже. Укажите климатическую зону вашего города и текущую летнюю температуру воздуха для точной калибровки тепловой модели.
    3. **Выделение зоны (Кроппинг):** С помощью зеленой рамки выделите конкретный квартал для анализа. Система исходит из допущения, что стандартный выделенный квартал равен **~10 Гектарам**.
    4. **Анализ ИИ:** Алгоритм компьютерного зрения распознает типы поверхностей (Асфальт, Бетон, Зелень) и рассчитает локальный нагрев.
    5. **Конструктор:** Следуйте сгенерированным рекомендациям (в га, кв.м. и шт.), применяйте решения по озеленению и смотрите, как изменится температура в сводном отчете.
    """)

# --- ВВОД ДАННЫХ ---
st.write("### ⚙️ 1. Настройки среды")
cfg_cols = st.columns(2)
with cfg_cols[0]:
    climate = st.selectbox("Климатическая зона:", ["Умеренный", "Тропики", "Пустыня", "Арктика / Зима"])
with cfg_cols[1]:
    t_air = st.number_input("Базовая температура воздуха (°C):", value=28, step=1)

uploaded_file = st.file_uploader("📥 Загрузите спутниковый снимок (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    img_raw = auto_enhance_image(Image.open(uploaded_file))
    st.write("### 🎯 2. Выделение зоны анализа")
    st.markdown("*Растяните зеленую рамку на нужный квартал (расчеты ведутся из предположения, что зона = 10 Га)*")
    
    cropped_img = st_cropper(img_raw, realtime_update=True, box_color='#00ff88', aspect_ratio=None)
    
    if cropped_img:
        processed_img, stats = process_thermal(cropped_img, t_air, climate)
        
        # БАЗОВЫЕ РАСЧЕТЫ ПЛОЩАДЕЙ (Допущение: участок = 10 Га)
        total_area_ha = 10.0
        area_heat = (stats['road']['p'] / 100) * total_area_ha
        area_warm = (stats['build']['p'] / 100) * total_area_ha
        area_cool = (stats['eco']['p'] / 100) * total_area_ha

        st.divider()
        st.write("### 🌡️ 3. Результаты сканирования")
        
        # Статус перегрева
        if stats['avg_t'] > stats['danger_limit']:
            st.markdown(f'<div class="danger-alert"><b>⚠️ КРИТИЧЕСКИЙ ТЕПЛОВОЙ ОСТРОВ: {stats["avg_t"]:.1f}°C</b><br>Температура поверхности значительно превышает норму. Требуется срочное вмешательство.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-alert"><b>✅ КОМФОРТНАЯ ЗОНА: {stats["avg_t"]:.1f}°C</b><br>Температурный режим в пределах допустимых значений.</div>', unsafe_allow_html=True)

        # Сравнение фото
        img_col1, img_col2 = st.columns(2)
        with img_col1: 
            st.image(cropped_img, caption="Оригинальный снимок", use_container_width=True)
        with img_col2: 
            st.image(processed_img, caption="Распознанные тепловые зоны (Красный - жар, Голубой - прохлада)", use_container_width=True)

        # --- ТОЧНЫЕ РЕКОМЕНДАЦИИ ИИ ---
        st.divider()
        st.write("### 💡 4. Диагностика и точные рекомендации ИИ")
        
        # Логика генерации советов
        rec_trees_ha = round(area_heat * 0.35, 1) # 35% дорог закрыть тенью
        rec_fountains = max(1, int(area_heat / 1.5)) # 1 фонтан на 1.5 га жары
        rec_vertical_sqm = int((area_warm * 10000) * 0.15) # 15% площади зданий (в кв.м)
        rec_albedo_ha = round(area_heat * 0.4, 1) # 40% асфальта осветлить
        
        st.markdown(f"""
        <div class="info-panel">
        <b>Анализ территории (Общая площадь ~10 Га):</b><br>
        🔴 Зона экстремального нагрева (Асфальт/Открытый грунт): <b>{area_heat:.1f} Га</b> ({stats['road']['p']:.1f}%)<br>
        🟠 Зона накопления тепла (Здания/Бетон): <b>{area_warm:.1f} Га</b> ({stats['build']['p']:.1f}%)<br>
        🔵 Зона естественного охлаждения (Парки/Вода): <b>{area_cool:.1f} Га</b> ({stats['eco']['p']:.1f}%)<br><br>
        <b>Рекомендуемый план действий для нормализации климата:</b>
        <ul>
            <li>🌳 <b>Озеленение:</b> Высадить деревья с широкой кроной на площади не менее <b>{rec_trees_ha} Га</b> для затенения теплоемких поверхностей.</li>
            <li>💧 <b>Водные объекты:</b> Установить сухие фонтаны или искусственные водоемы (рекомендуемое количество: <b>{rec_fountains} шт.</b>).</li>
            <li>🌿 <b>Вертикальное озеленение:</b> Интегрировать вьющуюся растительность на фасадах зданий (около <b>{rec_vertical_sqm} кв.м.</b> стен).</li>
            <li>🛣️ <b>Альбедо поверхностей:</b> Заменить темный асфальт на светлую плитку или применить отражающее покрытие на площади <b>{rec_albedo_ha} Га</b>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        # --- КОНСТРУКТОР БЛАГОУСТРОЙСТВА ---
        st.write("### 🛠️ 5. Интерактивный конструктор решений")
        st.markdown("Примените инструменты ниже, чтобы увидеть, как изменится температура района.")
        
        col_tool1, col_tool2 = st.columns(2)
        
        with col_tool1:
            st.markdown("**🌿 Биологические решения**")
            user_trees_ha = st.slider("Высадка крупномеров (Га)", 0.0, float(area_heat + area_warm), 0.0, step=0.1)
            user_vertical = st.slider("Вертикальное озеленение (кв.м фасадов)", 0, int(area_warm * 10000 * 0.5), 0, step=500)
            
        with col_tool2:
            st.markdown("**🏗️ Инженерные решения**")
            user_fountains = st.slider("Водные объекты (шт)", 0, 10, 0)
            user_albedo_ha = st.slider("Осветление асфальта (Га)", 0.0, float(area_heat), 0.0, step=0.1)

        # --- ФИЗИЧЕСКИЙ РАСЧЕТ ОХЛАЖДЕНИЯ ---
        # 1 Га деревьев снижает общую темп на ~0.8 градуса для 10 Га
        # 1 Фонтан снижает на ~0.3 градуса
        # 1000 кв.м вертикалки = ~0.1 градуса
        # 1 Га светлого асфальта = ~0.6 градуса
        
        t_drop = (user_trees_ha * 0.8) + (user_fountains * 0.3) + ((user_vertical / 1000) * 0.1) + (user_albedo_ha * 0.6)
        new_avg_t = stats['avg_t'] - t_drop

        # --- ИТОГОВЫЙ ОТЧЕТ И ГРАДУСНИК ---
        st.divider()
        st.write("### 📊 6. Прогноз эффективности")
        
        res_col1, res_col2 = st.columns([1, 3])
        
        with res_col1:
            # Визуализация градусника
            fill_percent = min(100, max(5, (new_avg_t / 60) * 100))
            t_color = "#ff4b4b" if new_avg_t > stats['danger_limit'] else "#00ff88"
            st.markdown(f'''
            <div class="thermo-container">
                <div class="thermo-fill" style="height:{fill_percent}%; background:{t_color};"></div>
            </div>
            ''', unsafe_allow_html=True)
            st.write(f"<center><h3 style='margin-top:10px;'>{new_avg_t:.1f}°C</h3></center>", unsafe_allow_html=True)
            
        with res_col2:
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Текущая T° района", f"{stats['avg_t']:.1f} °C")
            metric_col2.metric("Снижение температуры", f"-{t_drop:.1f} °C", delta_color="inverse")
            
            st.markdown("#### 📝 Заключение системы:")
            if t_drop == 0:
                st.write("Ожидание ввода параметров. Используйте ползунки в конструкторе для применения решений.")
            elif new_avg_t <= stats['danger_limit'] and t_drop > 2:
                st.markdown(f"<span style='color:#00ff88; font-weight:bold;'>УСПЕШНАЯ МОДЕРНИЗАЦИЯ:</span> Вы успешно вывели район из зоны теплового риска. Внедренные {user_trees_ha} Га деревьев и инженерные решения создали устойчивый микроклимат. Рекомендуется передать проект в работу.", unsafe_allow_html=True)
            elif t_drop > 0:
                st.markdown(f"<span style='color:#00bfff; font-weight:bold;'>ЕСТЬ УЛУЧШЕНИЯ:</span> Температура снизилась, но район все еще накапливает избыточное тепло. Рекомендуется увеличить площадь озеленения до {rec_trees_ha} Га.", unsafe_allow_html=True)

            # Таблица "до/после"
            st.write("")
            report_df = pd.DataFrame({
                "Метрика": ["Средняя температура поверхности", "Эффективная площадь охлаждения", "Статус микроклимата"],
                "До изменений": [f"{stats['avg_t']:.1f} °C", f"{area_cool:.1f} Га", "Критический" if stats['avg_t'] > stats['danger_limit'] else "В норме"],
                "После изменений (Прогноз)": [f"{new_avg_t:.1f} °C", f"{(area_cool + user_trees_ha):.1f} Га", "Комфортный" if new_avg_t <= stats['danger_limit'] else "Перегрев"]
            })
            st.table(report_df)
