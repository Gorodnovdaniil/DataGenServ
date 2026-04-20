"""
app.py - Frontend приложение VibeData на Streamlit.

Интерфейс для генерации синтетических данных на основе имитационного моделирования.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from backend.main import orchestrator
from backend.rag_engine import rag_engine


# Настройка страницы
st.set_page_config(
    page_title="VibeData - Генератор синтетических данных",
    page_icon="📊",
    layout="wide"
)

# Заголовок
st.title("📊 VibeData")
st.markdown("**Система генерации синтетических данных на основе имитационного моделирования**")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Выбор распределения
    distributions = orchestrator.get_all_distributions()
    dist_options = {d["name"]: d["id"] for d in distributions}
    
    selected_dist_name = st.selectbox(
        "Выберите распределение:",
        options=list(dist_options.keys())
    )
    selected_dist_id = dist_options[selected_dist_name]
    
    # Размер выборки
    sample_size = st.slider("Размер выборки:", min_value=10, max_value=10000, value=100)
    
    # Seed для воспроизводимости
    use_seed = st.checkbox("Использовать seed")
    seed_value = None
    if use_seed:
        seed_value = st.number_input("Seed:", min_value=0, value=42)
    
    st.divider()
    
    # Информация о распределении
    dist_info = orchestrator.get_distribution_info(selected_dist_id)
    with st.expander("ℹ️ О распределении"):
        st.write(f"**{dist_info.get('name', selected_dist_id)}**")
        st.write(dist_info.get('description', ''))
        st.latex(dist_info.get('formula', ''))
        
        if 'parameters' in dist_info:
            st.write("**Параметры:**")
            for param_name, param_info in dist_info['parameters'].items():
                st.write(f"- `{param_name}`: {param_info.get('description', '')} (по умолчанию: {param_info.get('default', 'N/A')})")

# Основная область
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎲 Параметры генерации")
    
    # Динамические поля для параметров распределения
    params = {}
    
    if selected_dist_id == "uniform":
        col_a, col_b = st.columns(2)
        with col_a:
            params["a"] = st.number_input("Минимум (a):", value=0.0)
        with col_b:
            params["b"] = st.number_input("Максимум (b):", value=1.0)
            
    elif selected_dist_id == "exponential":
        params["lambda"] = st.number_input("Интенсивность (λ):", min_value=0.01, value=1.0, step=0.1)
        
    elif selected_dist_id == "gamma":
        col_alpha, col_beta = st.columns(2)
        with col_alpha:
            params["alpha"] = st.number_input("Форма (α):", min_value=0.01, value=1.0, step=0.1)
        with col_beta:
            params["beta"] = st.number_input("Масштаб (β):", min_value=0.01, value=1.0, step=0.1)
            
    elif selected_dist_id == "normal":
        col_mu, col_sigma = st.columns(2)
        with col_mu:
            params["mu"] = st.number_input("Мат. ожидание (μ):", value=0.0)
        with col_sigma:
            params["sigma"] = st.number_input("Стд. отклонение (σ):", min_value=0.01, value=1.0, step=0.1)
            
    elif selected_dist_id == "poisson":
        params["lambda"] = st.number_input("Интенсивность (λ):", min_value=0.01, value=1.0, step=0.1)
        
    elif selected_dist_id == "triangular":
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            params["a"] = st.number_input("Минимум (a):", value=0.0)
        with col_b:
            params["b"] = st.number_input("Максимум (b):", value=1.0)
        with col_c:
            params["c"] = st.number_input("Мода (c):", value=0.5)
    
    # Кнопка генерации
    generate_btn = st.button("🚀 Сгенерировать данные", type="primary", use_container_width=True)

with col2:
    st.subheader("📈 Статистика")
    
    if generate_btn or 'last_result' in st.session_state:
        if generate_btn:
            # Генерация данных
            result = orchestrator.generate_data(
                distribution=selected_dist_id,
                n=sample_size,
                params=params,
                seed=seed_value
            )
            st.session_state.last_result = result
        
        result = st.session_state.last_result
        
        if result.get("success"):
            stats = result.get("statistics", {})
            
            st.metric("Среднее", f"{stats.get('mean', 0):.4f}")
            st.metric("Стд. отклонение", f"{stats.get('std', 0):.4f}")
            st.metric("Медиана", f"{stats.get('median', 0):.4f}")
            
            col_min, col_max = st.columns(2)
            with col_min:
                st.metric("Минимум", f"{stats.get('min', 0):.4f}")
            with col_max:
                st.metric("Максимум", f"{stats.get('max', 0):.4f}")
            
            # Для Пуассона показываем уникальные значения
            if selected_dist_id == "poisson" and "value_counts" in stats:
                st.write("**Распределение значений:**")
                vc_df = pd.DataFrame({
                    "Значение": list(stats["value_counts"].keys()),
                    "Частота": list(stats["value_counts"].values())
                })
                st.dataframe(vc_df, hide_index=True, use_container_width=True)
        else:
            st.error(f"Ошибка генерации: {result.get('error', 'Неизвестная ошибка')}")

# Результаты генерации
if generate_btn or 'last_result' in st.session_state:
    result = st.session_state.get('last_result')
    
    if result and result.get("success"):
        st.divider()
        st.subheader("📋 Результаты")
        
        # Таблица с данными (первые 100 значений)
        data = result.get("data", [])
        
        if len(data) > 0:
            # Создаем DataFrame
            df = pd.DataFrame({"№": range(1, len(data) + 1), "Значение": data})
            
            # Показываем таблицу с пагинацией
            st.write(f"**Сгенерировано значений:** {result.get('full_data_count', len(data))}")
            st.dataframe(df, hide_index=True, use_container_width=True, height=300)
            
            # Кнопка скачивания
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Скачать CSV",
                data=csv,
                file_name=f"vibedata_{selected_dist_id}_{sample_size}.csv",
                mime="text/csv"
            )
        
        # Гистограмма
        st.subheader("📊 Визуализация")
        chart_data = pd.DataFrame({"Значения": data})
        st.bar_chart(chart_data)

# RAG поиск по лекциям
st.divider()
st.subheader("📚 Поиск в лекциях")

with st.expander("🔍 Поиск теории по имитационному моделированию"):
    query = st.text_input("Введите запрос (например, 'метод обратных функций'):")
    
    if query:
        results = rag_engine.search(query, top_k=3)
        
        for i, res in enumerate(results, 1):
            with st.container():
                st.markdown(f"**{i}.** {res['text']}")
                st.caption(f"📄 {res['source']}, стр. {res['page']} | Релевантность: {res['relevance_score']:.2f}")
                st.divider()

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>VibeData © 2024 | Генерация синтетических данных методами имитационного моделирования</p>
    </div>
    """,
    unsafe_allow_html=True
)