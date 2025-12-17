import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

# Configuración de la Página
st.set_page_config(
    page_title="Predicción Energética: Campus UBB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilos
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1a5276;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Carga de Recursos
@st.cache_resource
def load_resources():
    resources = {}
    try:
        resources['model'] = joblib.load('mlp_model_entrenado.pkl')
    except:
        resources['model'] = None
        
    try:
        resources['loss_history'] = joblib.load('loss_history.pkl')
    except:
        resources['loss_history'] = None
    return resources

# Carga de Dataset
@st.cache_data
def load_dataset():
    """Carga el dataset original para análisis detallado"""
    try:
        base_path = "dataset"
        if not os.path.exists(base_path):
            base_path = "."
        df_consumption = pd.read_csv(os.path.join(base_path, 'building_consumption.csv'))
        df_weather = pd.read_csv(os.path.join(base_path, 'weather_data.csv'))
        df_meta = pd.read_csv(os.path.join(base_path, 'building_meta.csv'))
        return df_consumption, df_weather, df_meta
    except:
        return None, None, None

# Generador Datos Sintéticos
@st.cache_data
def get_synthetic_data():
    """Datos falsos solo para la pestaña de Lab de Entrenamiento"""
    N = 1000
    dates = pd.date_range(start="2025-01-01", periods=N, freq="h")
    data = {
        'air_temperature': np.random.uniform(5, 35, N), 
        'gross_floor_area': np.random.uniform(500, 5000, N), 
        'hour': dates.hour
    }
    df = pd.DataFrame(data)
    df['consumption'] = (df['air_temperature'] - 20)**2 + df['gross_floor_area']*0.05 + np.random.normal(0,5,N) + 200
    return df

# Inicialización
resources = load_resources()
final_pipeline = resources['model']
loss_history_offline = resources['loss_history']
df_consumption, df_weather, df_meta = load_dataset() 

# Funciones para graficos

def generate_hourly_profile(model, temp, wind, area, category, day_of_week, is_holiday):
    hours = list(range(24))
    predictions = []
    for hour in hours:
        input_df = pd.DataFrame({
            'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [wind],
            'gross_floor_area': [area], 'hour': [hour], 'day_of_week': [day_of_week],
            'month': [6], 'is_holiday': [is_holiday], 'category': [category]
        })
        pred = model.predict(input_df)[0]
        predictions.append(max(0, pred))
    return hours, predictions

def generate_weekly_profile(model, temp, wind, area, category):
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    daily_consumption = []
    for day in range(7):
        day_total = 0
        for hour in range(24):
            input_df = pd.DataFrame({
                'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [wind],
                'gross_floor_area': [area], 'hour': [hour], 'day_of_week': [day],
                'month': [6], 'is_holiday': [0 if day >= 5 else 1], 'category': [category]
            })
            pred = model.predict(input_df)[0]
            day_total += max(0, pred)
        daily_consumption.append(day_total)
    return days, daily_consumption

def generate_temperature_sensitivity(model, area, category, hour=14):
    temps = np.linspace(-5, 45, 50)
    predictions = []
    for temp in temps:
        input_df = pd.DataFrame({
            'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [10],
            'gross_floor_area': [area], 'hour': [hour], 'day_of_week': [2],
            'month': [6], 'is_holiday': [0], 'category': [category]
        })
        pred = model.predict(input_df)[0]
        predictions.append(max(0, pred))
    return temps, predictions

def compare_building_types(model, temp, hour):
    categories = ['office', 'teaching', 'library', 'mixed use', 'other']
    labels = ['Oficinas', 'Aulas', 'Biblioteca', 'Uso Mixto', 'Otros']
    predictions = []
    for cat in categories:
        input_df = pd.DataFrame({
            'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [10],
            'gross_floor_area': [2000], 'hour': [hour], 'day_of_week': [2],
            'month': [6], 'is_holiday': [0], 'category': [cat]
        })
        pred = model.predict(input_df)[0]
        predictions.append(max(0, pred))
    return labels, predictions

# Estructura Principal

tab_lab, tab_sim, tab_analysis, tab_data = st.tabs([
    "Lab de Entrenamiento", 
    "Simulador", 
    "Análisis Predictivo",
    "Explorador de Datos"
]) 

# Demo sintetico
with tab_lab:
    st.header("Laboratorio de Entrenamiento en Vivo")
    st.info("Entrena una versión simplificada del modelo para probar hiperparámetros en tiempo real.")
    
    col1, col2 = st.columns(2)
    with col1:
        h_layers = eval(st.selectbox("Arquitectura (Capas Ocultas)", ['(50,)', '(100, 50)', '(64, 32)'], index=1))
    with col2:
        iters = st.slider("Iteraciones (Épocas)", 50, 1000, 200)

    if st.button("Entrenar Modelo Demo"):
        df_synth = get_synthetic_data()
        X = df_synth[['air_temperature', 'gross_floor_area', 'hour']]
        y = df_synth['consumption']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        with st.spinner(f"Entrenando red neuronal con {h_layers}..."):
            model = MLPRegressor(hidden_layer_sizes=h_layers, max_iter=iters, random_state=42)
            model.fit(X_scaled, y)
        
        st.subheader("Curva de Aprendizaje Resultante")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(model.loss_curve_, color='#2980b9', linewidth=2)
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Loss Function (Error)")
        ax.set_title("Convergencia del Modelo Demo")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.success(f"Entrenamiento finalizado. Loss Final: {model.loss_:.4f}")

# Simulador Completo
with tab_sim:
    st.header("Simulador de Consumo: Escenario Campus")
    
    if final_pipeline:
        mode = st.radio("Modo de Simulación:", 
                        ["Edificio Individual", "Campus Completo", "Simulación Temporal"],
                        horizontal=True)
        st.markdown("---")

        # Individual
        if "Individual" in mode:
            col_inputs, col_results = st.columns([1, 2])
            
            with col_inputs:
                st.subheader("Parámetros de Entrada")
                
                with st.expander("Condiciones Ambientales", expanded=True):
                    temp = st.slider("Temperatura (°C)", -5.0, 45.0, 22.0, 0.5)
                    humidity = st.slider("Humedad Relativa (%)", 0, 100, 50)
                    wind = st.slider("Velocidad del Viento (km/h)", 0.0, 50.0, 10.0, 0.5)
                
                with st.expander("Contexto Temporal", expanded=True):
                    hora = st.slider("Hora del Día", 0, 23, 14)
                    dia = st.selectbox("Día de la Semana", range(7), 
                                      format_func=lambda x: ['Lunes','Martes','Miércoles','Jueves','Viernes','Sábado','Domingo'][x])
                    mes = st.selectbox("Mes", range(1, 13),
                                      format_func=lambda x: ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'][x-1],
                                      index=5)
                    feriado = st.checkbox("¿Es día feriado?")
                
                with st.expander("Caracteristicas del Edificio", expanded=True):
                    area = st.number_input("Área bruta (m²)", 100.0, 50000.0, 2000.0, 100.0)
                    categoria = st.selectbox("Tipo de Edificio", 
                                            ['office', 'teaching', 'library', 'mixed use', 'other'],
                                            format_func=lambda x: {'office': 'Oficinas', 'teaching': 'Aulas', 
                                                                 'library': 'Biblioteca', 'mixed use': 'Uso Mixto', 
                                                                 'other': 'Otros'}[x])
                
                predict_btn = st.button("Calcular Predicción", type="primary", use_container_width=True)

            with col_results:
                if predict_btn:
                    input_df = pd.DataFrame({
                        'air_temperature': [temp],
                        'relative_humidity': [humidity],
                        'wind_speed': [wind],
                        'gross_floor_area': [area],
                        'hour': [hora],
                        'day_of_week': [dia],
                        'month': [mes],
                        'is_holiday': [1 if feriado else 0],
                        'category': [categoria]
                    })
                    prediction = final_pipeline.predict(input_df)[0]
                    prediction = max(0, prediction)
                    
                    st.subheader("Resultado de la Predicción")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1: st.metric("Consumo Estimado", f"{prediction:.2f} kWh")
                    with col_m2: st.metric("Estimado Diario", f"{prediction * 24:.0f} kWh")
                    with col_m3: st.metric("Costo Hora (est.)", f"${prediction * 0.15:.2f}")
                    
                    st.markdown("---")
                    
                    col_g1, col_g2 = st.columns(2)
                    with col_g1:
                        st.subheader("Perfil Horario (24h)")
                        hours, hourly_pred = generate_hourly_profile(
                            final_pipeline, temp, wind, area, categoria, dia, 1 if feriado else 0
                        )
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        ax1.fill_between(hours, hourly_pred, alpha=0.3, color='#3498db')
                        ax1.plot(hours, hourly_pred, 'o-', color='#2980b9', linewidth=2)
                        ax1.axvline(x=hora, color='red', linestyle='--')
                        ax1.scatter([hora], [prediction], color='red', s=100, zorder=5)
                        ax1.set_xlabel("Hora del día")
                        ax1.set_ylabel("kWh")
                        ax1.grid(True, alpha=0.3)
                        st.pyplot(fig1)
                    
                    with col_g2:
                        st.subheader("Perfil Semanal")
                        days, weekly_pred = generate_weekly_profile(final_pipeline, temp, wind, area, categoria)
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        colors = ['#3498db' if i < 5 else '#e74c3c' for i in range(7)]
                        ax2.bar(days, weekly_pred, color=colors, edgecolor='white')
                        ax2.axhline(y=np.mean(weekly_pred), color='green', linestyle='--')
                        ax2.set_xlabel("Día")
                        ax2.grid(True, alpha=0.3, axis='y')
                        st.pyplot(fig2)
                else:
                    st.info("Configure los parámetros y presione 'Calcular Predicción' para ver los resultados.") 

        # Campus entero
        elif "Campus" in mode:
            st.subheader("Configuración del Campus")
            col_config, col_ambient = st.columns([2, 1])
            
            with col_ambient:
                st.markdown("*Condiciones Ambientales*")
                temp_c = st.slider("Temperatura (°C)", -5.0, 45.0, 22.0, key="tc")
                wind_c = st.slider("Viento (km/h)", 0.0, 50.0, 10.0, key="wc")
                hora_c = st.slider("Hora del día", 0, 23, 14, key="hc")
                dia_c = st.selectbox("Día", range(7), format_func=lambda x: ['Lun','Mar','Mié','Jue','Vie','Sáb','Dom'][x], key="dc")
                feriado_c = st.checkbox("¿Feriado?", key="fc")
            
            with col_config:
                st.markdown("*Composición del Campus*")
                campus_data = []
                edificios = [("Oficinas", "office", 5, 1500), ("Aulas", "teaching", 8, 2500), ("Biblioteca", "library", 1, 4000), 
                             ("Uso Mixto", "mixed use", 3, 3000), ("Otros", "other", 2, 1000)]
                
                for nombre, cat, cant_def, area_def in edificios:
                    cols = st.columns(3)
                    with cols[0]: st.markdown(f"**{nombre}**")
                    with cols[1]: cant = st.number_input(f"Cant", 0, 50, cant_def, key=f"c_{cat}", label_visibility="collapsed")
                    with cols[2]: area = st.number_input(f"Área", 100, 50000, area_def, key=f"a_{cat}", label_visibility="collapsed")
                    campus_data.append({'nombre': nombre, 'cat': cat, 'count': cant, 'area': area})
            
            if st.button("Simular Campus", type="primary", use_container_width=True):
                total_consumption = 0
                breakdown = {}
                details = []
                
                for item in campus_data:
                    if item['count'] > 0:
                        input_df = pd.DataFrame({'air_temperature': [temp_c], 'relative_humidity': [50], 'wind_speed': [wind_c],
                            'gross_floor_area': [item['area']], 'hour': [hora_c], 'day_of_week': [dia_c], 'month': [6],
                            'is_holiday': [1 if feriado_c else 0], 'category': [item['cat']]})
                        pred_unit = max(0, final_pipeline.predict(input_df)[0])
                        total_cat = pred_unit * item['count']
                        total_consumption += total_cat
                        breakdown[item['nombre']] = total_cat
                        details.append({'Tipo': item['nombre'], 'Cantidad': item['count'], 'Área Unit.': f"{item['area']:,} m²",
                            'Consumo Unit.': f"{pred_unit:.2f} kWh", 'Consumo Total': f"{total_cat:.2f} kWh"})
                
                st.markdown("---")
                col_r1, col_r2, col_r3, col_r4 = st.columns(4)
                with col_r1: st.metric("Consumo Total", f"{total_consumption:,.2f} kWh")
                with col_r2: st.metric("Diario Est.", f"{total_consumption * 24 / 1000:,.2f} MWh")
                with col_r3: st.metric("Edificios", sum([d['count'] for d in campus_data]))
                with col_r4: st.metric("Costo Hora", f"${total_consumption * 0.15:,.2f}")
                
                col_pie, col_bar = st.columns(2)
                with col_pie:
                    st.subheader("Distribución por Tipo")
                    if breakdown:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(breakdown.values(), labels=breakdown.keys(), autopct='%1.1f%%', explode=[0.02]*len(breakdown), shadow=True)
                        st.pyplot(fig)
                with col_bar:
                    st.subheader("Comparativa")
                    if breakdown:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.barh(list(breakdown.keys()), list(breakdown.values()))
                        ax.set_xlabel("Consumo (kWh)")
                        st.pyplot(fig)
                st.subheader("Detalle por Tipo de Edificio")
                st.dataframe(pd.DataFrame(details), use_container_width=True, hide_index=True)   

        # Modo temporal
        else:
            st.subheader("Simulación en Rango de Fechas")
            c1, c2 = st.columns(2)
            with c1: start_date = st.date_input("Fecha inicio", pd.to_datetime("2025-06-01"))
            with c2: end_date = st.date_input("Fecha fin", pd.to_datetime("2025-06-07"))
            
            st.markdown("**Configuración del Campus**")
            campus_temp = {
                'office': {'count': st.number_input("Oficinas", 0, 50, 5, key="to"), 'area': 1500},
                'teaching': {'count': st.number_input("Aulas", 0, 50, 8, key="tt2"), 'area': 2500},
                'library': {'count': st.number_input("Bibliotecas", 0, 10, 1, key="tl"), 'area': 4000},
                'mixed use': {'count': st.number_input("Mixto", 0, 20, 3, key="tm"), 'area': 3000},
                'other': {'count': st.number_input("Otros", 0, 20, 2, key="toth"), 'area': 1000}
            }
            
            if st.button("Ejecutar Simulación", type="primary", use_container_width=True):
                if start_date <= end_date:
                    with st.spinner("Simulando..."):
                        full_range = pd.date_range(start=start_date, end=end_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
                        sim_df = pd.DataFrame({'timestamp': full_range})
                        sim_df['hour'] = sim_df['timestamp'].dt.hour
                        sim_df['day_of_week'] = sim_df['timestamp'].dt.dayofweek
                        sim_df['month'] = sim_df['timestamp'].dt.month
                        sim_df['is_holiday'] = (sim_df['day_of_week'] >= 5).astype(int)
                        
                        campus_load = np.zeros(len(sim_df))
                        for cat, params in campus_temp.items():
                            if params['count'] > 0:
                                features_df = pd.DataFrame({
                                    'air_temperature': [22] * len(sim_df),
                                    'relative_humidity': [50] * len(sim_df),
                                    'wind_speed': [10] * len(sim_df),
                                    'gross_floor_area': [params['area']] * len(sim_df),
                                    'hour': sim_df['hour'], 'day_of_week': sim_df['day_of_week'],
                                    'month': sim_df['month'], 'is_holiday': sim_df['is_holiday'],
                                    'category': [cat] * len(sim_df)
                                })
                                pred_vector = np.maximum(final_pipeline.predict(features_df), 0)
                                campus_load += pred_vector * params['count']
                        sim_df['consumption'] = campus_load
                        
                        total = sim_df['consumption'].sum()
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: st.metric("Total", f"{total/1000:,.2f} MWh")
                        with c2: st.metric("Promedio Hora", f"{sim_df['consumption'].mean():,.2f} kWh")
                        with c3: st.metric("Máx", f"{sim_df['consumption'].max():,.2f} kWh")
                        with c4: st.metric("Mín", f"{sim_df['consumption'].min():,.2f} kWh")
                        
                        st.subheader("Evolución del Consumo")
                        fig, ax = plt.subplots(figsize=(14, 5))
                        ax.fill_between(sim_df['timestamp'], sim_df['consumption'], alpha=0.3, color='#3498db')
                        ax.plot(sim_df['timestamp'], sim_df['consumption'], color='#2980b9')
                        ax.set_title("Perfil de Demanda Energética")
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                else:
                    st.error("Fecha fin debe ser posterior a inicio.")
    else:
        st.error("Modelo no cargado. Ejecuta 'entrenar_modelo.py'.")

# Anailisis Predictivo
with tab_analysis:
    st.header("Análisis Predictivo del Modelo")
    
    if final_pipeline:
        col_params, col_viz = st.columns([1, 3])
        
        with col_params:
            st.subheader("Parámetros")
            area_an = st.number_input("Área (m²)", 500, 10000, 2000, key="area_an")
            cat_an = st.selectbox("Tipo", ['office', 'teaching', 'library', 'mixed use', 'other'], key="cat_an")
            hora_an = st.slider("Hora Ref.", 0, 23, 14, key="hora_an")
            temp_an = st.slider("Temp Ref. (°C)", -5.0, 45.0, 22.0, key="temp_an")
        
        with col_viz:
            t1, t2, t3, t4 = st.tabs(["Sensibilidad Térmica", "Comparativa Edificios", "Patrones", "Entrenamiento Real"])
            
            with t1:
                st.subheader("Sensibilidad a la Temperatura")
                temps, preds = generate_temperature_sensitivity(final_pipeline, area_an, cat_an, hora_an)
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.fill_between(temps, preds, alpha=0.3, color='#e74c3c')
                ax.plot(temps, preds, color='#c0392b', linewidth=2)
                ax.axvline(x=temp_an, color='blue', linestyle='--', label=f'Ref: {temp_an}°C')
                ax.set_ylabel("Consumo (kWh)")
                ax.set_xlabel("Temp (°C)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with t2:
                st.subheader("Comparativa por Categoría")
                labels, vals = compare_building_types(final_pipeline, temp_an, hora_an)
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.bar(labels, vals, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
                ax.set_ylabel("Consumo (kWh)")
                ax.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.1f}', ha='center', va='bottom')
                st.pyplot(fig)
            
            with t3:
                st.subheader("Patrones Temporales")
                hours, hourly = generate_hourly_profile(final_pipeline, temp_an, 10, area_an, cat_an, 2, 0)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                ax1.plot(hours, hourly, 'o-', color='#2980b9')
                ax1.set_title("Patrón Diario")
                ax1.grid(True, alpha=0.3)
                
                days, weekly = generate_weekly_profile(final_pipeline, temp_an, 10, area_an, cat_an)
                ax2.bar(days, weekly, color='#3498db')
                ax2.set_title("Patrón Semanal")
                st.pyplot(fig)

            with t4:
                st.subheader("Convergencia del Entrenamiento (Modelo Real)")
                if loss_history_offline:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(loss_history_offline, label='Loss', color='#8e44ad', linewidth=2)
                    ax.set_xlabel("Iteraciones")
                    ax.set_ylabel("Loss Function")
                    ax.set_title("Evolución del Error (Datos Reales)")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    st.metric("Loss Final", f"{loss_history_offline[-1]:.4f}")
                    st.pyplot(fig)
                    st.info("Gráfica generada a partir del historial guardado en 'loss_history.pkl' durante el entrenamiento offline.")
                else:
                    st.warning("No se encontró historial de entrenamiento.")
    else:
        st.error("Modelo no cargado.")

# Explorador de Datos
with tab_data:
    st.header("Explorador de Dataset")
    
    if df_consumption is not None:
        tab_overview, tab_weather, tab_buildings = st.tabs(["Consumo", "Clima", "Edificios"])
        
        with tab_overview:
            st.subheader("Datos de Consumo Energético")
            st.dataframe(df_consumption.head(100), use_container_width=True)
            
            col_s1, col_s2 = st.columns(2)
            with col_s1: st.metric("Total Registros", f"{len(df_consumption):,}")
            with col_s2: st.metric("Edificios Únicos", df_consumption['meter_id'].nunique())
            
            st.subheader("Estadísticas de Consumo")
            stats = df_consumption['consumption'].describe()
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Promedio", f"{stats['mean']:.2f}")
            with c2: st.metric("Mediana", f"{stats['50%']:.2f}")
            with c3: st.metric("Mín", f"{stats['min']:.2f}")
            with c4: st.metric("Máx", f"{stats['max']:.2f}")
        
        with tab_weather:
            st.subheader("Datos Meteorológicos")
            st.dataframe(df_weather.head(100), use_container_width=True)
            if 'air_temperature' in df_weather.columns:
                st.subheader("Distribución de Temperatura")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(df_weather['air_temperature'].dropna(), bins=50, color='#3498db', edgecolor='white')
                ax.set_title("Histograma de Temperaturas")
                st.pyplot(fig)
        
        with tab_buildings:
            st.subheader("Metadatos de Edificios")
            st.dataframe(df_meta, use_container_width=True)
            if 'category' in df_meta.columns:
                st.subheader("Distribución por Categoría")
                counts = df_meta['category'].value_counts()
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(counts.index, counts.values, color='#3498db')
                st.pyplot(fig)
    else:
        st.warning("No se pudieron cargar los datos del directorio 'dataset'.")