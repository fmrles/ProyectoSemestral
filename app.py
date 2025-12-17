import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

# corresponde a la configuración de la página
st.set_page_config(
    page_title="Predicción Energética: Campus UBB",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------ Recarga de Recursos ---------------------
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

resources = load_resources()
final_pipeline = resources['model']
loss_history_offline = resources['loss_history']

@st.cache_data
def load_dataset():
    """Carga el dataset original para análisis"""
    try:
        df_consumption = pd.read_csv('dataset/building_consumption.csv')
        df_weather = pd.read_csv('dataset/weather_data.csv')
        df_meta = pd.read_csv('dataset/building_meta.csv')
        return df_consumption, df_weather, df_meta
    except:
        return None, None, None

resources = load_resources()
final_pipeline = resources['model']
loss_history_offline = resources['loss_history']
df_consumption, df_weather, df_meta = load_dataset() 

# ---------------------------------------------------------------------------------


# --------------------- Nuevos estilos CSS para páginas ---------------------------
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
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------------



# ----------------------------- Llamadas a Métodos ----------------------------------- 
st.markdown('<p class="main-header"> Predicción Energética - Campus Universitario</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">Sistema de predicción de consumo energético basado en Redes Neuronales </p>', unsafe_allow_html=True)

# Tabs principales
tab_sim, tab_analysis, tab_data = st.tabs([
    "Simulador", 
    "Análisis Predictivo",
    "Explorador de Datos"
]) 

# -------------------------------------------------------------------------------------------

# ------------------ Nuevas funciones base que ayudan en la generación de los gráficos ------------------

def generate_hourly_profile(model, temp, wind, area, category, day_of_week, is_holiday):
    """Genera perfil de consumo para las 24 horas"""
    hours = list(range(24))
    predictions = []
    for hour in hours:
        input_df = pd.DataFrame({
            'air_temperature': [temp],
            'relative_humidity': [50],
            'wind_speed': [wind],
            'gross_floor_area': [area],
            'hour': [hour],
            'day_of_week': [day_of_week],
            'month': [6],
            'is_holiday': [is_holiday],
            'category': [category]
        })
        pred = model.predict(input_df)[0]
        predictions.append(max(0, pred))
    return hours, predictions

def generate_weekly_profile(model, temp, wind, area, category):
    """Genera perfil de consumo semanal"""
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    daily_consumption = []
    for day in range(7):
        day_total = 0
        for hour in range(24):
            input_df = pd.DataFrame({
                'air_temperature': [temp],
                'relative_humidity': [50],
                'wind_speed': [wind],
                'gross_floor_area': [area],
                'hour': [hour],
                'day_of_week': [day],
                'month': [6],
                'is_holiday': [1 if day >= 5 else 0],
                'category': [category]
            })
            pred = model.predict(input_df)[0]
            day_total += max(0, pred)
        daily_consumption.append(day_total)
    return days, daily_consumption

def generate_temperature_sensitivity(model, area, category, hour=14):
    """Genera curva de sensibilidad a temperatura"""
    temps = np.linspace(-5, 45, 50)
    predictions = []
    for temp in temps:
        input_df = pd.DataFrame({
            'air_temperature': [temp],
            'relative_humidity': [50],
            'wind_speed': [10],
            'gross_floor_area': [area],
            'hour': [hour],
            'day_of_week': [2],
            'month': [6],
            'is_holiday': [0],
            'category': [category]
        })
        pred = model.predict(input_df)[0]
        predictions.append(max(0, pred))
    return temps, predictions

def compare_building_types(model, temp, hour):
    """Compara consumo entre tipos de edificios"""
    categories = ['office', 'teaching', 'library', 'mixed use', 'other']
    labels = ['Oficinas', 'Aulas', 'Biblioteca', 'Uso Mixto', 'Otros']
    predictions = []
    for cat in categories:
        input_df = pd.DataFrame({
            'air_temperature': [temp],
            'relative_humidity': [50],
            'wind_speed': [10],
            'gross_floor_area': [2000],
            'hour': [hour],
            'day_of_week': [2],
            'month': [6],
            'is_holiday': [0],
            'category': [cat]
        })
        pred = model.predict(input_df)[0]
        predictions.append(max(0, pred))
    return labels, predictions
# ---------------------------------------------------------------------------------------------------


st.title("Predicción Energética - Campus Universitario")



with tab_exp:
    st.header("Laboratorio de Entrenamiento en Vivo")
    st.info("Entrena una versión simplificada del modelo para probar hiperparámetros.")
    
    col1, col2 = st.columns(2)
    with col1:
        h_layers = eval(st.selectbox("Capas Ocultas", ['(50,)', '(100, 50)', '(64, 32)'], index=0))
    with col2:
        iters = st.slider("Iteraciones", 50, 500, 200)

    if st.button("Entrenar Demo"):
        df_synth = get_synthetic_data()
        X = df_synth[['air_temperature', 'gross_floor_area', 'hour']]
        y = df_synth['consumption']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = MLPRegressor(hidden_layer_sizes=h_layers, max_iter=iters, random_state=42)
        model.fit(X_scaled, y)
        
        st.subheader("Curva de Aprendizaje")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(model.loss_curve_)
        ax.set_xlabel("Iteraciones")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.success(f"Entrenamiento finalizado. Loss: {model.loss_:.4f}")

# ----------------------------- Tab Simulador de consumo energétito --------------------------------------------
with tab_sim:
    st.header("Simulador de Consumo: Escenario Campus")
    
    if final_pipeline:
        mode = st.radio(
            "Seleccione modo de simulación:",
            ["Edificio Individual", "Campus Completo", "Simulación Temporal"],
            horizontal=True
        )
        
        st.markdown("---")

        # MODO DE EDIFICIO INVIDIDUAL
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
                                      format_func=lambda x: ['Enero','Febrero','Marzo','Abril','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'][x-1],
                                      index=5)
                    feriado = st.checkbox("¿día feriado?")
                
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
                    # Realizar predicción
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
                    
                    # Mostrar resultado principal en pantalla
                    st.subheader("Resultado de la Predicción")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Consumo Estimado", f"{prediction:.2f} kWh")
                    with col_m2:
                        consumo_diario_est = prediction * 24
                        st.metric("Estimado Diario", f"{consumo_diario_est:.0f} kWh")
                    with col_m3:
                        costo_est = prediction * 0.15  # Precio estimado kWh
                        st.metric("Costo Hora (est.)", f"${costo_est:.2f}")
                    
                    st.markdown("---")
                    
                    # Gráficos
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        st.subheader("Perfil Horario (24h)")
                        hours, hourly_pred = generate_hourly_profile(
                            final_pipeline, temp, wind, area, categoria, dia, 1 if feriado else 0
                        )
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        ax1.fill_between(hours, hourly_pred, alpha=0.3, color='#3498db')
                        ax1.plot(hours, hourly_pred, 'o-', color='#2980b9', linewidth=2, markersize=4)
                        ax1.axvline(x=hora, color='red', linestyle='--', label=f'Hora actual ({hora}:00)')
                        ax1.scatter([hora], [prediction], color='red', s=100, zorder=5)
                        ax1.set_xlabel("Hora del día")
                        ax1.set_ylabel("Consumo (kWh)")
                        ax1.set_title("Evolución del Consumo Durante el Día")
                        ax1.set_xticks(range(0, 24, 2))
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        st.pyplot(fig1)
                    
                    with col_g2:
                        st.subheader("Perfil Semanal")
                        days, weekly_pred = generate_weekly_profile(
                            final_pipeline, temp, wind, area, categoria
                        )
                        fig2, ax2 = plt.subplots(figsize=(8, 4))
                        colors = ['#3498db' if i < 5 else '#e74c3c' for i in range(7)]
                        bars = ax2.bar(days, weekly_pred, color=colors, edgecolor='white', linewidth=1.5)
                        ax2.axhline(y=np.mean(weekly_pred), color='green', linestyle='--', label=f'Promedio: {np.mean(weekly_pred):.0f} kWh')
                        ax2.set_xlabel("Día de la semana")
                        ax2.set_ylabel("Consumo diario (kWh)")
                        ax2.set_title("Distribución Semanal del Consumo")
                        ax2.legend()
                        ax2.grid(True, alpha=0.3, axis='y')
                        st.pyplot(fig2)
                else:
                    st.info("Debe configurar los parámetros y presione 'Calcular Predicción' para ver los resultados.")
