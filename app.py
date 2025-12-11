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

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Predicción Energética: Campus UBB",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 2. FUNCIONES DE CARGA ---
@st.cache_resource
def load_resources():
    """Carga el modelo y el historial de entrenamiento."""
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

# --- DATOS SINTÉTICOS PARA DEMO VIVA ---
@st.cache_data
def get_synthetic_data():
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

st.title(" Predicción Energética - Campus Universitario")

# --- TABS ---
tab_exp, tab_sim, tab_context = st.tabs(["1. Lab de Entrenamiento", "2. Simulador de Campus", "3. Contexto Analítico"])

# ====================================================================
# TAB 1: LABORATORIO
# ====================================================================
with tab_exp:
    st.header(" Laboratorio de Entrenamiento en Vivo")
    st.info("Entrena una versión simplificada del modelo para probar hiperparámetros.")
    
    col1, col2 = st.columns(2)
    with col1:
        h_layers = eval(st.selectbox("Capas Ocultas", ['(50,)', '(100, 50)', '(64, 32)'], index=0))
    with col2:
        iters = st.slider("Iteraciones", 50, 500, 200)

    if st.button(" Entrenar Demo"):
        df_synth = get_synthetic_data()
        X = df_synth[['air_temperature', 'gross_floor_area', 'hour']]
        y = df_synth['consumption']
        
        # Scaling simple manual para la demo
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


# ====================================================================
# TAB 2: SIMULADOR DE CAMPUS (LÓGICA ACTUALIZADA)
# ====================================================================
with tab_sim:
    st.header(" Simulador de Consumo: Escenario Campus")
    
    if final_pipeline:
        # --- SELECCIÓN DE MODO ---
        mode = st.radio("Modo de Simulación:", 
                        [" Edificio Individual (Instantáneo)", 
                         " Campus Completo (Instantáneo)",
                         " Simulación Temporal (Rango de Fechas)"], # <--- NUEVO MODO
                        horizontal=True)
        st.markdown("---")

        # -----------------------------------------------------------
        # MODOS INSTANTÁNEOS (Lógica anterior reutilizada)
        # -----------------------------------------------------------
        if mode in [" Edificio Individual (Instantáneo)", " Campus Completo (Instantáneo)"]:
            
            # PARÁMETROS AMBIENTALES
            st.subheader("1. Condiciones Ambientales y Temporales")
            c1, c2, c3, c4 = st.columns(4)
            with c1: temp = st.slider("Temperatura (°C)", -5.0, 45.0, 22.0)
            with c2: wind = st.number_input("Viento (km/h)", 0.0, 100.0, 12.0)
            with c3: hora = st.slider("Hora del Día", 0, 23, 14) 
            with c4: dia = st.selectbox("Día Semana", range(7), format_func=lambda x: ['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][x])
            feriado = st.checkbox("¿Es Feriado?")

            # MODO A: EDIFICIO INDIVIDUAL
            if "Edificio Individual" in mode:
                st.subheader("2. Detalles del Edificio")
                cc1, cc2 = st.columns(2)
                with cc1: area = st.number_input("Área (m²)", 100.0, 10000.0, 1500.0)
                with cc2: cat = st.selectbox("Categoría", ['office', 'teaching', 'library', 'mixed use', 'other'])
                
                if st.button("Calcular Consumo Individual", type="primary"):
                    input_df = pd.DataFrame({
                        'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [wind],
                        'gross_floor_area': [area], 'hour': [hora], 'day_of_week': [dia],
                        'month': [6], 'is_holiday': [1 if feriado else 0], 'category': [cat]
                    })
                    pred = final_pipeline.predict(input_df)[0]
                    st.metric("Consumo Estimado", f"{pred:.2f} kWh")

            # MODO B: CAMPUS COMPLETO
            else:
                st.subheader("2. Composición del Campus")
                st.info("Define la cantidad y área específica para cada tipo de edificio.")
                
                col_a, col_b = st.columns(2)
                campus_config = {}
                with col_a:
                    campus_config['office'] = {'count': st.number_input("Cant. Oficinas", 0, 20, 5), 'area': st.number_input("Área Oficina", 100, 10000, 1500)}
                    campus_config['teaching'] = {'count': st.number_input("Cant. Aulas", 0, 20, 8), 'area': st.number_input("Área Aulas", 100, 10000, 2500)}
                with col_b:
                    campus_config['library'] = {'count': st.number_input("Cant. Bibliotecas", 0, 5, 1), 'area': st.number_input("Área Biblioteca", 100, 20000, 4000)}
                    campus_config['mixed use'] = {'count': st.number_input("Cant. Uso Mixto", 0, 10, 2), 'area': st.number_input("Área Mixto", 100, 15000, 3000)}
                    campus_config['other'] = {'count': st.number_input("Cant. Otros", 0, 10, 2), 'area': st.number_input("Área Otros", 100, 5000, 1000)}

                if st.button("🏗️ Simular Campus Completo", type="primary"):
                    total_consumption = 0
                    breakdown = {}
                    
                    for cat, params in campus_config.items():
                        count = params['count']
                        area_user = params['area']
                        
                        if count > 0:
                            input_df = pd.DataFrame({
                                'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [wind],
                                'gross_floor_area': [area_user], 'hour': [hora], 'day_of_week': [dia],
                                'month': [6], 'is_holiday': [1 if feriado else 0], 'category': [cat]
                            })
                            pred_unit = final_pipeline.predict(input_df)[0]
                            total_cat = pred_unit * count
                            total_consumption += total_cat
                            breakdown[cat] = total_cat

                    st.markdown("---")
                    c_res1, c_res2 = st.columns([1, 2])
                    with c_res1:
                        st.metric("Consumo TOTAL", f"{total_consumption:,.2f} kWh", delta="Instantáneo")
                    with c_res2:
                        if total_consumption > 0:
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.pie(list(breakdown.values()), labels=list(breakdown.keys()), autopct='%1.1f%%', startangle=90)
                            ax.axis('equal') 
                            st.pyplot(fig)

       # -----------------------------------------------------------
        # MODO C: SIMULACIÓN TEMPORAL (RANGO DE FECHAS) - CORREGIDO
        # -----------------------------------------------------------
        else:
            st.subheader(" Planificación Energética Temporal")
            st.info("Simula el comportamiento del campus a lo largo de días o semanas.")

            # 1. Configuración del Campus
            with st.expander(" Configuración del Campus (Click para editar)", expanded=False):
                col_a, col_b = st.columns(2)
                campus_config = {}
                with col_a:
                    campus_config['office'] = {'count': st.number_input("Cant. Oficinas", 0, 20, 5, key="t_off"), 'area': st.number_input("Área Oficina", 100, 10000, 1500, key="ta_off")}
                    campus_config['teaching'] = {'count': st.number_input("Cant. Aulas", 0, 20, 8, key="t_tea"), 'area': st.number_input("Área Aulas", 100, 10000, 2500, key="ta_tea")}
                with col_b:
                    campus_config['library'] = {'count': st.number_input("Cant. Bibliotecas", 0, 5, 1, key="t_lib"), 'area': st.number_input("Área Biblioteca", 100, 20000, 4000, key="ta_lib")}
                    campus_config['mixed use'] = {'count': st.number_input("Cant. Mixto", 0, 10, 2, key="t_mix"), 'area': st.number_input("Área Mixto", 100, 15000, 3000, key="ta_mix")}
                    campus_config['other'] = {'count': st.number_input("Cant. Otros", 0, 10, 2, key="t_oth"), 'area': st.number_input("Área Otros", 100, 5000, 1000, key="ta_oth")}

            # 2. Selección de Fechas
            c_date1, c_date2 = st.columns(2)
            with c_date1: start_date = st.date_input("Fecha Inicio", pd.to_datetime("2025-06-01"))
            with c_date2: end_date = st.date_input("Fecha Fin", pd.to_datetime("2025-06-07"))

            if start_date <= end_date:
                days_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # 3. Editor de Calendario
                st.subheader(" Editor de Eventos Diarios")
                st.write("Marca los días que son Feriados o si habrá un Corte de Luz.")
                
                default_data = {
                    "Fecha": days_range.date,
                    "Es Feriado": [d.dayofweek >= 5 for d in days_range],
                    "Corte de Luz": [False] * len(days_range)
                }
                schedule_df = pd.DataFrame(default_data)
                
                edited_schedule = st.data_editor(
                    schedule_df,
                    column_config={
                        "Fecha": st.column_config.DateColumn("Fecha", disabled=True, format="YYYY-MM-DD"),
                        "Es Feriado": st.column_config.CheckboxColumn("¿Es Feriado?"),
                        "Corte de Luz": st.column_config.CheckboxColumn(" Corte Programado")
                    },
                    hide_index=True,
                    num_rows="fixed"
                )

                if st.button(" Simular Periodo", type="primary"):
                    
                    # --- CORRECCIÓN 1: Generación de Rango Horario (Usa 'h' minúscula) ---
                    full_range = pd.date_range(start=start_date, end=end_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
                    
                    sim_df = pd.DataFrame({'timestamp': full_range})
                    # Aseguramos que la columna 'date' sea del mismo tipo que las claves del mapa (datetime.date)
                    sim_df['date'] = sim_df['timestamp'].dt.date
                    sim_df['hour'] = sim_df['timestamp'].dt.hour
                    sim_df['day_of_week'] = sim_df['timestamp'].dt.dayofweek
                    sim_df['month'] = sim_df['timestamp'].dt.month
                    
                    # Generar Clima Sintético
                    sim_df['air_temperature'] = 15 + 10 * np.sin(2 * np.pi * (sim_df['hour'] - 9) / 24) + np.random.normal(0, 1, len(sim_df))
                    sim_df['relative_humidity'] = 60 - 20 * np.sin(2 * np.pi * (sim_df['hour'] - 9) / 24)
                    sim_df['wind_speed'] = 12 + np.random.normal(0, 2, len(sim_df))
                    
                    # --- CORRECCIÓN 2: Lógica de Búsqueda Segura (Evita KeyError) ---
                    # Convertimos el calendario editado a un diccionario para búsqueda rápida
                    # Aseguramos que las claves sean objetos date
                    schedule_map = edited_schedule.set_index("Fecha").to_dict('index')

                    def get_safe_event(date_obj, column):
                        """Busca la fecha en el mapa. Si no existe, devuelve False (valor por defecto)."""
                        try:
                            return 1 if schedule_map[date_obj][column] else 0
                        except KeyError:
                            return 0 # Asume NO Feriado / NO Corte si la fecha no coincide

                    sim_df['is_holiday'] = sim_df['date'].apply(lambda x: get_safe_event(x, 'Es Feriado'))
                    sim_df['power_outage'] = sim_df['date'].apply(lambda x: get_safe_event(x, 'Corte de Luz'))

                    # Iterar por edificios y sumar consumos
                    campus_total_load = np.zeros(len(sim_df))
                    
                    for cat, params in campus_config.items():
                        count = params['count']
                        area_user = params['area']
                        
                        if count > 0:
                            features_df = sim_df[['air_temperature', 'relative_humidity', 'wind_speed', 'hour', 'day_of_week', 'month', 'is_holiday']].copy()
                            features_df['gross_floor_area'] = area_user
                            features_df['category'] = cat
                            
                            input_cols = ['air_temperature', 'relative_humidity', 'wind_speed', 'gross_floor_area', 'hour', 'day_of_week', 'month', 'is_holiday', 'category']
                            pred_vector = final_pipeline.predict(features_df[input_cols])
                            campus_total_load += (pred_vector * count)

                    # Aplicar Corte de Luz
                    final_load = np.where(sim_df['power_outage'] == 1, 0, campus_total_load)
                    
                    # Resultados
                    total_energy = np.sum(final_load)
                    
                    st.markdown("---")
                    col_res1, col_res2 = st.columns([1, 3])
                    with col_res1:
                        st.metric("Energía Total Periodo", f"{total_energy/1000:,.2f} MWh")
                        st.caption(f"Simulación de {len(full_range)} horas")

                    with col_res2:
                        st.subheader(" Evolución del Consumo")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(sim_df['timestamp'], final_load, label="Consumo (kWh)", color="#1f77b4")
                        
                        # Marcar cortes
                        outage_times = sim_df[sim_df['power_outage'] == 1]['timestamp']
                        if not outage_times.empty:
                            ax.scatter(outage_times, [0]*len(outage_times), color='red', label="Corte Programado", zorder=5, s=15)
                        
                        ax.set_ylabel("Potencia (kW)")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
            else:
                st.error("La fecha de fin debe ser posterior a la de inicio.")


# ====================================================================
# TAB 3: CONTEXTO (Mismo código anterior)
# ====================================================================
with tab_context:
    st.header(" Contexto del Modelo (Pre-Entrenamiento)")
    
    col_metrics, col_graph = st.columns([1, 2])
    with col_metrics:
        st.subheader("Arquitectura")
        if final_pipeline:
            mlp = final_pipeline.named_steps['regressor']
            st.code(f"Capas: {mlp.hidden_layer_sizes}\nActiv: {mlp.activation}\nIter: {mlp.n_iter_}")
    with col_graph:
        st.subheader("Convergencia Real")
        if loss_history_offline:
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            ax3.plot(loss_history_offline, color='green')
            ax3.set_ylabel("Loss")
            st.pyplot(fig3)