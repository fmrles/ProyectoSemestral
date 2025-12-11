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
# TAB 1: LABORATORIO (Mismo código anterior)
# ====================================================================
with tab_exp:
    st.header(" Laboratorio de Entrenamiento en Vivo")
    st.info("Entrena una versión simplificada del modelo para probar hiperparámetros.")
    
    col1, col2 = st.columns(2)
    with col1:
        h_layers = eval(st.selectbox("Capas Ocultas", ['(50,)', '(100, 50)', '(64, 32)'], index=0))
    with col2:
        iters = st.slider("Iteraciones", 50, 500, 200)

    if st.button("⚡ Entrenar Demo"):
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
        mode = st.radio("Modo de Simulación:", [" Edificio Individual", " Campus Completo (Personalizado)"], horizontal=True)
        st.markdown("---")

        # --- PARÁMETROS AMBIENTALES (Comunes) ---
        st.subheader("1. Condiciones Ambientales y Temporales")
        c1, c2, c3, c4 = st.columns(4)
        with c1: temp = st.slider("Temperatura (°C)", -5.0, 45.0, 22.0)
        with c2: wind = st.number_input("Viento (km/h)", 0.0, 100.0, 12.0)
        with c3: hora = st.slider("Hora del Día", 0, 23, 14) 
        with c4: dia = st.selectbox("Día Semana", range(7), format_func=lambda x: ['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][x])
        feriado = st.checkbox("¿Es Feriado?")

        # -----------------------------------------------------------
        # MODO A: EDIFICIO INDIVIDUAL
        # -----------------------------------------------------------
        if mode == " Edificio Individual":
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

        # -----------------------------------------------------------
        # MODO B: CAMPUS COMPLETO (PERSONALIZADO)
        # -----------------------------------------------------------
        else:
            st.subheader("2. Composición Personalizada del Campus")
            st.info("Define la cantidad y el área específica para cada tipo de edificio.")
            
            # Contenedor para la configuración
            campus_config = []
            
            # Definimos columnas para cabecera visual
            h1, h2, h3 = st.columns([1, 1, 2])
            with h1: st.markdown("**Tipo de Edificio**")
            with h2: st.markdown("**Cantidad**")
            with h3: st.markdown("**Área Unitaria (m²)**")

            # --- INPUTS PARA CADA TIPO ---
            
            # 1. OFICINAS
            c_off1, c_off2, c_off3 = st.columns([1, 1, 2])
            with c_off1: st.markdown(" Oficinas")
            with c_off2: q_off = st.number_input("Cant. Oficinas", 0, 50, 5, label_visibility="collapsed")
            with c_off3: a_off = st.number_input("Área Oficina", 100, 10000, 1500, label_visibility="collapsed")
            campus_config.append({'cat': 'office', 'count': q_off, 'area': a_off})

            # 2. AULAS (TEACHING)
            c_tea1, c_tea2, c_tea3 = st.columns([1, 1, 2])
            with c_tea1: st.markdown(" Aulas")
            with c_tea2: q_tea = st.number_input("Cant. Aulas", 0, 50, 8, label_visibility="collapsed")
            with c_tea3: a_tea = st.number_input("Área Aulas", 100, 10000, 2500, label_visibility="collapsed")
            campus_config.append({'cat': 'teaching', 'count': q_tea, 'area': a_tea})

            # 3. BIBLIOTECAS
            c_lib1, c_lib2, c_lib3 = st.columns([1, 1, 2])
            with c_lib1: st.markdown(" Bibliotecas")
            with c_lib2: q_lib = st.number_input("Cant. Bibliotecas", 0, 10, 1, label_visibility="collapsed")
            with c_lib3: a_lib = st.number_input("Área Biblioteca", 100, 20000, 4000, label_visibility="collapsed")
            campus_config.append({'cat': 'library', 'count': q_lib, 'area': a_lib})

            # 4. USO MIXTO
            c_mix1, c_mix2, c_mix3 = st.columns([1, 1, 2])
            with c_mix1: st.markdown(" Uso Mixto")
            with c_mix2: q_mix = st.number_input("Cant. Mixto", 0, 20, 2, label_visibility="collapsed")
            with c_mix3: a_mix = st.number_input("Área Mixto", 100, 15000, 3000, label_visibility="collapsed")
            campus_config.append({'cat': 'mixed use', 'count': q_mix, 'area': a_mix})

            # 5. OTROS
            c_oth1, c_oth2, c_oth3 = st.columns([1, 1, 2])
            with c_oth1: st.markdown(" Otros")
            with c_oth2: q_oth = st.number_input("Cant. Otros", 0, 20, 2, label_visibility="collapsed")
            with c_oth3: a_oth = st.number_input("Área Otros", 100, 5000, 1000, label_visibility="collapsed")
            campus_config.append({'cat': 'other', 'count': q_oth, 'area': a_oth})


            st.markdown("---")
            if st.button(" Simular Campus Completo", type="primary"):
                total_consumption = 0
                breakdown = {}
                
                # Iteramos sobre la configuración ingresada por el usuario
                for item in campus_config:
                    count = item['count']
                    area_user = item['area']
                    cat = item['cat']
                    
                    if count > 0:
                        # Preparamos el input para 1 edificio de este tipo y tamaño específico
                        input_df = pd.DataFrame({
                            'air_temperature': [temp], 'relative_humidity': [50], 'wind_speed': [wind],
                            'gross_floor_area': [area_user], 'hour': [hora], 'day_of_week': [dia],
                            'month': [6], 'is_holiday': [1 if feriado else 0], 'category': [cat]
                        })
                        
                        # Predicción unitaria * cantidad
                        pred_unit = final_pipeline.predict(input_df)[0]
                        total_cat = pred_unit * count
                        
                        total_consumption += total_cat
                        breakdown[cat] = total_cat

                # --- VISUALIZACIÓN DE RESULTADOS ---
                c_res1, c_res2 = st.columns([1, 2])
                
                with c_res1:
                    st.metric("Consumo TOTAL Campus", f"{total_consumption:,.2f} kWh", delta="Simulación instantánea")
                    st.caption(f"Calculado para {hora}:00 hrs con condiciones actuales.")
                
                with c_res2:
                    st.write("#### Distribución del Consumo por Tipo")
                    if total_consumption > 0:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        labels = list(breakdown.keys())
                        values = list(breakdown.values())
                        # Colores personalizados para cada categoría
                        colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
                        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(labels)])
                        ax.axis('equal') 
                        st.pyplot(fig)
                    else:
                        st.warning("Configura al menos un edificio para ver resultados.")

    else:
        st.error(" Modelo offline no encontrado. Ejecuta 'entrenar_modelo.py' primero.")


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