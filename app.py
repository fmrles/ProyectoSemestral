import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib # Para cargar el modelo OFFLINE
import os
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="MLP: Experimentación y Predicción",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. CARGA DE DATOS (Necesario para el entrenamiento en vivo) ---
@st.cache_data
def get_synthetic_data():
    """Genera un dataset sintético ligero para la experimentación en vivo."""
    N = 2000
    dates = pd.date_range(start="2025-01-01", periods=N, freq="h")
    data = {
        'timestamp': dates,
        'air_temperature': np.random.uniform(5, 35, N), 
        'gross_floor_area': np.random.uniform(500, 5000, N), 
        'category': np.random.choice(['office', 'teaching', 'library'], N),
        'hour': dates.hour
    }
    df = pd.DataFrame(data)
    # Target: Consumo (con efecto U de la temperatura + factor de área)
    df['consumption'] = (
        (df['air_temperature'] - 20)**2 * 1.5 + 
        df['gross_floor_area'] * 0.05 + 
        np.random.normal(0, 5, N) + 500
    )
    return df

# --- 2. CARGA DEL MODELO FINAL (Modelo OFFLINE) ---
@st.cache_resource
def load_final_model():
    """Carga el modelo final entrenado con el script externo (entrenar_modelo.py)."""
    model_path = 'mlp_model_entrenado.pkl'
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        return None

# --- 3. INICIALIZACIÓN DE VARIABLES GLOBALES ---
df_synth = get_synthetic_data()
final_pipeline = load_final_model()
target_col = 'consumption'
feature_cols = ['air_temperature', 'gross_floor_area', 'hour'] # Subconjunto para el simulador

# Si el modelo offline está cargado, usamos sus features para el simulador
if final_pipeline:
    # Intenta obtener los nombres de features del pipeline para el simulador
    try:
        feature_names = final_pipeline.named_steps['preprocessor'].transformers_[0][2]
    except:
        feature_names = feature_cols
    
# TABS PRINCIPALES
tab_exp, tab_sim = st.tabs(["🧪 1. Experimento de Entrenamiento", "🔮 2. Simulador y Predicción"])


# ====================================================================
# PESTAÑA 1: EXPERIMENTACIÓN INTERACTIVA (TRAINING EN VIVO)
# ====================================================================
with tab_exp:
    st.header("1. Experimentación Interactiva (MLP)")
    st.info("Utiliza este panel para entrenar el modelo en vivo, ajustar hiperparámetros y observar la convergencia (Curva de Error).")

    # A. CONTROLES DE HIPERPARÁMETROS
    st.subheader("Configuración del Perceptrón Multicapa")
    col1, col2, col3 = st.columns(3)
    
    # Control para Capas Ocultas
    with col1:
        h_layers_choice = st.selectbox(
            "Capas Ocultas (Tamaño)",
            options=['(100, 50)', '(64, 32)', '(50,)', '(20, 20, 20)'],
            index=1,
            help="Define el número de neuronas por capa. Ej: (100, 50) son 2 capas."
        )
        # Convertir el string de capas a una tupla
        h_layers = eval(h_layers_choice)
        
    # Control para Activación
    with col2:
        activ = st.selectbox("Función de Activación", ["relu", "tanh", "logistic"], index=0)
        
    # Control para Iteraciones
    with col3:
        iters = st.slider("Máx. Iteraciones (Épocas)", 50, 1000, 300)
    
    
    if st.button("🚀 Entrenar y Graficar Curva de Error", type="primary"):
        # Preparación de datos (Feature Engineering simple para la demo)
        X_synth = df_synth[['air_temperature', 'gross_floor_area', 'hour']]
        y_synth = df_synth[target_col]
        
        # División y Escalado (Necesario para entrenamiento)
        X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, test_size=0.2, random_state=42)
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.values.reshape(-1, 1))
        
        X_train_s = scaler_X.transform(X_train)
        y_train_s = scaler_y.transform(y_train.values.reshape(-1, 1)).ravel() # .ravel() para MLPRegressor
        
        # --- MODELO Y ENTRENAMIENTO ---
        model_exp = MLPRegressor(
            hidden_layer_sizes=h_layers,
            activation=activ,
            max_iter=iters,
            random_state=42,
            solver='sgd', # Usar SGD para ver mejor la curva de error por epoch
            learning_rate_init=0.01,
            warm_start=True # Permitir entrenar y guardar la curva de error
        )
        
        # Entrenar en ciclos para capturar la CURVA DE ERROR (MSE/Loss)
        loss_history = []
        status_text = st.empty()
        
        for i in range(iters):
            model_exp.partial_fit(X_train_s, y_train_s)
            current_loss = model_exp.loss_
            loss_history.append(current_loss)
            
            # Actualizar barra de progreso en vivo
            status_text.progress((i + 1) / iters)
        
        # --- EVALUACIÓN Y VISUALIZACIÓN ---
        st.subheader("📊 Resultados de la Experimentación")
        
        # 1. Gráfico de la Curva de Error (Requisito)
        fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
        ax_loss.plot(loss_history, label='Curva de Error (Loss)')
        ax_loss.set_title(f"Convergencia del Modelo (Activación: {activ})")
        ax_loss.set_xlabel("Épocas")
        ax_loss.set_ylabel("Error Cuadrático (Loss)")
        ax_loss.grid(True, alpha=0.3)
        st.pyplot(fig_loss)
        
        # 2. Métricas Finales
        y_pred_test_s = model_exp.predict(scaler_X.transform(X_test))
        y_pred_test = scaler_y.inverse_transform(y_pred_test_s.reshape(-1, 1))
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)
        
        st.metric("RMSE Final (Error Cuadrático)", f"{rmse:.2f} kWh", delta_color="off")
        st.metric("R2 Score", f"{r2:.3f}", delta_color="off")
        
        st.success("✅ Experimento completado y métricas calculadas.")


# ====================================================================
# PESTAÑA 2: SIMULADOR Y PREDICCIÓN (USANDO EL MODELO FINAL OFFLINE)
# ====================================================================
with tab_sim:
    st.header("2. Simulador de Predicción (Modelo Final)")
    st.markdown("Utiliza el modelo final pre-entrenado para estimar el consumo en tiempo real.")

    if final_pipeline is not None:
        
        # --- PARÁMETROS DEL SIMULADOR ---
        col_c, col_t = st.columns(2)
        
        with col_c:
            st.subheader("Condiciones de la Predicción")
            # Los nombres de las variables deben coincidir con el entrenamiento offline
            temp = st.slider("Temperatura del Aire (°C)", -5.0, 45.0, 20.0, key="sim_temp")
            area = st.number_input("Área Bruta (m²)", 100.0, 10000.0, 1000.0, key="sim_area")
            wind = st.number_input("Velocidad del Viento (km/h)", 0.0, 100.0, 10.0, key="sim_wind")
            
        with col_t:
            st.subheader("Contexto Temporal y Edificio")
            hora = st.slider("Hora del día", 0, 23, 12, key="sim_hour")
            dia_sem = st.selectbox("Día de la Semana", range(7), key="sim_day", format_func=lambda x: ['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][x])
            es_feriado = st.checkbox("¿Es Feriado?", value=False, key="sim_holiday")
            
            # Categorías deben coincidir con las usadas en el entrenamiento
            categorias = ['office', 'teaching', 'library', 'mixed use', 'other'] 
            cat_edificio = st.selectbox("Categoría del Edificio", categorias, key="sim_cat")

        if st.button("🔮 Generar Predicción", use_container_width=True, type="primary"):
            
            # Crear DataFrame de entrada con las 8 features (Nombres exactos del entrenamiento)
            input_data = pd.DataFrame({
                'air_temperature': [temp],
                'relative_humidity': [50.0], # Asumir valor fijo si no se incluye en input
                'wind_speed': [wind],
                'gross_floor_area': [area],
                'hour': [hora],
                'day_of_week': [dia_sem],
                'month': [6], # Asumir mes fijo
                'is_holiday': [1 if es_feriado else 0],
                'category': [cat_edificio]
            })
            
            # --- PREDICCIÓN Y VISUALIZACIÓN ---
            try:
                # El pipeline se encarga del escalado y one-hot encoding
                prediccion_kwh = final_pipeline.predict(input_data)[0]
                
                st.markdown("---")
                st.subheader("Resultado Estimado")
                st.metric(label="Consumo Predicho", value=f"{prediccion_kwh:,.2f} kWh", delta_color="off")
                
                # Gráfico de Consumo Predictivo (Requisito)
                st.markdown("### 📈 Visualización del Ciclo Diario Proyectado")
                
                # Crear un ciclo diario variando solo la hora
                horas_proyectadas = list(range(0, 24))
                df_proy = input_data.loc[input_data.index.repeat(24)].reset_index(drop=True)
                df_proy['hour'] = horas_proyectadas
                
                # Predecir el ciclo completo
                consumo_proyectado = final_pipeline.predict(df_proy)
                
                df_vis = pd.DataFrame({'Hora': horas_proyectadas, 'Consumo (kWh)': consumo_proyectado})
                
                # Marcar la hora de predicción actual
                df_vis['Actual'] = np.where(df_vis['Hora'] == hora, df_vis['Consumo (kWh)'], np.nan)
                
                fig_cycle, ax_cycle = plt.subplots(figsize=(10, 5))
                ax_cycle.plot(df_vis['Hora'], df_vis['Consumo (kWh)'], marker='o', linestyle='-', label='Ciclo Proyectado')
                ax_cycle.scatter(df_vis['Hora'], df_vis['Actual'], color='red', s=100, zorder=5, label='Predicción Actual')
                
                ax_cycle.set_title(f"Consumo Proyectado del Edificio '{cat_edificio}' para el día {['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][dia_sem]}")
                ax_cycle.set_xlabel("Hora del Día")
                ax_cycle.set_ylabel("Consumo Estimado (kWh)")
                ax_cycle.grid(True, alpha=0.3)
                ax_cycle.legend()
                st.pyplot(fig_cycle)
                
            except Exception as e:
                st.error(f"Error en la predicción. Asegúrate de que los inputs sean consistentes con el entrenamiento: {e}")
                
    else:
        st.warning("El simulador está inactivo. Asegúrate de que el archivo 'mlp_model_entrenado.pkl' exista en el directorio raíz.")