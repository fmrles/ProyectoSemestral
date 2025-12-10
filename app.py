import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import os

st.set_page_config(page_title="Predicción Energética UBB", layout="wide")

# Cargar datos
@st.cache_data
def load_data(uploaded_file):
    """
    Prioridad de carga:
    1. Archivo subido por el usuario en la interfaz.
    2. Archivo 'building_consumption.csv' local en la carpeta.
    3. Datos sintéticos (fallback).
    """
    df = None
    source_type = ""

    # Carga subida manual
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            source_type = "Archivo Subido Manualmente"
        except Exception as e:
            st.error(f"Error al leer el archivo subido: {e}")
    
    # Carga archivo local
    elif os.path.exists('building_consumption.csv'):
        try:
            df = pd.read_csv('building_consumption.csv')
            source_type = "Archivo Local (building_consumption.csv)"
        except Exception as e:
            st.warning(f"Se encontró el archivo local pero hubo error al leerlo: {e}")

    # Generacion si falla
    if df is None:
        source_type = "Datos Sintéticos (Demo)"
        dates = pd.date_range(start="2023-01-01", periods=1000, freq="h")
        data = {
            'timestamp': dates,
            'Temperature': np.random.uniform(5, 35, 1000), 
            'Occupancy': np.random.randint(0, 500, 1000), 
            'Building_ID': np.random.choice(['Biblioteca', 'Aulas', 'Gimnasio'], 1000)
        }
        df = pd.DataFrame(data)
        # Consumo simulado
        df['Energy_Consumption'] = (
            (df['Temperature'] - 20)**2 * 1.5 + 
            df['Occupancy'] * 0.5 + 
            np.random.normal(0, 5, 1000) + 100
        )

    return df, source_type

# Procesamiento y limpieza
def preprocess_data(df, time_col, target_col):
    """
    Limpia y prepara los datos basándose en las columnas seleccionadas.
    """
    df = df.copy()
    
    # Convertir fecha
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        st.error(f"Error convirtiendo la columna de fecha: {e}")
        return None, None

    # Feature Engineering básico
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['month'] = df[time_col].dt.month
    
    # Manejo de variables categóricas
    # Excluimos la columna de fecha y el objetivo
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != time_col and c != target_col]
    
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Eliminar columna timestamp original para el modelo numérico
    df_model = df.drop(columns=[time_col])
    
    # Manejo de NaNs (rellenar con media para no perder datos en demo)
    df_model = df_model.fillna(df_model.mean())
    
    return df, df_model

# Interfaz
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Logo_UBB.jpg/640px-Logo_UBB.jpg", width=100)
st.sidebar.title("Configuración")

# Opción de carga manual
uploaded_file = st.sidebar.file_uploader("Cargar Dataset (Opcional)", type=["csv"])
df_raw, source_msg = load_data(uploaded_file)

st.sidebar.success(f"Fuente: {source_msg}")

st.title(" Predicción de Consumo Energético - Avance 2")

tab1, tab2, tab3, tab4 = st.tabs(["Datos", "Modelo", "Predicción", "Info"])

# Variables de estado para selectores de columnas
if 'target_col' not in st.session_state: st.session_state.target_col = None
if 'time_col' not in st.session_state: st.session_state.time_col = None

with tab1:
    st.header("1. Exploración de Datos")
    
    if df_raw is not None:
        st.write("Vista previa de los datos cargados:")
        st.dataframe(df_raw.head())
        
        st.subheader("Configuración de Columnas")
        col1, col2 = st.columns(2)
        
        # Intentar adivinar columnas
        all_cols = df_raw.columns.tolist()
        
        # Heurística simple para adivinar
        default_time = next((c for c in all_cols if 'time' in c.lower() or 'date' in c.lower()), all_cols[0])
        default_target = next((c for c in all_cols if 'energy' in c.lower() or 'consum' in c.lower() or 'value' in c.lower()), all_cols[-1])
        
        with col1:
            time_col = st.selectbox("Selecciona la Columna de Fecha/Hora", all_cols, index=all_cols.index(default_time))
        with col2:
            target_col = st.selectbox("Selecciona la Columna Objetivo (Consumo)", all_cols, index=all_cols.index(default_target))
            
        st.session_state.time_col = time_col
        st.session_state.target_col = target_col
        
        # Procesar
        df_view, df_model = preprocess_data(df_raw, time_col, target_col)
        
        if df_model is not None:
            st.success("✅ Datos procesados correctamente")
            st.write("Datos listos para el modelo (Variables dummies creadas y fechas descompuestas):")
            st.dataframe(df_model.head())
    else:
        st.error("No se pudieron cargar datos.")

# Variables globales para el modelo
model = None
scaler_X = StandardScaler()
scaler_y = StandardScaler()

with tab2:
    st.header("2. Entrenamiento MLP")
    
    if df_raw is not None and df_model is not None:
        # Configuración Hiperparámetros
        c1, c2, c3 = st.columns(3)
        h_layers = c1.selectbox("Capas Ocultas", [(64, 32), (100,), (50, 50)], index=0)
        activ = c2.selectbox("Activación", ["relu", "tanh"], index=0)
        iters = c3.slider("Max Iteraciones", 200, 1000, 500)
        
        target = st.session_state.target_col
        
        if target in df_model.columns:
            X = df_model.drop(columns=[target])
            y = df_model[[target]]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            if st.button("🚀 Entrenar Modelo"):
                with st.spinner('Entrenando... (esto puede tardar unos segundos)'):
                    # Escalar
                    X_train_s = scaler_X.fit_transform(X_train)
                    X_test_s = scaler_X.transform(X_test)
                    y_train_s = scaler_y.fit_transform(y_train)
                    y_test_s = scaler_y.transform(y_test)
                    
                    model = MLPRegressor(hidden_layer_sizes=h_layers, activation=activ, max_iter=iters, random_state=42)
                    model.fit(X_train_s, y_train_s.ravel())
                    
                    # Guardar en estado
                    st.session_state['model'] = model
                    st.session_state['scaler_X'] = scaler_X
                    st.session_state['scaler_y'] = scaler_y
                    st.session_state['features'] = X.columns.tolist()
                    
                    # Métricas
                    score = model.score(X_test_s, y_test_s)
                    st.success(f"Modelo Entrenado! R2 Score: {score:.4f}")
                    
                    preds = scaler_y.inverse_transform(model.predict(X_test_s).reshape(-1,1))
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(y_test.values[:50], label='Real')
                    ax.plot(preds[:50], label='Predicho', linestyle='--')
                    ax.legend()
                    st.pyplot(fig)

with tab3:
    st.header("3. Simulador")
    if 'model' in st.session_state:
        st.info("Ingresa valores para simular:")
        inputs = {}
        cols = st.columns(3)
        for i, col in enumerate(st.session_state['features']):
            with cols[i%3]:
                inputs[col] = st.number_input(col, value=0.0)
        
        if st.button("Predecir"):
            in_df = pd.DataFrame([inputs])
            in_scaled = st.session_state['scaler_X'].transform(in_df)
            pred_s = st.session_state['model'].predict(in_scaled)
            res = st.session_state['scaler_y'].inverse_transform(pred_s.reshape(-1,1))
            st.metric("Consumo Predicho", f"{res[0][0]:.2f}")
    else:
        st.warning("Entrena el modelo primero en la pestaña 2.")

with tab4:
    st.header("Documentación")
    st.markdown("### Flujo del Proyecto")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        Data -> Clean -> Split -> MLP -> Pred;
        Data [shape=cylinder];
        MLP [style=filled, color=lightblue];
    }
    """)
    st.markdown(f"**Dataset usado:** {source_msg}")