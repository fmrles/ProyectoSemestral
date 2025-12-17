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








st.title("Predicción Energética - Campus Universitario")

tab_exp, tab_sim, tab_context = st.tabs(["1. Lab de Entrenamiento", "2. Simulador de Campus", "3. Contexto Analítico"])

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

with tab_sim:
    st.header("Simulador de Consumo: Escenario Campus")
    
    if final_pipeline:
        mode = st.radio("Modo de Simulación:", 
                        ["Edificio Individual (Instantáneo)", 
                         "Campus Completo (Instantáneo)",
                         "Simulación Temporal (Rango de Fechas)"],
                        horizontal=True)
        st.markdown("---")

        if mode in ["Edificio Individual (Instantáneo)", "Campus Completo (Instantáneo)"]:
            
            st.subheader("1. Condiciones Ambientales y Temporales")
            c1, c2, c3, c4 = st.columns(4)
            with c1: temp = st.slider("Temperatura (°C)", -5.0, 45.0, 22.0)
            with c2: wind = st.number_input("Viento (km/h)", 0.0, 100.0, 12.0)
            with c3: hora = st.slider("Hora del Día", 0, 23, 14) 
            with c4: dia = st.selectbox("Día Semana", range(7), format_func=lambda x: ['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][x])
            feriado = st.checkbox("¿Es Feriado?")

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

            else:
                st.subheader("2. Composición Personalizada del Campus")
                st.info("Define la cantidad de edificios y el área unitaria para cada tipo.")
                
                h1, h2, h3 = st.columns([1, 1, 2])
                with h1: st.markdown("**Tipo de Edificio**")
                with h2: st.markdown("**Cantidad**")
                with h3: st.markdown("**Área Unitaria (m²)**")

                campus_config = []
                
                c_off1, c_off2, c_off3 = st.columns([1, 1, 2])
                with c_off1: st.markdown("Oficinas")
                with c_off2: q_off = st.number_input("Cant. Oficinas", 0, 50, 5, key="q_off", label_visibility="collapsed")
                with c_off3: a_off = st.number_input("Área Oficina", 100, 10000, 1500, key="a_off", label_visibility="collapsed")
                campus_config.append({'cat': 'office', 'count': q_off, 'area': a_off})

                c_tea1, c_tea2, c_tea3 = st.columns([1, 1, 2])
                with c_tea1: st.markdown("Aulas")
                with c_tea2: q_tea = st.number_input("Cant. Aulas", 0, 50, 8, key="q_tea", label_visibility="collapsed")
                with c_tea3: a_tea = st.number_input("Área Aulas", 100, 10000, 2500, key="a_tea", label_visibility="collapsed")
                campus_config.append({'cat': 'teaching', 'count': q_tea, 'area': a_tea})

                c_lib1, c_lib2, c_lib3 = st.columns([1, 1, 2])
                with c_lib1: st.markdown("Bibliotecas")
                with c_lib2: q_lib = st.number_input("Cant. Bibliotecas", 0, 10, 1, key="q_lib", label_visibility="collapsed")
                with c_lib3: a_lib = st.number_input("Área Biblioteca", 100, 20000, 4000, key="a_lib", label_visibility="collapsed")
                campus_config.append({'cat': 'library', 'count': q_lib, 'area': a_lib})

                c_mix1, c_mix2, c_mix3 = st.columns([1, 1, 2])
                with c_mix1: st.markdown("Uso Mixto")
                with c_mix2: q_mix = st.number_input("Cant. Mixto", 0, 20, 2, key="q_mix", label_visibility="collapsed")
                with c_mix3: a_mix = st.number_input("Área Mixto", 100, 15000, 3000, key="a_mix", label_visibility="collapsed")
                campus_config.append({'cat': 'mixed use', 'count': q_mix, 'area': a_mix})

                c_oth1, c_oth2, c_oth3 = st.columns([1, 1, 2])
                with c_oth1: st.markdown("Otros")
                with c_oth2: q_oth = st.number_input("Cant. Otros", 0, 20, 2, key="q_oth", label_visibility="collapsed")
                with c_oth3: a_oth = st.number_input("Área Otros", 100, 5000, 1000, key="q_oth_area", label_visibility="collapsed")
                campus_config.append({'cat': 'other', 'count': q_oth, 'area': a_oth})
                
                st.markdown("---")

                if st.button("Simular Campus Completo", type="primary"):
                    total_consumption = 0
                    breakdown = {}
                    
                    for item in campus_config:
                        count = item['count']
                        area_user = item['area']
                        cat = item['cat']
                        
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
                            colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0']
                            ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(labels)])
                            ax.axis('equal') 
                            st.pyplot(fig)
                        else:
                            st.warning("Configura al menos un edificio para ver resultados.")


        else:
            st.subheader("1. Condiciones Ambientales Fijas y Temporales")
            
            col_t_clim, col_t_wind, col_t_hum = st.columns(3)
            with col_t_clim: temp_t = st.slider("Temperatura Fija (°C)", -5.0, 45.0, 22.0, key="temp_t")
            with col_t_hum: hum_t = st.slider("Humedad Fija (%)", 0, 100, 50, key="hum_t")
            with col_t_wind: wind_t = st.number_input("Viento Fijo (km/h)", 0.0, 100.0, 12.0, key="wind_t")
            
            st.markdown("---")
            
            col_date1, col_date2 = st.columns(2)
            with col_date1: start_date = st.date_input("Fecha Inicio", pd.to_datetime("2025-06-01"))
            with col_date2: end_date = st.date_input("Fecha Fin", pd.to_datetime("2025-06-07"))

            st.subheader("2. Configuración del Campus")
            
            
            h1_t, h2_t, h3_t = st.columns([1, 1, 2])
            with h1_t: st.markdown("**Tipo de Edificio**")
            with h2_t: st.markdown("**Cantidad**")
            with h3_t: st.markdown("**Área Unitaria (m²)**")
            
            campus_config_temp = {}
            
            c_off1, c_off_inputs = st.columns([1, 3]) 
            with c_off1: st.markdown("Oficinas")
            with c_off_inputs:
                c_q, c_a = st.columns([1, 2]) 
                with c_q: q_off = st.number_input("Cant. Oficinas", 0, 50, 5, key="tq_off", label_visibility="collapsed")
                with c_a: a_off = st.number_input("Área Oficina", 100, 10000, 1500, key="taq_off", label_visibility="collapsed")
                campus_config_temp['office'] = {'count': q_off, 'area': a_off}

            c_tea1, c_tea_inputs = st.columns([1, 3])
            with c_tea1: st.markdown("Aulas")
            with c_tea_inputs:
                c_q, c_a = st.columns([1, 2])
                with c_q: q_tea = st.number_input("Cant. Aulas", 0, 50, 8, key="tq_tea", label_visibility="collapsed")
                with c_a: a_tea = st.number_input("Área Aulas", 100, 10000, 2500, key="taq_tea", label_visibility="collapsed")
                campus_config_temp['teaching'] = {'count': q_tea, 'area': a_tea}

            c_lib1, c_lib_inputs = st.columns([1, 3])
            with c_lib1: st.markdown("Bibliotecas")
            with c_lib_inputs:
                c_q, c_a = st.columns([1, 2])
                with c_q: q_lib = st.number_input("Cant. Bibliotecas", 0, 10, 1, key="tq_lib", label_visibility="collapsed")
                with c_a: a_lib = st.number_input("Área Biblioteca", 100, 20000, 4000, key="taq_lib", label_visibility="collapsed")
                campus_config_temp['library'] = {'count': q_lib, 'area': a_lib}

            c_mix1, c_mix_inputs = st.columns([1, 3])
            with c_mix1: st.markdown("Uso Mixto")
            with c_mix_inputs:
                c_q, c_a = st.columns([1, 2])
                with c_q: q_mix = st.number_input("Cant. Mixto", 0, 20, 2, key="tq_mix", label_visibility="collapsed")
                with c_a: a_mix = st.number_input("Área Mixto", 100, 15000, 3000, key="taq_mix", label_visibility="collapsed")
                campus_config_temp['mixed use'] = {'count': q_mix, 'area': a_mix}

            c_oth1, c_oth_inputs = st.columns([1, 3])
            with c_oth1: st.markdown("Otros")
            with c_oth_inputs:
                c_q, c_a = st.columns([1, 2])
                with c_q: q_oth = st.number_input("Cant. Otros", 0, 20, 2, key="tq_oth", label_visibility="collapsed")
                with c_a: a_oth = st.number_input("Área Otros", 100, 5000, 1000, key="taq_oth", label_visibility="collapsed")
                campus_config_temp['other'] = {'count': q_oth, 'area': a_oth}
            
            
            st.subheader("3. Editor de Eventos Diarios")

            if start_date <= end_date:
                days_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                default_data = {
                    "Fecha": days_range.date,
                    "Es Feriado": [d.dayofweek >= 5 for d in days_range],
                    "Corte de Luz": [False] * len(days_range)
                }
                schedule_df = pd.DataFrame(default_data)
                
                edited_schedule = st.data_editor(
                    schedule_df,
                    column_config={
                        "Fecha": st.column_config.DateColumn("Fecha", disabled=True),
                        "Es Feriado": st.column_config.CheckboxColumn("¿Es Feriado?"),
                        "Corte de Luz": st.column_config.CheckboxColumn("Corte Programado")
                    },
                    hide_index=True,
                    num_rows="fixed"
                )

                if st.button("Simular Periodo", type="primary"):
                    
                    full_range = pd.date_range(start=start_date, end=end_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1), freq='h')
                    
                    sim_df = pd.DataFrame({'timestamp': full_range})
                    sim_df['date'] = sim_df['timestamp'].dt.date
                    sim_df['hour'] = sim_df['timestamp'].dt.hour
                    sim_df['day_of_week'] = sim_df['timestamp'].dt.dayofweek
                    sim_df['month'] = sim_df['timestamp'].dt.month
                    
                    sim_df['air_temperature'] = temp_t
                    sim_df['relative_humidity'] = hum_t
                    sim_df['wind_speed'] = wind_t
                    
                    schedule_map = edited_schedule.set_index("Fecha").to_dict('index')
                    
                    def get_safe_event(date_obj, column):
                        try:
                            return 1 if schedule_map[date_obj][column] else 0
                        except KeyError:
                            return 0 

                    sim_df['is_holiday'] = sim_df['date'].apply(lambda x: get_safe_event(x, 'Es Feriado'))
                    sim_df['power_outage'] = sim_df['date'].apply(lambda x: get_safe_event(x, 'Corte de Luz'))

                    campus_total_load = np.zeros(len(sim_df))
                    
                    for cat, params in campus_config_temp.items():
                        count = params['count']
                        area_user = params['area']
                        
                        if count > 0:
                            features_df = sim_df[['air_temperature', 'relative_humidity', 'wind_speed', 'hour', 'day_of_week', 'month', 'is_holiday']].copy()
                            features_df['gross_floor_area'] = area_user
                            features_df['category'] = cat
                            
                            input_cols = ['air_temperature', 'relative_humidity', 'wind_speed', 'gross_floor_area', 'hour', 'day_of_week', 'month', 'is_holiday', 'category']
                            pred_vector = final_pipeline.predict(features_df[input_cols])
                            campus_total_load += (pred_vector * count)

                    final_load = np.where(sim_df['power_outage'], 0, campus_total_load)
                    
                    total_energy = np.sum(final_load)
                    
                    st.markdown("---")
                    col_res1, col_res2 = st.columns([1, 3])
                    
                    with col_res1:
                        st.metric("Energía Total Periodo", f"{total_energy/1000:,.2f} MWh")
                        st.caption(f"Periodo: {len(days_range)} días")

                    with col_res2:
                        st.subheader("Evolución del Consumo (Simulación)")
                        
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(sim_df['timestamp'], final_load, label="Consumo Estimado (kWh)", color="#1f77b4")
                        
                        outage_times = sim_df[sim_df['power_outage'] == 1]['timestamp']
                        if not outage_times.empty:
                            ax.scatter(outage_times, [0]*len(outage_times), color='red', label="Corte Programado", zorder=5, s=10)
                        
                        ax.set_ylabel("Potencia (kW)")
                        ax.set_title("Perfil de Demanda Energética del Campus")
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)
            else:
                st.error("La fecha de fin debe ser posterior a la de inicio.")
                
            
    else:
        st.error("Modelo offline no encontrado. Ejecuta 'entrenar_modelo.py' primero.")


with tab_context:
    st.header("Contexto del Modelo (Pre-Entrenamiento)")
    
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