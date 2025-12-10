import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("🚀 Iniciando proceso de entrenamiento...")

# --- 1. CARGA DE DATOS ---
def load_data():
    base_path = "dataset" 
    
    
    print("   Cargando archivos CSV...")
    try:
        
        df_consumption = pd.read_csv(os.path.join(base_path, 'building_consumption.csv'))
        df_weather = pd.read_csv(os.path.join(base_path, 'weather_data.csv'))
        df_meta = pd.read_csv(os.path.join(base_path, 'building_meta.csv'))
        df_calendar = pd.read_csv(os.path.join(base_path, 'calender.csv'))
        return df_consumption, df_weather, df_meta, df_calendar
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró algún archivo. Verifica las rutas. {e}")
        exit()

df_consumo, df_clima, df_meta, df_calendario = load_data()

# --- 2. PREPROCESAMIENTO Y MERGE (Fusión) ---
print("   Fusionando datasets...")

# Convertir a datetime
df_consumo['timestamp'] = pd.to_datetime(df_consumo['timestamp'])
df_clima['timestamp'] = pd.to_datetime(df_clima['timestamp'])
df_calendario['date'] = pd.to_datetime(df_calendario['date'])

# A. Unir con Metadatos del Edificio (para obtener la categoría)
df_merged = pd.merge(df_consumo, df_meta[['id', 'category', 'gross_floor_area']], 
                     left_on='meter_id', right_on='id', how='left')

# B. Unir con Clima
# Unimos por 'timestamp' y 'campus_id' para ser precisos
df_merged = pd.merge(df_merged, df_clima, on=['timestamp', 'campus_id'], how='left')

# C. Unir con Calendario
# Creamos una columna 'date' solo para el merge
df_merged['date'] = df_merged['timestamp'].dt.normalize()
df_merged = pd.merge(df_merged, df_calendario, on='date', how='left')

# D. Feature Engineering (Extraer características de tiempo)
df_merged['hour'] = df_merged['timestamp'].dt.hour
df_merged['day_of_week'] = df_merged['timestamp'].dt.dayofweek
df_merged['month'] = df_merged['timestamp'].dt.month

# Limpieza de nulos (si quedaron huecos tras el merge)
df_merged = df_merged.dropna(subset=['consumption']) 
df_merged['gross_floor_area'] = df_merged['gross_floor_area'].fillna(df_merged['gross_floor_area'].mean())

print(f"   Dataset final creado con {df_merged.shape[0]} registros.")

# --- 3. DEFINICIÓN DE FEATURES Y TARGET ---
numeric_features = ['air_temperature', 'relative_humidity', 'wind_speed', 
                    'gross_floor_area', 'hour', 'day_of_week', 'month', 'is_holiday']
categorical_features = ['category']

target = 'consumption'

X = df_merged[numeric_features + categorical_features]
y = df_merged[target]

# --- 4. PIPELINE DE TRANSFORMACIÓN ---
# Esto automatiza el escalado y el one-hot encoding
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 5. ENTRENAMIENTO DEL MODELO ---
print(" Entrenando MLP Regressor (esto puede tardar)...")

# Pipeline completo: Preprocesamiento -> Modelo
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), 
                               activation='relu', 
                               solver='adam', 
                               max_iter=500, 
                               random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

# --- 6. EVALUACIÓN ---
print(" Evaluando modelo...")
y_pred = model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"   RMSE: {rmse:.4f}")
print(f"   R2 Score: {r2:.4f}")

# --- 7. GUARDAR ---
print(" Guardando modelo...")
joblib.dump(model_pipeline, 'mlp_model_entrenado.pkl')
print(" ¡Listo! Modelo guardado como 'mlp_model_entrenado.pkl'")