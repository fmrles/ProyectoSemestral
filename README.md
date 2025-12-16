# Predicción de Consumo Energético - Campus Universitario

## Descripción
Este es un sistema de predicción de consumo energético para campus universitarios utilizando Redes Neuronales (MLP - Perceptrón Multicapa) 

## Características
- Predicción de consumo energético individual por edificio
- Simulación de campus completo con múltiples edificios
- Simulación temporal con rango de fechas
- Análisis de sensibilidad térmica
- Comparativa entre tipos de edificios
- Laboratorio de experimentación MLP vs Regresión Lineal
- Visualización de métricas y convergencia del modelo
- Explorador del dataset

## Instalación

```bash
# Clonar o descargar el proyecto
cd ProyectoSemestral

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (solo la primera vez)
python entrenar_modelo.py

# Ejecutar la aplicación
streamlit run app.py
```

## Estructura del Proyecto
```
ProyectoSemestral/
├── app.py                    # Aplicación Streamlit principal
├── entrenar_modelo.py        # Script de entrenamiento del MLP
├── mlp_model_entrenado.pkl   # Modelo entrenado (generado)
├── loss_history.pkl          # Historial de convergencia (generado)
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
└── dataset/
    ├── building_consumption.csv
    ├── weather_data.csv
    ├── building_meta.csv
    └── calender.csv
```

## Tecnologías Utilizadas
- **Python 3.9+**
- **Streamlit** - Interfaz web interactiva
- **Scikit-learn** - MLPRegressor y preprocesamiento
- **Pandas** - Manipulación de datos
- **Matplotlib** - Visualizaciones
- **NumPy** - Operaciones numéricas

## Variables del Modelo
| Variable | Descripción |
|----------|-------------|
| air_temperature | Temperatura del aire (°C) |
| relative_humidity | Humedad relativa (%) |
| wind_speed | Velocidad del viento (km/h) |
| gross_floor_area | Área bruta del edificio (m²) |
| hour | Hora del día (0-23) |
| day_of_week | Día de la semana (0-6) |
| month | Mes del año (1-12) |
| is_holiday | Indicador de feriado (0/1) |
| category | Tipo de edificio |

## Métricas del Modelo
- **Arquitectura**: (100, 50) neuronas
- **Activación**: ReLU
- **Iteraciones**: 146 (convergencia temprana)
- **R² Score**: ~0.88

## Autores
- Javier Gutiérrez Fuentes
- Francisco Morales Muñoz
- Gonzalo Matus Muñoz