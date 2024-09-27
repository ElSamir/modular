import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar el dataset de propiedades con las asignaciones de estudiantes
df = pd.read_csv(
    'propiedades_asignadas_con_estudiantes_por_ciudad.csv', low_memory=False)

# Seleccionar las columnas que se usarán para la predicción, incluyendo 'city'
features = ['bedrooms', 'beds', 'Waterfront', 'Elevator', 'Pets allowed', 'Smoking allowed',
            'Wheelchair accessible', 'Pool', 'TV', 'Microwave', 'Internet', 'Heating']

# Incluir la columna 'city' en el DataFrame
df = df[['student_id', 'id', 'city'] + features]

# Convertir la columna 'city' a variables dummies (one-hot encoding)
df = pd.get_dummies(df, columns=['city'], drop_first=False)

# Crear la lista completa de características incluyendo las columnas dummies de 'city'
city_features = [col for col in df.columns if col.startswith('city_')]
features += city_features

# Filtrar el DataFrame para incluir solo las columnas necesarias
df = df[['student_id', 'id'] + features]

# Agrupar por student_id y calcular las características promedio de las 5 propiedades de cada estudiante
student_features = df.groupby('student_id').mean().reset_index()

# Dividir los datos en características y el target
X = student_features[features]
y = X  # En este caso, predeciremos todas las características

# Entrenar el modelo de regresión lineal múltiple
model = LinearRegression()
model.fit(X, y)

# Realizar predicciones sobre el conjunto de datos de entrenamiento
y_pred = model.predict(X)

# Calcular las métricas de precisión
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Precisión del modelo:")
print(f"- Error Cuadrático Medio (MSE): {mse}")
print(f"- Error Absoluto Medio (MAE): {mae}")
print(f"- Coeficiente de Determinación (R²): {r2}")

# Función para hacer la predicción y recomendar un id para un estudiante específico


def recomendar_propiedad(student_id, city_weight=10):
    # Verificar si el student_id existe
    if student_id not in student_features['student_id'].values:
        print(f"El student_id {student_id} no existe en el conjunto de datos.")
        return

    # Obtener las características promedio del estudiante
    student_data = student_features[student_features['student_id']
                                    == student_id][features]

    # Hacer la predicción de la propiedad recomendada
    predicted_features = model.predict(student_data)

    # Calcular la distancia ponderada entre la propiedad predicha y todas las propiedades existentes
    # Primero, calcular la distancia para las características de la ciudad y multiplicarla por el peso
    city_distance = ((df[city_features].values - predicted_features[0]
                     [len(features) - len(city_features):]) ** 2).sum(axis=1) * city_weight

    # Luego, calcular la distancia para las demás características (amenidades)
    amenity_distance = ((df[features[:-len(city_features)]].values -
                        predicted_features[0][:-len(city_features)]) ** 2).sum(axis=1)

    # Sumar ambas distancias para obtener la distancia total
    df['distance'] = np.sqrt(city_distance + amenity_distance)

    # Encontrar la propiedad más cercana
    recommended_property = df.loc[df['distance'].idxmin()]

    # Obtener el nombre de la ciudad de la propiedad recomendada
    recommended_city = [col.replace(
        'city_', '') for col in city_features if recommended_property[col] == 1]

    # Mostrar la recomendación
    print(
        f"Recomendación de propiedad para el student_id {student_id} es el id: {recommended_property['id']} en la ciudad: {recommended_city[0] if recommended_city else 'Desconocida'}")
    return recommended_property['id']


# Solicitar el ID del estudiante al usuario
student_id = int(input(
    "Ingrese el ID del estudiante para obtener una recomendación de propiedad: "))
recommended_property_id = recomendar_propiedad(student_id)
