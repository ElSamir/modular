# Importación de las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Configuración inicial
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Carga de los datos
data = pd.read_csv(
    "recomendaciones_propiedades_ml_actualizado_1000.csv", encoding='ISO-8859-1')

# Visualización inicial del dataset
print(data.head())
print(data.describe())

# Preprocesamiento de los datos
# Transformación de variables categóricas a numéricas con Label Encoding
le_universidad = LabelEncoder()
le_property_id = LabelEncoder()
le_arrendo_anteriomente = LabelEncoder()

data['Universidad'] = le_universidad.fit_transform(data['Universidad'])
data['PropertyID'] = le_property_id.fit_transform(data['PropertyID'])
data['ArrendóAnteriormente'] = le_arrendo_anteriomente.fit_transform(
    data['ArrendóAnteriormente'])

# Definición de las variables independientes (X) y la variable dependiente (y)
X = data[['Edad', 'Universidad', 'CostoMensual', 'NúmeroHabitaciones',
          'ArrendóAnteriormente', 'Calificación', 'DuraciónArrendamiento',
          'Agua', 'Electricidad', 'Internet', 'Gas', 'Calefacción', 'Limpieza']]
y = data['PropertyID']
# Guardar IDs de estudiante para asociar con las predicciones
estudiante_ids = data['EstudianteID']

# División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, estudiante_ids, test_size=0.3, random_state=42)

# Escalado de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción con el modelo
y_pred = model.predict(X_test)

# Evaluación del modelo
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Visualización en 2D de las predicciones vs valores reales
plt.scatter(y_test, y_pred, color='blue', s=30, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test),
         max(y_test)], color='red', linewidth=2)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Predicciones vs Valores Reales')
plt.show()

# Asociar predicciones con IDs de estudiante
predictions_df = pd.DataFrame({'EstudianteID': ids_test, 'Predicción': y_pred})
print(predictions_df.head())

# Predicción de la propiedad para un nuevo estudiante
nuevo_estudiante = [[25, le_universidad.transform(['CUCEI'])[
    0], 1200, 3, le_arrendo_anteriomente.transform(['Sí'])[0], 4.5, 12, 1, 1, 1, 1, 1, 0]]
nuevo_estudiante = scaler.transform(nuevo_estudiante)
prediccion = model.predict(nuevo_estudiante)
print("La propiedad recomendada para el nuevo estudiante es:",
      le_property_id.inverse_transform([int(prediccion[0])])[0])
