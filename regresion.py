# Importación de las bibliotecas necesarias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np


# Configuración inicial
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Carga de los datos
# Carga de los datos con un códec alternativo
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

# División del dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Escalado de los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenamiento del modelo
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predicción con el modelo
y_pred = clf.predict(X_test)

# Evaluación del modelo
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualización de la importancia de las características
importances = clf.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.title('Importancia de las Características')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Importancia Relativa')
plt.show()

# Predicción de la propiedad para un nuevo estudiante
nuevo_estudiante = [[25, le_universidad.transform(['CUCEI'])[
    0], 1200, 3, le_arrendo_anteriomente.transform(['Sí'])[0], 4.5, 12, 1, 1, 1, 1, 1, 0]]
nuevo_estudiante = scaler.transform(nuevo_estudiante)
prediccion = clf.predict(nuevo_estudiante)
print("La propiedad recomendada para el nuevo estudiante es:",
      le_property_id.inverse_transform(prediccion)[0])
