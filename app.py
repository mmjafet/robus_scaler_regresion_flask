from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Ruta del archivo de dataset específico
DATASET_PATH = 'C:/Users/Suseth Sandoval/Downloads/AndroidAdware2017/TotalFeatures-ISCXFlowMeter.csv'

# Ruta para la página de inicio y procesamiento de datos
@app.route('/')
def index():
    # Cargar y procesar el dataset
    df = pd.read_csv(DATASET_PATH)
    
    # Identificar y convertir columnas categóricas
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

    # Separar características (X) y etiquetas (y)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Escalado de características
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Separar en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Definir el número de datos de entrenamiento que deseas utilizar
    num_training_samples = 5000  # Cambia este valor según lo necesites
    X_train = X_train[:num_training_samples]
    y_train = y_train[:num_training_samples]

    # Entrenamiento del modelo
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Predicciones
    y_pred = model.predict(X_test)

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Resultados de Predicción')
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)

    return render_template('result.html', plot_url=plot_path)


if __name__ == '__main__':
    app.run(debug=True)