import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    data = pd.read_excel('Datos_PrActica_1_BPNN.xls')

    data.replace(' ', np.nan, inplace=True)
    data.dropna(inplace=True)

    data['ACCIDENTE'] = pd.factorize(data['ACCIDENTE'])[0]
    data['ESTADO_CARRETERA'] = pd.factorize(data['ESTADO_CARRETERA'])[0]
    data['INTENSIDAD_PRECIPITACION'] = pd.factorize(data['INTENSIDAD_PRECIPITACION'])[0]
    data['TIPO_PRECIPITACION'] = pd.factorize(data['TIPO_PRECIPITACION'])[0]

    return data


# Preprocesado de los datos para facilitar el aprendizaje de la red
def preprocess(X, y):
    X = X / X.max(axis=0)  # Escalamos los valores de cada columna entre 0 y 1

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    # Equilibramos el número de filas de no accidentes
    y_no = np.where(y_train == 0)[0]  # Todas las filas que no acabaron en accidente
    y_yes = np.where(y_train == 1)[0]  # Todas las filas que sí acabaron en accidente
    random_no = np.random.choice(y_no, len(y_train[y_yes]),
                                 replace=False)  # Coge aleatoriamente tantas filas de no accidente como haya de sí accidente
    x_train = np.concatenate((x_train[random_no], x_train[y_yes]), 0)  # Unimos las filas de no y sí
    y_train = np.concatenate((y_train[random_no], y_train[y_yes]), 0)

    return x_train, x_test, y_train, y_test