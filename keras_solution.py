import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf

def predict(input):
    model = load_model("model.h5")
    model.predict(input)

def load_data():
    data = pd.read_excel('Datos_PrActica_1_BPNN.xls')

    data.replace(' ', np.nan, inplace=True)
    data.dropna(inplace=True)

    data['ACCIDENTE'] = pd.factorize(data['ACCIDENTE'])[0]
    data['ESTADO_CARRETERA'] = pd.factorize(data['ESTADO_CARRETERA'])[0]
    data['INTENSIDAD_PRECIPITACION'] = pd.factorize(data['INTENSIDAD_PRECIPITACION'])[0]
    data['TIPO_PRECIPITACION'] = pd.factorize(data['TIPO_PRECIPITACION'])[0]
    return data

def train_and_save(X, y):
    model = Sequential()
    model.add(Dense(12, input_dim=X.shape[1], activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='SGD',
                  metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.binary_accuracy])

    model.fit(X, y, epochs=100)

    model.save("model.h5")

data = load_data()

X = np.array(data.iloc[:, 1:13], dtype='float32')
y = np.array(data.iloc[:, 13], dtype='float32')

train_and_save(X, y)




