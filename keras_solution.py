import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from data_processing import load_data, preprocess


def predict(input_data):
    model = load_model("model.h5")
    if input_data[0].shape == ():
        input_data = input_data.reshape((1, 12))
    return np.round(model.predict(input_data))


def train_and_save(training_X, training_y, validation_X, validation_y):
    model = Sequential([
            Dense(12, input_dim=X.shape[1], activation='sigmoid'),
            Dense(8, activation='sigmoid'),
            Dense(1) ])

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
                  metrics=[tf.keras.metrics.mean_squared_error, tf.keras.metrics.binary_accuracy])

    print(model.summary())

    history = model.fit(training_X, training_y, validation_data=(validation_X, validation_y), batch_size=64, epochs=250)

    model.save("model.h5")

    plt.subplot(1,2,1)
    plt.plot(history.history['mean_squared_error'])
    plt.title('Error cuadrático medio')
    plt.xlabel('Epoch')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['binary_accuracy'])
    plt.title('Precisión binaria')
    plt.xlabel('Epoch')
    plt.show()


data = load_data()

X = np.array(data.iloc[:, 1:13], dtype='float32')
y = np.array(data.iloc[:, 13], dtype='float32')

x_train, x_test, y_train, y_test = preprocess(X, y)

train_and_save(x_train, y_train, x_test, y_test)
print(predict(x_test))