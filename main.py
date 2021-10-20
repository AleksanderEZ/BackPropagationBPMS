import pandas as pd
import numpy as np
#from keras import Sequential
#from keras.layers import Dense

data = pd.read_excel('datosBPNN.xls')
data['ACCIDENTE'] = pd.factorize(data['ACCIDENTE'])[0]
X = np.array(data.iloc[:, 1:13])
y = np.array(data.iloc[:, 13])
print(y)

#model = Sequential()
#model.add(Dense(12, input_dim=11, activation='sigmoid'))
#model.add(Dense(1, activation='sigmoid'))

#model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['loss', 'accuracy'])

#model.fit(X, y, epochs=1500)