import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#데이터 불러오기
raw_data = pd.read_excel('titanic.xls')
raw_data.info()
raw_data.describe()
#데이터 전처리
tmp = []
for each in raw_data['sex']:
    if each == 'female':
        tmp.append(1)
    elif each == 'male':
        tmp.append(0)
    else:
        tmp.append(np.nan)

raw_data['sex'] = tmp

raw_data['survived'] = raw_data['survived'].astype('float')
raw_data['pclass'] = raw_data['pclass'].astype('float')
raw_data['sex'] = raw_data['sex'].astype('float')
raw_data['sibsp'] = raw_data['sibsp'].astype('float')
raw_data['parch'] = raw_data['parch'].astype('float')
raw_data['fare'] = raw_data['fare'].astype('float')

raw_data = raw_data[raw_data['age'].notnull()]
raw_data = raw_data[raw_data['sibsp'].notnull()]
raw_data = raw_data[raw_data['parch'].notnull()]
raw_data = raw_data[raw_data['fare'].notnull()]

raw_data.info()
#데이터 나누기
x_data = raw_data.values[:, [0,3,4,5,6,8]]
y_data = raw_data.values[:, [1]]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, 
                                               test_size=0.1, random_state=7)
#
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
np.random.seed(7)

print('tensorflow version : ', tf.__version__)
print('keras version : ', keras.__version__)
#모델 구성BN
sgd = optimizers.SGD(lr=0.005)
model = Sequential()
model.add(Dense(5, input_shape=(6,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense((1), activation='sigmoid'))
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1500)


#모델 구성 그냥
model = Sequential()
model.add(Dense(5, input_shape=(6,), activation='relu'))
model.add(Dense((1), activation='sigmoid'))
model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
model.summary()

hist2 = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1500)

# acc-epoch 비교
plt.figure(figsize=(12,8))
plt.plot(hist2.history['acc'])
plt.plot(hist.history['acc'])
plt.legend(['acc','acc_BN'])
plt.show()