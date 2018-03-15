import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

import os

weights_name = "weights.hdf5"
numpy.random.seed(0)

dataframe = pandas.read_csv("iris.csv", header=None)
dataset = dataframe.values

X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

X_train, X_test, y_train, y_test = \
          train_test_split(X, Y, test_size=0.2, random_state=42)

# Preprocess the labels
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_y_train = encoder.transform(y_train)

encoder.fit(y_test)
encoded_y_test = encoder.transform(y_test)

# to_categorical converts the numbered labels into a one-hot vector
y_train = np_utils.to_categorical(encoded_y_train)
y_test = np_utils.to_categorical(encoded_y_test)

model = Sequential()
model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

if not os.path.isfile(weights_name):
  model.fit(X_train, y_train, epochs=200, batch_size=5)
  model.save_weights(weights_name)
else:
  model.load_weights(weights_name)

# Explicar qué es cross validation
# Crear una grafica aquí
# from sklearn.cross_validation import cross_val_score
# from sklearn.cross_validation import KFold

# model.predict()

score = model.evaluate(X_test, y_test)
print("Loss:", score[0])
print("Accuracy: {}%".format(score[1] * 100))

# kfold = KFold(n=len(X), n_folds=10, shuffle=True, random_state=0)
# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
