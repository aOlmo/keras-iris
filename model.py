import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

numpy.random.seed(0)

dataframe   = pandas.read_csv("iris.csv", header=None)
dataset     = dataframe.values

X = dataset[:,0:4].astype(float)
Y = dataset[:,4]

# Preprocess the labels

# LabelEncoder from scikit-learn turns each text label
# (e.g "Iris-setosa", "Iris-versicolor") into a vector
# In this case, each of the three labels are just assigned
# a number from 0-2.
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# to_categorical converts the numbered labels into a one-hot vector
dummy_Y = np_utils.to_categorical(encoded_Y)

def baseline_model():
    model = Sequential()
    model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
    model.add(Dense(3, init='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

baseline_model()
