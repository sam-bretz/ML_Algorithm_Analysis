"""
Author: Samuel Bretz (sbretz3)
Email: bretzsam@gmail.com
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from API import API
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline



def getCMC():
	dataframe = pd.read_csv("../data/cmc.csv")
	dataset = dataframe.values
	X = dataset[:,0:9].astype(float)
	Y = dataset[:,9]
	return (X, Y)

def getSC():
	dataframe = pd.read_csv("../data/SkillCraft1_Dataset.csv")
	dataset = dataframe.values
	X = dataset[:,0:18]
	Y = dataset[:,19]
	return (X, Y)

def cmcNet():
	seed = 7
	np.random.seed(seed)

	# load dataset
	X, Y = getCMC()

	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)

	# define baseline model
	def baseline_model():
		# create model
		model = Sequential()
		model.add(Dense(12, input_dim=9, activation='relu'))
		model.add(Dense(8, activation='relu'))
		model.add(Dense(3, activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	model = baseline_model()
	history = model.fit(X, dummy_y, validation_split=0.33, epochs=50, batch_size=10, verbose=1)
	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'], 'o-', color="r", label="Training Score")
	plt.plot(history.history['val_acc'], 'o-', color="g", label="Testing Score")
	plt.title('Neural Network CMC Dataset')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc="best")
	plt.show()

def scNet():
	seed = 7
	np.random.seed(seed)

	# load dataset
	X, Y = getSC()

	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_y = np_utils.to_categorical(encoded_Y)

	# define baseline model
	def baseline_model():
		# create model
		model = Sequential()
		model.add(Dense(32, input_dim=18, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(7, activation='softmax'))
		# Compile model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	model = baseline_model()
	history = model.fit(X, dummy_y, validation_split=0.33, epochs=50, batch_size=10, verbose=1)
	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['acc'], 'o-', color="r", label="Training Score")
	plt.plot(history.history['val_acc'], 'o-', color="g", label="Testing Score")
	plt.title('Neural Network SC Dataset')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(loc="best")
	plt.show()


if __name__ == '__main__':
	cmcNet()
	scNet()

