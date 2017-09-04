"""
Author: Samuel Bretz (sbretz3)
Date: 8/31/17
Email: bretzsam@gmail.com
"""

import numpy as np
import random
import sklearn
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import math
import graphviz
from DataPrep import DataPrep



"""
Function to prepare and clean data.
RETURN VALUE: Tuple in the following form: ((X_train, X_test),(y_train, y_test))
"""
def prepData():
	dataPrep = DataPrep()
	data, classCodeSamples = dataPrep.dTreeData()

	#split into training sets and test sets
	X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
	data, classCodeSamples, test_size=0.33, random_state=42)
	return ((X_train, X_test),(y_train, y_test))

"""Build and fit the tree.
	RETURN VALUE: sklearn decision tree classifier object"""
def Tree(data):
	X, y = data
	X_train, X_test = X
	y_train, y_test = y
	classifier = tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
	classifier = classifier.fit(X_train, y_train)
	return classifier

"""Determine the accuracy of the model.
	RETURN VALUE: None"""
def accuracy(classifier, X_test, y_test):
	preds = []
	for i in range(len(X_test)):
		preds.append(classifier.predict(np.array(X_test[i]).reshape(1, -1))[0])
	print accuracy_score(preds, y_test)

"""Renders an image of the decision tree, outputs to a pdf file.
	RETURN VALUE: None"""
def displayTree(classifier):
	dot_data = tree.export_graphviz(classifier, out_file=None)
	graph = graphviz.Source(dot_data)
	graph.render("arryhthmia")


if __name__ == '__main__':
	X, y = prepData()
	X_train, X_test = X
	y_train, y_test = y
	classifier = Tree(prepData())
	accuracy(classifier, X_test, y_test)
	displayTree(classifier)


