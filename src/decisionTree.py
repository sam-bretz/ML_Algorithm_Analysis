"""
Author: Samuel Bretz (sbretz3)
Email: bretzsam@gmail.com
"""

import numpy as np
import sklearn
from sklearn import tree
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import math
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from API import API
from dataVis import dataVis



"""Determine the best hyperparameters of the decision tree model.
	RETURN VALUE:
"""
def hyperTune(tree, X, y):
	#determine what hyperparameters to tune
	criterion = ['gini','entropy']
	splitter = ['best','random']
	maxDepth = range(1,15)
	min_samples_split = range(2,15)
	min_samples_leaf = range(1,15)
	max_leaf_nodes = range(2,15)
	param_grid = dict(criterion=criterion, splitter=splitter, max_depth=maxDepth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)
	grid = RandomizedSearchCV(tree, param_grid, cv=10, scoring='accuracy', n_iter=10, random_state=5)
	grid.fit(X, y)
	print grid.best_score_
	print grid.best_params_

"""Renders an image of the decision tree, outputs to a pdf file.
	RETURN VALUE: None"""
def displayTree(classifier):
	dot_data = sklearn.tree.export_graphviz(classifier, out_file=None)
	graph = graphviz.Source(dot_data)
	graph.render("tree")

"""Plot the data curve with a vanillaTree
	RETURN VALUE: None"""
def treeDepthPlot(depth, func):
	X, y = func()
	train_score, test_scores, depths = treeVary(X, y, depth)
	plt.plot(train_score, 'r--', test_scores, '--b')
	plt.axis([0,depth,0,1])
	red_patch = mpatches.Patch(color='red', label='Training Accuracy')
	blue_patch = mpatches.Patch(color='blue', label='Testing Accuracy')
	plt.legend(handles=[red_patch, blue_patch],bbox_to_anchor=(1,0.3))
	plt.show()


if __name__ == '__main__':
	api = API()
	X, y = api.getCMCData()

	api.plot_learning_curve(sklearn.tree.DecisionTreeClassifier(max_depth=3),
	 "Decision Tree Learning Curve CMC Dataset", X, y, cv=10, n_jobs=-1)
	api.plot_validation_curve(sklearn.tree.DecisionTreeClassifier(presort=True), X, y, (0.0,1.0), None,
	                         range(1,45), 'max_depth', 'Decision Tree Validation Curve CMC Dataset',
	                        'Max Depth', 'Accuracy', -1)


	####################
	#  hyperTuned Plots #
	####################
	api.plot_learning_curve(sklearn.tree.DecisionTreeClassifier(splitter='best', max_leaf_nodes=14, min_samples_leaf=3,
	                                           criterion="gini", min_samples_split=13),
	                        "Pruned Decision Tree Learning Curve CMC Dataset", X, y, cv=10, n_jobs=-1)
	api.plot_validation_curve(sklearn.tree.DecisionTreeClassifier(splitter='best', max_leaf_nodes=14, min_samples_leaf=3,
	                                           criterion="gini", min_samples_split=13), X, y, (0.0,1.0), None,
	                         range(1,15), 'max_depth', 'Decision Tree Validation Curve CMC Dataset',
	                        'Max Depth', 'Accuracy', -1)



	X, y = api.getSCData()
	api.plot_learning_curve(sklearn.tree.DecisionTreeClassifier(max_depth=3),
	                        "Decision Tree Learning Curve SC Dataset", X, y, cv=10, n_jobs=-1)
	api.plot_validation_curve(sklearn.tree.DecisionTreeClassifier(presort=True), X, y, (0.0, 1.0), None,
	                          range(1, 45), 'max_depth', 'Decision Tree Validation Curve SC Dataset',
	                          'Max Depth', 'Accuracy', -1)

	####################
	#  hyperTuned Plots #
	####################
	api.plot_learning_curve(sklearn.tree.DecisionTreeClassifier(splitter='best', max_leaf_nodes=14, min_samples_leaf=3,
	                                                            criterion="gini", min_samples_split=13),
	                        "Pruned Decision Tree Learning Curve SC Dataset", X, y, cv=10, n_jobs=-1)
	api.plot_validation_curve(
		sklearn.tree.DecisionTreeClassifier(splitter='best', max_leaf_nodes=14, min_samples_leaf=3,
		                                    criterion="gini", min_samples_split=13), X, y, (0.0, 1.0), None,
		range(1, 15), 'max_depth', 'Decision Tree Validation Curve SC Dataset',
		'Max Depth', 'Accuracy', -1)
	plt.show()

