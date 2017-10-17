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

class dataVis():
	""" Class for making data visualization plots."""

	def boxPlot(self, data):
		data = pd.read_csv("../data/" + data + ".csv")
		plt.style.use('seaborn-dark')
		data.boxplot()

	def histogram(self, data):
		data = pd.read_csv("../data/" + data + ".csv")
		data.hist()

if __name__ == '__main__':
	__package__ == '__dataVis__'