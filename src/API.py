"""
Author: Samuel Bretz (sbretz3)
Email: bretzsam@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection


class API():
	"""
	Function to prepare and clean data.
	RETURN VALUE: Tuple in the following form: (Training Data, Testing Data)
	"""

	def getCMCData(self):
		data = pd.read_csv("../data/cmc.csv")
		print data.shape
		X = data[
			['AGEW', 'EDUCATIONW', 'EDUCATIONH', 'NUMCHILDREN', 'RELIGION', 'WORKING', 'OCCUPATIONH', 'STANDOFLIVING',
			 'MEDIA']]
		y = data['CLASS']
		return (X, y)


	def getSCData(self):
		data = pd.read_csv("../data/SkillCraft1_Dataset.csv")
		X = data[['Age', 'HoursPerWeek', 'TotalHours', 'APM',
		          'SelectByHotkeys', 'AssignToHotkeys', 'UniqueHotkeys', 'MinimapAttacks', 'MinimapRightClicks',
		          'NumberOfPACs', 'GapBetweenPACs',
		          'ActionLatency', 'ActionsInPAC', 'TotalMapExplored', 'WorkersMade', 'UniqueUnitsMade',
		          'ComplexUnitsMade', 'ComplexAbilitiesUsed']]
		y = data['LeagueIndex']
		return (X, y)


	def plot_learning_curve(self, estimator, title, X, y, ylim=None, cv=None,
	                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = model_selection.learning_curve(
			estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
		         label="Training score")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
		         label="Cross-validation score")

		plt.legend(loc="best")
		return plt




	"""Plot validation curve.
		RETURN VALUE: sklearn plt object"""
	def plot_validation_curve(self, estimator, X, y, ylim, cv, param_range, paramName, title, x_label, y_label,
	                          n_jobs):
		plt.figure()
		plt.title(title)
		plt.ylim(ylim)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		train_scores, test_scores = model_selection.validation_curve(
			estimator, X, y, param_name=paramName, param_range=param_range, cv=cv, scoring='accuracy', n_jobs=n_jobs)
		train_scores_mean = np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()
		plt.plot(train_scores_mean, 'o-', color="r",
		         label="Training score")
		plt.plot(test_scores_mean, 'o-', color="g",
		         label="Cross-validation score")

		plt.legend(loc="best")
		return plt


if __name__ == '__main__':
	__package__ = '__API__'
