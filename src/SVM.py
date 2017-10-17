"""
Author: Samuel Bretz (sbretz3)
Email: bretzsam@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from API import API

if __name__ == '__main__':
	api = API()
	X, y = api.getSCData()
	XCMC, yCMC = api.getCMCData()
	api.plot_validation_curve(SVC(kernel='rbf', cache_size=7000, max_iter=10000), X, y, (0.0, 1.0), None,
	                          np.logspace(-6, -1, 8),
	                          'gamma', 'SVM Validation Curve SC kernel=rbf', '10^-$\gamma$', 'accuracy', -1)
	api.plot_learning_curve(SVC(kernel='rbf', cache_size=7000), "SVM Learning Curve SC kernel = linear", X, y, cv=None, n_jobs=-1)

	api.plot_validation_curve(SVC(kernel='poly', cache_size=7000, max_iter=10000), X, y, (0.0, 1.0), None,
	                          np.logspace(-6, -1, 8),
	                          'gamma', 'SVM Validation Curve SC kernel=poly', '10^-$\gamma$', 'accuracy', -1)
	api.plot_learning_curve(SVC(kernel='poly', cache_size=7000), "SVM Learning Curve SC kernel = linear", X, y, cv=None,
	                        n_jobs=-1)



	api.plot_validation_curve(SVC(kernel='rbf', cache_size=7000, max_iter=10000), XCMC, yCMC, (0.0, 1.0), None,
	                          np.logspace(-6, -1, 8),
	                          'gamma', 'SVM Validation Curve SC kernel=rbf', '10^-$\gamma$', 'accuracy', -1)
	api.plot_learning_curve(SVC(kernel='rbf', cache_size=7000), "SVM Learning Curve SC kernel = linear", XCMC, yCMC, cv=None,
	                        n_jobs=-1)

	api.plot_validation_curve(SVC(kernel='poly', cache_size=7000, max_iter=10000), XCMC, yCMC, (0.0, 1.0), None,
	                          np.logspace(-6, -1, 8),
	                          'gamma', 'SVM Validation Curve SC kernel=poly', '10^-$\gamma$', 'accuracy', -1)
	api.plot_learning_curve(SVC(kernel='poly', cache_size=7000), "SVM Learning Curve SC kernel = linear", XCMC, yCMC, cv=None,
	                        n_jobs=-1)

	plt.show()
