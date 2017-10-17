"""
Author: Samuel Bretz (sbretz3)
Email: bretzsam@gmail.com
"""

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
import matplotlib as plt
from API import API

if __name__ == '__main__':
    api = API()
    X, y = api.getCMCData()
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8), algorithm="SAMME")

    api.plot_learning_curve(bdt, "Adaboost Learning Curve SC", X, y, n_jobs=-1)
    api.plot_validation_curve(bdt, X, y, (0.0,1.), None, range(1,25), 'n_estimators',
                              "Adaboost Validation Curve SC Weak Learner: Decision Tree (max depth = 8)", "# of Learners", "Accuracy", n_jobs=-1)

    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8), algorithm="SAMME")

    api.plot_learning_curve(bdt, "Adaboost Learning Curve CMC", X, y, n_jobs=-1)
    api.plot_validation_curve(bdt, X, y, (0.0, 1.), None, range(1, 25), 'n_estimators',
                              "Adaboost Validation Curve CMC Weak Learner: Decision Tree (max depth = 8)",
                              "# of Learners", "Accuracy", n_jobs=-1)
    X, y = api.getSCData()
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8), algorithm="SAMME")

    api.plot_learning_curve(bdt, "Adaboost Learning Curve SC", X, y, n_jobs=-1)
    api.plot_validation_curve(bdt, X, y, (0.0, 1.), None, range(1, 25), 'n_estimators',
                              "Adaboost Validation Curve SC Weak Learner: Decision Tree (max depth = 8)",
                              "# of Learners", "Accuracy", n_jobs=-1)

    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=8), algorithm="SAMME")

    api.plot_learning_curve(bdt, "Adaboost Learning Curve SC", X, y, n_jobs=-1)
    api.plot_validation_curve(bdt, X, y, (0.0, 1.), None, range(1, 25), 'n_estimators',
                              "Adaboost Validation Curve SC Weak Learner: Decision Tree (max depth = 8)",
                              "# of Learners", "Accuracy", n_jobs=-1)
    plt.show()