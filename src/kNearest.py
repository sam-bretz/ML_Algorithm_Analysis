"""
Author: Samuel Bretz (sbretz3)
Email: bretzsam@gmail.com
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier as KNN
from API import API

"""Function for fitting a k-nearest neighbors classifier.
    RETURN VALUE: sklearn k-nearest neighbors object"""
def kNNClassifier(X, y, numNeighbors):
    knn = KNN(n_neighbors=numNeighbors)
    knn.fit(X, y)
    return knn

"""Determine the accuracy of the model using 10 fold cv.
    RETURN VALUE: Tuple of the following form: (train_score, test_score) where both are
    type List"""
def accuracyCV(classifier, X, y):
    scores = model_selection.cross_validate(classifier, X, y, cv=10, scoring='accuracy')
    return (scores['train_score'].mean(), scores['test_score'].mean())

"""Vary the number of neighbors used with the knn classifier
    RETURN VALUE: (list of training score, list of test scores, numNeighbors)"""
def knnVary(X, y, numNeighbors):
    train_scoreList = []
    test_scoreList = []
    neighbors = range(1, numNeighbors)
    for i in range(1, numNeighbors):
        print i
        classifier = kNNClassifier(X, y, numNeighbors)
        train_score, test_score = accuracyCV(classifier, X, y)
        train_scoreList.append(train_score)
        test_scoreList.append(test_score)
    return (train_scoreList, test_scoreList, numNeighbors)

"""Plot the data curve with a knn classifier
    RETURN VALUE: None"""
def knnPlot(numNeighbors, func):
    X, y = func()
    train_score, test_scores, numNeighbors = knnVary(X, y, numNeighbors)
    plt.plot(train_score, 'r--', test_scores, '--b')
    plt.axis([0,numNeighbors,0,1])
    red_patch = mpatches.Patch(color='red', label='Training Accuracy')
    blue_patch = mpatches.Patch(color='blue', label='Testing Accuracy')
    plt.legend(handles=[red_patch, blue_patch],bbox_to_anchor=(1,0.3))
    plt.show()


if __name__ == '__main__':
    api = API()
    #show validation curve
    X, y = api.getSCData()
    api.plot_validation_curve(KNN(), X, y, (0.0,1.0), 10, range(1,25), 'n_neighbors',
                              'K-nn Validation Curve SC Data', '# of Neighbors', 'Accuracy', -1)
    X, y = api.getCMCData()
    api.plot_validation_curve(KNN(), X, y, (0.0, 1.0), 10, range(1, 25), 'n_neighbors',
                              'K-nn Validation Curve CMC Data', '# of Neighbors', 'Accuracy', -1)
    plt.show()
