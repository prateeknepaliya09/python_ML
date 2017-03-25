# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 15:04:21 2016

@author: Prateek
"""

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
print(clf_pf.predict([[-0.8, -1]]))

"""
#!/usr/bin/env python

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Input training data
training_points = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
training_labels = [1, 1, 1, 2, 2, 2]
X = np.array(training_points)
Y = np.array(training_labels)

# Create Naive Bayes classifier
clf = GaussianNB()
clf.fit(X, Y)

# Classify test data with the classifier
test_points = [[1, 1], [2, 2], [3, 3], [4, 3]]
test_labels = [2, 2, 2, 1]
predicts = clf.predict(test_points)

# Calculate Accuracy Rate manually
count = len(["ok" for idx, label in enumerate(test_labels) if label == predicts[idx]])
print "Accuracy Rate, which is calculated manually is: %f" % (float(count) / len(test_labels))

# Calculate Accuracy Rate by using accuracy_score()
print "Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(test_labels, predicts)
"""