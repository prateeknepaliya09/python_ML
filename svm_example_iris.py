# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 01:13:08 2017

@author: Prateek
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
iris.data
X = iris.data[:,2:]
y = iris.target

sets=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False,tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
svc = svm.SVC(kernel='linear', C=1,gamma=0).fit(X, y)
svc2 = svm.SVC(kernel='rbf', C=1,gamma=0).fit(X, y)
svc3 = svm.SVC(kernel='rbf', C=1,gamma=10).fit(X, y)
svc.fit(X,y)
svc.score(X,y)

X[:,0]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

h=(x_max-x_min)/100
np.meshgrid(np.arange(x_min, x_max,h))

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
plt.subplot(1.2,1,1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy,Z, cmap='rainbow_r', alpha=0.8)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma',s=30)
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVM with Linear kernel')
plt.show()
