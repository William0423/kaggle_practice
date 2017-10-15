# coding:utf-8


from sklearn.cross_validation import KFold
import numpy as np
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
kf = KFold(4, n_folds=2)
print len(kf)
print kf

# X_train, X_test = [], []
# y_train, y_test = [], []
for train_index, test_index in kf:
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    print X_train, X_test
    y_train, y_test = y[train_index], y[test_index]
    print y_train, y_test
