# coding:utf-8
from data_processing_tool import pre_processing
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
import numpy as np


def classification_model(model, data, prediction_var, outcome_var):
    print "============= %s ==============" % model.__class__.__name__
    model.fit(data[prediction_var], data[outcome_var])
    prediction_result = model.predict(data[prediction_var])

    accuracy = accuracy_score(data[outcome_var], prediction_result)
    print "the model accuracy: "
    print (model.__class__.__name__, "Accuracy : %s" %
           "{0:.3%}".format(accuracy))

    print "KFold result: "
    kf = KFold(data.shape[0], n_folds=5)  # data.shape[0]: all rows
    error = []
    for train_index, test_index in kf:
        # print train_index
        # different in KFold_l.py demo
        train_X = data[prediction_var].iloc[train_index, :]
        print train_X
        train_y = data[outcome_var].iloc[train_index]
        model.fit(train_X, train_y)

        test_X = data[prediction_var].iloc[test_index, :]
        test_y = data[outcome_var].iloc[test_index]

        # result of texs_X compare to test_y
        model_accurcy = model.score(test_X, test_y)
        error.append(model_accurcy)
        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))


if __name__ == '__main__':
    path = './data/data.csv'
    data = pd.read_csv(path)
    data = pre_processing(data)
    prediction_var = ['radius_mean', 'perimeter_mean',
                      'area_mean', 'compactness_mean', 'concave points_mean']
    # label
    outcome_var = 'diagnosis'

    model = DecisionTreeClassifier()
    classification_model(model, data, prediction_var, outcome_var)
