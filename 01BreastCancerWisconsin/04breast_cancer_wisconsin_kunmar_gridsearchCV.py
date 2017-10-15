
from data_processing_tool import pre_processing
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def classification_model_gridsearchCV(model, param_grid, data_X, data_y):
    clf = GridSearchCV(model, param_grid, cv=10, scoring="accuracy")
    clf.fit(data_X, data_y)
    print("The best parameter found on development set is :")
    # this will gie us our best parameter to use
    print(clf.best_params_)
    print("the bset estimator is ")
    print(clf.best_estimator_)
    print("The best score is ")
    # this is the best score that we can achieve using these parameters#
    print(clf.best_score_)


if __name__ == '__main__':

    path = './data/data.csv'
    data = pd.read_csv(path)
    data = pre_processing(data)
    prediction_key = ['radius_mean', 'perimeter_mean',
                      'area_mean', 'compactness_mean', 'concave points_mean']
    label_key = "diagnosis"
    data_X = data[prediction_key]
    data_y = data[label_key]

    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    model = DecisionTreeClassifier()
    classification_model_gridsearchCV(model, param_grid, data_X, data_y)

    model = KNeighborsClassifier()
    k_range = list(range(1, 30))
    leaf_size = list(range(1, 30))
    weight_options = ['uniform', 'distance']
    param_grid = {'n_neighbors': k_range,
                  'leaf_size': leaf_size, 'weights': weight_options}
    classification_model_gridsearchCV(model, param_grid, data_X, data_y)
