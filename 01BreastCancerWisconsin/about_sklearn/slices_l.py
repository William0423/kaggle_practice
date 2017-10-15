import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold


# data = pd.Series(np.arange(20, 30), index=range(10))
# kf = KFold(len(data), n_folds=2)
# print data
# # print data.iloc[: 3, ]
# for train_in, test_in in kf:
#     print train_in, test_in
#     print data.values[train_in]


path = '../data/data.csv'
data = pd.read_csv(path, nrows=20)
prediction_var = ['radius_mean', 'perimeter_mean',
                  'area_mean', 'compactness_mean', 'concave points_mean']
print data.shape[0]
kf = KFold(data.shape[0], n_folds=5)
# print data.iloc[: 3, ]
for train_in, test_in in kf:
    print train_in, test_in
    print data[prediction_var].ix[train_in, :]
    # print data[prediction_var].loc[train_in, :]
    # print data[prediction_var].iloc[train_in, :] 
