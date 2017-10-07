# coding:utf-8

import pandas as pd

from sklearn.model_selection import train_test_split


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



def pre_processing(train):
	# print train.describe()
	# print train.info()
	# print train.head(5)

	# 判断每列是否为空：
	###### 统计每一列是否有空数据，如果有空值，应该如何处理？###########
	# 方法一：
	# total = train.isnull().sum().sort_values(ascending=False)
	# print total
	# 方法二：
	# trainInx =  train.columns
	# # print trainInx
	# for col in trainInx:
	# 	if train[col].isnull().sum() > 0:
	# 		print col
	# 		print train[col].isnull().sum()

	train['label'] = train['diagnosis'].apply(lambda x : (1 if x=='M' else 0))
	# 删除最后一列：
	train.drop(['Unnamed: 32','diagnosis', 'id'], inplace=True,axis=1)

	y = train['label'].values
	# y = train['label']
	print y
	train.drop(['label'], inplace=True,axis=1)
	x = train.values
	# x = train
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) # train_test_split()这个方法中参数的格式可以是numpy.ndarray和dataframe
	print type(X_train)
	print len(X_train), len(y_train), len(X_test), len(y_test)
	return X_train, X_test, y_train, y_test

# def train_model(train):

def bagging_train(X_train, X_test, y_train, y_test):
	print "===============start bagging==============="
	bag_clf = BaggingClassifier(
		DecisionTreeClassifier(), n_estimators=500,
		max_samples=100, bootstrap=True, n_jobs=-1 # 并行计算必须写作main方法里面，否则出错
	) # 有500个分类器，每个分类器最多有100个样本
	bag_clf.fit(X_train, y_train)
	y_pred = bag_clf.predict(X_test)
	print (bag_clf.__class__.__name__, accuracy_score(y_test, y_pred, normalize=True))
	return bag_clf

def tree_train(X_train, X_test, y_train, y_test):
	print "=============start tree=============="
	tree_clf = DecisionTreeClassifier(random_state=42)
	tree_clf.fit(X_train, y_train)
	y_pred_tree = tree_clf.predict(X_test)
	print (accuracy_score(y_test, y_pred_tree))
	return tree_clf

if __name__ == '__main__':
	path = './data/data.csv'
	train = pd.read_csv(path)
	X_train, X_test, y_train, y_test = pre_processing(train)
	bagging_train(X_train, X_test, y_train, y_test)
	tree_train(X_train, X_test, y_train, y_test)


	'''
总结，我直接使用了所有数据，没有了解每列数据的含义；没有分析各列之间的相关性进行降维
	'''

