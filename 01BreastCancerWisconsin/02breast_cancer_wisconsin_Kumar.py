#coding:utf-8

import pandas as pd
import seaborn as sns # used for plot interactive graph. I like it most for plot
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier # for random forest classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm # for Support Vector Machine




def pre_processing(train):
	# train['label'] = train['diagnosis'].apply(lambda x : (1 if x=='M' else 0))
	# 删除最后一列：
	train.drop(['Unnamed: 32', 'id'], inplace=True,axis=1)
	# 第二章标签二值转化的方法
	train['diagnosis']=train['diagnosis'].map({'M':1,'B':0})
	# print train.head(5)
	# print train.columns
	features_mean= list(train.columns[1:11])
	features_se= list(train.columns[11:20])
	features_worst=list(train.columns[21:31])
	print(features_mean)
	print("-----------------------------------")
	print(features_se)
	print("------------------------------------")
	print(features_worst)

	##########   统计两类标签的总数:对于类别数量相差比较大的情况，在做交叉验证的时候注意的问题？？？   ##################
	# lets get the frequency of cancer stages
	# sns.countplot(train['diagnosis'],label="Count")
	# plt.show()

	############     进行相关性分析，以达到降维的目的      ###############
	corrmat = train[features_mean].corr()
	plt.subplots(figsize=(14,14))
	sns.heatmap(corrmat,square=True , annot=True, fmt= '.2f',annot_kws={'size': 15}, xticklabels= features_mean, yticklabels= features_mean,cmap= 'coolwarm')
	plt.xticks(rotation='90')
	plt.yticks(rotation=0)
	# plt.show()

	return train

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
	print (tree_clf.__class__.__name__, accuracy_score(y_test, y_pred_tree))

	############     查看最重要的特征  ###########
	# featimp = pd.Series(tree_clf.feature_importances_, index=features_mean).sort_values(ascending=False)
	# print(featimp) # this is the property of Random Forest classifier that it provide us the importance

	return tree_clf

def rand_train(X_train, X_test, y_train, y_test):
	print "=============rand_train =============="
	randoem_clf=RandomForestClassifier(n_estimators=100)# a simple random forest model
	randoem_clf.fit(X_train, y_train)
	y_pred_tree = randoem_clf.predict(X_test)
	print (randoem_clf.__class__.__name__, accuracy_score(y_test, y_pred_tree))

	############     查看最重要的特征  ###########
	# featimp = pd.Series(randoem_clf.feature_importances_, index=features_mean).sort_values(ascending=False)
	# print(featimp) # this is the property of Random Forest classifier that it provide us the importance 
	
	return randoem_clf


def svm_train(X_train, X_test, y_train, y_test):
	print "=============svm_train =============="
	svm_clf=svm.SVC()
	svm_clf.fit(X_train, y_train)
	y_pred_tree = svm_clf.predict(X_test)
	print (svm_clf.__class__.__name__, accuracy_score(y_test, y_pred_tree))
	return svm_clf


def plot_train_data(train, features_mean):
	color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B
	colors = train["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column
	pd.scatter_matrix(train[features_mean], c=colors, alpha = 0.5, figsize = (15, 15)); #将df各列分别组合绘制散点图
	plt.xticks(rotation='90')
	plt.yticks(rotation=0)
	plt.show()


if __name__ == '__main__':

	path = './data/data.csv'
	'''
Here Mean means the means of the all cells, standard Error of all cell and worst means the worst cell
	三类数据：所有细胞的平均值，所有细胞的标准值，坏细胞的平均值
	'''

	train = pd.read_csv(path)
	train = pre_processing(train)

	prediction_var = ['perimeter_mean', 'texture_mean','smoothness_mean','compactness_mean','symmetry_mean']
	features_mean = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']
	# features_se = ['radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se']
	# features_worst = ['radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
	# all_meanvalue = True

	plot_train_data(train, features_mean)

	train_data, test_data = train_test_split(train, test_size=0.3, random_state=0) # train_test_split()这个方法中参数的格式可以是numpy.ndarray和dataframe
	########### 对于mean，选取 #########
	# col_list = ['radius_mean','texture_mean','smoothness_mean','compactness_mean','concavity_mean','symmetry_mean','fractal_dimension_mean']
	####     参考答案选取的：0.68以上的都剔除  ##########




	# if all_meanvalue:
	# 	prediction_var = features_mean

	########### 选取随机森林中最重要的五个特征：############
	# prediction_var=['concave points_mean','perimeter_mean' , 'concavity_mean' , 'radius_mean','area_mean']

	#### 最重要的5个不相关的特征 ##########
	# prediction_var = ['concave points_mean', 'texture_mean','smoothness_mean','compactness_mean','symmetry_mean']

	X_train = train_data[prediction_var]# taking the training data input 
	y_train=train_data.diagnosis# This is output of our training data
	X_test= test_data[prediction_var] # taking test data inputs
	y_test =test_data.diagnosis   #output value of test dat


	# bagging_train(X_train, X_test, y_train, y_test)
	# tree_train(X_train, X_test, y_train, y_test)
	# rand_train(X_train, X_test, y_train, y_test)
	# svm_train(X_train, X_test, y_train, y_test)