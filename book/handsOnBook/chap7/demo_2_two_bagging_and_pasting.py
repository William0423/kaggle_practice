# coding:utf-8

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

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


def plot_classifier(tree_clf, bag_clf, X, y):
	plt.figure(figsize=(11, 4))
	plt.subplot(1,2,1)

	plot_decision_boundary(tree_clf, X, y)
	plt.title("Decision Tree", fontsize=14)
	plt.subplot(1,2,2)
	plot_decision_boundary(bag_clf, X, y)
	plt.title("Decision Trees with Bagging", fontsize=14)
	plt.show()

def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], alpha=0.5, contour=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap, linewidth=10)
    if contour:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=alpha)
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=alpha)
    plt.axis(axes)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18, rotation=0)

if __name__ == '__main__':
	X, y = make_moons(n_samples=500, noise=0.3, random_state=42) # random_state为随机数种子
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	bag_clf = bagging_train(X_train, X_test, y_train, y_test)
	tree_clf = tree_train(X_train, X_test, y_train, y_test)
	plot_classifier(tree_clf, bag_clf, X, y)