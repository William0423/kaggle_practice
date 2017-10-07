# coding:utf-8


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_moons
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def rf_bag_train(X_train, X_test, y_train, y_test):
	bag_clf = BaggingClassifier(
		DecisionTreeClassifier(splitter="random", max_leaf_nodes=16, random_state=42),
		n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1, random_state=42)
	bag_clf.fit(X_train, y_train)
	y_pred = bag_clf.predict(X_test)


	rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
	rnd_clf.fit(X_train, y_train)

	y_pred_rf = rnd_clf.predict(X_test)


	print np.sum(y_pred == y_pred_rf) / float(len(y_pred))  # almost identical predictions



def plot_digit(data):
	import matplotlib
	image = data.reshape(28, 28)
	plt.imshow(image, cmap = matplotlib.cm.hot, interpolation="nearest")
	plt.axis("off")

from sklearn.datasets.mldata import fetch_mldata
def learn_feature_importance():
'''
读取数据出错解决办法：http://scikit-learn.org/stable/datasets/mldata.html
'''
	mnist = fetch_mldata('MNIST original')
	rnd_clf = RandomForestClassifier(random_state=42)
	rnd_clf.fit(mnist["data"], mnist["target"])

	plot_digit(rnd_clf.feature_importances_)

	cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
	cbar.ax.set_yticklabels(['Not important', 'Very important'])
	plt.show()

if __name__ == '__main__':
	# X, y = make_moons(n_samples=500, noise=0.3, random_state=42) # random_state为随机数种子
	# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
	# rf_bag_train(X_train, X_test, y_train, y_test)

	learn_feature_importance()