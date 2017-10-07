# coding:utf-8

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
	'''
	axes：横坐标和纵坐标其实和结束刻度
	'''
	plt.plot(X[:, 0], y, data_style, label=data_label) # 原始点

	x1 = np.linspace(axes[0], axes[1], 500) # 指定区间返回间隔的数字：-0.5~0.5
	y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
	plt.plot(x1, y_pred, style, linewidth=2, label=label) # 预测点绘图

	if label or data_label:
		plt.legend(loc="upper center", fontsize=16)
	plt.axis(axes)

def simple_tree_train_plot(X, y):
	################ 注意y1,y2,y3的取值  #############
	# 训练1：
	tree_reg1 = DecisionTreeRegressor(max_depth=2)
	tree_reg1.fit(X, y)
	# 训练2：
	y2 = y-tree_reg1.predict(X)
	tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
	tree_reg2.fit(X, y2)
	y_pred = sum(tree.predict(X) for tree in (tree_reg1, tree_reg2))
	# 训练3：
	y3 = y2 - tree_reg2.predict(X)
	tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
	tree_reg3.fit(X, y3)

	# 测试样例
	# X_new = np.array([[0.8]])
	# print tree_reg1.predict(X_new)
	# print tree_reg2.predict(X_new)
	# print tree_reg3.predict(X_new)
	# # 预测结果
	# y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
	print y_pred

	plt.subplot(3,2,1)
	x1 = np.linspace(-0.5, 0.5, 500)
	plt.plot(X[:, 0],y2,"g-", color='blue')  
	plt.plot(X, tree_reg2.predict(X),color='red',linewidth=2)  
	plt.axes([-0.5, 0.5, -0.1, 0.8])

	###########作图 ##########
	# plt.figure(figsize=(11,11))
	# plt.subplot(3,2,1)
	# plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
	# plt.ylabel("$y$", fontsize=16, rotation=0)
	# plt.title("Residuals and tree predictions", fontsize=16)

	# plt.subplot(3,2,2)
	# plot_predictions([tree_reg1], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
	# plt.ylabel("$y$", fontsize=16, rotation=0)
	# plt.title("Ensemble predictions", fontsize=16)

	plt.subplot(3,2,3)
	plot_predictions([tree_reg2], X, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
	plt.ylabel("$y - h_1(x_1)$", fontsize=16)

	# plt.subplot(3,2,4)
	# plot_predictions([tree_reg1, tree_reg2], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
	# plt.ylabel("$y$", fontsize=16, rotation=0)

	# plt.subplot(3,2,5)
	# plot_predictions([tree_reg3], X, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
	# plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
	# plt.xlabel("$x_1$", fontsize=16)

	# plt.subplot(3,2,6)
	# plot_predictions([tree_reg1, tree_reg2, tree_reg3], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
	# plt.xlabel("$x_1$", fontsize=16)
	# plt.ylabel("$y$", fontsize=16, rotation=0)

	plt.show()


########### 画图 ###########
# plt.scatter(X,y,color='blue')  
# # plt.plot(X, tree_reg1.predict(X),color='red',linewidth=4)  
# plt.xticks(())  
# plt.yticks(())  
# plt.show()





##################使用集成方法：#############
def GB_train_plot(X, y):
	from sklearn.ensemble import GradientBoostingRegressor
	gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
	gbrt.fit(X, y)

	gbrt_slow = GradientBoostingRegressor(max_depth=2, n_estimators=200, learning_rate=0.1, random_state=42)
	gbrt_slow.fit(X, y)

	plt.figure(figsize=(11,4))

	plt.subplot(1,2,1)
	plot_predictions([gbrt], X, y, axes=[-0.5, 0.5, -0.1, 0.8], label="Ensemble predictions")
	plt.title("learning_rate={}, n_estimators={}".format(gbrt.learning_rate, gbrt.n_estimators), fontsize=14)

	plt.subplot(122)
	plot_predictions([gbrt_slow], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
	plt.title("learning_rate={}, n_estimators={}".format(gbrt_slow.learning_rate, gbrt_slow.n_estimators), fontsize=14)

	plt.show()

if __name__ == '__main__':
	np.random.seed(42) # 使得随机数据可预测：http://blog.csdn.net/xylin1012/article/details/71931900?utm_source=itdadao&utm_medium=referral
	# 数据：
	X = np.random.rand(100, 1) - 0.5
	print X[:, 0]
	y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
	simple_tree_train_plot(X, y)
	# GB_train_plot(X, y)