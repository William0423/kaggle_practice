# coding: utf-8

from sklearn import datasets
import numpy as np
iris = datasets.load_iris()

X = iris["data"][:,3:] # 取最后一列特征；
print iris['feature_names']
# print iris['target_names']
# print iris['target']
y = (iris['target'] == 2).astype(np.int)
print y

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
# 训练模型：
log_reg.fit(X, y)

# 测试数据集：
X_new = np.linspace(0,3,1000).reshape(-1,1)
# print X_new
# 预测：
y_proba = log_reg.predict_proba(X_new) # 输入结果为概率
print y_proba

import matplotlib.pyplot as plt
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")

plt.show()





