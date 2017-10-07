# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt


def demo_zero_practic():
	heads_proba = 0.51 # 问题，加上这个参数和不加有什么区别？
	coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32) # 生成10000行x10列0~1之间的随机数，然后和0.51进行比较返回true或者false，然后转化成数字
	# coin_tosses = np.random.rand(10000, 10) # 生成10000行x10列0~1之间的随机数，然后和0.51进行比较返回true或者false，然后转化成数字
	coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
	cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0) / np.arange(1, 10001).reshape(-1, 1)
	cumulative_heads_ratio = np.cumsum(coin_tosses, axis=0).astype(float) / np.arange(1, 10001).reshape(-1,1).astype(float) # np.arange(1, 10001).reshape(-1,1)
	# for i in coin_tosses:
	# 	print i

	# print "######################"
	# print "######################"
	# for i in np.cumsum(coin_tosses, axis=0):
	# 	print i

	# for i in cumulative_heads_ratio:
	# 	print i

	plt.figure(figsize=(8,3.5))
	plt.plot(cumulative_heads_ratio)
	plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
	plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
	plt.xlabel("Number of coin tosses")
	plt.ylabel("Heads ratio")
	plt.legend(loc="lower right") # 把图例放在右下角
	plt.axis([0, 10000, 0.42, 0.58])
	plt.show()


def np_practice():
	# a = np.array([[1,2,3], [4,5,6], [7,8,9]])
	# print a
	# print np.cumsum(a, axis=0)

	print float(5141)/float(10000)
	# print np.arange(1, 10001).reshape(-1,1)

if __name__ == '__main__':
	# np_practice()
	demo_zero_practic()