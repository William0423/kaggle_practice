# coding:utf-8

from numpy.random import randn
import matplotlib.pyplot as plt

def plot_one():

	########### 第一种  ########
	# fig = plt.figure()
	# ax1 = fig.add_subplot(2,2,1)
	# ax2 = fig.add_subplot(2,2,2)
	# ax1.hist(randn(100), bins = 100 , color = 'blue' , alpha = 0.8) # alpha = 0.8表示颜色深度  
	# plt.show()

	############对应的第二种作图方法
	# fig, axes = plt.subplots(2,2, sharex=True, sharey=True) # 生成2x2的图
	# for i in range(2):
	# 	for j in range(2):
	# 		axes[i, j].hist(randn(500), bins = 50, color='k', alpha=0.5)
	# plt.subplots_adjust(wspace=1, hspace=1) # 第一个
	# plt.show()

	########## 标题、轴标签、刻度、刻度标签、
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.hist(randn(1000).cumsum())
	ticks = ax.set_xticks([0,250,500,750,1000]) # 刻度
	labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five']) # 刻度对应的标题名称
	ax.set_title('My first matplotlib plot')
	ax.set_xlabel('Stages')
	plt.show()


	

if __name__ == '__main__':
	print "#############start#######"
	plot_one()
	print "=============end=============="