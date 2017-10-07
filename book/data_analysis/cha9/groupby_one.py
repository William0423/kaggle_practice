# coding:utf-8

import pandas as pd
import numpy as np

def demo_one():
	df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],'key2' : ['one', 'two', 'one', 'two', 'one'],'data1' : np.random.randn(5),'data2' : np.random.randn(5)})
	print df
	grouped = df['data1'].groupby(df['key1'])
	print grouped.size()


if __name__ == '__main__':
	demo_one()