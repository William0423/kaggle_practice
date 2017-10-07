# coding: utf-8

import pandas as pd 

def demo_one():
	date = pd.date_range('2017-05-14', periods=10, freq='1h')
	print date
	
if __name__ == '__main__':
	demo_one()