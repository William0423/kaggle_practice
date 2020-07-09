#coding: utf-8
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

if __name__ == '__main__':
    hour_list = []
    hour = datetime.datetime.now().hour
    print("10".zfill(2))
    for i in range(hour+1, 24):
        print(str(i))
