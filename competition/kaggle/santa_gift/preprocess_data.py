# coding:utf-8

import numpy as np
import pandas as pd


def preprocessing_data(child_wishlist_data_path, gift_goodkids_data_path):
    child_wishlist_data = pd.read_csv(child_wishlist_data_path, header=None)
    gift_goodkids_data = pd.read_csv(gift_goodkids_data_path, header=None)

    print child_wishlist_data.info()
    print gift_goodkids_data.info()
    print gift_goodkids_data.describe()
    # print child_wishlist_data.head(5)
    # print child_wishlist_data.describe()

    # print gift_goodkids_data.head(5)
    # print child_wishlist_data[2]


if __name__ == '__main__':
    child_wishlist_data_path = './data/child_wishlist.csv'
    gift_goodkids_data_path = './data/gift_goodkids.csv'
    preprocessing_data(child_wishlist_data_path, gift_goodkids_data_path)
