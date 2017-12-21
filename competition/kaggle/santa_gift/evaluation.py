# coding:utf-8
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "./data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


import pandas as pd
import numpy as np
from collections import Counter

n_children = 1000000  # n children to give
n_gift_type = 1000  # n types of gifts available
n_gift_quantity = 1000  # each type of gifts are limited to this quantity
n_gift_pref = 10  # number of gifts a child ranks
n_child_pref = 1000  # number of children a gift ranks
# 0.4% of all population, rounded to the closest even number
twins = int(0.004 * n_children)  # 4000
ratio_gift_happiness = 2
ratio_child_happiness = 2


gift_pref = pd.read_csv('./data/child_wishlist.csv',
                        header=None).drop(0, 1).values
child_pref = pd.read_csv('./data/gift_goodkids.csv',
                         header=None).drop(0, 1).values


def avg_normalized_happiness(pred, child_pref, gift_pref):

    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= n_gift_quantity

    # check if twins have the same gift
    for t1 in range(0, twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        assert twin1[1] == twin2[1]

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)

    for row in pred:
        child_id = row[0]
        gift_id = row[1]

        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0
        assert gift_id >= 0

        # ChildHappiness = 2 * GiftOrder if the gift is found in the wish list of the child.
        # GiftOrder为该孩子想要的10种礼物的位置，比如一位孩子想要的礼物为[5,2,3,1,4]，现在给的礼物是3,那么礼物的下表是2；
        # 如果给的礼物是4,那么礼物的下标是4。所以，对分母固定的请情况下，下标越远，child_happiness的值越小
        child_happiness = (
            n_gift_pref - np.where(gift_pref[child_id] == gift_id)[0]) * ratio_child_happiness
        if len(np.where(gift_pref[child_id] == gift_id)[0]) != 0:
            print n_gift_pref - np.where(gift_pref[child_id] == gift_id)[0]
        if not child_happiness:
            child_happiness = -1

        # GiftHappiness = 2 * ChildOrder if the child is found in the good kids list of the gift.
        gift_happiness = (
            n_child_pref - np.where(child_pref[gift_id] == child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness

    # print(max_child_happiness, max_gift_happiness
    print('normalized child happiness=', float(total_child_happiness) / (float(n_children) * float(max_child_happiness)),
          ', normalized gift happiness', np.mean(total_gift_happiness) / float(max_gift_happiness * n_gift_quantity))

    # my goal is to maximize the: Average Normalized Happiness (ANH) = AverageNormalizedChildHappiness (ANCH) + AverageNormalizedSantaHappiness (ANSH)
    return float(total_child_happiness) / (float(n_children) * float(max_child_happiness)) + np.mean(total_gift_happiness) / float(max_gift_happiness * n_gift_quantity)


# the value is [childid, giftid]
random_sub = pd.read_csv('./data/sample_submission_random.csv').values.tolist()

print(avg_normalized_happiness(random_sub, child_pref, gift_pref))
