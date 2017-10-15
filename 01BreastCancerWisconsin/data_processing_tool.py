
# coding:utf-8


def pre_processing(train):
    train.drop(['Unnamed: 32', 'id'], inplace=True, axis=1)
    train['diagnosis'] = train['diagnosis'].map({'M': 1, 'B': 0})
    return train
