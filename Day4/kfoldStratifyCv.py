from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np

iris = load_iris()
kf_data = iris.keys()
print("<< kf_data >>")
print(kf_data)

kf_data = iris.data
kf_label = iris.target
kf_columns = iris.feature_names

# alt + shift + E
kf_data = pd.DataFrame(kf_data, columns=kf_columns)
print("<< kf_label >>")
print(pd.value_counts(kf_label))
print(kf_label.sum())
print(kf_label.dtype)

def Kfold():
    kf = KFold(n_splits=5, random_state=0)
    # split()는 학습용과 검증용의 데이터 인덱스 출력
    for i, (train_idx, valid_idx) in enumerate(kf.split(kf_data.values, kf_label)):
        train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
        valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]

        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))

def Stratified_KFold():
    kf = StratifiedKFold(n_splits=5, random_state=0)
    for i, (train_idx, valid_idx) in enumerate(kf.split(kf_data.values, kf_label)):
        train_data, train_label = kf_data.values[train_idx, :], kf_label[train_idx]
        valid_data, valid_label = kf_data.values[valid_idx, :], kf_label[valid_idx]

        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))
        # print("Cross Validation Score:{:.2f}%".format(np.mean(val_scores)))

if __name__ == '__main__':
    Kfold()