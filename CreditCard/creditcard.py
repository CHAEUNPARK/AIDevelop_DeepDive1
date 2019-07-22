import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 데이터 로드
data = pd.read_csv('./CreditCard/creditcard.csv')
print(data.head())
print(data.columns) #컬럼명 확인
# 데이터 빈도수 확인
# print(pd.value_counts(data['Class']))
# print(data.shape)
# pd.value_counts(data['Class']).plot.bar()
# plt.title('Fraud class histogram')
# plt.xlabel('Class')
# plt.ylabel('Frequency')
# plt.show()

y = data['Class']
X = data.drop(columns=['Time', 'Class'])
y = y.values.reshape((-1, 1))
print(X.shape)
print(y.shape)
# 데이터 train, test 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

# 데이터 불균형 맞추기

print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {}\n".format(sum(y_train==0)))
print("y_train", y_train)
print("y_train.ravel", y_train.ravel())

