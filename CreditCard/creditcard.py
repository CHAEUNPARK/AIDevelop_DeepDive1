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


# #####################################################################################
# y = data['Class']
# X = data.drop(columns=['Time', 'Class'])
# y = y.values.reshape((-1, 1))
# print(X.shape)
# print(y.shape)
# ######################################################################################

#  amount standardscaler 전처리
sdscaler = StandardScaler()
data['normAmount'] = sdscaler.fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1) #불필요한 컬럼 삭제
print(data.head())

X = np.array(data.ix[:, data.columns != 'Class'])  #독립변수
y = np.array(data.ix[:, data.columns == 'Class'])  #종속변수

print('Shape of X: {}'.format(X.shape))
print('Shape of y: {}'.format(y.shape))
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

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {}'.format(y_train_res.shape))

print("After OverSampling, counts of y_train_res '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of y_train_res '0': {}".format(sum(y_train_res==0)))

print("After OverSampling, counts of test_X: {}".format(X_test.shape))
print("After OverSampling, counts of test_y: {}".format(y_test.shape))

# 실제 정확도를 알아보기 위한 새로운 데이터 갯수
print("before OverSampling, counts of label '1': {}".format(sum(y_test==1)))
print("before OverSampling, counts of label '0': {}".format(sum(y_test==0)))

