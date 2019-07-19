from sklearn.datasets import load_boston
import os
from os.path import join
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

boston = load_boston()
print(boston.DESCR)
data = boston.data
label = boston.target

columns = boston.feature_names
print(label)
print(data)
print(columns)

# 데이터 프레임 변화
data = pd.DataFrame(data, columns=columns)
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())

# x변수 RM변수, y는 주택가격
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=2019)

sim_lr = LinearRegression()
sim_lr.fit(x_train['RM'].values.reshape((-1, 1)),y_train)
y_pred = sim_lr.predict(x_test['RM'].values.reshape((-1, 1)))

plt.scatter(x_test['RM'],y_test, s=10, c='black')
plt.plot(x_test['RM'], y_pred, c='red')
plt.legend(['Regression line', 'x_test'], loc='upper left')
plt.show()

# 다중회귀분석
mul_lr = LinearRegression()
mul_lr.fit(x_train.values, y_train)
y_pred = mul_lr.predict(x_test.values)
print('다중 선형 회귀, R2 : {:.4f}'.format(r2_score(y_test, y_pred)))

from mglearn.plots import plot_animal_tree
plot_animal_tree()
plt.show()
