import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
# from util.logfile import logger
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()                  # 붓꽃 데이터셋은 꽃받침의 길이, 너비 꽃잎의 길이, 너비인 4개의 변수, 3개의 붓꽃 종을 라벨데이터

# 데이터 입력
data = iris.data
label = iris.target
columns = iris.feature_names

data = pd.DataFrame(data, columns=columns)

# 모델 정확도 확인

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True, stratify=label, random_state=2019)

lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

# 커널 linear(선형), ploy(다항식), RBF(방사기저), Hyper-tangent(쌍곡선 탄젠트 함수)
svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

##################################### dt ###########################################

dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
dt_y_pred = dt.predict(x_test)

# 데이터프레임 만들기
df = pd.DataFrame(dt.feature_importances_.reshape((1, -1)),columns=columns, index=['feature_importnaces_'])

