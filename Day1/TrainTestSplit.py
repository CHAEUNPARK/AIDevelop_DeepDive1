import numpy as np
from sklearn.model_selection import train_test_split

X = [[0,1],[2,3],[4,5],[6,7],[8,9]]
Y = [0,1,2,3,4]

# 데이터(X)만 넣었을 경우
X_train, X_test = train_test_split(X, test_size=0.2, random_state=123)

# 데이터(X)와 레이블(Y)을 넣었을 경우
