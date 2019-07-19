from sklearn.tree import DecisionTreeRegressor
import Day2.boston as boston
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

# noe overfitting
dt_regr = DecisionTreeRegressor(max_depth=5)

# 모델 학습
dt_regr.fit(boston.x_train, boston.y_train)

# 예측
# y_pred = dt_regr.predict(boston.x_test['RM'].values.reshape(-1, 1))
#
# print("단순 결정 트리 회귀:{:.4f}".format(r2_score(boston.y_test, y_pred)))

import graphviz
from sklearn.tree import export_graphviz
export_graphviz(dt_regr, out_file='boston.dot',         # 학습모델, 파일
                class_names=boston.label,               # 라벨, 타겟, 종속변수
                feature_names=boston.columns,           # 컬럼
                impurity=False,                         # gini 미출력
                filled=True)                            # filled: node의 색깔을 다르게
with open('boston.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph)                        # dot_graph의 source 저장
dot.render(filename='boston.png')                       # png로 저장