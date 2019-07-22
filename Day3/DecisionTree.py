from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import graphviz
from sklearn.tree import export_graphviz
wine = load_wine()
print(wine.DESCR)

data = wine.data
label = wine.target
# columns = wine.feature_names

x_train, x_test, y_train, y_test = train_test_split(data, label, stratify=label, random_state=0)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train, y_train)
score_tr = tree.score(x_train, y_train)
score_te = tree.score(x_test, y_test)
print('DT훈련 세트 정확도{:.3f}'.format(score_tr))
print('DT테스트 세트 정확도{:.3f}'.format(score_te))

tree1 = DecisionTreeClassifier(max_depth=2, random_state=0)
tree1.fit(x_train, y_train)
score_tr1 = tree1.score(x_train, y_train)
score_te1 = tree1.score(x_test, y_test)
print('DT훈련 depth세트 정확도{:.3f}'.format(score_tr1))
print('DT테스트 depth세트 정확도{:.3f}'.format(score_te1))

export_graphviz(tree1, out_file='./Day3/tree1.dot',
                class_names=wine.target_names,
                feature_names=wine.feature_names,
                impurity=False,                 #gini 미출력
                filled=True)                    #filled : node의 색깔을 다르게

with open('./Day3/tree1.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph)                #dot_graph의 source 저장
dot.render(filename='./Day3/tree1.png')         #png로 저장