from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from Titanic import train as preTrain
from sklearn.model_selection import GridSearchCV

train = preTrain.train
test = preTrain.test

# print(train.shape)
# print(test.shape)

Y_train_data = pd.DataFrame(train['Survived'].values.reshape(-1,1), columns=['Survived'])
X_train_data = train.drop(['Survived'], axis=1)

# print(X_train_data.shape)
# print(Y_train_data.shape)

# X_train, X_test, y_train, y_test = train_test_split(X_train_data, Y_train_data, test_size=0, random_state=0)

X_train = X_train_data
y_train = Y_train_data
classifier = RandomForestClassifier(n_estimators=200, random_state=0)

all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
print(all_accuracies)


grid_param = {
    'n_estimators' : [50, 100, 300, 500, 800, 1000],
    'criterion' : ['gini', 'entropy'],
    'bootstrap' : [True, False],
}

gs = GridSearchCV(estimator=classifier,
                  param_grid=grid_param,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs.fit(X_train, y_train)

print(gs.best_params_)
print(gs.best_score_)

