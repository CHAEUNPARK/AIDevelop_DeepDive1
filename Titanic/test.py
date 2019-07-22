from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from Titanic import train as preTrain

train = preTrain.train
test = preTrain.test

print(train.shape)
print(test.shape)

Y_train_data = pd.DataFrame(train['Survived'].values.reshape(-1,1), columns=['Survived'])
X_train_data = train.drop(['Survived'], axis=1)

print(X_train_data.shape)
print(Y_train_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train_data, Y_train_data, test_size=0.3, random_state=0)

classifier = RandomForestClassifier(n_estimators=500, random_state=5)

all_accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=5)
print(all_accuracies)