from Titanic import train as preTrain
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train = preTrain.train
test = preTrain.test

Y_train_data = pd.DataFrame(train['Survived'].values.reshape(-1,1), columns=['Survived'])
X_train_data = train.drop(['Survived'], axis=1)

X_train = X_train_data
y_train = Y_train_data

X_train['FareBand'] = X_train['FareBand'].astype('int64')

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=5)


X_test = test
X_test['FareBand'] = X_test['FareBand'].astype('int64')

# Y_test_data = pd.DataFrame(test['Survived'].values.reshape(-1,1), columns=['Survived'])
# X_test_data = test.drop(['Survived'], axis=1)
#
# X_test = X_test_data
# y_test = Y_test_data
dtrain = xgb.DMatrix(X_train, label=y_train)

param = {
    'max_depth' : 5,
    'eta' : 1,
    'objective' : 'binary:logistic'
}
param['nthread'] = 4
param['eval_metric'] = 'auc'
num_round = 10
# xgb.XGBClassifier

# xgb = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
#                         objective='multi:softprob', gamma=0,
#                         max_delta_step=0, subsample=0.9, colsample_bytree=0.9)
# xgb.fit(X_train,y_train)
# xgb.cv(param, dtrain, num_round, nfold=5,
#        metrics={'auc'}, seed=0, callbacks=[xgb.callback.print_evaluation(show_stdv=False)])

# bst = xgb.train(param, dtrain, num_round)
# y_pred = xgb.predict_proba()

# xgb_model = xgb.XGBClassifier(objective='multi:softprob',
#                               random_state=0,
#                               min_child_weight=5,
#                               max_depth=10,
#                               gamma=0.1)
xgb_model = xgb.XGBClassifier()
xgb_model.fit(x_train, y_train)

y_pred = xgb_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : {:.2f}%".format(accuracy*100))

