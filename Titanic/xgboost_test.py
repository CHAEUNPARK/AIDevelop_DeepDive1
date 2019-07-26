from Titanic import train as preTrain
from sklearn.model_selection import GridSearchCV
import pandas as pd
import joblib
import xgboost as xgb

train = preTrain.train
test = preTrain.test

Y_train_data = pd.DataFrame(train['Survived'].values.reshape(-1,1), columns=['Survived'])
X_train_data = train.drop(['Survived'], axis=1)

X_train = X_train_data
y_train = Y_train_data

X_train['FareBand'] = X_train['FareBand'].astype('int64')
dtrain = xgb.DMatrix(X_train, label=y_train)

param = {
    'max_depth' : 5,
    'eta' : 1,
    'objective' : 'binary:logistic'
}
param['nthread'] = 4
param['eval_metric'] = 'auc'
num_round = 10
xgb.XGBClassifier
# xgb.cv(param, dtrain, num_round, nfold=5,
#        metrics={'auc'}, seed=0, callbacks=[xgb.callback.print_evaluation(show_stdv=False)])

bst = xgb.train(param, dtrain, num_round)