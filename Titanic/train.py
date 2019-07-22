import pandas as pd

train = pd.read_csv('./Titanic/train.csv')
test = pd.read_csv('./Titanic/test.csv')
gender_submission = pd.read_csv('./Titanic/gender_submission.csv')

del train['Cabin']
del train['Ticket']
del test['Cabin']
del test['Ticket']

train = train.fillna({"Embarked" : "S"})

embarked_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)