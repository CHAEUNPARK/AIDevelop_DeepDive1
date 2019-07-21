import pandas as pd

train = pd.read_csv('./Titanic/train.csv')
test = pd.read_csv('./Titanic/test.csv')
gender_submission = pd.read_csv('./Titanic/gender_submission.csv')

del train['Cabin']

