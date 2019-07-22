import pandas as pd
#  data load
train = pd.read_csv('./Titanic/train.csv')
test = pd.read_csv('./Titanic/test.csv')
gender_submission = pd.read_csv('./Titanic/gender_submission.csv')

# remove columns
del train['Cabin']
del train['Ticket']
del test['Cabin']
del test['Ticket']

# fill missing data
train = train.fillna({"Embarked" : "S"})

# mapping
embarked_mapping = {"S":1, "C":2, "Q":3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)


# Title Preprocessing
combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major',
                                                 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)


# Name Drop
del train['Name']
del train['PassengerId']
del test['Name']
combine = [train, test]

# Sex Preprocessing
sex_mapping = {"male" : 0, "female" : 1}
for dataset in combine :
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

