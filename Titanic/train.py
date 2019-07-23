import pandas as pd
import numpy as np
#  data load
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
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

# Age Preprocessing
train['Age'] = train['Age'].fillna(-0.5)
test['Age'] = test['Age'].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

for x in range(len(train['AgeGroup'])):
    if train['AgeGroup'][x] == "Unknown":
        train['AgeGroup'][x] = age_title_mapping[train["Title"][x]]

for x in range(len(test['AgeGroup'])):
    if test['AgeGroup'][x] == "Unknown":
        test['AgeGroup'][x] = age_title_mapping[test["Title"][x]]

age_mapping = {'Baby' : 1, 'Child' : 2, 'Teenager' : 3, 'Student' : 4, 'Young Adult' : 5, 'Adult' : 6, 'Senior' : 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)

# Fare Preprocessing
train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])

train = train.drop(['Fare'], axis=1)
test = test.drop(['Fare'], axis=1)
