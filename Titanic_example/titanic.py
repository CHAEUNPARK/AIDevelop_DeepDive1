import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print('train data shape: ', train.shape)
print('test data shape: ', test.shape)
print('-----------train info------------')
print(train.info())
print('-----------test info ------------')
print(test.info())

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    print("feature_size\n", feature_size)
    print("feature_index\n", feature_index)
    print("survived count\n", survived)
    print("dead count\n", dead)

    plt.plot()
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead     = train[train['Survived'] == 0][feature].value_counts()
    df       = pd.DataFrame([survived, dead])
    print("survived", survived)
    print("dead", dead)
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True)
    plt.show()

# Preprocessing
# Sex
dataset_ = [train, test]
for dataset in dataset_:
    dataset['Sex'] = dataset['Sex'].astype(str)

# Embarked Feature
print("train.isnull().sum()", train.isnull().sum())
train['Embarked'].value_counts(dropna=False)

for dataset in dataset_:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)

print("train.isnull().sum()", train.isnull().sum())

# Age
for dataset in dataset_:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)

print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

for dataset in dataset_:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0: 'Child', 1: 'Young', 2: 'Middle', 3: 'Prime', 4: 'Old'}).astype(str)

# sibsp&Parch Feature
for dataset in dataset_:
    dataset['Family'] = dataset["Parch"] + dataset['SibSp']
    dataset['Family'] = dataset['Family'].astype(int)

feature_drop = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(feature_drop, axis=1)
test = test.drop(feature_drop, axis=1)

train = train.drop(['PassengerId', 'AgeBand'], axis=1)

print(train.head())
print(test.head())
print("train.isnull().sum()", train.isnull().sum())

train = pd.get_dummies(train)
test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)
test_data = test.drop('PassengerId', axis=1).copy()

def train_and_test(model, train_data, train_label):
    from sklearn.metrics import accuracy_score
    X_train, X_test, Y_train, Y_test = train_test_split(train_data, train_label, test_size=0.2, shuffle=True, random_state=5)
    model.fit(X_train, Y_train)
    prediction = model.predict(X_test)
    accuracy = round(accuracy_score(Y_test, prediction) * 100, 2)
    print("Accuracy: ", model, accuracy, "%")
    return prediction

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

log_pred = train_and_test(LogisticRegression(), train_data, train_label)
svm_pred = train_and_test(SVC(), train_data, train_label)
rf_pred = train_and_test(RandomForestClassifier(n_estimators=100), train_data, train_label)


# if __name__ == '__main__':
#     # pie_chart('Sex')
#     # pie_chart('Pclass')
#     # pie_chart('Embarked')
#     # bar_chart('SibSp')
#     # bar_chart('Parch')
#     # bar_chart('Embarked')
#     pass