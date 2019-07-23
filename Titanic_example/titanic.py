import pandas as pd
import numpy as np

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



if __name__ == '__main__':
    # pie_chart('Sex')
    # pie_chart('Pclass')
    # pie_chart('Embarked')
    # bar_chart('SibSp')
    # bar_chart('Parch')
    # bar_chart('Embarked')
    pass