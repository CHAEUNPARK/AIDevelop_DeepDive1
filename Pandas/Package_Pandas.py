import numpy as np
from scipy.stats import mode
import statistics as sta
import pandas as pd

def p_dataframe():
    x = np.array([-2.1, -1, 1, 1, 4.3])
    print(np.mean(x))
    print(np.median(x))
    print(mode(x))

    x_m = np.mean(x)
    x_a = x - x_m
    x_p = np.power(x_a, 2)

    print(" Variance x ")
    print(np.var(x))
    print(sta.pvariance(x))
    print(sta.variance(x))

def p_dict():
    obj1 = pd.DataFrame(data = np.arange(16).reshape(4,4), index=['a', 'b', 'c', 'd'], columns=['a', 'b', 'c', 'd'])
    print(obj1.index)
    print(obj1.columns)
    print(obj1.values)
    print(obj1.dtypes)

def add_dataframe():
    d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = pd.DataFrame(d)
    df.columns = ['Rev']
    print(df)

def copy_columns():
    d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = pd.DataFrame(d)
    df.columns = ['Rev']
    df['col'] = df['Rev']
    print(df)

def rm_columns():
    d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = pd.DataFrame(d)
    df.columns = ["Rev"]
    df['NewCol'] = 5
    df1 = df.drop("NewCol", axis = 1)
    print(df1)

def df_attribute():
    names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
    births = [968, 155, 77, 578, 973]
    BabyDataSet = list(zip(names, births))
    print(BabyDataSet)

    df = pd.DataFrame(data = BabyDataSet, columns = ['Names', 'Births'])
    print(df)
    print(df.shape)
    print(df.index)
    print(df.columns)
    print(df.axes)

def search_col():
    d = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df = pd.DataFrame(d)
    df.columns = ['Rev']
    df['col'] = df['Rev']
    i = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    df.index = i
    print(df['Rev'])
    print(df[['Rev', 'col']])

def check_null():
    df = pd.DataFrame(data=np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['f', 'g', 'h', 'i'])
    print(df)
    df2 = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    print(df2)
    print(df2['f'].isnull())
    print(df2['f'].notnull())

def df_replace():
    obj1 = pd.DataFrame(data=np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['a', 'b', 'c', 'd'])
    obj1.replace(to_replace = 0, value = 999, inplace = True)
    print(obj1)
    obj1.replace(to_replace = 2, value = 888, inplace = True)
    print(obj1)
    obj1['d'].replace(3, 777, inplace=True)
    print(obj1)

def df_replace1():
    obj1 = pd.DataFrame(data=np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['a', 'b', 'c', 'd'])
    obj1.replace(to_replace = (0, 1), value = 999, inplace = True)
    print(obj1)
    obj1.replace(to_replace = [3, 4, 5], value = 888, inplace = True)
    print(obj1)
    obj1['d'].replace((10, 11), 777, inplace=True)
    print(obj1)

def df_group():
    d = {'one': [1, 1, 1, 1, 1], 'two': [2, 2, 2, 2, 2], 'letter': ['a', 'a', 'b', 'b', 'c']}
    df1 = pd.DataFrame(d)
    print(df1)
    one = df1.groupby('letter')
    print(one)
    print(one.sum())

    letterone = df1.groupby(['letter', 'one']).sum()
    print(letterone)

    lettertwo = df1.groupby(['letter', 'one'], as_index=False).sum()
    print(lettertwo)
    print(lettertwo.index)

def describe():
    df = pd.DataFrame(np.arange(30).reshape(5, 6))
    print(df)
    print(df.describe())
    print(df[0].describe())

def operation():
    df = pd.DataFrame(np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['f', 'g', 'h', 'i'])

    print(df)
    print(df.sum(axis=0))
    print(df.mean(axis=0))
    print(df.std(axis=0))
    print(df.var(axis=0))

    print(df)
    print(df.sum(axis=1))
    print(df.mean(axis=1))
    print(df.std(axis=1))
    print(df.var(axis=1))

    print(df)
    print(df.min(axis=0))
    print(df.max(axis=0))

    print(df)
    print(df.min(axis=1))
    print(df.max(axis=1))

    print(pd.concat([df, df]))
    print(pd.concat([df, df], axis=1))

def merge():
    df = pd.DataFrame(np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['f', 'g', 'h', 'i'])
    df1 = pd.DataFrame(np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['f', 'g', 'h', 'i'])
    print(pd.merge(df, df1))
    print(pd.merge(df, df1, left_on='f', right_on='f'))

def merge1():
    raw_data = {
        'subject_id' : ['1', '2', '3', '4', '5'],
        'first_name' : ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name' : ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']
    }
    df_a = pd.DataFrame(raw_data)

    raw_data = {
        'subject_id' : ['4', '5', '6', '7', '8'],
        'first_name' : ['Billy', 'Brain', 'Bran', 'Bryce', 'Betty'],
        'last_name' : ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']
    }
    df_b = pd.DataFrame(raw_data)

    print(df_a)
    print(df_b)
    print(pd.merge(df_a, df_b, on='subject_id'))
    print(pd.merge(df_a, df_b, on='subject_id', how='inner'))


def merge2():
    raw_data = {
        'subject_id': ['1', '2', '3', '4', '5'],
        'first_name': ['Alex', 'Amy', 'Allen', 'Alice', 'Ayoung'],
        'last_name': ['Anderson', 'Ackerman', 'Ali', 'Aoni', 'Atiches']
    }
    df_a = pd.DataFrame(raw_data, columns=['subject_id', 'first_name', 'last_name'])

    raw_data = {
        'subject_id': ['4', '5', '6', '7', '8'],
        'first_name': ['Billy', 'Brain', 'Bran', 'Bryce', 'Betty'],
        'last_name': ['Bonder', 'Black', 'Balwner', 'Brice', 'Btisan']
    }
    df_b = pd.DataFrame(raw_data, columns=['subject_id', 'first_name', 'last_name'])

    print(pd.merge(df_a, df_b, on='subject_id', how='outer'))

if __name__ == '__main__':
    p_dataframe()
    p_dict()
    add_dataframe()
    copy_columns()
    rm_columns()
    df_attribute()
    search_col()
    check_null()
    df_replace()
    df_replace1()
    df_group()
    describe()
    operation()
    merge()
    merge1()
    merge2()