import os
from os.path import join
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def importData():
    abalone_path=join('../data','abalone.txt')
    column_path=join('../data', 'abalone_attributes.txt')

    abalone_columns = list()
    for l in open(column_path):
        abalone_columns.append(l.strip())

    data = pd.read_csv(abalone_path, header=None, names=abalone_columns)
    data.shape
    label = data['Sex']
    del data['Sex']
    return data

def fsdscalar(data):
    sdscaler= StandardScaler()
    sdscaler.fit(data)
    sdscaler_data=sdscaler.transform(data)
    sdscaler_pd = pd.DataFrame(sdscaler_data,columns=data.columns)
    print("sdscaler_data=>",sdscaler_pd)

if __name__ == '__main__':
    fsdscalar(importData())