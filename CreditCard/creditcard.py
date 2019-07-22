import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 데이터 로드
data = pd.read_csv('./CreditCard/creditcard.csv')
print(data.head())
print(data.columns) #컬럼명 확인
# 데이터 빈도수 확인
print(pd.value_counts(data['Class']))

pd.value_counts(data['Class']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()