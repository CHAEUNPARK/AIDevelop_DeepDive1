import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame([['yellow', 'M', '23', 'a'],
                   ['red', 'L', '26', 'b'],
                   ['blue', 'XL', '20', 'c']])
df.columns = ['color', 'size', 'price', 'type']
print(df)

# 데이터셋 필요한 부분 숫자 변경
x = df[['color', 'size', 'price', 'type']].values           #데이터 프레임에서 numpy.narray로 변경

shop_le = LabelEncoder()
x[:,0] = shop_le.fit_transform(x[:,0])
x[:,1] = shop_le.fit_transform(x[:,1])
x[:,2] = x[:,2].astype(dtype=float)
x[:,3] = shop_le.fit_transform(x[:,3])

print("라벨인코더 변환값\n",x)