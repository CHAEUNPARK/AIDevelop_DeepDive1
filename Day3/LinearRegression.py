import Day2.boston as boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

sim_lr = LinearRegression()
sim_lr.fit(boston.x_train['RM'].values.reshape((-1, 1)), boston.y_train)
y_pred = sim_lr.predict(boston.x_test['RM'].values.reshape((-1,1)))

# 결과 불러오기
print('단순 선형 회귀 {:.4f}'.format(r2_score(boston.y_test, y_pred)))
print('단순 선형회귀, 계수(w):{:.4f}, 절편 (b):{:.4f}'.format(sim_lr.coef_[0], sim_lr.intercept_) )

plt.scatter(boston.x_test['RM'], boston.y_test, s=10, c='black')
plt.plot(boston.x_test['RM'], y_pred, c='red')
plt.legend(['Regression line', 'x_test'],loc='upper left')
plt.show()

mul_lr = LinearRegression()
mul_lr.fit(boston.x_train.values, boston.y_train)
y_pred = mul_lr.predict(boston.x_test.values)

print('다중 선형회귀 계수 : {}, 절편 : {:.4f}'.format(mul_lr.coef_, mul_lr.intercept_))
print('다중 선형 회귀, R2 : {:.4f}'.format(r2_score(boston.y_test, y_pred)))
