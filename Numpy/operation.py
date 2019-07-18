import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import statistics as sta
######### 기본 사칙연산 및 특수 연산자 ############
x = np.array([1, 5, 2])
y = np.array([7, 4, 1])

print(x + y)
print(x * y)
print(x - y)
print(x / y)
print(x % y)

############## dot 함수 ##################

bb = np.array([1, 2, 3])
cc = np.array([-7, 8, 9])
print(np.dot(bb, cc))

xs = np.array(((2, 3), (3, 5)))
ys = np.array(((1, 2), (5, -1)))
print(np.dot(xs, ys), type(np.dot(xs, ys)))

############# 배열 ####################

l33 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np33 = np.array(l33, dtype= int)

print(np33.shape)
print(np33.ndim)
print(np33)
print("first row : ", np33[0])
print("first column : ", np33[:, 0])

np33 = np.array(l33, int)
print(np33)

print(np33[:2, 1:])

############ argmax #################

arr = np.array([9, 18, 29, 39, 49])

print(" index ")
print(arr.argmax())
print(arr.argmin())

print(" value ")
print(arr[np.argmax(arr)])
print(arr[np.argmin(arr)])


################ axis ######################

a = np.arange(6)
b = np.arange(6).reshape(2, 3)
a[5] = 100

print(a)
print(b)
print(a[np.argmax(a)])

print(np.argmax(b, axis=0))
print(np.argmax(b, axis=1))

################ random ######################

a = np.random.rand(3, 2)
print(a)

b = np.random.rand(3,3,3)
print(b)

outcome = np.random.randint(1, 7, size=10)
print(outcome)
print(type(outcome))
print(len(outcome))

print(np.random.randint(2, size=10))
print(np.random.randint(1, size=10))
print(np.random.randint(5, size=(2, 4)))

a = np.random.randn(3, 2)
print(a)
b = np.random.randn(3, 3, 3)
print(b)
plt.plot(a)
plt.show()

arr = np.arange(10)
print(arr)
np.random.shuffle(arr)
print(arr)

arr2 = np.arange(9).reshape((-1, 3))
print(arr2)
np.random.shuffle(arr2)
print(arr2)

################### 통계함수 ####################

x = np.array([-2.1, -1, 1, 1, 4.3])
print(np.mean(x))
print(np.median(x))
print(mode(x))

x = np.array([-2.1, -1, 1, 1, 4.3])
print(np.mean(x))
print(np.median(x))
print(mode(x))

x_m = np.mean(x)
x_a = x - x_m
x_p = np.power(x_a, 2)

print("Variance x")
print("np.var(x)")
print(sta.pvariance(x))
print(sta.variance(x))

x = np.array([-2.1, -1, 1, 1, 4.3])
print(np.mean(x))
print(np.median(x))
print(mode(x))

x_m = np.mean(x)
x_a = x - x_m
x_p = np.power(x_a, 2)
print(np.var(x))
print(sta.pvariance(x))
print(sta.variance(x))

print(np.std(x))
print(sta.pstdev(x))
print(sta.stdev(x))
