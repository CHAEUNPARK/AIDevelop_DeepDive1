import numpy as np

npa = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(npa.size)
print(npa.shape)
print(len(npa))

npl = np.array([1, 100, 42, 42, 42, 6, 7])
print(npl.size)
print(len(npl))
print(npl.shape)
print(npl.ndim)

############ zeros ############

a = np.zeros(3)
print(a.ndim)
print(a.shape)
print(a)

########### eye ############

np.eye(2, dtype=int)
print(np.eye(3))
print(np.eye(3, k=1))
print(np.eye(3, k=-1))

########### indentity ############

print(np.identity(5))

########### linespace #############

start = 0
end = 10
linspace = np.linspace(start, end, num=8, endpoint=True, retstep=False)
print(linspace)