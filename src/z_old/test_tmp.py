import numpy as np


# a = np.empty((8,1),dtype="int32")

# b = np.ones((8, 1), dtype="int32")

# c = np.concatenate((a,b), axis=1)

# print(c)


a = np.array([1,2,3])
b = np.array([2,4])

c = np.sum(a,b)
print(c)