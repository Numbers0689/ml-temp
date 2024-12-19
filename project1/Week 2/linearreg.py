import numpy as np
import matplotlib.pyplot as plt
import copy, math
from utils import *

x_train, y_train = load_data()

print("Type of X: ", type(x_train))
print("values of X, 5: ", x_train[:5])
print("Type of Y: ", type(y_train))
print("values of Y, 5: ", y_train[:5])

print("X shape: ", x_train.shape)
print("Y shape: ", y_train.shape)
print("no. of training samples, m: ", len(x_train))

plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("profits vs population in city")
plt.xlabel("population in 10000")
plt.ylabel("profits in 10000$")
plt.show()