import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b

    return f_wb


x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

m = x_train.shape[0]
print(f"number of samples: {m}")

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

w = 200
b = 100
tmp_f_wb = compute_model_output(x_train, w, b)
plt.plot(x_train, tmp_f_wb, c='b', label="prediction")
plt.scatter(x_train, y_train, marker='x', c='r', label="Actual value")
plt.title("Housing price prediction")
plt.xlabel("Sq feet (in 1000s)")
plt.ylabel("price (in 1000$)")
plt.legend()
plt.show()

x_i = 1.4
cost1200sq_f_wb = w * x_i + b
print(f"cost of 1200sqft : {cost1200sq_f_wb} $")
