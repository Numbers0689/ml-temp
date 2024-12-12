import numpy as np
import copy, math
import matplotlib.pyplot as plt
np.set_printoptions(precision=2) 

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)


b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# def predict(x, w, b):
#     return np.dot(x, w) + b

# x_vec = X_train[0]
# f_wb = predict(x_vec, w_init, b_init)
# print(f"prediction: {f_wb}")

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i])**2

    cost /= 2*m
    return cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    dj_db /= m
    dj_dw /= m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_fn, gradient_fn, alpha, num_iters):
    J_his = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_fn(X, y, w, b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

        if i < 100000:
            J_his.append(cost_fn(X, y, w, b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_his[-1]:8.2f}   ")

    return w, b, J_his


cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"cost: {cost}")

tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')

