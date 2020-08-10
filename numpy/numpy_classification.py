# %%
import numpy as np
import pandas as pd

# %%
x_data = np.array(
    [2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
).reshape(-1, 1)
x_data
t_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).reshape(
    -1, 1
)
t_data
print(
    f"x_data = {x_data} , shape = {x_data.shape} , t_data = {t_data} , shape = {t_data.shape}"
)


# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def loss_func(x, t):
    delta = 1e-7
    z = np.dot(x, w) + b
    y = sigmoid(z)

    return -np.sum(t * np.log(y + delta) + (1 - t) * np.log((1 - y) + delta))


# %%


def numerical_derivative(f, x):
    delta_x = 1e-4
    grid = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index

        temp_val = x[idx]
        x[idx] = float(temp_val) + delta_x
        fx1 = f(x)

        x[idx] = float(temp_val) - delta_x
        fx2 = f(x)

        grid[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = temp_val

        it.iternext()
    return grid


# %%
# %%
def error_val(x, t):
    y = np.dot(x, w) + b
    return (np.sum((t - y) ** 2)) / len(x)


def predict(x):
    z = np.dot(x, w) + b
    y = sigmoid(z)

    if y >= 0.5:
        result = 1
    else:
        result = 0
    return y, result


# %%
w = np.random.rand(1, 1)
b = np.random.rand(1)
print(f"w = {w} , shape = {w.shape}, b = {b}, shape = {b.shape}")

# %%
learning_rate = 1e-6
f = lambda x: loss_func(x_data, t_data)
print(f"initial error value = {error_val(x_data,t_data)}, w = {w} , b  = {b}")
for step in range(50000):

    w -= learning_rate * numerical_derivative(f, w)
    b -= learning_rate * numerical_derivative(f, b)
    if step % 10000 == 0:
        print(
            f"step = {step}, error value = {error_val(x_data,t_data)}, w = {w}, b = {b}"
        )


# %%
real, logicl = predict(5)
print(f"{real}, {logicl}")

# %%
