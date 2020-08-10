# %%
import numpy as np
from numpy.core.fromnumeric import shape

x_data = np.array([1, 2, 3, 4, 5]).reshape(5, 1)
t_data = np.array([2, 3, 4, 5, 6]).reshape(5, 1)

print(f"x_data.shape = {x_data.shape}, t_data.shape = {t_data.shape}")
x_data

# %%
raw_data = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
x_data = raw_data[0:, 0].reshape(5, 1)
x_data
t_data = raw_data[0:, 1].reshape(5, 1)
t_data

# %%
w = np.random.rand(1, 1)
b = np.random.rand(1)
print(f"w = {w} , shape = {w.shape}, b = {b}, shape = {b.shape}")

# %%
def loss_func(x, t):
    y = np.dot(x, w) + b
    return (np.sum((t - y) ** 2)) / len(x)


# %%
def numerical_derivative(f, x):
    delta_x = 1e-4
    grid = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        # print(f"idx = {idx}, x[idx] = {x[idx]}")

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
def error_val(x, t):
    y = np.dot(x, w) + b
    return (np.sum((t - y) ** 2)) / len(x)


def predict(x):
    y = np.dot(x, w) + b
    return y


# %%
learning_rate = 1e-2
f = lambda x: loss_func(x_data, t_data)
print(f"initial error value = {error_val(x_data,t_data)}, w = {w} , b  = {b}")
for step in range(8001 * 2):

    w -= learning_rate * numerical_derivative(f, w)
    b -= learning_rate * numerical_derivative(f, b)
    if step % 400 == 0:
        print(
            f"step = {step}, error value = {error_val(x_data,t_data)}, w = {w}, b = {b}"
        )


# %%
predict(43)


# %%
