# %%
def numerical_derivative(f, x):
    delta_x = 1e-4
    grid = np.zeros_like(x)
    print("초기 value =", x)
    print("초기 grid =", grid)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        print(f"idx = {idx}, x[idx] = {x[idx]}")

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
# 입력이 1개
def func1(input_obj):
    return input_obj[0] ** 2


# %%
# 입력이 2개
def func1(input_obj):
    x = input_obj[0]
    y = input_obj[1]
    return 2 * x + 3 * x * y + pow(y, 3)


# %%
# 입력이 4개
def func1(input_obj):
    w = input_obj[0, 0]
    x = input_obj[0, 1]
    y = input_obj[1, 0]
    z = input_obj[1, 1]

    return w * x + x * y * z + 3 * w + z * y ** 2


# %%
import numpy as np


def func2(x):
    return 3 * x * (np.exp(x))


# %%
print(numerical_derivative(func1, np.array([[1.0, 2.0], [2.0, 4.0]])))

# %%
print(numerical_derivative(func2, 2))

# %%
