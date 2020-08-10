# %%
import numpy as np
import pandas as pd

#%%
df = pd.read_excel("./peopel.xlsx")
#%%
drop_df = df[["①_003_키", "ⓞ_13_체지방량", "①_031_몸무게"]].dropna(axis=0)
# drop_df
# drop_df.isnull().sum()
data_np = np.array(drop_df)
data_np
data_np.shape
# %%
# from matplotlib import pyplot as plt

# plt.scatter(data_np[:, 0].tolist(), data_np[:, 1].tolist())
# plt.show()
# %%
# 키 체지방량
x_data = data_np[:, 0:-1].reshape(-1, 1)
# 몸무게
t_data = data_np[:, -1].reshape(-1, 1)
print(f"shape = {x_data.shape}  , shape = {t_data.shape}")


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
w = np.random.rand(2, 1)
b = np.random.rand(1)
print(f"w = {w} , shape = {w.shape}, b = {b}, shape = {b.shape}")

# %%
learning_rate = 1e-9
f = lambda x: loss_func(x_data, t_data)
print(f"initial error value = {error_val(x_data,t_data)}, w = {w} , b  = {b}")
for step in range(40000):

    w -= learning_rate * numerical_derivative(f, w)
    b -= learning_rate * numerical_derivative(f, b)
    if step % 4000 == 0:
        print(
            f"step = {step}, error value = {error_val(x_data,t_data)}, w = {w}, b = {b}"
        )


# %%


# %%
