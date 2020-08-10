# %%
import numpy as np
import pandas as pd
from scipy.sparse.construct import random

#%%
df = pd.read_excel("./peopel.xlsx")
# %%
drop_df = df[["①_003_키", "ⓞ_13_체지방량", "①_031_몸무게"]].dropna(axis=0)
# drop_df
data_np = np.array(drop_df)
data_np
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# %%

x_data = data_np[:, 0:-1].reshape(-1, 2)
# 몸무게
t_data = data_np[:, -1].reshape(-1, 1)
x_data
# %%
x_train, x_test, y_train, y_test = train_test_split(
    x_data, t_data, test_size=0.3,
)

# %%
x_train

# %%
y_train

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(x_train, y_train)


# %%
y_pred = lr.predict(x_test)

# %%
y_pred

# %%
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(mse, rmse)


# %%
r2_score(y_test, y_pred)


# %%
