#%%
from sklearn.datasets import load_iris
import pandas as pd

# %%
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df.head()

# %%
iris_df.mean()

# %%
iris_df.var()

# %%
from sklearn.preprocessing import StandardScaler


# %%
scaler = StandardScaler()

# %%
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)


# %%
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

# %%
print(iris_df_scaled.mean())
print(iris_df_scaled.var())

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
scaler = MinMaxScaler()

# %%
scaler.fit(iris_df)

# %%
iris_scaled = scaler.transform(iris_df)

# %%
iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)

# %%
print(iris_df_scaled.mean())
print(iris_df_scaled.var())
# %%
print(iris_df_scaled.min())
print(iris_df_scaled.max())
# %%
