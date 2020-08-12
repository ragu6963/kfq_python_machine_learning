#%%
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# %%
# scaler = StandardScaler()
scaler = MinMaxScaler()

scaler.fit(iris_df)

iris_df_scaled = scaler.transform(iris_df)
# %%
train_data, test_data, train_label, test_label = train_test_split(
    iris_df_scaled, iris.target, test_size=0.4, random_state=121
)

# %%
df_clf = DecisionTreeClassifier()
df_clf.fit(train_data, train_label)

# %%
pred = df_clf.predict(test_data)
print(accuracy_score(test_label, pred))

# %%
