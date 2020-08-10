# %%
from scipy.sparse.construct import random
import sklearn

print(sklearn.__version__)


# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# %%
iris = load_iris()
iris

# %%
iris_data = iris.data
iris_target = iris.target
iris_features_label = iris.feature_names
iris_target_label = iris.target_names
# %%

iris_df = pd.DataFrame(data=iris_data, columns=iris_features_label,)
iris_df["label"] = iris_target
iris_df
# %%
x_train, x_test, y_train, y_test = train_test_split(
    iris_data, iris_target, test_size=0.2, random_state=11
)

# %%
x_train

# %%
y_train

# %%
df_clf = DecisionTreeClassifier(random_state=11)
df_clf.fit(x_train, y_train)


# %%
pred = df_clf.predict(x_test)
pred
# %%
y_test

# %%
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, pred))

# %%
0