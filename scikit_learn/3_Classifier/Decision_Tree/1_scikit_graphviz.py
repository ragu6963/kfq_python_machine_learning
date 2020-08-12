#%%
from sklearn.datasets import load_iris
import pandas as pd

# %%
from sklearn.model_selection import train_test_split

iris = load_iris()

train_data, test_data, train_label, test_label = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=11
)


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier(max_depth=3)
dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(test_data)
print(accuracy_score(test_label, pred))
# %%
import seaborn as sns
import numpy as np

print(f"Feature importances:\n{np.round(dt_clf.feature_importances_,3)}")
for name, value in zip(iris.feature_names, dt_clf.feature_importances_):
    print(f"{name} : {value:0.3f}")

sns.barplot(x=dt_clf.feature_importances_, y=iris.feature_names)
# %%
from sklearn.tree import export_graphviz

export_graphviz(
    dt_clf,
    out_file="tree.dot",
    class_names=iris.target_names,
    feature_names=iris.feature_names,
    impurity=True,
    filled=True,
)


# %%
import graphviz

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# %%
