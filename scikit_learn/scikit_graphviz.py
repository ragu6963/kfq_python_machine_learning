#%%
from sklearn.datasets import load_iris
import pandas as pd

# %%
from sklearn.model_selection import train_test_split

iris = load_iris()

train_data, test_data, train_label, test_label = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=121
)


# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier()
dt_clf.fit(train_data, train_label)
pred = dt_clf.predict(test_data)
print(accuracy_score(test_label, pred))
|

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
