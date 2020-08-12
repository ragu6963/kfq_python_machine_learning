# %%
from scipy.sparse.construct import random
import sklearn

print(sklearn.__version__)
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
iris_data = load_iris()
train_data, test_data, train_label, test_label = train_test_split(
    iris_data.data, iris_data.target, test_size=0.4, random_state=121
)

# %%
df_clf = DecisionTreeClassifier()
df_clf.fit(train_data, train_label)
pred = df_clf.predict(test_data)
# print(pred == test_label)
print(accuracy_score(test_label, pred))
# %%
