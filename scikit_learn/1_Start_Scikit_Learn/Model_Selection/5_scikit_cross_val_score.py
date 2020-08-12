"""
KFold를 통해 수행하는 교차 검증을 보다 간편하게 해주는
### cross_val_score() ###
"""
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

# %%
iris = load_iris()
df_clf = DecisionTreeClassifier(random_state=156)

# %%
data = iris.data
label = iris.target

# %%
scores = cross_val_score(df_clf, data, label, scoring="accuracy", cv=3)

print(f"교차 검증별 정확도 : {np.round(scores,4)}")
print(f"평균 검증 정확도 : {np.round(np.mean(scores),4)}")


# %%
# accuracy, f1_micro
scoring = ["accuracy", "f1_micro"]
scores = cross_validate(df_clf, data, label, scoring=scoring, cv=3)
scores


# %%
scores["test_accuracy"]
# %%
