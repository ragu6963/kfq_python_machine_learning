"""
교차 검증과 하이퍼 파라미터 튜닝을 한번에
"""
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# %%
iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=121
)
# %%
dtree = DecisionTreeClassifier(random_state=156)


# %%
para = {"max_depth": range(1, 10), "min_samples_split": range(1, 10)}

# %%
grid_dtree = GridSearchCV(dtree, param_grid=para, cv=3, refit=True)


# %%
grid_dtree.fit(x_train, y_train)
# %%
scores_df = pd.DataFrame(grid_dtree.cv_results_)

# %%
scores_df[
    [
        "params",
        "mean_test_score",
        "rank_test_score",
        "split0_test_score",
        "split1_test_score",
        "split2_test_score",
    ]
]


# %%
print(grid_dtree.best_params_)
print(grid_dtree.best_score_)

# %%
estimator = grid_dtree.best_estimator_


# %%
pred = estimator.predict(x_test)

# %%
print(accuracy_score(y_test, pred))


# %%
