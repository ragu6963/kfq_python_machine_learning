"""
KFold 와 StratifiedKFold 의 데이터 분포도 비교
교차 검증 방법
train_test_split 방법은 과적합(overfitting)에 취약한 약점을 가질 수 있다.
과적합이란 모델이 학습 데이터에만 과도하게 최적화되어, 실제 예측을 다른 데이터로 수행할 경우에는 성능이 떨어지는 것을 말한다.
"""
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


# %%
iris = load_iris()

# %%
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["label"] = iris.target
iris_df["label"].value_counts()
# %%
iris_df
# %%
features = iris.data
label = iris.target
# %%
df_clf = DecisionTreeClassifier(random_state=156)

# %%
kfold = KFold(n_splits=5, shuffle=True)
skf = StratifiedKFold(n_splits=5)

n_iter = 0
# for train_index, test_index in skf.split(features, label):
for train_index, test_index in kfold.split(features):
    n_iter += 1
    label_train = iris_df["label"].iloc[train_index]
    label_test = iris_df["label"].iloc[test_index]
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    print(f"{n_iter} 교차검증")
    print(f"학습 레이블 데이터 분포\n{label_train.value_counts()}")
    print(f"검증 레이블 데이터 분포\n{label_test.value_counts()}")


# %%
