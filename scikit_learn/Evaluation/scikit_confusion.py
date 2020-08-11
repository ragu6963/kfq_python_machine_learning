"""
단순히 accuracy만 보는것만이 아니라
오차행렬을 볼 필요가 있다
오차행렬에서 각 값의 뜻은
TN : N이라 예측하고, 실제 값이 N인 경우
FN : N이라 예측하고, 실제 값이 P인 경우
TP : P이라 예측하고, 실제 값이 P인 경우
FP : P이라 예측하고, 실제 값이 N인 경우
"""
# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def get_clf_eval(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print("오차행렬")
    print(confusion)
    print(f"정확도 : {accuracy}, 정밀도 : {precision}, 재현율 : {recall}")


# %%
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
class MyDummy(BaseEstimator):
    def fit(self, x, y=None):
        pass

    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)


# %%
from sklearn.datasets import load_digits

digits = load_digits()
y = (digits["target"] == 7).astype(int)
x_train, x_test, y_train, y_test = train_test_split(
    digits["data"], y, random_state=11
)


# %%
fackclf = MyDummy()
fackclf.fit(x_train, y_train)
pred = fackclf.predict(x_test)
get_clf_eval(y_test, pred)


# %%
