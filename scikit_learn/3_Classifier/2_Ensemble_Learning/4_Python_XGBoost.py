"""
파이썬 XGBoost
"""
# %%
# 오차행렬, 정확도, 정밀도, 재현율, F1 스코어, ROC AUC 출력
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import Binarizer


def get_clf_eval(y_test, pred=None, pred_proba=None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    print("오차행렬")
    print(confusion)
    print(
        f"정확도 : {accuracy:0.4f}, 정밀도 : {precision:0.4f}, 재현율 : {recall:0.4f}, F1 : {f1:0.4f}, AUC:{roc_auc:0.4f}"
    )


# 임계값에 따라 변화하는 정밀도 및 재현율 출력
thresholds = [0.4, 0.45, 0.46, 0.47, 0.48]


def get_eval_by_threshold(y_test, pred_proba, thresholds):
    pred_proba_c1 = pred_proba.reshape(-1, 1)
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(pred_proba_c1)
        pred = binarizer.transform(pred_proba_c1)
        print("임계값 :", threshold)
        get_clf_eval(y_test, pred, pred_proba)


custom_threshold = 0.48


def get_eval_by_binarizer(y_test, pred_proba, custom_threshold):
    pred_proba_c1 = pred_proba.reshape(-1, 1)

    binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_c1)
    custom_predict = binarizer.transform(pred_proba_c1)

    get_clf_eval(y_test, custom_predict, pred_proba)


# get_clf_eval(y_test, pred, pred_proba)
# get_eval_by_threshold(y_test, pred_proba, thresholds)
# get_eval_by_binarizer(y_test, pred_proba, custom_threshold)
# %%
# 정밀도 재현율 시각화
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def precision_recall_curve_plot(y_test, pred_proba_c1):
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, pred_proba_c1
    )

    plt.figure(figsize=(8, 6))
    threshold_boundary = thresholds.shape[0]
    plt.plot(
        thresholds,
        precisions[0:threshold_boundary],
        linestyle="--",
        label="precision",
    )
    plt.plot(thresholds, recalls[0:threshold_boundary], label="recall")

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel("Threshold value")
    plt.ylabel("Precision and Revall value")
    plt.legend()
    plt.grid()
    plt.show()


# precision_recall_curve_plot(y_test, pred_proba)
# %%
# ROC 곡선 시각화
from sklearn.metrics import roc_curve


def roc_curve_plot(y_test, pred_proba_c1):
    # 임계값에 따른 FPR, TPR 값
    fprs, tprs, _ = roc_curve(y_test, pred_proba_c1)
    # ROC 곡선 그래프
    plt.plot(fprs, tprs, label="ROC")
    # 가운데 대각선 직선(최악의 경우)
    plt.plot([0, 1], [0, 1], "k--", label="Random")

    start, end = plt.xlim()
    plt.xticks(np.round(np.arange(start, end, 0.1), 2))

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.show()


# roc_curve_plot(y_test, pred_proba)
# %%
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
X_features = dataset.data
y_label = dataset.target
cancer_df = pd.DataFrame(data=X_features, columns=dataset.feature_names)
cancer_df["target"] = y_label

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_label, test_size=0.2, random_state=156
)
# %%
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

# %%
params = {
    "max_depth": 3,
    "eta": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "early_stoppings": 100,
}
num_rounds = 400


# %%
wlist = [(dtrain, "train"), (dtest, "eval")]
xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    early_stopping_rounds=100,
    evals=wlist,
)


# %%
pred_probas = xgb_model.predict(dtest)
print(f"predict() 수행 결과 확률값 10개 표시\n{np.round(pred_probas[:10],3)}")
preds = [1 if x > 0.5 else 0 for x in pred_probas]
print(f"예측값 10개 표시")
print(np.round(preds[:10]))
get_clf_eval(y_test, preds, pred_probas)
# %%
from xgboost import plot_importance
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax)
