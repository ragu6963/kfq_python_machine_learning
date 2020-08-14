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
    roc_auc = roc_auc_score(y_test, pred_proba, average="macro")
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
import pandas as pd
import numpy as np

cust_df = pd.read_csv("./train.csv", encoding="latin-1")

# %%
cust_df.head()
cust_df.info()
cust_df.isnull().sum().sum()

# %%
print(f"불만족 고객 : {cust_df[cust_df.TARGET==1].TARGET.count()}")
print(f"만족 고객 : {cust_df[cust_df.TARGET==0].TARGET.count()}")

print(
    f"불만족 비율 : {(cust_df[cust_df.TARGET==1].TARGET.count()/cust_df.TARGET.count())}"
)
# %%
# 이상치 확인
# var3 : -999999 확인 가능
cust_df.describe()
# %%
# 이상치 값 수정 , ID 컬럼 제거
# 최다 값이 2로 수정
cust_df.var3.replace(-999999, 2, inplace=True)
cust_df.drop("ID", axis=1, inplace=True)
# %%
# 피처 와 라벨 분리
X_features = cust_df.iloc[:, :-1]
y_labels = cust_df.iloc[:, -1]


# %%
# 학습데이터와 검증데이터 분리
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_labels, test_size=0.2, random_state=0
)
# %%
# XGBoost 모델 학습
from xgboost import XGBClassifier
import time

start = time.time()
xgb_clf = XGBClassifier(n_estimators=500, random_state=156)
evals = [(X_train, y_train), (X_test, y_test)]
xgb_clf.fit(
    X_train,
    y_train,
    early_stopping_rounds=100,
    eval_metric="auc",
    eval_set=evals,
    verbose=True,
)
preds = xgb_clf.predict(X_test)
pred_proba = xgb_clf.predict_proba(X_test)[:, 1]
end = time.time()
print(f"소모 시간 : {end-start}")
get_clf_eval(y_test, preds, pred_proba)

# %%
# XGBoost 하이퍼 파라미터 튜닝
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb_clf = XGBClassifier(n_estimators=100)
params = {
    "max_depth": [5, 7],
    "min_child_weight": [1, 3],
    "colsample_bytree": [0.5, 0.75],
}
evals = [(X_train, y_train), (X_test, y_test)]

gridcv = GridSearchCV(xgb_clf, param_grid=params, cv=3)
gridcv.fit(
    X_train,
    y_train,
    early_stopping_rounds=30,
    eval_metric="auc",
    eval_set=evals,
    verbose=True,
)
print(f"최적 파라미터 : {gridcv.best_estimator_}")
get_clf_eval(y_test, gridcv.predict_proba(X_test[:, 1]))


# %%

