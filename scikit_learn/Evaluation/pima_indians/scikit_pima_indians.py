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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

pima_indians_df = pd.read_csv("pima_indians.csv")


# %%
# 데이터 정보 확인
pima_indians_df.isnull().sum()
pima_indians_df.info()
pima_indians_df.describe()

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

x_pima_indians_df = pima_indians_df.drop("Outcome", axis=1)
y_pima_indians_df = pima_indians_df["Outcome"]

scaler = MinMaxScaler()
scaler.fit(x_pima_indians_df)
x_pima_indians_df = scaler.transform(x_pima_indians_df)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_pima_indians_df,
    y_pima_indians_df,
    test_size=0.2,
    random_state=156,
    stratify=y_pima_indians_df,
)
# %%


# 로지스틱 회긔 예측 모델
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_test)
pred_proba = lr_clf.predict_proba(x_test)[:, 1]
# get_clf_eval(y_test, pred, pred_proba)
# get_eval_by_threshold(y_test, pred_proba, thresholds)
# precision_recall_curve_plot(y_test, pred_proba)
# roc_curve_plot(y_test, pred_proba)
get_eval_by_binarizer(y_test, pred_proba, 0.48)

# %%
# 히스토그램
import matplotlib.pyplot as plt

plt.hist(pima_indians_df["Glucose"], bins=10)

# %%
# min == 0 의 비율 확인
zero_features = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

total_count = pima_indians_df["Glucose"].count()

for feature in zero_features:
    zero_count = pima_indians_df[pima_indians_df[feature] == 0][feature].count()
    print(f"{feature}의 0 비율은 {zero_count/total_count*100:0.2f} %")
# %%
# min == 0 인 값 평균값으로 대체
mean_zero_features = pima_indians_df[zero_features].mean()
pima_indians_df[zero_features] = pima_indians_df[zero_features].replace(
    0, mean_zero_features
)

# %%
