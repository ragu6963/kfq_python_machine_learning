# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

card_df = pd.read_csv("./creditcard.csv")

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
from sklearn.preprocessing import StandardScaler


def get_preprocessed_df(df=None):
    df_copy = df.copy()
    """
    정규 분포 형태로 피처값 변화
    """
    # scaler = StandardScaler()
    # amount_n = scaler.fit_transform(df_copy["Amount"].values.reshape(-1, 1))
    # df_copy.insert(0, "Aomunt_Scaled", amount_n)
    """
    로그 변환
    """
    amount_n = np.log1p(df_copy["Amount"])
    df_copy.insert(0, "Aomunt_Scaled", amount_n)
    """
    이상치 데이터 삭제
    """
    outlier_index = get_outlier(df=df_copy, column="V14", weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    df_copy.drop(["Time", "Amount"], axis=1, inplace=True)
    return df_copy


from sklearn.model_selection import train_test_split


def get_train_test_dataset(df=None):
    df_copy = get_preprocessed_df(df)
    X_features = df_copy.iloc[:, :-1]
    y_labels = df_copy.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.3, random_state=0, stratify=y_labels
    )
    return X_train, X_test, y_train, y_test


"""
IQR을 이용한 이상치 데이터 제거
"""
import numpy as np


def get_outlier(df=None, column=None, weight=1.5):
    fraud = df[df["Class"] == 1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)
    # IQR을 구하고, IQR에 1.5를 곱해 최댓값과 최솟값 지점 구함
    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight
    # 최댓값보다 크거나, 최솟값보다 작은 값을 이상치 데이터로 설정하고 index 반환
    outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
    return outlier_index


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

#%%
X_train, X_test, y_train, y_test = get_train_test_dataset(card_df)
#%%
"""
SMOTE 오버 샘플링 : 학습데이터(train)만 오버 샘플링
"""
from imblearn.over_sampling import SMOTE

print(f"SMOTE 적용 전 학습용 데이터 Shape : {X_train.shape} , {y_train.shape}")
smote = SMOTE(random_state=0)
X_train, y_train = smote.fit_sample(X_train, y_train)
print(f"SMOTE 적용 후 학습용 데이터 Shape : {X_train.shape} , {y_train.shape}")

# %%
# 로지스틱 회긔 예측 모델
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
pred = lr_clf.predict(X_test)
pred_proba = lr_clf.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, pred, pred_proba)
#%%
# ROC 곡선 / 정밀도 재현율
# roc_curve_plot(y_test, pred_proba)
print("로지스틱 회긔")
precision_recall_curve_plot(y_test, pred_proba)
print("-------------------------------------------")
# %%
# LightGBM 예측 모델
from lightgbm import LGBMClassifier

lgbm_clf = LGBMClassifier(
    n_estimators=1000, num_leaves=64, n_jobs=-1, boost_from_average=False
)
evals = [(X_test, y_test)]
lgbm_clf.fit(X_train, y_train)
preds = lgbm_clf.predict(X_test)
pred_proba = lgbm_clf.predict_proba(X_test)[:, 1]
get_clf_eval(y_test, preds, pred_proba)
#%%
# ROC 곡선 / 정밀도 재현율
print("LightGBM")
# roc_curve_plot(y_test, pred_proba)
precision_recall_curve_plot(y_test, pred_proba)
print("-------------------------------------------")
# %%
"""
데이터 분포도 확인
"""
import seaborn as sns

plt.figure(figsize=(8, 4))
plt.xticks(range(0, 30000, 1000), rotation=60)
sns.distplot(card_df["Amount"])


# %%

import seaborn as sns

plt.figure(figsize=(9, 9))
corr = card_df.corr()
sns.heatmap(corr, cmap="RdBu")


# # %%
# """
# IQR을 이용한 이상치 데이터 제거
# """
# import numpy as np


# def get_outlier(df=None, column=None, weight=1.5):
#     fraud = df[df["Class"] == 1][column]
#     quantile_25 = np.percentile(fraud.values, 25)
#     quantile_75 = np.percentile(fraud.values, 75)
#     # IQR을 구하고, IQR에 1.5를 곱해 최댓값과 최솟값 지점 구함
#     iqr = quantile_75 - quantile_25
#     iqr_weight = iqr * weight
#     lowest_val = quantile_25 - iqr_weight
#     highest_val = quantile_75 + iqr_weight
#     # 최댓값보다 크거나, 최솟값보다 작은 값을 이상치 데이터로 설정하고 index 반환
#     outlier_index = fraud[(fraud < lowest_val) | (fraud > highest_val)].index
#     return outlier_index


# %%
outlier_index = get_outlier(df=card_df, column="V14", weight=1.5)
print("이상치 데이터 인덱스 :", outlier_index)
# %%
