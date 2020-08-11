"""
재현율과 정밀도
재현율 : 실제 P인 대상 중에 TP의 비율
재현율 = TP / (TP + FN)

정밀도 : 예측 P인 대상 중에 TP의 비율
정밀도 = TP / (TP + FP)

재현율 정밀도 조절하기
재현율 정밀도 확인하기
재현율과 정밀도를 결합한 F1 스코어
ROC 곡선과 AUC
ROC 곡선은 FPR의 변할 때 TPR이 어떻게 변하는지 나타내는 곡선이다.
TPR : 재현율(민감도)
FPR : 실제 N인 대상 중에 P로 예측한 값(N으로 예측하지 않은 값, 예측이 틀린 비율)
FPR = 1 - 특이성 
특이성 : N을 예측한 비율, FPR -> N을 예측하지 못한 비율
"""
# %%
from sklearn import preprocessing

# 결측값 처리
def fillna(df):
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Cabin"].fillna("N", inplace=True)
    df["Embarked"].fillna("N", inplace=True)
    df["Fare"].fillna(0, inplace=True)
    return df


# 불필요 속성(name,id,ticket) 제거
def drop_features(df):
    df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    return df


# 레이블 인코딩
def encode_features(df):
    df["Cabin"] = df["Cabin"].str[:1]
    features = ["Cabin", "Sex", "Embarked"]
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


# 함수 모두 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = encode_features(df)
    return df


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
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]


def get_eval_by_threshold(y_test, pred_proba_c1, thresholds):
    for threshold in thresholds:
        binarizer = Binarizer(threshold=threshold).fit(pred_proba_c1)
        custom_predict = binarizer.transform(pred_proba_c1)
        print("임계값 :", threshold)
        get_clf_eval(y_test, custom_predict, pred_proba[:, 1])


# get_eval_by_threshold(y_test, pred_proba[:, 1].reshape(-1, 1), thresholds)


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

titanic_df = pd.read_csv("train.csv")
y_titanic_df = titanic_df["Survived"]
x_titanic_df = titanic_df.drop("Survived", axis=1)
x_titanic_df = transform_features(x_titanic_df)
x_train, x_test, y_train, y_test = train_test_split(
    x_titanic_df, y_titanic_df, test_size=0.20, random_state=11
)


lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
pred = lr_clf.predict(x_test)
pred_proba = lr_clf.predict_proba(x_test)
get_clf_eval(y_test, pred, pred_proba[:, 1])


# %%
# 칼럼들의 Negative , Positive  확률 구하기 -> pred_proba
# 첫 번째 칼럼 Negative 확율, 두 번째 칼럼 Positive 확율
# 두 번째 칼럼(Positive)가 0.5 면 결과 값이 1(Positive)
import numpy as np

pred_proba = lr_clf.predict_proba(x_test)
pred = lr_clf.predict(x_test)
print(f"proba결과 shape : {pred_proba.shape}")
print(f"proba결과 3개 추출 : {pred_proba[:3]}")

pred_proba_result = np.concatenate([pred_proba, pred.reshape(-1, 1)], axis=1)
print(f"두 개의 class 중에서 더 큰 확률을 클래스 값으로 예측 \n {pred_proba_result[:3]}")

# %%
# 임계값을 사용하여 결과 조절
# Binarizer 사용 예시
from sklearn.preprocessing import Binarizer

X = [[1, -1, 2], [2, 0, 0], [0, 1.1, 1.2]]
binarizer = Binarizer(threshold=1.1)
print(binarizer.fit_transform(X))

# %%
# 재현율 정밀도 조절하기 (트레이드 오프)
# 임계값을 조절(Binarizer)하여 정밀도, 재현율 조절
# 임계값이 작아지면 재현율 UP
# 임계값이 커지면 정밀도 UP

custom_threshold = 0.6
pred_proba_1 = pred_proba[:, 1].reshape(-1, 1)

binarizer = Binarizer(threshold=custom_threshold).fit(pred_proba_1)
custom_predict = binarizer.transform(pred_proba_1)

get_clf_eval(y_test, custom_predict)

# %%
# precision_recall_curve 를 사용한 조절
# API 이용
from sklearn.metrics import precision_recall_curve

pred_proba_class1 = lr_clf.predict_proba(x_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(
    y_test, pred_proba_class1
)
print(f"반환된 분류 결정 임계값 배열의 shape : {thresholds.shape}")

thr_index = np.arange(0, thresholds.shape[0], 15)
print(f"샘플 추출의 위한 임계값 배열의 index 10개 : {thr_index}")
print(f"샘플용 10개의 임계값 : {np.round(thresholds[thr_index],2)}")

print(f"샘플 임계값별 정밀도 : {np.round(precisions[thr_index],3)}")
print(f"샘플 임계값별 재현율 : {np.round(recalls[thr_index],3)}")

# %%
# 정밀도 재현율 시각화
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


pred_proba_class1 = lr_clf.predict_proba(x_test)[:, 1]
precision_recall_curve_plot(y_test, pred_proba_class1)

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


roc_curve_plot(y_test, pred_proba[:, 1])


# %%

