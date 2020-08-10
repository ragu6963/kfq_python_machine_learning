# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# %%
titanic_df = pd.read_csv("train.csv")

# %%
# 결측값 처리
def fillna(df):
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Cabin"].fillna("N", inplace=True)
    df["Embarked"].fillna("N", inplace=True)
    df.isnull().sum().sum()
    return df


# %%
# 불필요 속성(name,id,ticket) 제거
def drop_features(df):
    # df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
    df.drop(["PassengerId", "Name", "Ticket", "Fare"], axis=1, inplace=True)
    return df


# %%
# 레이블 인코딩
def encode_features(df):
    titanic_df["Cabin"] = titanic_df["Cabin"].str[:1]
    features = ["Cabin", "Sex", "Embarked"]
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df[feature])
        df[feature] = le.transform(df[feature])
    return df


# %%
# 함수 모두 호출
def transform_features(df):
    df = fillna(df)
    df = drop_features(df)
    df = encode_features(df)
    return df


# %%
# target 분류 및 데이터 전처리
titanic_df = pd.read_csv("train.csv")
y_titanic_df = titanic_df["Survived"]
x_titanic_df = titanic_df.drop("Survived", axis=1)

x_titanic_df = transform_features(x_titanic_df)
# %%
# 학습용 & 검증용 데이터 나누기
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_titanic_df, y_titanic_df, test_size=0.2, random_state=11
)
# %%
# module import
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# %%
# 각 학습 Class 학습 생성 예측 평가
dt_clf = DecisionTreeClassifier(random_state=11)
rf_clf = RandomForestClassifier(random_state=11)
lr_clf = LogisticRegression()

dt_clf.fit(x_train, y_train)
dt_pred = dt_clf.predict(x_test)
print(f"{accuracy_score(y_test, dt_pred):0.4f}")

rf_clf.fit(x_train, y_train)
rf_pred = rf_clf.predict(x_test)
print(f"{accuracy_score(y_test, rf_pred):0.4f}")

lr_clf.fit(x_train, y_train)
lr_pred = lr_clf.predict(x_test)
print(f"{accuracy_score(y_test, lr_pred):0.4f}")

# %%
# cross_val_score() 교차검증
from sklearn.model_selection import cross_val_score

scores = cross_val_score(dt_clf, x_titanic_df, y_titanic_df, cv=5)
for iter_count, accuracy in enumerate(scores):
    print(f"{iter_count}회 : {accuracy}")
print(f"평균 {np.mean(scores)}")
# %%
# GridSearchCV 결정 트리
from sklearn.model_selection import GridSearchCV

dt_clf = DecisionTreeClassifier(random_state=11)

parameters = {
    "max_depth": [2, 3, 5, 10],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 5, 8],
}
grid_dclf = GridSearchCV(
    dt_clf, param_grid=parameters, scoring="accuracy", cv=5
)
grid_dclf.fit(x_train, y_train)
print(f"최적 파라미터 : {grid_dclf.best_params_}")
print(f"최고 정확도 : {grid_dclf.best_score_}")
best_dclf = grid_dclf.best_estimator_
pred = best_dclf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)

# %%
# GridSearchCV 랜덤 포레스트
rf_clf = RandomForestClassifier(random_state=11)

parameters = {
    "max_depth": [2, 3, 5, 10],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 5, 8],
}
grid_rflf = GridSearchCV(
    rf_clf, param_grid=parameters, scoring="accuracy", cv=5
)
grid_rflf.fit(x_train, y_train)
print(f"최적 파라미터 : {grid_rflf.best_params_}")
print(f"최고 정확도 : {grid_rflf.best_score_}")
best_rflf = grid_rflf.best_estimator_
pred = best_rflf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print(accuracy)
# %%
# test 파일 예측
y_test_df = pd.read_csv("gender_submission.csv")
y_test_df = y_test_df.drop("PassengerId", axis=1)

x_test_df = pd.read_csv("test.csv")
x_test_df = transform_features(x_test_df)

# %%
# 결정트리 예측
pred = best_dclf.predict(x_test_df)
accuracy = accuracy_score(y_test_df, pred)
print(accuracy)
# %%
# 랜덤포레스트 예측
pred = best_rflf.predict(x_test_df)
accuracy = accuracy_score(y_test_df, pred)
print(accuracy)

# %%
# =================================================================
# 이하 아래 데이터 확인용 코드
# 문자열 분포도
print(titanic_df["Sex"].value_counts())
print(titanic_df["Embarked"].value_counts())
print(titanic_df["Cabin"].value_counts())


# %%
# 성별과 생존율 연관성
titanic_df.groupby(["Sex", "Survived"])["Survived"].count()
sns.barplot(x="Sex", y="Survived", data=titanic_df)

# %%
# 재산(티켓클래스)과 생존율 연관성
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=titanic_df)


# %%
# 나이별 카테고리 나누기
def get_category(age):
    cat = ""
    if age <= -1:
        cat = "Unknown"
    elif age <= 5:
        cat = "Baby"
    elif age <= 12:
        cat = "Child"
    elif age <= 18:
        cat = "Teenager"
    elif age <= 25:
        cat = "Student"
    elif age <= 35:
        cat = "Young Adult"
    elif age <= 60:
        cat = "Adult"
    else:
        cat = "Elderly"
    return cat


# %%
plt.figure(figsize=(10, 6))
group_names = [
    "Unknown",
    "Baby",
    "Child",
    "Teenager",
    "Student",
    "Young Adult",
    "Adult",
    "Elderly",
]
titanic_df["Age_cat"] = titanic_df["Age"].apply(lambda x: get_category(x))
sns.barplot(
    x="Age_cat", y="Survived", hue="Sex", data=titanic_df, order=group_names
)
titanic_df.drop("Age_cat", axis=1, inplace=True)


titanic_df = encode_features(titanic_df)
titanic_df.head()
# %%
