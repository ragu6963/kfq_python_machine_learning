"""
0 혹은 1로 치중되어 있는 데이터의 경우에는
아무것도 하지 않아도 accuracy 가 높게 나올 수 있다
아래 코드는 그 예시
"""
# %%
from sklearn.base import BaseEstimator
from sklearn import preprocessing
import numpy as np

# %%
class MyDummyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            # 남성
            if X["Sex"].iloc[i] == 1:
                # 사망
                pred[i] = 0
            else:
                # 생존
                pred[i] = 1
        return pred


# %%
# 결측값 처리
def fillna(df):
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df["Cabin"].fillna("N", inplace=True)
    df["Embarked"].fillna("N", inplace=True)
    df.isnull().sum().sum()
    return df


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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

titanic_df = pd.read_csv("train.csv")
y_titanic_df = titanic_df["Survived"]
x_titanic_df = titanic_df.drop("Survived", axis=1)
x_titanic_df = transform_features(x_titanic_df)
x_train, x_test, y_train, y_test = train_test_split(
    x_titanic_df, y_titanic_df, test_size=0.2, random_state=0
)


# %%
myclf = MyDummyClassifier()
myclf.fit(x_train, y_train)

mypred = myclf.predict(x_test)
print(accuracy_score(y_test, mypred))

# %%
class MyDummy(BaseEstimator):
    def fit(self, x, y=None):
        pass

    def predict(self, x):
        return np.zeros((len(x), 1), dtype=bool)


# %%
from sklearn.datasets import load_digits

digits = load_digits()
y = (digits.target == 7).astype(int)
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, y, random_state=11
)


# %%
fackclf = MyDummy()
fackclf.fit(x_train, y_train)
pred = fackclf.predict(x_test)
print(accuracy_score(y_test, pred))

# %%
