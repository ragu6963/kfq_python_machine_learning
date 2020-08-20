# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bike_df = pd.read_csv("./train.csv")
#%%
bike_df.head()

# %%
bike_df.info()
# %%
bike_df.isnull().sum()
# %%
# 문자열 datetime 타입으로 변환
bike_df.datetime = bike_df.datetime.apply(pd.to_datetime)
# %%
# 날짜 추출 및 날짜 삭제
bike_df["year"] = bike_df.datetime.apply(lambda x: x.year)
bike_df["month"] = bike_df.datetime.apply(lambda x: x.month)
bike_df["day"] = bike_df.datetime.apply(lambda x: x.day)
bike_df["hour"] = bike_df.datetime.apply(lambda x: x.hour)

# %%
# 필요없는 데이터 삭제
drops = ["datetime", "casual", "registered"]
bike_df.drop(drops, axis=1, inplace=True)
# %%
# RMSLE 측정 도구
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmsle(y, pred):
    log_y = np.log1p(y)
    log_pred = np.log1p(pred)
    squared_error = (log_y - log_pred) ** 2
    rmsle = np.sqrt(np.mean(squared_error))
    return rmsle


def rmse(y, pred):
    return np.sqrt(mean_squared_error(y, pred))


def evaluate_regr(y, pred):
    rmsle_val = rmsle(y, pred)
    rmse_val = rmse(y, pred)
    mae_val = mean_absolute_error(y, pred)
    print(
        f"RMSLE : {rmsle_val:0.3f}, RMSE : {rmse_val:0.3f}, MAE : {mae_val:0.3f}"
    )


# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso

y_target = bike_df["count"]
X_features = bike_df.drop(["count"], axis=1, inplace=False)
#%%
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target, test_size=0.3, random_state=0
)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

evaluate_regr(y_test, pred)

# %%
# 오차가 큰 column 확인
def get_top_error_data(y, pred, n_tops=5):
    result_df = pd.DataFrame(y_test.values, columns=["real_count"])
    result_df["predicted_count"] = np.round(pred)
    result_df["diff"] = abs(
        result_df["real_count"] - result_df["predicted_count"]
    )
    print(result_df.sort_values("diff", ascending=False)[:n_tops])


get_top_error_data(y_test, pred)
# %%
# target value 분포도 확인
y_target.hist()

# %%
# log1p()를 사용하여 왜곡 수정, 정규분포 형태로 변환
y_target_transform = np.log1p(y_target)
y_target_transform.hist()
# %%
# 재학습
y_target_log = np.log1p(y_target)
X_train, X_test, y_train, y_test = train_test_split(
    X_features, y_target_log, test_size=0.3, random_state=0
)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

y_test_exp = np.expm1(y_test)
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp, pred_exp)

# %%
# features 회귀 계수 값 확인
coef = pd.Series(lr_reg.coef_, index=X_features.columns)
coef_sort = coef.sort_values(ascending=False)
sns.barplot(x=coef_sort, y=coef_sort.index)
# %%
# 날짜 feature One-Hot Encoding
X_features_ohe = pd.get_dummies(
    X_features,
    columns=[
        "year",
        "month",
        "day",
        "hour",
        "holiday",
        "workingday",
        "season",
        "weather",
    ],
)
#%%
X_features_ohe
# %%
#  ne-Hot Encoding 후 재학습
y_target_log = np.log1p(y_target)
X_train, X_test, y_train, y_test = train_test_split(
    X_features_ohe, y_target_log, test_size=0.3, random_state=0
)
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
pred = lr_reg.predict(X_test)

y_test_exp = np.expm1(y_test)
pred_exp = np.expm1(pred)

evaluate_regr(y_test_exp, pred_exp)
# %%
# 모델에 따른 평가 수치 반환 함수
def get_model_predict(model, features, target, is_expm1=False):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.3, random_state=0
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1:
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print(f"### {model} ###")
    evaluate_regr(y_test, pred)
    print("-----------------------------------")


# %%
# 회귀모델
lr_reg = LinearRegression()
ridge_reg = Ridge(alpha=10)
lasso_reg = Lasso(alpha=0.01)
get_model_predict(lr_reg, X_features_ohe, y_target_log, True)
get_model_predict(ridge_reg, X_features_ohe, y_target_log, True)
get_model_predict(lasso_reg, X_features_ohe, y_target_log, True)

# 트리모델
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

rf_reg = RandomForestRegressor(n_estimators=500)
gbm_reg = GradientBoostingRegressor(n_estimators=500)
xgb_reg = XGBRegressor(n_estimators=500)
lgbm_reg = LGBMRegressor(n_estimators=500)
get_model_predict(rf_reg, X_features_ohe, y_target_log, True)
get_model_predict(gbm_reg, X_features_ohe, y_target_log, True)
get_model_predict(xgb_reg, X_features_ohe, y_target_log, True)
get_model_predict(lgbm_reg, X_features_ohe, y_target_log, True)

#%%
# features 회귀 계수 값 확인
coef = pd.Series(lr_reg.coef_, index=X_features_ohe.columns)
coef_sort = coef.sort_values(ascending=False)[:20]
sns.barplot(x=coef_sort, y=coef_sort.index)
#%%
test_data = pd.read_csv("./test.csv")
test_data.info()
