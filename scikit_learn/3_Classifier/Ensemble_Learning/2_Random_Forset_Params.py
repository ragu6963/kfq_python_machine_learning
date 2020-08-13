"""
랜덤 포레스트 하이퍼 파라미터 및 튜닝
"""
# %%
# 중복 피처명 처리
def get_new_feature_name_df(old_feature_name_df):
    feature_dup_df = pd.DataFrame(
        data=old_feature_name_df.groupby("column_name").cumcount(),
        columns=["dup_cnt"],
    )
    feature_dup_df = feature_dup_df.reset_index()
    new_feature_name_df = pd.merge(
        old_feature_name_df.reset_index(), feature_dup_df, how="outer"
    )
    new_feature_name_df["column_name"] = new_feature_name_df[
        ["column_name", "dup_cnt"]
    ].apply(lambda x: x[0] + "_" + str(x[1]) if x[1] > 0 else x[0], axis=1)
    new_feature_name_df = new_feature_name_df.drop(["index"], axis=1)
    return new_feature_name_df


# %%
import pandas as pd


def get_human_dataset():
    feature_name_df = pd.read_csv(
        "./human_activity/features.txt",
        sep="\s+",
        header=None,
        names=["column_index", "column_name"],
    )
    new_feature_name_df = get_new_feature_name_df(feature_name_df)
    feature_name = new_feature_name_df.iloc[:, 1].values.tolist()
    X_train = pd.read_csv(
        "./human_activity/train/X_train.txt", sep="\s+", names=feature_name,
    )
    X_test = pd.read_csv(
        "./human_activity/test/X_test.txt", sep="\s+", names=feature_name,
    )
    y_train = pd.read_csv(
        "./human_activity/train/y_train.txt",
        sep="\s+",
        header=None,
        names=["action"],
    )
    y_test = pd.read_csv(
        "./human_activity/test/y_test.txt",
        sep="\s+",
        header=None,
        names=["action"],
    )
    return X_train, X_test, y_train, y_test


# %%
feature_name_df = pd.read_csv(
    "./human_activity/features.txt",
    sep="\s+",
    header=None,
    names=["column_index", "column_name"],
)
feature_name = feature_name_df.iloc[:, 1].values.tolist()
feature_name[:10]

# %%
params = {
    "n_estimators": [100],
    "max_depth": [6, 8, 10, 12],
    "min_samples_leaf": [8, 12, 18],
    "min_samples_split": [8, 16, 20],
}
X_train, X_test, y_train, y_test = get_human_dataset()
#%%
# 로지스틱 회긔 예측 모델

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
grid_cv = GridSearchCV(rf_clf, param_grid=params, cv=2)
grid_cv.fit(X_train, y_train)

print(f"최적 하이퍼 라라미터 : \n{grid_cv.best_params_}")
print(f"최고 예측 정확도 : \n{grid_cv.best_score_}")


# %%
from sklearn.metrics import accuracy_score

rf_clf1 = RandomForestClassifier(
    random_state=0,
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=8,
    min_samples_split=8,
)
rf_clf1.fit(X_train, y_train)
# %%
pred = rf_clf1.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(f"정확도 : {accuracy:0.4f}")


# %%
