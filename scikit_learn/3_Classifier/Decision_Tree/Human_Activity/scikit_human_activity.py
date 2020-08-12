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
X_train, X_test, y_train, y_test = get_human_dataset()

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
# %%
pred = dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"결정 트리 예측 정확도 : {accuracy:0.4f}")
print(f"결정 트리 기본 하이퍼 파라미터\n{dt_clf.get_params()}")

# %%
from sklearn.model_selection import GridSearchCV

params = {
    "max_depth": [6, 8, 10],
    "min_samples_split": [16, 24],
    "min_samples_leaf": [4, 8, 12],
}
grid_cv = GridSearchCV(
    dt_clf, param_grid=params, cv=5, scoring="accuracy", verbose=1
)
grid_cv.fit(X_train, y_train)
print(f"최고 평균 정확도 수치 : {grid_cv.best_score_:0.4f}")
print(f"최고 하이퍼 파라미터 : {grid_cv.best_params_}")

# %%
# max_depth 별 정확도
max_depth = [6, 8, 10, 12, 16, 20, 24]
for depth in max_depth:
    dt_clf = DecisionTreeClassifier(max_depth=depth, random_state=156)
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"max_depth = {depth} 정확도 : {accuracy:0.4f}")

# %%
best_dt_clf = grid_cv.best_estimator_
pred1 = best_dt_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred1)
print(f"결정 트리 예측 정확도 : {accuracy:0.4f}")