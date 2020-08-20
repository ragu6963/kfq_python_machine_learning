"""
GBM 
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
X_train, X_test, y_train, y_test = get_human_dataset()


#%%
from sklearn.ensemble import GradientBoostingClassifier
import time
from sklearn.metrics import accuracy_score

start_time = time.time()

gb_clf = GradientBoostingClassifier(random_state=0)
gb_clf.fit(X_train, y_train)
pred = gb_clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
end_time = time.time()
print(f"GBM 정확도 : {accuracy:0.4f}")
print(f"GBM 수행 시간 : {end_time - start_time }")


# %%

# %%
