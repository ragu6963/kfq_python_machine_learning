#%%
import pandas as pd

# 데이터 로드
uselog = pd.read_csv("./use_log.csv")
customer = pd.read_csv("./customer_join.csv")
# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# %%
# 데이터 정규화
sc = StandardScaler()
# 정규화할 컬럼
customer_clustering = customer[
    ["mean", "median", "max", "min", "membership_period"]
]
# 정규화
customer_clustering_sc = sc.fit_transform(customer_clustering)

# %%
# 군집화
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(customer_clustering_sc)

#%%
customer_clustering["cluster"] = kmeans.labels_
# %%
customer_clustering["cluster"].unique()
# %%
customer_clustering.columns = [
    "월평균값",
    "월중앙값",
    "월최대값",
    "월최소값",
    "회원기간",
    "cluster",
]
# %%
customer_clustering.head()
# %%
# 군집별 데이터 분포
customer_clustering.groupby("cluster").mean()
# %%
# 차원 축소
from sklearn.decomposition import PCA

x = customer_clustering_sc
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.transform(x)
#%%
pca_df = pd.DataFrame(x_pca)
# %%
pca_df
# %%
pca_df["cluster"] = customer_clustering["cluster"]
#%%
customer_clustering["cluster"].unique()
# %%
import matplotlib.pyplot as plt

for i in customer_clustering["cluster"].unique():
    tmp = pca_df.loc[pca_df["cluster"] == i]
    plt.scatter(tmp[0], tmp[1])

plt.show()
# %%
# 군집화 결과와 차원 축소 결과 결합
customer_clustering = pd.concat([customer_clustering, customer], axis=1)

# %%
customer_clustering.head()
# %%
customer_clustering.groupby(["cluster", "is_deleted"], as_index=False).count()[
    ["cluster", "is_deleted", "customer_id",]
]

# %%
customer_clustering.groupby(["cluster", "routine_flg"], as_index=False).count()[
    ["cluster", "routine_flg", "customer_id",]
]

# %%
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
# %%
uselog
# %%
uselog["년월"] = uselog["usedate"].dt.strftime("%Y%m")

# %%
uselog_months = uselog.groupby(["년월", "customer_id"], as_index=False).count()

# %%
uselog_months
# %%
del uselog_months["usedate"]
# %%
uselog_months.rename(columns={"log_id": "count"}, inplace=True)
# %%
uselog_months
# %%
year_months = list(uselog_months["년월"].unique())
# %%
year_months
# %%
predict_data = pd.DataFrame()

# %%
for i in range(6, len(year_months)):
    # 각 월별 데이터 임시저장
    tmp = uselog_months.loc[uselog_months["년월"] == year_months[i]]
    # print(tmp)
    tmp.rename(columns={"count": "count_pred"}, inplace=True)
    for j in range(1, 7):
        # 최대 이전 6개월까지 저장
        tmp_before = uselog_months.loc[
            uselog_months["년월"] == year_months[i - j]
        ]
        # print(tmp_before)
        del tmp_before["년월"]
        tmp_before.rename(
            columns={"count": "count_{}".format(j - 1)}, inplace=True
        )
        tmp = pd.merge(tmp, tmp_before, on="customer_id", how="left")
    predict_data = pd.concat([predict_data, tmp], ignore_index=True)

# %%
predict_data
# %%
predict_data = predict_data.dropna()
#%%
predict_data.groupby(["customer_id"]).count()
# %%
predict_data = predict_data.reset_index(drop=True)

# %%
predict_data
# %%
customer.info()
# %%

predict_data = pd.merge(
    predict_data, customer[["customer_id", "start_date"]], on="customer_id"
)

# %%
predict_data
# %%
predict_data["now_date"] = pd.to_datetime(predict_data["년월"], format=("%Y%m"))

# %%
predict_data["start_date"] = pd.to_datetime(predict_data["start_date"])

# %%
predict_data.info()
# %%
predict_data["period"] = None


# %%
from dateutil.relativedelta import relativedelta

for i in range(len(predict_data)):
    delta = relativedelta(
        predict_data["now_date"][i], predict_data["start_date"][i]
    )
    predict_data["period"] = delta.years * 12 + delta.months
# %%
predict_data.info()
# %%
predict_data_1 = predict_data.loc[
    predict_data["start_date"] >= pd.to_datetime("20180401")
]

# %%
predict_data_1.reset_index(drop=True)
# %%
y_target = predict_data_1["count_pred"]
X_feature = predict_data_1[
    ["count_0", "count_1", "count_2", "count_3", "count_4", "count_5"]
]


# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_feature, y_target, test_size=0.25, random_state=156
)

# %%
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
lr.score(X_train, y_train)
#%%
X_test

# %%

x1 = [
    3,
    4,
    4,
    6,
    8,
    7,
]
x2 = [
    2,
    2,
    3,
    3,
    4,
    6,
]

x_pred = [x1, x2]
#%%
lr.predict(x_pred)


# %%
