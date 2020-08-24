#%%
import pandas as pd

# %%
uselog = pd.read_csv("./use_log.csv")

# %%
uselog.shape
# %%
uselog.head()
# %%
customer = pd.read_csv("./customer_master.csv")
# %%
customer.shape
# %%
customer.info()
# %%
class_master = pd.read_csv("./class_master.csv")

# %%
class_master.shape
# %%
class_master
# %%
campaign_master = pd.read_csv("./campaign_master.csv")

# %%
campaign_master.shape
# %%
campaign_master
# %%
# merge customer and class_master
customer_join = pd.merge(customer, class_master, on="class")
# %%
# merge customer_join and class_master
customer_join = pd.merge(customer_join, campaign_master, on="campaign_id")
# %%
customer_join.head()
# %%
customer_join.shape
# %%
customer_join.isnull().sum()
# %%
customer_join.groupby("class_name").count()
# %%
customer_join.groupby("campaign_name").count()

# %%
customer_join.groupby("gender").count()["customer_id"]

# %%
# start_date 타입 date로 변환
customer_join["start_date"] = pd.to_datetime(customer_join["start_date"])

# %%
customer_join.info()
# %%
customer_start = customer_join.loc[
    customer_join["start_date"] > pd.to_datetime("20180401")
]
# %%
customer_start.head()
# %%
# end_date 타입 date로 변환
customer_join["end_date"] = pd.to_datetime(customer_join["end_date"])

# %%
customer_join.info()

# %%
customer_newer = customer_join.loc[
    (customer_join["end_date"] >= pd.to_datetime("20190331"))
    | (customer_join["end_date"].isna())
]

# %%
customer_newer["end_date"].unique()

# %%
customer_newer.groupby("class_name").count()["customer_id"]

# %%
uselog["usedate"] = pd.to_datetime(uselog["usedate"])
# %%
uselog.info()
# %%
# 년월까지 끊기()
uselog["year_month"] = uselog["usedate"].dt.strftime("%Y%m")

# %%
uselog.groupby(["year_month", "customer_id"]).count()

# %%
uselog_months = uselog.groupby(
    ["year_month", "customer_id"], as_index=False
).count()

# %%
# 컬럼명 변경
uselog_months.rename(columns={"log_id": "count"}, inplace=True)


# %%
uselog_months.head()

# %%
# 컬럼 삭제
del uselog_months["usedate"]
# %%
uselog_customer = uselog_months.groupby("customer_id").agg(
    ["mean", "median", "max", "min"]
)["count"]
# %%
uselog_customer
# %%
uselog["weekday"] = uselog["usedate"].dt.weekday


# %%
uselog.groupby(["customer_id", "year_month", "weekday"], as_index=False).count()
# %%
uselog_week = uselog.groupby(
    ["customer_id", "year_month", "weekday"], as_index=False
).count()[["customer_id", "year_month", "weekday", "log_id"]]
# %%
uselog_week.rename(columns={"log_id": "count"}, inplace=True)
# %%
uselog_week = uselog_week.groupby(["customer_id"], as_index=False).max()[
    ["customer_id", "count"]
]


# %%
uselog_week["routine_flg"] = 0

# %%
uselog_week["routine_flg"] = uselog_week["routine_flg"].where(
    uselog_week["count"] < 4, 1
)
# %%
customer_join.head()
# %%
customer_join = pd.merge(customer_join, uselog_week, on="customer_id")
# %%
customer_join = pd.merge(customer_join, uselog_customer, on="customer_id")

# %%
customer_join.drop("count", axis=1, inplace=True)
# %%
customer_join.isnull().sum()
# %%
from dateutil.relativedelta import relativedelta

# %%
customer_join["calc_date"] = customer_join["end_date"]
# %%
customer_join["calc_date"] = customer_join["calc_date"].fillna(
    pd.to_datetime("20190430")
)


# %%
customer_join.info()
# %%
customer_join["membership_period"] = 0

# %%
for i in range(len(customer_join)):
    delta = relativedelta(
        customer_join.iloc[i]["calc_date"], customer_join.iloc[i]["start_date"]
    )
    customer_join["membership_period"].iloc[i] = delta.years * 12 + delta.months
    # print(delta.years * 12 + delta.months)

# %%
customer_join.head()

# %%
customer_join[["mean", "median", "max", "min"]].describe()


# %%
customer_join.groupby("routine_flg").count()["customer_id"]
# %%
import matplotlib.pyplot as plt

# %%
plt.hist(customer_join["membership_period"])

# %%
customer_end = customer_join.loc[customer_join["is_deleted"] == 1]

# %%
customer_end.describe()
# %%
customer_stay = customer_join.loc[customer_join["is_deleted"] == 0]
# %%
customer_stay.describe()

# %%
customer_join.to_csv("customer_join.csv", index=False)
# %%

