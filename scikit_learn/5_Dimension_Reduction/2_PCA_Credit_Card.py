#%%
import pandas as pd
from scipy.sparse.construct import rand

df = pd.read_excel("credit_card_data.xls", header=1, sheet_name="Data").iloc[
    0:, 1:
]

# %%
df.head()
# %%
df.rename(
    columns={"PAY_0": "PAY_1", "default payment next month": "default"},
    inplace=True,
)
y_target = df["default"]
X_features = df.drop("default", axis=1)


# %%
df.head()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

corr = X_features.corr()
plt.figure(figsize=(14, 14))
sns.heatmap(corr, annot=True, fmt=".1g")
# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

cols_bill = ["BILL_AMT" + str(i) for i in range(1, 7)]

scaler = StandardScaler()
df_cols_scaled = scaler.fit_transform(X_features[cols_bill])
pca = PCA(n_components=2)
pca.fit(df_cols_scaled)
print(f"PCA 컴포넌트별 변동성 : {pca.explained_variance_ratio_}")
# %%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rcf = RandomForestClassifier(n_estimators=300, random_state=156)
# %%
# 원본 데이터 교차검증
scores = cross_val_score(rcf, X_features, y_target, scoring="accuracy", cv=3)
print("개별 정확도 :", scores)
print("평균 정확도 :", np.mean(scores))

# %%
# PCA(components = 6) 변환 데이터 교차검증
scaler = StandardScaler()
df_scaled = scaler.fit_transform(X_features)

pca = PCA(n_components=6)
df_pca = pca.fit_transform(df_scaled)
scores_pca = cross_val_score(rcf, df_pca, y_target, scoring="accuracy", cv=3)

print("개별 정확도 :", scores_pca)
print("평균 정확도 :", np.mean(scores_pca))

#%%
df_cols_scaled.shape
# %%
