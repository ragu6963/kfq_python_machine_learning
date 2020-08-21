#%%
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt

# %%
iris = load_iris()
# %%
columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_df = pd.DataFrame(iris.data, columns=columns)
iris_df["target"] = iris.target
iris_df.head()
# %%
markers = ["^", "s", "o"]

for i, marker in enumerate(markers):
    x_axis_data = iris_df[iris_df["target"] == i]["sepal_length"]
    y_axis_data = iris_df[iris_df["target"] == i]["sepal_width"]
    plt.scatter(
        x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i]
    )
plt.legend()
plt.show()

# %%
# PCA를 적용하기 전에 개별 속성을 함께 스케일링
from sklearn.preprocessing import StandardScaler

iris_scaled = StandardScaler().fit_transform(iris_df.iloc[:, :-1])

# %%

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

# fit 과 transform을 호출해 PCA 변환
pca.fit(iris_scaled)
iris_pca = pca.transform(iris_scaled)
print(iris_pca.shape)
# %%
# PCA 변환된 데이터로 다시 데이터프레임으로
pca_columns = ["pca_component_1", "pca_component_2", "pca_component_3"]
iris_df_pca = pd.DataFrame(iris_pca, columns=pca_columns)
iris_df_pca["target"] = iris.target

# %%
markers = ["^", "s", "o"]

for i, marker in enumerate(markers):
    x_axis_data = iris_df_pca[iris_df_pca["target"] == i]["pca_component_1"]
    y_axis_data = iris_df_pca[iris_df_pca["target"] == i]["pca_component_3"]
    plt.scatter(
        x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i]
    )
plt.legend()
plt.show()
# %%
# PCA 컴포넌트별로 차지하는 변동성 비율
print(pca.explained_variance_ratio_)
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

rcf = RandomForestClassifier(random_state=156)
scores = cross_val_score(rcf, iris.data, iris.target, scoring="accuracy", cv=3)
print("개별 정확도 :", scores)
print("평균 정확도 :", np.mean(scores))
# %%
pca_X = iris_df_pca[["pca_component_1", "pca_component_2"]]
scores_pca = cross_val_score(rcf, pca_X, iris.target, scoring="accuracy", cv=3)
print("개별 정확도 :", scores_pca)
print("평균 정확도 :", np.mean(scores_pca))
