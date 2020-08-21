#%%
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = load_iris()
iris_df = pd.DataFrame(
    data=iris.data,
    columns=["sepal_length", "sepal_width", "petal_length", "petal_width"],
)
iris_df.head()
# %%
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300, random_state=0)
kmeans.fit(iris_df)
# %%
print(kmeans.labels_)

# %%
iris_df["target"] = iris.target
iris_df["cluster"] = kmeans.labels_
iris_result = iris_df.groupby(["target", "cluster"])["sepal_length"].count()
print(iris_result)

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_transformed = pca.fit_transform(iris.data)

iris_df["pca_x"] = pca_transformed[:, 0]
iris_df["pca_y"] = pca_transformed[:, 1]
iris_df.head()
# %%
marker0_ind = iris_df[iris_df["cluster"] == 0].index
marker1_ind = iris_df[iris_df["cluster"] == 1].index
marker2_ind = iris_df[iris_df["cluster"] == 2].index


plt.scatter(
    x=iris_df.loc[marker0_ind, "pca_x"],
    y=iris_df.loc[marker0_ind, "pca_y"],
    marker="o",
)

plt.scatter(
    x=iris_df.loc[marker1_ind, "pca_x"],
    y=iris_df.loc[marker1_ind, "pca_y"],
    marker="s",
)
plt.scatter(
    x=iris_df.loc[marker2_ind, "pca_x"],
    y=iris_df.loc[marker2_ind, "pca_y"],
    marker="^",
)

plt.show()

