#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


# %%
X, y = make_blobs(
    n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0
)
print(X.shape, y.shape)
# %%
unique, counts = np.unique(y, return_counts=True)
print(unique, counts)
# %%
import pandas as pd

cluster_df = pd.DataFrame(data=X, columns=["ftr1", "ftr2"])
cluster_df["target"] = y
cluster_df.head()
# %%
target_list = np.unique(y)
markers = ["o", "s", "^"]

for target in target_list:
    target_cluster = cluster_df[cluster_df["target"] == target]
    plt.scatter(
        x=target_cluster["ftr1"],
        y=target_cluster["ftr2"],
        edgecolor="k",
        marker=markers[target],
    )
plt.show()

# %%
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=200, random_state=0)
cluster_labels = kmeans.fit_predict(X)
cluster_df["kmeans_label"] = cluster_labels

centers = kmeans.cluster_centers_
unique_labels = np.unique(cluster_labels)


# %%
markers = ["o", "s", "^"]

for label in unique_labels:
    label_cluster = cluster_df[cluster_df["kmeans_label"] == label]
    center_x_y = centers[label]
    plt.scatter(
        x=label_cluster["ftr1"],
        y=label_cluster["ftr2"],
        edgecolor="k",
        marker=markers[label],
    )
    plt.scatter(
        x=center_x_y[0],
        y=center_x_y[1],
        s=200,
        color="white",
        alpha=0.9,
        edgecolor="k",
        marker=markers[label],
    )
    plt.scatter(
        x=center_x_y[0],
        y=center_x_y[1],
        s=200,
        color="white",
        alpha=0.9,
        edgecolor="k",
        marker=markers[label],
    )

    plt.scatter(
        x=center_x_y[0],
        y=center_x_y[1],
        s=70,
        color="k",
        alpha=0.9,
        edgecolor="k",
        marker="$%d$" % label,
    )
plt.show()

# %%
cluster_df.groupby("target")["kmeans_label"].value_counts()

