#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

iris = load_iris()
iris_scaled = StandardScaler().fit_transform(iris.data)
lda = LinearDiscriminantAnalysis(n_components=2)
# PCA와 다르게 target 값 필요
lda.fit(iris_scaled, iris.target)
iris_lda = lda.transform(iris_scaled)
print(iris_lda.shape)


# %%
import pandas as pd
import matplotlib.pyplot as plt

lda_columns = ["lda_1", "lda_2"]
iris_df_lda = pd.DataFrame(iris_lda, columns=lda_columns)
iris_df_lda["target"] = iris.target

markers = ["^", "s", "o"]

for i, marker in enumerate(markers):
    x_axis_data = iris_df_lda[iris_df_lda["target"] == i]["lda_1"]
    y_axis_data = iris_df_lda[iris_df_lda["target"] == i]["lda_2"]
    plt.scatter(
        x_axis_data, y_axis_data, marker=marker, label=iris.target_names[i]
    )
plt.legend()
plt.show()
                                                                                    # %%
