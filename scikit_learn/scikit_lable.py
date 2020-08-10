# %%
from sklearn.preprocessing import LabelEncoder

items = ["TV", "냉장고", "전자레인지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]

# %%
encoder = LabelEncoder()
encoder.fit(items)

# %%
labels = encoder.transform(items)

# %%
print(labels)
print(encoder.classes_)
# %%
from sklearn.preprocessing import OneHotEncoder

items = ["TV", "냉장고", "전자레인지", "컴퓨터", "선풍기", "선풍기", "믹서", "믹서"]


# %%
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1, 1)

# %%
oh_encoder = OneHotEncoder()

oh_encoder.fit(labels)

# %%
oh_labels = oh_encoder.transform(labels)


# %%
print(oh_labels.toarray())

# %%
