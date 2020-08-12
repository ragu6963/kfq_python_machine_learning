"""
레이블 인코딩(Label Encoding)과 원-핫 인코딩(One Hot Encoding)
문자형 피처를 코드형 숫자 값으로 변환하는것
"""
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
