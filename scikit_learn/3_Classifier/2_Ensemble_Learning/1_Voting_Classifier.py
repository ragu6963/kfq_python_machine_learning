"""
보통 방식의 앙상블을 구현한 VotingClassifier 클래스
"""
#%%
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head()

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier

lr_clf = LogisticRegression()
knn_clf = KNeighborsClassifier(n_neighbors=8)

vo_clf = VotingClassifier(
    estimators=[("LR", lr_clf), ("KNN", knn_clf)], voting="soft"
)

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=156
)

vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
print(f"Voting 정확도 : {accuracy_score(pred,y_test):0.4f}")

classifiers = [lr_clf, knn_clf]

for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    print(f"{classifier} 정확도 : {accuracy_score(pred,y_test):0.4f}")

# %%
