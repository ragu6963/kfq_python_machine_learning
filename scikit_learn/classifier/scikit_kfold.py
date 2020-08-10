# KFold 와 StratifiedKFold 의 비교
# %%
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import StratifiedKFold


# %%
iris = load_iris()

# %%
features = iris.data
label = iris.target
# %%
df_clf = DecisionTreeClassifier(random_state=156)

# %%
# shuffle = True, index 섞임(default = False)
# %%


# %%
kfold = KFold(n_splits=5, shuffle=True)
skf = StratifiedKFold(n_splits=5)
cv_accuracy = []

n_iter = 0
for train_index, test_index in skf.split(features, label):
    # for train_index, test_index in kfold.split(features):
    # print(train_index)
    # print(test_index)
    # print("-------------------")
    x_train, x_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]
    df_clf.fit(x_train, y_train)
    pred = df_clf.predict(x_test)
    accuracy = np.round(accuracy_score(y_test, pred), 4)
    # print(accuracy)
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    print(
        f"\n{n_iter+1} 교차검증 정확도 : {accuracy}, 학습데이터 크기 : {train_size}, 검증데이터 크기 : {test_size}"
    )
    print(f"{n_iter+1} 검증 세트 인덱스 : {test_index}")
    n_iter += 1
    cv_accuracy.append(accuracy)


# %%
cv_accuracy

# %%
