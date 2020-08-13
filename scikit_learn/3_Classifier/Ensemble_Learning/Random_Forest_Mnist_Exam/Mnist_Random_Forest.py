"""
숫자 인식 랜덤포레스트 예제
"""

# %%
""" 
데이터 불러와서 csv로 저장
"""
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784")
import pandas as pd

mnist_df = pd.DataFrame(data=mnist.data, columns=mnist.feature_names)
mnist_df["target"] = mnist.target

mnist_df.to_csv("./mnist.csv", index=False)

# %%
"""
csv 데이터 불러오기
"""
import pandas as pd

mnist_df = pd.read_csv("./mnist.csv")
#%%
X_data = mnist_df.drop("target", axis=1)
y_data = mnist_df["target"]
#%%
X_data.head()

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=156
)
# %%
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
# %%
from sklearn.metrics import accuracy_score

pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print(f"랜덤포레스트 예측 정확도 : {accuracy:0.4f}")
#%%

#%%
from PIL import Image
from matplotlib import pyplot as plt
import glob
import numpy as np

for image_path in glob.glob("./*.png"):
    img = Image.open(image_path)
    img = img.convert("L")
    img = np.array(img)
    img = img.reshape(28 * 28)
    for index in range(0, 784):
        if img[index] == 0:
            img[index] = 255
        elif img[index] == 255:
            img[index] = 0
        else:
            img[index] = 255 - img[index]

    img_pred = rf_clf.predict([img])
    print(img_pred)
    # plt.imshow(img)


# %%
