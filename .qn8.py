import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

df_train = pd.read_csv("./fashion-mnist_train.csv")
df_test = pd.read_csv("./fashion-mnist_test.csv")

X_train = df_train.drop(columns=["label"])
y_train = df_train["label"]

X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

X_train = X_train/255.0
X_test = X_test/255.0

k_values = [1,3,5,7,9]

for k in k_values:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train,y_train)
        y_predict = knn_model.predict(X_test)
        
        print(f"K={k} : \n")
        print(f"Accuracy : {accuracy_score(y_test,y_predict)}")
        print(f"F1-Score : {f1_score(y_test,y_predict,average="weighted")}")
