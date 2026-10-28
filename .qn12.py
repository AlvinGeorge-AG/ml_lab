import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score,accuracy_score

df_train = pd.read_csv("./fashion-mnist_train.csv")
df_test = pd.read_csv("./fashion-mnist_test.csv")

df_train = df_train.sample(n=5000, random_state=42)
df_test = df_test.sample(n=2000, random_state=42)

X_train = df_train.drop(columns=["label"])
y_train = df_train["label"]

X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

X_train = X_train/255.0
X_test = X_test/255.0

models = ['linear','poly','rbf']

for model in models:
        svm_model = make_pipeline(
                StandardScaler(),
                SVC(kernel=model)
        )
        svm_model.fit(X_train,y_train)
        y_predict = svm_model.predict(X_test)
        
        print(f"{model} SVM Model : ")
        print(f"Accuracy : {accuracy_score(y_test,y_predict)}")
        print(f"F1_score : {f1_score(y_test,y_predict,average="weighted")}")
