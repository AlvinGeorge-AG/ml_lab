import time
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

df_train = pd.read_csv("./mnist_train.csv")
df_test = pd.read_csv("./mnist_test.csv")

df_train = df_train.sample(n=5000, random_state=42)
df_test = df_test.sample(n=2000, random_state=42)

X_train = df_train.drop(columns=["label"])
y_train = df_train["label"]
X_test = df_test.drop(columns=["label"])
y_test = df_test["label"]

activations = ['logistic','relu','tanh']
results = {}

for activation in activations:
        print(f"Training {activation}... ",flush=True)
        model = make_pipeline(
                StandardScaler(),
                MLPClassifier(activation=activation,max_iter=100,random_state=42,hidden_layer_sizes=(50,50))
        )

        time_start = time.time()
        model.fit(X_train,y_train)
        time_stop = time.time()

        y_predict = model.predict(X_test)

        results[activation] = {
                "accuracy_score":accuracy_score(y_test,y_predict),
                "f1_score":f1_score(y_test,y_predict,average='weighted'),
                "confusion_matrix":confusion_matrix(y_test,y_predict),
                "time":time_stop-time_start
        }

for i,data in results.items():
        print(f"Activation : {i}")
        print(f"accuracy_score : {data["accuracy_score"]}")
        print(f"f1_score : {data["f1_score"]}")
        print(f"Training Time : {data["time"]}")
        print(f"confusion_matrix : \n {data["confusion_matrix"]}")
