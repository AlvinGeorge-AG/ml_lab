import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

df = pd.read_csv("./WineQT.csv")
print(df.head())

X = df.drop(columns=["quality","Id"])
y = df["quality"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

hidden_layer_sizes = [(10,),(50,),(50,50),(90,10),(100,25)]

for layer in hidden_layer_sizes:
        model = make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=layer,random_state=42,max_iter=1000)
        )
        model.fit(X_train,y_train)
        y_predict = model.predict(X_test)
        
        print(f"NN layer = {(layer)} :\n")
        print(f"Accuracy : {accuracy_score(y_test,y_predict)}")
        print(f"F1_Score : {f1_score(y_test,y_predict,average='weighted')}")
        print(f"Confusion Matrix  : \n {confusion_matrix(y_test,y_predict)}")
