import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("./iris.csv")
df["target"] = (df["Species"]=="Iris-setosa").astype(int)


X = df[["PetalLengthCm","PetalWidthCm"]]
y = df["target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = make_pipeline(
        StandardScaler(),
        SVC(kernel='linear',C=1)
)

model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print(f"Accuracy Score : {accuracy_score(y_test,y_predict)}")

xx,yy = np.meshgrid(
        np.linspace(X.iloc[:,0].min(),X.iloc[:,0].max(),100),
        np.linspace(X.iloc[:,1].min(),X.iloc[:,1].max(),100)
)

Z = model.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y)
plt.contour(xx,yy,Z,levels=[-1,0,1],colors="black",linestyles=['--','-','--'])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("SVM Decision Boundary and Margin")
plt.show()
