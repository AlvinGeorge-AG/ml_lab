import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000)
)

logistic_model.fit(X_train,y_train)
y_lr_predict = logistic_model.predict(X_test)

logistic_l1_model = make_pipeline(       #Lasso
        StandardScaler(),
        LogisticRegression(penalty='l1',solver="saga",max_iter=5000)
)

logistic_l1_model.fit(X_train,y_train)
y_l1_predict = logistic_l1_model.predict(X_test)

logistic_l2_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty='l2',max_iter=5000)
)

logistic_l2_model.fit(X_train,y_train)
y_l2_predict = logistic_l2_model.predict(X_test)


print("Classification Metrics :\n")
print("LogisticRegression : ")
print(f"Accuracy : {accuracy_score(y_test,y_lr_predict)}")
print(f"Classification report : \n",classification_report(y_test,y_lr_predict))

print("LogisticRegression L1: ")
print(f"Accuracy : {accuracy_score(y_test,y_l1_predict)}")
print(f"Classification report : \n",classification_report(y_test,y_l1_predict))


print("LogisticRegression L2: ")
print(f"Accuracy : {accuracy_score(y_test,y_l2_predict)}")
print(f"Classification report : \n",classification_report(y_test,y_l2_predict))
