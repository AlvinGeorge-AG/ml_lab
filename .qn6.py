import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_score,f1_score

df = pd.read_csv("./diabetes.csv")
df = df.dropna()

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = make_pipeline(
        StandardScaler(),
        SimpleImputer(strategy='median'),
        LogisticRegression(max_iter=1000)
)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

model2 = make_pipeline(
        SimpleImputer(strategy='median'),
        LogisticRegression(max_iter=1000)
)

model2.fit(X_train,y_train)
y_pred2 = model2.predict(X_test)

print("Metrics Without Feature Scaling :")
print(f"Accuracy : {accuracy_score(y_test,y_pred2)}")
print(f"Recall : {recall_score(y_test,y_pred2)}")
print(f"Precision : {precision_score(y_test,y_pred2)}")
print(f"F1-Score : {f1_score(y_test,y_pred2)}")
print(f"Classification Report : \n {classification_report(y_test,y_pred2)}")

print("Metrics With Feature Scaling :")
print(f"Accuracy : {accuracy_score(y_test,y_pred)}")
print(f"Recall : {recall_score(y_test,y_pred)}")
print(f"Precision : {precision_score(y_test,y_pred)}")
print(f"F1-Score : {f1_score(y_test,y_pred)}")
print(f"Classification Report : \n {classification_report(y_test,y_pred)}")
