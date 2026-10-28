import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

df = pd.read_csv("./adult.csv")
df.replace('?',np.nan,inplace=True)

X = df.drop(columns=["income"])
y = df["income"]

categorical_features = ['workclass','education','marital-status','occupation','relationship','race','gender','native-country']
numerical_features = ['age','fnlwgt','educational-num','capital-gain','capital-loss','hours-per-week']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

numerical_transformer = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
)

categorical_transformer = make_pipeline(
        OneHotEncoder(handle_unknown='ignore'),
        SimpleImputer(strategy='most_frequent')
)

logistic_preprocessor = ColumnTransformer([
        ('num',numerical_transformer,numerical_features),
        ('cat',categorical_transformer,categorical_features)
])

logistic_model = make_pipeline(
        logistic_preprocessor,
        LogisticRegression(max_iter=2000)
)

logistic_model.fit(X_train,y_train)
y_logistic_pred = logistic_model.predict(X_test)

tree_preprocessor = ColumnTransformer([
        ('num',SimpleImputer(strategy='median'),numerical_features),
        ('cat',categorical_transformer,categorical_features)
])

dt_model = make_pipeline(
        tree_preprocessor,
        DecisionTreeClassifier(random_state=42)
)

dt_model.fit(X_train,y_train)
y_dt_predict = dt_model.predict(X_test)


print("Metrics:\n")
print("LogisticRegression : ")
print(f"Accuracy : {accuracy_score(y_test,y_logistic_pred)}")
print(f"Precision : {precision_score(y_test,y_logistic_pred,average="weighted")}")
print(f"f1-Score : {f1_score(y_test,y_logistic_pred,average="weighted")}")
print(f"recall_score : {recall_score(y_test,y_logistic_pred,average="weighted")}")

print("DecisionTreeClassifier : ")
print(f"Accuracy : {accuracy_score(y_test,y_dt_predict)}")
print(f"Precision : {precision_score(y_test,y_dt_predict,average="weighted")}")
print(f"f1-Score : {f1_score(y_test,y_dt_predict,average="weighted")}")
print(f"recall_score : {recall_score(y_test,y_dt_predict,average="weighted")}")

plt.figure(figsize=(30,30))
plot_tree(decision_tree=dt_model.named_steps['decisiontreeclassifier'],feature_names=dt_model[:-1].get_feature_names_out(),class_names=dt_model.named_steps['decisiontreeclassifier'].classes_,filled=True,max_depth=2)
plt.show()
