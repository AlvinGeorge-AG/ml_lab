import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report

df = pd.read_csv("./OnlineRetail.csv")
df = df.dropna()
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

customer_df = df.groupby("CustomerID").agg(
        total_quantity=('Quantity','sum'),
        total_spent=('TotalPrice','sum'),
        number_of_invoices=('InvoiceNo','nunique'),
        number_of_unique_products=('StockCode','nunique')
).reset_index()

customer_df["customer_segmentation"] = pd.qcut(
        customer_df["total_spent"],
        q=3,
        labels=["low","medium","high"]
)
print(customer_df.head())

X = customer_df.drop(columns=["customer_segmentation"])
y = customer_df["customer_segmentation"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model = DecisionTreeClassifier(criterion='entropy',random_state=42)
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print("Metrics :\n")
print(f"Accuracy : {accuracy_score(y_test,y_predict)}")
print(f"Classification report : \n {classification_report(y_test,y_predict)}")
print(f"Feature Importances : {model.feature_importances_}")

plt.figure(figsize=(20,10))
plot_tree(decision_tree=model,feature_names=X.columns,class_names=model.classes_,filled=True)
plt.show()


