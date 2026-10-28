import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

df = pd.read_csv("./housing.csv")

print(df.head())
df = df.dropna()

df["Average"] = df["total_rooms"]/df["households"]
df["median_house_value"] = df["median_house_value"]/100000

X = df[["Average"]]
y = df["median_house_value"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)

print("Model Performances :\n")
print("Mean Squared Error : ",mean_squared_error(y_test,y_predict))
print("Mean Absolute Error : ",mean_absolute_error(y_test,y_predict))
print("R2-Score : ",r2_score(y_test,y_predict))

plt.scatter(X_test,y_test,label="Actual Data")
plt.plot(X_test,y_predict,color="red",label="Predicted")
plt.xlabel("Average")
plt.ylabel("median_house_value")
plt.legend()
plt.show()
