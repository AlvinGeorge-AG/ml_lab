import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

degrees = [2,3,4,5,6,7]
df = pd.read_csv("./auto-mpg.csv")

df = df.dropna()

X = df[["displacement"]]
y = df["mpg"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

fig , axes = plt.subplots(2,3,figsize=(15,8))
axes = axes.flatten()
X_plot = np.linspace(X.min().iloc[0],X.max().iloc[0],300).reshape(-1,1)

for i,d in enumerate(degrees):
        poly = PolynomialFeatures(degree=d,include_bias=False)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        poly_model = LinearRegression()
        poly_model.fit(X_train_poly,y_train)
        y_poly_predict = poly_model.predict(X_test_poly)
        
        X_plot_poly = poly.transform(X_plot)
        y_plot = poly_model.predict(X_plot_poly)

        axes[i].scatter(X,y)
        axes[i].plot(X_plot,y_plot,color="red")
        axes[i].set_title(f"Degree={d}")
        axes[i].set_xlabel("Displacement")
        axes[i].set_ylabel("MPG")
        print(f"Polynomial Regression Results (Degree={d}) : ")
        print("mean_squared_error :",mean_squared_error(y_test,y_poly_predict))
        print("r2_score :",r2_score(y_test,y_poly_predict))
plt.tight_layout()
plt.show()
