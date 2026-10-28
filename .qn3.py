import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,r2_score

data = load_diabetes(as_frame=True)

X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

linear_model = make_pipeline(
        StandardScaler(),
        LinearRegression()
)
linear_model.fit(X_train,y_train)
y_lr_predict = linear_model.predict(X_test)

ridge_pipeline = make_pipeline(
        StandardScaler(),
        Ridge()
)

ridge_params = {"ridge__alpha":[0.01,0.1,1,10,100]}

ridge_grid = GridSearchCV(
        ridge_pipeline,
        param_grid=ridge_params,
        cv=5
)

ridge_grid.fit(X_train,y_train)
y_ridge_predict = ridge_grid.predict(X_test)

lasso_pipeline = make_pipeline(
        StandardScaler(),
        Lasso()
)

lasso_params = {"lasso__alpha":[0.01,0.1,1,10,100]}

lasso_grid = GridSearchCV(
        lasso_pipeline,
        param_grid=lasso_params,
        cv=5
)
lasso_grid.fit(X_train,y_train)
y_lasso_predict = lasso_grid.predict(X_test)


print("best hyperparameters:")
print(f"Ridge : {ridge_grid.best_params_}")
print(f"Lasso : {lasso_grid.best_params_}")

print("Metrics :")
print("Linear Regression:")
print(f"Mean squared error : {mean_squared_error(y_test,y_lr_predict)}")
print(f"R2 score : {r2_score(y_test,y_lr_predict)}")

print("Lasso Regression:")
print(f"Mean squared error : {mean_squared_error(y_test,y_lasso_predict)}")
print(f"R2 score : {r2_score(y_test,y_lasso_predict)}")

print("Ridge Regression:")
print(f"Mean squared error : {mean_squared_error(y_test,y_ridge_predict)}")
print(f"R2 score : {r2_score(y_test,y_ridge_predict)}")
