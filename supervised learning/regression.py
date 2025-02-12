import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import kagglehub
from sklearn.linear_model import LinearRegression


# Linear regression is a statistical method that is used to predict a continuous dependent variable i.e target variable based on one or more independent variables. This technique assumes a linear relationship between the dependent and independent variables which means the dependent variable changes proportionally with changes in the independent variables.

df = pd.read_csv('datasets/insurance.csv')
numeric_df = df.select_dtypes(include="number")
# numeric_df.fillna(numeric_df.mean(),inplace=True)
numeric_df.dropna(inplace=True)
print(numeric_df)


# SIMPLE LINEAR REGRESSION (prediction with single feature)

# when you have arrays: the input, x, and the output, y. You should call .reshape() on x because this array must be two-dimensional, or more precisely, it must have one column and as many rows as necessary. That’s exactly what the argument (-1, 1) of .reshape() specifies.

# linear regression can be calculated using y = b0 + b1*x where b0 is intercept and b1 is slope

# The coefficient of determination, denoted as 𝑅², tells you which amount of variation in 𝑦 can be explained by the dependence on 𝐱, using the particular regression model. A larger 𝑅² indicates a better fit and means that the model can better explain the variation of the output with different inputs.
x = np.array(numeric_df['age']).reshape(-1,1)
y = np.array(numeric_df['charges'])
model = LinearRegression().fit(x,y)
# coefficient r2 can be determined using score method
r_sq = model.score(x,y)
print(f"R2 = {r_sq}")
print(f"Slope = {model.coef_}")
print(f"Intercept = {model.intercept_}")

predict_y = model.predict(np.array(32).reshape(-1,1))
print(f"predict: {predict_y}")