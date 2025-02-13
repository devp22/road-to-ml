import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import kagglehub
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

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

x = np.array([1,4,9,10,16,20,25]).reshape(-1,1)
y = np.array([4,8,18,22,34,50,60])
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.3)
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
plt.scatter(X_test,Y_test,c='Red')
plt.plot(X_test,Y_pred,c='Blue')
plt.show()
print("score: ",model.score(X_test,Y_test))
