import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('datasets/insurance.csv')
df["sex"] = [0 if i == "male" else 1 for i in df["sex"]]
df["smoker"] = [0 if i == "no" else 1 for i in df["smoker"]]
df["region"] = [0 if i == "southeast" else 1 if i == "southwest" else 2 if i == "northwest" else 3 for i in df["region"]]
numeric_df = df.select_dtypes(include="number")
numeric_df.fillna(numeric_df.mean(),inplace=True)

# BINOMIAL LOGISTIC REGRESSION
# y = df['sex']
# X = df.drop('sex',axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
# clf = LogisticRegression(max_iter=10000,random_state=0)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# score = accuracy_score(y_test,y_pred) * 100
# print(f"score: {score:.2f}%")

# MULTINOMIAL LOGISTIC REGRESSION

y = df['region']
X = df.drop('region',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
clf = LogisticRegression(max_iter=10000,random_state=0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test,y_pred) * 100
print(f"score: {score:.2f}%")