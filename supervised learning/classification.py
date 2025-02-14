import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv('datasets/insurance.csv')
# df["sex"] = [0 if i == "male" else 1 for i in df["sex"]]
# df["smoker"] = [0 if i == "no" else 1 for i in df["smoker"]]
# df["region"] = [0 if i == "southeast" else 1 if i == "southwest" else 2 if i == "northwest" else 3 for i in df["region"]]
# numeric_df = df.select_dtypes(include="number")
# numeric_df.fillna(numeric_df.mean(),inplace=True)

# ---------------------------LOGISTIC REGRESSION--------------------------

# 1. BINOMIAL LOGISTIC REGRESSION
# y = df['sex']
# X = df.drop('sex',axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
# clf = LogisticRegression(max_iter=10000,random_state=0)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# score = accuracy_score(y_test,y_pred) * 100
# print(f"score: {score:.2f}%")

# 2. MULTINOMIAL LOGISTIC REGRESSION

# y = df['region']
# X = df.drop('region',axis=1)
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
# clf = LogisticRegression(max_iter=10000,random_state=0)
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)
# score = accuracy_score(y_test,y_pred) * 100
# print(f"score: {score:.2f}%")

# ---------------------------DECISION TREE---------------------------

df = pd.read_csv('datasets/titanic.csv')
df.drop(['Name','Ticket','Fare','Cabin'],axis=1,inplace=True)
labelEncoder = LabelEncoder()
labels = labelEncoder.fit_transform(df['Sex'])
df['Sex'] = labels
labels = labelEncoder.fit_transform(df['Embarked'])
df['Embarked'] = labels
df.fillna(df.mean(),inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
y = df['Pclass']
X = df.drop('Pclass',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
dtc = DecisionTreeClassifier(random_state=1)
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
score = accuracy_score(y_pred=y_pred,y_true=y_test)
print("Score: ",score*100,"%")