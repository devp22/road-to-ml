import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder,StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("datasets/penguins_lter.csv")
df.drop(columns=['Comments'],inplace=True)
df.drop(columns=['Individual ID'],inplace=True)
df.drop(columns=['Date Egg'],inplace=True)
cols = df.select_dtypes(include='object').columns
df[cols] = df[cols].apply(lambda x: x.fillna(x.mode().iloc[0]))
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_data = encoder.fit_transform(df[cols])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cols))
df.dropna(inplace=True,axis=1)
df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
y = df['Sex_MALE']
X = df.drop(columns=['Sex_MALE','Sex_FEMALE'])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.05,random_state=99)

# SINGLE-LAYER PERCEPTRON
 
perceptron = Perceptron()
perceptron.fit(X_train,y_train)
y_pred = perceptron.predict(X=X_test)
score = accuracy_score(y_true=y_test,y_pred=y_pred)
print("Score of singl-layered neural network is: ",score)

# MULTI-LAYER PERCEPTRON


#MLP models are highly sensitive to unscaled input data. Since Perceptron is not affected much by scaling, the improvement in MLP is minimal without proper preprocessing.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.01,random_state=100)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(30,10,5),activation='logistic',max_iter=99,random_state=99)
mlp.fit(X=X_train_scaled,y=y_train)
y_pred = mlp.predict(X=X_test_scaled)
score = accuracy_score(y_pred=y_pred,y_true=y_test)
print("Score of multi-layered neural model is: ",score)