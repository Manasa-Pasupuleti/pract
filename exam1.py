import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df = pd.read_csv('Advertising.csv')
print(df.head(10))

X = df.iloc[:, 0:3]  # X = df['TV','Radio','Newspaper'] required 2D so
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=51)



lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
ans=lr.predict(X_test)

print("Predict Advertisment:\n",ans)  # All the record