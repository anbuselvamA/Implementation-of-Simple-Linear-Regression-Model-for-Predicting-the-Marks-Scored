# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: A.Anbuselvam
RegisterNumber:  22009081
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
```

## Output:
## Head

![Screenshot 2023-09-12 153046](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/6575f437-5692-44f0-8989-281e27e5307c)

## Tail

![Screenshot 2023-09-12 153120](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/77cf9b37-7d29-44e2-9f20-38c3338903e2)

## Array value of x

![Screenshot 2023-09-12 153157](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/1ebc8a32-35c5-43fa-a694-760cec016eed)

## Array value of y

![Screenshot 2023-09-12 153226](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/986414f9-4e9b-43cb-95fd-64290280189d)

## Values of y prediction

![Screenshot 2023-09-12 153307](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/1e5d184b-aea6-4d0f-a3f6-8110db3bcaab)

## Y test

![Screenshot 2023-09-12 153342](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/c5f8246f-47ff-4e5d-9396-41e651150136)

## Training set graph

![Screenshot 2023-09-12 153420](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/acffa966-6c09-4b44-a724-7a04f00cf048)

## Test set graph

![Screenshot 2023-09-12 153456](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/4e4ca144-0a35-482b-bfcf-571eee61ec72)

## Valus of MSE,MAE,RMSE

![Screenshot 2023-09-12 153521](https://github.com/anbuselvamA/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559871/5c53c794-9869-4973-bee3-8bec17855077)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
