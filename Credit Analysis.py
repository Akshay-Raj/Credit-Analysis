import pandas as pd
import numpy as np


train=pd.read_csv("/train.csv")
test=pd.read_csv("/test.csv")


#Sample of data
train.head(10)

#Types of data columns
train.dtypes

#Find missing values
train.isnull().sum()
test.isnull().sum()

#Impute missing values with mean value
train.fillna(train.mean(),inplace=True) 
train.isnull().sum() 

#Test data
test.fillna(test.mean(),inplace=True) 
test.isnull().sum()

#Impute missing values with mode value
train.Gender.fillna(train.Gender.mode()[0],inplace=True)
train.Married.fillna(train.Married.mode()[0],inplace=True)
train.Dependents.fillna(train.Dependents.mode()[0],inplace=True) 
train.Self_Employed.fillna(train.Self_Employed.mode()[0],inplace=True)  
train.isnull().sum() 

#Test data
test.Gender.fillna(test.Gender.mode()[0],inplace=True)
test.Dependents.fillna(test.Dependents.mode()[0],inplace=True) 
test.Self_Employed.fillna(test.Self_Employed.mode()[0],inplace=True)  
test.isnull().sum() 

#Outliers Treatment
train.Loan_Amount_Term=np.log(train.Loan_Amount_Term)


train=train.drop('Loan_ID',axis=1)
test=test.drop('Loan_ID',axis=1)

#Create target variable
X=train.drop('Loan_Status',1)
y=train.Loan_Status

#Build dummy variables for categorical variables
X=pd.get_dummies(X)
train=pd.get_dummies(train)
test=pd.get_dummies(test)

#Split train data for cross validation
from sklearn.model_selection import train_test_split
x_train,x_cv,y_train,y_cv = train_test_split(X,y,test_size=0.2)

#(1)LOGISTIC REGRESSION ALGORITHM
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

pred_cv=model.predict(x_cv)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
accuracy_score(y_cv,pred_cv) #78.86%
matrix=confusion_matrix(y_cv,pred_cv)
print(matrix)


#(2)DECISION TREE ALGORITHM

from sklearn import tree
dt=tree.DecisionTreeClassifier(criterion='gini')
dt.fit(x_train,y_train)

pred_cv1=dt.predict(x_cv)

accuracy_score(y_cv,pred_cv1) #71.54%
matrix1=confusion_matrix(y_cv,pred_cv1)
print(matrix1)


#(3)RANDOM FOREST ALGORITHM
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)

pred_cv2=rf.predict(x_cv)

accuracy_score(y_cv,pred_cv2) #77.23%
matrix2=confusion_matrix(y_cv,pred_cv2)
print(matrix2)


#(3)SUPPORT VECTOR MACHINE (SVM) ALGORITHM
from sklearn import svm
svm_model=svm.SVC()
svm_model.fit(x_train,y_train)

pred_cv3=svm_model.predict(x_cv)

accuracy_score(y_cv,pred_cv3) #64.23%
matrix3=confusion_matrix(y_cv,pred_cv3)
print(matrix3)


#(4)NAIVE BAYES ALGORITHM
from sklearn.naive_bayes import GaussianNB 
nb=GaussianNB()
nb.fit(x_train,y_train)

pred_cv4=nb.predict(x_cv)

accuracy_score(y_cv,pred_cv4) #80.49%
matrix4=confusion_matrix(y_cv,pred_cv4)
print(matrix4)


#(5)K-NEAREST NEIGHBOR(kNN) ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
kNN=KNeighborsClassifier()
kNN.fit(x_train,y_train)

pred_cv5=kNN.predict(x_cv)

accuracy_score(y_cv,pred_cv5) #64.23%
matrix5=confusion_matrix(y_cv,pred_cv5)
print(matrix5)


#(6)GRADIENT BOOSTING MACHINE ALGORITHM
from sklearn.ensemble import GradientBoostingClassifier
gbm=GradientBoostingClassifier()
gbm.fit(x_train,y_train)

pred_cv6=gbm.predict(x_cv)

accuracy_score(y_cv,pred_cv6) #78.86%
matrix6=confusion_matrix(y_cv,pred_cv6)
print(matrix6)

#Select best model in order of accuracy
#Naive Bayes - 80.49%
#Logistic Regression - 78.86%
#Gradient Boosting Machine -78.86%
#Random Forest - 77.23%
#Decision Tree - 71.54%
#Support Vector Machine - 64.23%
#k-Nearest Neighbors(kNN) - 64.23%

#Predict values using test data (Naive Bayes)
pred_test=nb.predict(test)

#Write test results in csv file
predictions=pd.DataFrame(pred_test, columns=['predictions']).to_csv('Credit_Predictions.csv')
