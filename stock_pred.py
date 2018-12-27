# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 01:21:33 2018

@author: vishn
"""
"""
STOCK MARKET PREDICTION USING SMALL DATASET

"""

#import the pandas library
import pandas as pd

#create a dataframe of the existing historical data 
df=pd.read_csv('reli.csv')
#Delete Unwanted rows
del df["Date"]

#Append a row at the start and fill with reverse order index
df.insert(0,'dates',0)
df['dates']=df.index
df['dates']=df.dates.values[::-1]

#Check for trends for the date
#Date vs Price
import matplotlib.pyplot as plt
x1=df['dates']
y1=df['Price']
plt.plot(x1,y1)
plt.xlabel("DATE")
plt.ylabel("PRICE")
plt.legend()
plt.show()

#Date vs Open
#Date vs Price
x1=df['dates']
y1=df['Open']
plt.plot(x1,y1)
plt.xlabel("DATE")
plt.ylabel("OPEN")
plt.legend()
plt.show()

#Date vs High
x1=df['dates']
y1=df['High']
plt.plot(x1,y1)
plt.xlabel("DATE")
plt.ylabel("HIGH")
plt.legend()
plt.show()

#Date vs Low
x1=df['dates']
y1=df['Low']
plt.plot(x1,y1)
plt.xlabel("DATE")
plt.ylabel("LOW")
plt.legend()
plt.show()

#Graph for Price,Open vs Date
x=df['dates']
y1=df['Open']
y2=df['Price']
plt.plot(x,y1,'g',label='OPEN PRICE',linewidth=1)
plt.plot(x,y2,'r',label='CLOSE PRICE',linewidth=1)
plt.title("OPEN VS CLOSE")
plt.xlabel("DATE")
plt.ylabel("PRICE")
plt.legend()
plt.show()

#Divide the dataset into features and labels
features=df.iloc[:,[0,2]]
labels=df.iloc[:,[1,6]]
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,
                                                                       test_size=0.2,shuffle=False)

#Apply the feature scaling
from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
features_train=standardscaler.fit_transform(features_train)
features_test=standardscaler.fit_transform(features_test)

#Fitting the model to multiple linear regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(features_train,labels_train)

#predicting the test set results
test_results=regressor.predict(features_test)

#For predicting single values,Scale them and then use
print("Enter an integer:")
ip_date=int(input())
print("Enter the opening price: ")
op=float(input())
import numpy as np
parameters=np.array([ip_date,op]).reshape(1,-1)
parameters=pd.DataFrame(parameters)
parameters=standardscaler.fit_transform(parameters)
para_result=regressor.predict(parameters)
print("The predicted price and change % is: ")
print(para_result)

#Getting score for the Multi Linear Regressor model
score_mlr=regressor.score(features_train,labels_train)










