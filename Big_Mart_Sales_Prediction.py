#----------------------------------about dataset
#this is shape of dataset (8523, 12)
#this is columns name of dataset ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility','Item_Type', 'Item_MRP', 'Outlet_Identifier','Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type','Outleype', 'Item_Outlet_Sales'],


#------------------------------------work flow
#data collection
#data pre-processing
#data anlysis
#train test data
#XGBoost Regression
#Evaluation
#--------------------------------------import labrary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder#it is used to convert string data into categories data

#-----------------------------------------data analysis
data = pd.read_csv("C:/Users/kunde/all vs code/ml prject/Train.csv")
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())
print(data.isnull().sum())#IT IS USED TO FIND ANY NULL VALUES IN DATASET
print(data.head(5))
print(data.tail(5))
#handing missing values
#we will replace missing value with mean of that columns
data["Item_Weight"].fillna(data["Item_Weight"].mean(), inplace=True)
#and also we can use missing value as mode (maximun repatation)
mode_of_outlet_size = data.pivot_table(values="Outlet_Size", columns="Outlet_Type", aggfunc=(lambda x:x.mode()[0]))
missing_value = data["Outlet_Size"].isnull()
data.loc[missing_value, "Outlet_Size"]= data.loc[missing_value, "Outlet_Type"].apply(lambda x: mode_of_outlet_size)
print(data.isnull().sum())
print(data.head(5))
sns.set()
sns.displot(data["Item_Weight"])
plt.show()
sns.displot(data["Item_Visibility"])
plt.show()
sns.countplot(x="Outlet_Establishment_Year", data=data)
plt.show()


#-----------------------------------------data separation

y = data["Item_Outlet_Sales"]
print(data["Item_Identifier"].value_counts())
print(data["Item_Fat_Content"].value_counts())
print(data["Outlet_Size"].value_counts())
print(data["Outlet_Location_Type"].value_counts())
print(data["Outlet_Type"].value_counts())
print(data["Item_Type"].value_counts())
print(data["Outlet_Identifier"].value_counts())
data.replace({"Item_Fat_Content":{"LF":"Low Fat", "reg":"Regular", "low fat":"Low Fat"}}, inplace=True)
#data.replace({"Outlet_Size": {"Medium": 1, "[Grocery store]" : 2, "Small":3, "High":4}}, inplace=True)
#data.replace({"Outlet_Location_Type": {"Tier 3":1, "Tier 2":2, "Tier" : 3}}, inplace=True)
#data.replace({"Outlet_Type":{"Supermarket Type1":1, "Grocery Store":2, "Supermarket Type3": 3, "Supermarket Type2": 4}}, inplace=True)
#data.replace({"Item_Type":{"Fruits and Vegetables":1,"Snack Foods":2,"Household":3,"Frozen Foods":4,"Dairy":5,"Canned":6,"Baking Goods":7,"Health and Hygiene":8, "Soft Drinks":9,"Meat":10,"Breads":11,"Hard Drinks":12,"Others":13,"Starchy Foods":14,"Breakfast":15,"Seafood":16}}, inplace=True)
#data.replace({"Outlet_Identifier": {"OUT027": 1,"OUT013": 2,"OUT049": 3,"OUT046": 4,"OUT035": 5,"OUT045": 6,"OUT018": 7,"OUT017": 8,"OUT010": 9,"OUT019": 10}}, inplace=True)
#x = data.drop(columns=["Item_Identifier", "Item_Outlet_Sales"], axis=1)
#print(x.head(5))
encoder = LabelEncoder()
data["Item_Identifier"]= encoder.fit_transform(data["Item_Identifier"])
data["Item_Type"]= encoder.fit_transform(data["Item_Type"])
data["Outlet_Type"]= encoder.fit_transform(data["Outlet_Type"])
#data["Outlet_Size"]= encoder.fit_transform(data["Outlet_Size"])
data["Item_Fat_Content"]= encoder.fit_transform(data["Item_Fat_Content"])
data["Outlet_Location_Type"]= encoder.fit_transform(data["Outlet_Location_Type"])
data["Outlet_Identifier"]= encoder.fit_transform(data["Outlet_Identifier"])
print(data.head(5))
x = data.drop(columns=["Outlet_Size", "Item_Outlet_Sales"], axis=1)

#-----------------------------------------tarin-test-split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
print(x.shape, x_train.shape, x_test.shape)
print(y.shape, y_train.shape, y_test.shape)

#---------------------------------------------model selection and use 
model = XGBRegressor()
model.fit(x_train, y_train)

#----------------------------------------------prediction of train data
y_tr = model.predict(x_train)
accur = metrics.r2_score(y_train, y_tr)
print(accur, "this is prediction of train data")

#----------------------------------------------prediction of test data
y_te = model.predict(x_test)
accur = metrics.r2_score(y_te, y_test)
print(accur, "this is test data prediction")
#---------------------------------------------single data predcitions
de = [1,9.3,0,0.016047301,4,249.8092,2,1999,0,0]#3735.138
arra = np.asarray(de)
x_valu = arra.reshape(1 , -1)
y_pred = model.predict(x_valu)
print(y_pred)

#----------------------------------------------if i use liner regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
y_tr = model.predict(x_train)
accur = metrics.r2_score(y_train, y_tr)
print(accur)
y_te = model.predict(x_test)
accur = metrics.r2_score(y_te, y_test)
print(accur)






