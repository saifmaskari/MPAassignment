#!/usr/bin/env python
# coding: utf-8

# # Estimation of Used Car Prices - Data Science project
# 

# ## Import Libraries
# All libraries are used for specific tasks including data preprocessing, visualization, transformation and evaluation

# In[163]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor 
import warnings
warnings.filterwarnings("ignore")


# ## Import Data
# ### Read Training Data
# The training set is read locally and the **head** function is used to display the data for intial understanding

# In[164]:


dataTrain=pd.read_csv('data_train.csv')
dataTrain.head()


# The **shape** function displays the number of rows and columns in the training set

# In[165]:


dataTrain.shape


# ### Read Testing Data
# The testing set is read locally and the **head** function is used to display the data for intial understanding

# In[166]:


dataTest=pd.read_csv('data_test.csv')
dataTest.head()


# The **shape** function displays the number of rows and columns in the testing set

# In[167]:


dataTest.shape


# Checking for null values in each column and displaying the sum of all null values in each column (Training Set)

# In[168]:


dataTrain.isnull().sum()


# Checking for null values in each column and displaying the sum of all null values in each column (Testing Set)

# In[169]:


dataTest.isnull().sum()


# Removing the rows with empty values since the number of empty rows are small. This is the best approach compared to replacing with mean or random values

# In[170]:


dataTrain=dataTrain.dropna()
dataTest=dataTest.dropna()


# Checking if null values are eliminated (Training set)

# In[171]:


dataTrain.isnull().sum()


# In[172]:


dataTrain.shape


# Checking if null values are eliminated (Testing set)

# In[173]:


dataTest.isnull().sum()


# In[174]:


dataTest.shape


# Checking the data types to see if all the data is in correct format. All the data seems to be in their required format.

# In[175]:


dataTrain.dtypes


# Checking the correlation between the numerical features

# ## EDA (Exploratory Data Analysis)
# Visualizations are used to understand the relationship between the target variable and the features, in addition to correlation coefficient and p-value. 
# The visuals include heatmap, scatterplot,boxplot etc.
# 

# ### Regression/scatter Plot
# This regression plot show the relation between **odometer** and **price**. A slight negative correlation is observed
# this shows that price is being affected by the change in odometer value.

# In[176]:


plt.figure(figsize=(10,6))
sns.regplot(x="odometer_value", y="price_usd", data=dataTrain)


# As observed in the plot, a **negative correlation** of -0.42 is obtained along with a p-value of 0. The p value confirms that the calculated correlation is **significant** hence this feature is significant to the prediction of used car price.

# In[177]:


from scipy import stats
pearson_coef, p_value = stats.pearsonr(dataTrain['odometer_value'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# The regression plot belowe shows a relationship between the year that the car is produced and the price of the car. A positive 
# correlation is observed between the two variables. This shows that the price increases with increase in production year of the car.

# In[178]:


plt.figure(figsize=(10,6))
sns.regplot(x="year_produced", y="price_usd", data=dataTrain)


# As observed above, a high positive correlation of 0.7 is calculated along with the p-value of 0. This indicates that the correlation between the variables is significant hence year produced feature can be used for prediction.

# In[179]:


pearson_coef, p_value = stats.pearsonr(dataTrain['year_produced'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In this plot, a minor correlation is observed between variables. It can only be confirmed with actual correlation which is to be calculated.

# In[180]:


plt.figure(figsize=(10,6))
sns.regplot(x="engine_capacity", y="price_usd", data=dataTrain)


# A 0.3 correlation is calculated which is very small with a p value of 0. This indicates that even though the correlation is small but its 30% of 100 which is significant hence this feature can be used for predicition.

# In[181]:


pearson_coef, p_value = stats.pearsonr(dataTrain['engine_capacity'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# This regression plot shows an minor positive correlation observed with the help of the best fit line. The calculation will confirm the actual value.

# In[182]:


plt.figure(figsize=(10,6))
sns.regplot(x="number_of_photos", y="price_usd", data=dataTrain)


# The correlation is 0.31 based on the calculation while the p-value calculated is zero. This is similar to the last feature hence the significant 31% of 100 correlation makes this feature eligble for prediction.

# In[183]:


pearson_coef, p_value = stats.pearsonr(dataTrain['number_of_photos'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# This plot shows no correlation with points all over the graph.

# In[184]:


plt.figure(figsize=(10,6))
sns.regplot(x="number_of_maintenance", y="price_usd", data=dataTrain)


# The calculation proves that a correlation is lesser than 0.1 percent is same as no correlation and the p-value of lesser than 0.01 confirms it. This feature is not significant enough for predicition

# In[185]:


pearson_coef, p_value = stats.pearsonr(dataTrain['number_of_maintenance'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# Similar to the last plot, this plot shows no correlation with points all over the graph.

# In[186]:


plt.figure(figsize=(10,6))
sns.regplot(x="duration_listed", y="price_usd", data=dataTrain)


# The calculated correlation is lesser than 0.1 which is considered negligible. The p-value lesser than 0.01 confirming the correlation value hence this feature is not suitable for prediction of price. 

# In[187]:


pearson_coef, p_value = stats.pearsonr(dataTrain['duration_listed'], dataTrain['price_usd'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)


# ### Box Plot
# These plots are used for categorical data to determine the importance of features for prediction. 

# In the given plot below, it is observed that the price range vary for automatic and manual transmisson. This indicates the categories can vary with price hence feature can be used for prediction

# In[188]:


sns.boxplot(x="transmission", y="price_usd", data=dataTrain)


# The box plot shows how prices vary based on different colors. This shows that color can be used as a feature for price prediction.

# In[189]:


plt.figure(figsize=(10,6))
sns.boxplot(x="color", y="price_usd", data=dataTrain)


# This plot shows engine fuel types and how they affect the price. Hybrid petroll with the highest price range while hybrid diesel with lowest price range. This feature can be used for prediction.

# In[190]:


sns.boxplot(x="engine_fuel", y="price_usd", data=dataTrain)


# The engine type (based on fuel type) shows that both categories have almost the same price range which will not bring differences in price when prediction is made. Hence this feature is not suitable for price prediction

# In[191]:


sns.boxplot(x="engine_type", y="price_usd", data=dataTrain)


# Thee box plot shows body type categories with varying prices per category hence this feature can be used for price prediction

# In[192]:


plt.figure(figsize=(10,6))
sns.boxplot(x="body_type", y="price_usd", data=dataTrain)


# Has warranty feature shows a huge difference in price ranges between cars with warrant and vice versa. This feature is very important for price prediction as the bigger the difference in range the better the feature.

# In[193]:


sns.boxplot(x="has_warranty", y="price_usd", data=dataTrain)


# This feature is similar to the feature above, all three categories have wider price ranges between one another. This feature is also crucial for price prediction.

# In[194]:


sns.boxplot(x="ownership", y="price_usd", data=dataTrain)


# Front and rear drive have **minimal price difference** while all drive shows a **greater difference** hence the feature can be used for prediction.

# In[195]:


sns.boxplot(x="type_of_drive", y="price_usd", data=dataTrain)


# With almost same price range between categories this feature is not suitable for prediction.

# In[196]:


sns.boxplot(x="is_exchangeable", y="price_usd", data=dataTrain)


# This plot shows that the manufacturer name is important when selling a car. The variety of price ranges for all categories prove that the feature is significant for price prediction.

# In[197]:


plt.figure(figsize=(10,6))
sns.boxplot(x="manufacturer_name", y="price_usd", data=dataTrain)


# Using Exploratory data aanalysis, few features can be dropped because they had no impact on the price prediction. Those features are removed with the functuion below.(Training set)

# In[198]:


dataTrain.drop(['number_of_maintenance', 'duration_listed', 'engine_type','is_exchangeable'], axis = 1, inplace = True)


# Same features are removed for testing set since the data will be used to train the model

# In[199]:


dataTest.drop(['number_of_maintenance', 'duration_listed', 'engine_type','is_exchangeable'], axis = 1, inplace = True)


# In[203]:


dataTrain.shape


# In[204]:


dataTest.shape


# A descriptive analysis to check incorrect entries and anormalies. This is also used to give an overview of the numerical data. It is observed that most of the data has no incorrect entries.

# In[205]:


dataTrain.describe()


# This is a check for categorical data, it is observed that all the data is within the range with no incorrect entries.

# In[206]:


dataTrain.describe(include=['object'])


# ### Data Transformation
# Label encoding of categorical features in the training set. Label encoding is converting categorical data into numerical data since the model cant understand textual data.

# In[211]:


labelencoder = LabelEncoder()
dataTrain.manufacturer_name = labelencoder.fit_transform(dataTrain.manufacturer_name)
dataTrain.transmission = labelencoder.fit_transform(dataTrain.transmission)
dataTrain.color = labelencoder.fit_transform(dataTrain.color)
dataTrain.engine_fuel = labelencoder.fit_transform(dataTrain.engine_fuel)
#dataTrain.engine_type = labelencoder.fit_transform(dataTrain.engine_type)
dataTrain.body_type = labelencoder.fit_transform(dataTrain.body_type)
dataTrain.has_warranty = labelencoder.fit_transform(dataTrain.has_warranty)
dataTrain.ownership = labelencoder.fit_transform(dataTrain.ownership)
dataTrain.type_of_drive = labelencoder.fit_transform(dataTrain.type_of_drive)
#dataTrain.is_exchangeable = labelencoder.fit_transform(dataTrain.is_exchangeable)


# Label encoding of all categorical data in the testing set.

# In[212]:


labelencoder1 = LabelEncoder()
dataTest.manufacturer_name = labelencoder1.fit_transform(dataTest.manufacturer_name)
dataTest.transmission = labelencoder1.fit_transform(dataTest.transmission)
dataTest.color = labelencoder1.fit_transform(dataTest.color)
dataTest.engine_fuel = labelencoder1.fit_transform(dataTest.engine_fuel)
#dataTest.engine_type = labelencoder1.fit_transform(dataTest.engine_type)
dataTest.body_type = labelencoder1.fit_transform(dataTest.body_type)
dataTest.has_warranty = labelencoder1.fit_transform(dataTest.has_warranty)
dataTest.ownership = labelencoder1.fit_transform(dataTest.ownership)
dataTest.type_of_drive = labelencoder1.fit_transform(dataTest.type_of_drive)
#dataTest.is_exchangeable = labelencoder1.fit_transform(dataTest.is_exchangeable)


# Checking on the remaining features and if label encoding is applied to all categorical features (Training set).

# In[214]:


dataTrain.head(10)


# Check on the remaining features and application of label encoding to all categorical features (Testing set).

# In[215]:


dataTest.head(10)


# --Data Transfornation (normalization) ----
# z-score used for scaling down the features between the range of -1 and 1. This helps the model make better prediction as it is easy to understand. The scaling is applied to the training and testing set

# In[216]:


# Calculate the z-score from with scipy
import scipy.stats as stats
dataTrain = stats.zscore(dataTrain)
dataTest = stats.zscore(dataTest)


# In[217]:


dataTrain


# In[218]:


dataTest


# Dividing the data for training and testing accordingly. X takes the all features while Y takes the target variable
# 
# We have 13 actual columns [0-12 index]; 12 are predictor variables and 1 is the target variable

# In[221]:


x_train=dataTrain.iloc[:,0:11]
y_train=dataTrain.iloc[:,12]
x_test=dataTest.iloc[:,0:11]
y_test=dataTest.iloc[:,12]


# In[222]:


x_train


# ## Fit Model
# ### Multiple Linear Regression
# Calling multiple linear regression model and fitting the training set

# In[223]:


rg = LinearRegression()
mdl=rg.fit(x_train,y_train)


# Making price prediction using the testing set (Fit to MLR)

# In[224]:


y_pred1 = rg.predict(x_test)


# ### MLR Evaluation
# Calculating the R-square for MLR model

# In[225]:


print('The R-square for Multiple Linear regression is: ', rg.score(x_train,y_train))


# Calculating the Mean Square Error for MLR model

# In[226]:


mse1 = mean_squared_error(y_test, y_pred1)
print('The mean square error for Multiple Linear Regression: ', mse1)


# Calculating the Mean Absolute Error for MLR model

# In[227]:


mae1= mean_absolute_error(y_test, y_pred1)
print('The mean absolute error for Multiple Linear Regression: ', mae1)


# ### Distribution Plot
# Comparison of actual values vs predicted values (Testing set)

# In[228]:


plt.figure(figsize=(10,6))

ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred1, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# ### Random Forest Regressor (Best Model)
# Calling the random forest model and fitting the training data

# In[229]:


rf = RandomForestRegressor()
model=rf.fit(x_train,y_train)


# Prediction of car prices using the testing data

# In[230]:


y_pred2 = rf.predict(x_test)


# ### Random Forest Evaluation
# Calculating the R square value for Random Forest Model (Highest R-square value)

# In[231]:


print('The R-square for Random Forest is: ', rf.score(x_train,y_train))


# Calculating the Mean Square Error for Random Forest Model (Lowest MSE value)

# In[232]:


mse2 = mean_squared_error(y_test, y_pred2)
print('The mean square error of price and predicted value is: ', mse2)


# Calculating the Mean Absolute Error for Random Forest Model (Lowest Mean Absolute Error)

# In[233]:


mae2= mean_absolute_error(y_test, y_pred2)
print('The mean absolute error of price and predicted value is: ', mae2)


# ### Distribution Plot
# Comparison of Actual and Predicted values of price for the testing set

# In[234]:


plt.figure(figsize=(10,6))

ax1 = sns.distplot(y_test, hist=False, color="r", label="Actual Value")
sns.distplot(y_pred2, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# ### LASSO Model 
# Calling the model and fitting the training data

# In[252]:


LassoModel=Lasso()
lm=LassoModel.fit(x_train,y_train)


# Price prediction uisng testing data

# In[253]:


y_pred3 = lm.predict(x_test)


# ### LASSO Evaluation
# R square value for the LASSO Model

# In[254]:


print('The R-square for LASSO is: ', lm.score(x_train,y_train))


# Mean Absolute Error for LASSO Model

# In[255]:


mae3= mean_absolute_error(y_test, y_pred3)
print('The mean absolute error of price and predicted value is: ', mae3)


# Mean Squared Error for the LASSO Model

# In[256]:


mse3 = mean_squared_error(y_test, y_pred3)
print('The mean square error of price and predicted value is: ', mse3)


# In[261]:


scores = [('MLR', mae1),
          ('Random Forest', mae2),
          ('LASSO', mae3)
         ]         


# In[262]:


mae = pd.DataFrame(data = scores, columns=['Model', 'MAE Score'])
mae


# In[263]:


mae.sort_values(by=(['MAE Score']), ascending=False, inplace=True)

f, axe = plt.subplots(1,1, figsize=(10,7))
sns.barplot(x = mae['Model'], y=mae['MAE Score'], ax = axe)
axe.set_xlabel('Mean Absolute Error', size=20)
axe.set_ylabel('Model', size=20)

plt.show()


# #Based on the MAE, it is concluded that the Random Forest is the best regression model for predicting the car price based on the 12 predictor variables 

# In[ ]:




