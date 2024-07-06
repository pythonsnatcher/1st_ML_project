# **My First ML Project**

# **Load Data**

import pandas as pd


df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# **Data Preparation**

## Data seperation as x and y


y = df['logS']


x = df.drop('logS', axis=1)


## Data splitting


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=100)

x_train

x_test

# **Model Building**

## Linear Regression

### **Training the model**

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

### Applying the model to make a prediction

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

y_lr_train_pred


y_lr_test_pred

### Evaluate model performance




from sklearn.metrics import mean_squared_error, r2_score
lr_train_mse = mean_squared_error(y_train,y_lr_train_pred)
lr_train_r2 = r2_score(y_train,y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test,y_lr_test_pred)
lr_test_r2 = r2_score(y_test,y_lr_test_pred)

print('LR MES (TRAIN): ',lr_train_mse)
print('LR MES (TEST): ',lr_test_mse)
print('LR MES (TRAIN): ',lr_train_mse)
print('LR MES (TEST): ',lr_test_mse)

lr_results = pd.DataFrame(['Linear Regression',lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()
lr_results.columns = ['Method','Training MSE','Training r2','Test MSE','Test r2']

lr_results

## **Random Forest**

### Training the model

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train,y_train)

### Applying the model to make a predicition

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

### Evaluate model performance

from sklearn.metrics import mean_squared_error, r2_score
rf_train_mse = mean_squared_error(y_train,y_rf_train_pred)
rf_train_r2 = r2_score(y_train,y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test,y_rf_test_pred)
rf_test_r2 = r2_score(y_test,y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest',rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results.columns = ['Method','Training MSE','Training r2','Test MSE','Test r2']
rf_results

## Model Comparison

df_models = pd.concat([lr_results,rf_results],).reset_index(drop=True)


# Data Visualisation of Prediction Results

import matplotlib.pyplot as plt
import numpy as np

plt.scatter(x=y_train, y = y_lr_train_pred,alpha =0.3)
plt.plot(y_train,p(y_train), c='black')

z = np.polyfit(y_train,y_lr_train_pred,1)
p = np.poly1d(z)
plt.ylabel('Predict logS')
plt.xlabel('Experimental logS')
