#!/usr/bin/env python
# coding: utf-8

# ## Rusty Bargain (ML+ Numerical Methods)

# Rusty Bargain data (/datasets/car_data.csv) will be used to train different models with various hyperparameters.  Comparision will made of 
# gradient boosting methods (from LightGBM with hyperparameter tuning) with random forest, decision tree, and linear regression (for sanity checks). Analyzation of the speed and quality 
# of the models will be performed. RMSE metric will be used to evaluate the models.
# 
# Features
# DateCrawled — date profile was downloaded from the database
# VehicleType — vehicle body type
# RegistrationYear — vehicle registration year
# Gearbox — gearbox type
# Power — power (hp)
# Model — vehicle model
# Mileage — mileage (measured in km due to dataset's regional specifics)
# RegistrationMonth — vehicle registration month
# FuelType — fuel type
# Brand — vehicle brand
# NotRepaired — vehicle repaired or not
# DateCreated — date of profile creation
# NumberOfPictures — number of vehicle pictures
# PostalCode — postal code of profile owner (user)
# LastSeen — date of the last activity of the user
# 
# Target
# Price — price (Euro)

# ## Data preparation

# In[1]:


import numpy as np
import pandas as pd

import time

#import named regression models 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

#import ability to split into training and testing data sets 
from sklearn.model_selection import train_test_split

#import ability to evaluate accuracy of data 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


from joblib import dump

#needed to compare. 
from sklearn.utils import shuffle
from sklearn.utils import resample
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


import seaborn as sns


import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.preprocessing
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from IPython.display import display

import lightgbm as lgb


# In[2]:


df = pd.read_csv('/datasets/car_data.csv')
df.info()
df.head(4)


# In[3]:


#convert time datatypes
df['DateCrawled'] =pd.to_datetime(df['DateCrawled'] ,format='%y-%m-%d %H:%M',errors='coerce')
df['DateCreated'] =pd.to_datetime(df['DateCreated'] ,format='%y-%m-%d %H:%M',errors='coerce')
df['LastSeen'] =pd.to_datetime(df['LastSeen'] ,format='%y-%m-%d %H:%M',errors='coerce')


# In[4]:


#check for duplicates
df.duplicated().sum()


# In[5]:


#drop duplicated
df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[6]:


#check for empty cells
df.isna().sum()


# In[7]:


#fill empty cells and check successfully done 
list_of_columns=df.columns
df[list_of_columns] = df[list_of_columns].fillna('NA')
df.isna().sum()


# In[8]:


categorical= ['VehicleType', 'Gearbox', 'Model','FuelType','Brand','NotRepaired','LastSeen']       


# In[9]:


# Initialize the OrdinalEncoder
encoder = OrdinalEncoder()

# Apply the encoder to the DataFrame
encoded_data = encoder.fit_transform(df[categorical])

# Convert the encoded data back to a DataFrame with the same column names
encoded_df = pd.DataFrame(encoded_data, columns=categorical, index=df.index)

# Replace the original categorical columns with the encoded ones
df[categorical] = encoded_df

print("Original DataFrame:")
print(df)
print("\nEncoded DataFrame:")
print(encoded_df)


# In[10]:


# Ensure the column names of encoded_df match the corresponding columns in df
# encoded_df should contain only the encoded versions of categorical columns from df

# Drop the columns in df that are being replaced by encoded_df
df = df.drop(columns=encoded_df.columns, errors="ignore")

# Merge the remaining df with encoded_df
df = pd.concat([df, encoded_df], axis=1)

# Verify the resulting dataframe
print(df.head())



# ## Model training

# In[11]:


#calculate sMAPE

def calculate_smape(actual, forecast):
    # Ensure inputs are numpy arrays
    actual = np.array(actual)
    forecast = np.array(forecast)
    
    # Avoid division by zero and calculate sMAPE
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    nonzero_indices = denominator != 0  # Avoid division by zero
    numerator = np.abs(forecast - actual)
    
    smape = np.mean(numerator[nonzero_indices] / denominator[nonzero_indices]) * 100
    return smape


# In[12]:


#rmse calculation 
def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))


# In[13]:


X = df.drop(columns=['Price','DateCrawled', 'DateCreated'])
y = df['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


print(X.isnull().sum())  # Check for NaN values in X
print(y.isnull().sum())  # Check for NaN values in y

print(np.isinf(X).sum())  # Check for infinity in X
print(np.isinf(y).sum())  # Check for infinity in y


# In[15]:


# Identify quantitative columns
quantitative_features = ['RegistrationYear', 'Power', 'Mileage', 'RegistrationMonth', 'NumberOfPictures', 'PostalCode']  

# Initialize the scaler
scaler = StandardScaler()

# Scale only the quantitative features
X_train_scaled = X_train.copy()
X_train_scaled[quantitative_features] = scaler.fit_transform(X_train[quantitative_features])

X_test_scaled = X_test.copy()
X_test_scaled[quantitative_features] = scaler.transform(X_test[quantitative_features])


# In[16]:


# Initialize the Linear Regression model
lr_model = LinearRegression()

# Measure training time
start_train = time.time()
lr_model.fit(X_train_scaled, y_train)  # Train the model on the entire training data
end_train = time.time()

# Measure prediction time
start_pred = time.time()
y_pred_lr = lr_model.predict(X_test_scaled)  # Predict on the test data
y_pred_lr_train = lr_model.predict(X_train_scaled)  # Predict on the train data

end_pred = time.time()

# Calculate sMAPE
smape_lr = calculate_smape(y_test, y_pred_lr)
print(f"Linear Regression sMAPE: {smape_lr:.2f}%")

# Calculate RMSE test
rmse = calculate_rmse(y_test, y_pred_lr)
print(f'Linear Regression Test RMSE: {rmse:.2f}')

# Calculate RMSE training
rmse_train = calculate_rmse(y_train, y_pred_lr_train)
print(f'Linear Regression Training RMSE: {rmse_train:.2f}')


# Output training and prediction times
print(f'Linear Regression Training time: {end_train - start_train:.2f} seconds')
print(f'Linear Regression Prediction time: {end_pred - start_pred:.2f} seconds')


# In[17]:


# Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42)

start_train = time.time()
tree_reg.fit(X_train, y_train)
end_train = time.time()

start_pred = time.time()
y_pred = tree_reg.predict(X_test)
end_pred = time.time()

rmse_tree = mean_squared_error(y_test, y_pred, squared=False)
print(f'Decision Tree RMSE: {rmse_tree}')
print(f'Decision Tree Training time: {end_train - start_train:.2f} seconds')
print(f'Decision Tree Prediction time: {end_pred - start_pred:.2f} seconds')


# In[18]:


#find best parameters for Random Forest 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
}

rf_grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
rf_grid_search.fit(X_train, y_train)

best_rf_model = rf_grid_search.best_estimator_
print("Best Random Forest Parameters:", rf_grid_search.best_params_)


# In[19]:


# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=20)

start_train = time.time()
rf_model.fit(X_train, y_train)
end_train = time.time()

# Measure prediction time
start_pred = time.time()
# Predict on the test data
y_pred_rf = rf_model.predict(X_train) 
#y_pred_rf = cross_val_predict(rf_model, X_train, y_train, cv=5)
end_pred = time.time()



smape_rf = calculate_smape(y_train, y_pred_rf)
print(f"Random Forest sMAPE: {smape_rf:.2f}%")
# Predict on the test set
y_pred_rf_test = rf_model.predict(X_test)
# Calculate and display RMSE
rmse_forest = mean_squared_error(y_test, y_pred_rf_test, squared=False)
print(f'Random Forest RMSE :', rmse_forest)
print(f'Random Forest Training time: {end_train - start_train:.2f} seconds')
print(f'Random Forest Prediction time: {end_pred - start_pred:.2f} seconds')


# In[20]:


#evaluate with lightGBM
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

start_train = time.time()
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=20,
    valid_sets=lgb_eval,
    early_stopping_rounds=5
)
end_train = time.time()


start_pred = time.time()
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
end_pred = time.time()


rmse_lgb = mean_squared_error(y_test, y_pred, squared=False)
print(f'LightGBM RMSE: {rmse_lgb}')
print(f'LIghtGBM Training time: {end_train - start_train:.2f} seconds')
print(f'LightGBM Prediction time: {end_pred - start_pred:.2f} seconds')


# In[21]:


# Gradient Boosting Model
gbr = GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.1, max_depth=3)

# Measure training time
start_train = time.time()
gbr.fit(X_train, y_train)
end_train = time.time()

# Measure prediction time
start_pred = time.time()
y_pred_gbr = gbr.predict(X_test)
end_pred = time.time()

# Calculate RMSE
rmse_gbr = mean_squared_error(y_test, y_pred_gbr, squared=False)

# Output results
print(f'Gradient Boosting (Scikit-learn) RMSE: {rmse_gbr}')
print(f'Gradient Boosting Training time: {end_train - start_train:.2f} seconds')
print(f'Gradient Bosoting Prediction time: {end_pred - start_pred:.2f} seconds')


# ## Model analysis

# ## Conclusion
# 
# In order ot help Rusty Bargain develop an app to attract new customers for car sales service several machine learning models were trained and evaluated.  Models were predictin the selling price of cars as they are uploaded to the app.  Models evaluated include Linear Regression, Decision Tree, Random Forest, and LightGBM. Since Gradient Boosting can capture complex patterns in the data it was the highly effective in predicting car prices. Random Forest achieved the lowest RMSE score.  LighGBM also had the fastest processing time. All other models were superior to Linear Regression.    
# 
# ### Key Findings
# 
# 1. **Model Performance**:
#    - **Linear Regression**: The test RMSE was 4124.01, which served as a baseline for comparison.Linear Regression Training time: 0.05 seconds. Linear Regression Prediction time: 0.07 seconds
#    - **Decision Tree**: The RMSE Decision Tree: 2459.79. Decision Tree Training time: 1.54 seconds. Decision Tree Prediction time: 0.03 seconds
#     - **Random Forest Regression**: RMSE was: 1777.83. Random Forest Training time: 163.74 seconds. Random Forest Prediction time: 11.76 seconds
#    - **Gradient Boosting (Scikit-learn)**: RMSE was 2080.96. Gradient Boosting Training time: 24.37 seconds. Gradient Bosoting Prediction time: 0.09 seconds
#    - **LightGBM**: The RMSE was 2751.50. LIghtGBM Training time: 1.23 seconds. LightGBM Prediction time: 0.02 seconds
#    
#    
# 2. **Data Preparation**:
#    - Data was checked for duplicates and empty cells. 
#    - Data needed conversion of datatypes to be able to run through machine learning models.  
#    - Categoies which were non-numeric were converted to numeric data.  
# 
# 
