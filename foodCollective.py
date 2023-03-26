#to import data
import shap
import pandas as pd
import numpy as np
import xgboost as xgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

CONSTRAINT = 0.3
MAX_USERS = 22

# select user input
#user_input = int(input("Which user's result do you want to look at?: "))

# read the dataframe
data = pd.read_csv('data.csv')

# drop all Null data (filtering null values)
data.dropna(inplace=True)

# split the dataset into training and test data
X = data.iloc[:,:-2]
y = data["symptom_value"]

# create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X, label=y)

# specify XGBoost parameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror'
}

# train the model
model = xgb.train(params, dtrain)

# create an explainer object for SHAP
explainer = shap.Explainer(model, X)

# calculate SHAP values for each feature for each instance
shap_values = explainer(X)

# compute the sum of SHAP values for each food category across all instances
food_shap_sum = shap_values.values[:,:-2].sum(axis=0)

# keep track of the total SHAP value for each food category across all users
food_shap_total = np.zeros_like(food_shap_sum)

for i in range(MAX_USERS):
    df = data[data['user_number'] == i]
    X_user = df.iloc[:,:-2]
    
    # calculate SHAP values for the user's data
    shap_values_user = explainer(X_user)
    
    # accumulate the SHAP values for each food category across all users
    food_shap_total += shap_values_user.values[:,:-2].sum(axis=0)

# plot the total SHAP values for each food category
food_labels = X.columns[:-2]
plt.bar(food_labels, food_shap_total)
plt.xlabel('Food categories')
plt.ylabel('Total SHAP value')
plt.show()

