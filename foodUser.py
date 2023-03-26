#imports
import shap
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

CONSTRAINT = 0.3

def bars(shap_values):
    # summarize the SHAP values for each feature
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)

    # plot the SHAP values for each feature
    fig, ax = plt.gcf(), plt.gca()
    ax.set_xlim(-0.5, 2)
    ax.set_title("Food Group Impact")
    plt.show()

# select user input
user_input = int(input("Which user's result do you want to look at?: "))

# read the dataframe
data = pd.read_csv('data.csv')
df = data[data['user_number'] == user_input]

# drop all Null data (filtering null values)
df.dropna()

# feature seletion to determine correllation between the colummn and symtpom val
# using corrwidth to compare all values of columns F1-F20 to the symptom_value
# for each data point. Using method - "Spearman Correlation"
correlation = df.iloc[:,:-2].corrwith(df['symptom_value'],method='spearman').abs()
# filtering all food categories with vorrelation value >0.3 and indexing to get names from df header
high_corr_categories = correlation[correlation>CONSTRAINT].index.tolist()

# print the dataframe
df2 = df[high_corr_categories]
df2 = df2.join(df.iloc[:,-2:])
print("Highest affecting food group categories:", high_corr_categories)
print(df2)

# split the dataset into training and test data
X = df.iloc[:,:-2]
y = df["symptom_value"]

# create a DMatrix for XGBoost
dtrain = xgb.DMatrix(X, label=y)

# specify XGBoost parameters
params = {'max_depth': 3, 'eta': 0.1, 'objective': 'reg:squarederror'}

# train the model
model = xgb.train(params, dtrain)

# create an explainer object for SHAP
explainer = shap.Explainer(model, X)

# calculate SHAP values for each feature for each instance
shap_values = explainer(X)

# call the bars() function
bars(shap_values)

# plot the heatmap
shap.plots.beeswarm(shap_values)


