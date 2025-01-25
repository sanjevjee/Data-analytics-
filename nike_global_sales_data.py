

- Month: Month of sale.

- Region: Geographical region of the sale.

- Main_Category: Main product category (e.g., Equipment, Apparel,
  Footwear).

- Sub_Category: Subcategories within main categories.

- Product_Line: Specific product lines (e.g., Gym Sack, Hats, Tech    Fleece).

- Price_Tier: Price range of the product (Budget, Mid-Range, Premium).

- Units_Sold: Number of units sold.

- Revenue_USD: Total revenue in USD.

- Online_Sales_Percentage: Percentage of sales made online.

- Retail_Price: Retail price of the product.
"""

# importing necessary libraries
# manupulate and visulize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

# to build model and check performance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.preprocessing import StandardScaler

# to ignore warnings

import warnings
warnings.filterwarnings('ignore')

# Giving access to google drive
from google.colab import drive
drive.mount('/content/drive')

# importing data from google drive
data=pd.read_csv('/content/drive/MyDrive/nike_sales_2024.csv')
df=data.copy()
df.head()

# checking shape of data
df.shape

"""- there are 1000 rows and 10 columns."""

# checking the unique values in every column
df.nunique().to_frame('Unique Values').T

# checking data type in dataset
df.info()

"""- all data type is correct."""

# checking statical summary of data
df.describe().T

# checking null values in dataset
df.isna().sum().to_frame('Null Values').T

"""- there are no missing values which is good.

## EDA

### Univariate

#### Month
"""

# checking month feature details
df.Month.unique()

# check it on bar chart
sns.countplot(df.Month,color="y")
plt.show()

"""#### Region"""

# checking Region feature details
df.Region.unique()

# check it on bar chart
sns.countplot(df.Region,color="r")
plt.show()

"""#### Main_Category"""

# checking main_category feature details
df.Main_Category.unique()

# check it with bar chart
sns.countplot(df,x='Main_Category',color="g")
plt.show()

"""#### Sub_Category"""

# checking Sub_Category feature details
df.Sub_Category.unique()

# check it with bar chart
sns.countplot(df,x='Sub_Category',color="b")
plt.xticks(rotation=90)
plt.show()

"""#### Product_Line"""

# checking product_line feature details
df.Product_Line.unique()

# check it with bar chart
sns.countplot(df,x='Product_Line',color="y")
plt.xticks(rotation=90)
plt.show()

"""#### Price_Tier"""

# checking price_tier feature details
df.Price_Tier.unique()

# check it with bar chart
sns.countplot(df,x='Price_Tier',color="c")
plt.xticks(rotation=90)
plt.show()

"""#### Units_Sold"""

# checking Units_Sold feature details
df.Units_Sold.nunique()

# check it with bar chart
sns.boxplot(df,x='Units_Sold',color="y")
plt.xticks(rotation=90)
plt.show()

"""#### Revenue_USD"""

# checking Revenue_USD feature details
df.Revenue_USD.nunique()

# check it with bar chart
sns.histplot(df,x='Revenue_USD',color="r")
plt.xticks(rotation=90)
plt.show()

"""#### Online_Sales_Percentage"""

# checking Online_Sales_Percentage feature details
df.Online_Sales_Percentage.nunique()

# check it with bar chart
sns.histplot(df,x='Online_Sales_Percentage',color="c")
plt.xticks(rotation=90)
plt.show()

"""#### Retail_Price"""

# checking Retail_Price feature details
df.Retail_Price.nunique()

# check it with bar chart
sns.histplot(df,x='Retail_Price',color="m")
plt.xticks(rotation=90)
plt.show()

"""### Biovariate"""

# checking columns
df.columns

"""#### Month vs Retail_Price"""

# checking the relationship between month and Retail pricax.bar(df,x='Month',y='Retail_Price',color='Month')
px.bar(df,x='Month',y='Retail_Price',color='Month')

"""#### Region vs Retail_Price"""

px.bar(df,x='Region',y='Retail_Price',color='Region')

"""#### Product_Line vs Retail_Price"""

px.bar(df,x='Product_Line',y='Retail_Price',color='Product_Line')

"""#### Main_Category vs Retail_Price"""

px.bar(df,x='Main_Category',y='Retail_Price',color='Main_Category')

"""#### Sub_Category vs Retail_Price"""

px.bar(df,x='Sub_Category',y='Retail_Price',color='Sub_Category')

"""#### Month vs Unit_Sold"""

px.bar(df,x='Month',y='Units_Sold',color='Month')

"""## Statical analysis

#### Q1. Does the percentage of online sales significantly impact total sales or revenue?
"""

from scipy.stats import pearsonr

# Calculate the Pearson correlation coefficient and p-value
correlation, p_value = pearsonr(df['Online_Sales_Percentage'], df['Revenue_USD'])

print(f"Pearson correlation coefficient: {correlation}")
print(f"P-value: {p_value}")

# Interpret the results
alpha = 0.05  # Significance level

if p_value < alpha:
    print("There is a statistically significant correlation between online sales percentage and revenue.")
else:
    print("There is no statistically significant correlation between online sales percentage and revenue.")

"""### Q2. Are there significant seasonal variations in units sold or revenue?"""

# Group data by month and sum units sold and revenue
monthly_sales = df.groupby('Month').agg({'Units_Sold': 'sum', 'Revenue_USD': 'sum'})

# Create line plots for units sold and revenue over time
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales['Units_Sold'], label='Units Sold')
plt.plot(monthly_sales.index, monthly_sales['Revenue_USD'], label='Revenue')
plt.xlabel('Month')
plt.ylabel('Values')
plt.title('Seasonal Variations in Units Sold and Revenue')
plt.legend()
plt.show()

"""- so we are looking revenue and unint sold variation so i use levens test to see the variation."""

# checking levense test for variation
# alpha = 0.05
import scipy.stats as stats
from scipy.stats import levene
stats.levene(df['Units_Sold'], df['Revenue_USD'])

"""- p_value is greater then 0.5 so we have enough evidence to say that there are no variation in unit sold and revenue.

## Data Preprocessing
"""

# checking duplicates in data set
df.duplicated().sum()

"""- there are no duplicates"""

# checking missing values
df.isnull().sum()

df.info()

# checking outlier in dataset
sns.boxplot(df)
plt.xticks(rotation=90)
plt.show()

"""- there are only few outlier in revenue_usd feature. assuming all datapoint are real."""

# checking correleation of dataset
corr=df.select_dtypes(include=np.number)
corr.corr()

# check it with heatmap
sns.heatmap(corr.corr(),annot=True)

"""- unit sold is strongly positive coreleated with revenue_usd.
- revenue_usd is strongly positive coreleated with retail_price.
"""

# checking all feature distribution with pairplot
sns.pairplot(df)

# copying data
df1=df.copy()

# lets seprate independent and dependent variables
X=df1.drop(['Revenue_USD'],axis=1)
y=df1['Revenue_USD']
X=pd.get_dummies(X)

# replace true as 1 and false as 0 in X
X=X.replace({True: 1, False: 0})
X = sm.add_constant(X)

# lets seprate train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from statsmodels.formula.api import ols # Import ols from statsmodels
import statsmodels.api as sm
model = sm.OLS(y, X).fit()

print(model.summary())

"""- R2 is around 91 and adjusting r2 is around 91. which is good.
- coefficient is  -2.373e+06 .
- if 1 unit increase in units_sold 178 unit increase in revenue_usd  
  or vice-versa.
- there are multicolarity because of dummy data.
- if 1 unit online_sale_percentage 1058.6186  unit decrease in revenue_usd and so on.

## Check performance
"""

# function to compute adjusted R-squared
def adj_r2_score(predictors, targets, predictions):
    r2 = r2_score(targets, predictions)
    n = predictors.shape[0]
    k = predictors.shape[1]
    return 1 - ((1 - r2) * (n - 1) / (n - k - 1))


# function to compute MAPE
def mape_score(targets, predictions):
    return np.mean(np.abs(targets - predictions) / targets) * 100


# function to compute different metrics to check performance of a regression model
def model_performance_regression(model, predictors, target):
    """
    Function to compute different metrics to check regression model performance

    model: regressor
    predictors: independent variables
    target: dependent variable
    """

    # predicting using the independent variables
    pred = model.predict(predictors)

    r2 = r2_score(target, pred)  # to compute R-squared
    adjr2 = adj_r2_score(predictors, target, pred)  # to compute adjusted R-squared
    rmse = np.sqrt(mean_squared_error(target, pred))  # to compute RMSE
    mae = mean_absolute_error(target, pred)  # to compute MAE
    mape = mape_score(target, pred)  # to compute MAPE

    # creating a dataframe of metrics
    df_perf = pd.DataFrame(
        {
            "RMSE": rmse,
            "MAE": mae,
            "R-squared": r2,
            "Adj. R-squared": adjr2,
            "MAPE": mape,
        },
        index=[0],
    )

    return df_perf

# checking model performance on train set (seen 80% data)
print("Training Performance\n")
olsmodel_train_perf = model_performance_regression(model, X_train, y_train)
olsmodel_train_perf

# checking model performance on test set (seen 20% data)
print("Test Performance\n")
olsmodel_test_perf = model_performance_regression(model, X_test, y_test)
olsmodel_test_perf

"""- mape Training: 35.45%
- mape Test: 26.33%
- The test set MAPE is much lower, which is a good sign, as it indicates that the model's percentage error decreases with unseen data.

- R2 is good performing on testset.

## Checking Linear Regression Assumption

We will be checking the following Linear Regression assumptions:

1. **No Multicollinearity**

2. **Linearity of variables**

3. **Independence of error terms**

4. **Normality of error terms**

5. **No Heteroscedasticity**

#### Check Multicolarity
"""

from statsmodels.stats.outliers_influence import variance_inflation_factor


def checking_vif(predictors):
    vif = pd.DataFrame()
    vif["feature"] = predictors.columns

    # calculating VIF for each feature
    vif["VIF"] = [
        variance_inflation_factor(predictors.values, i)
        for i in range(len(predictors.columns))
    ]
    return vif

checking_vif(X_train)

def treating_multicollinearity(predictors, target, high_vif_columns):
    """
    Checking the effect of dropping the columns showing high multicollinearity
    on model performance (adj. R-squared and RMSE)

    predictors: independent variables
    target: dependent variable
    high_vif_columns: columns having high VIF
    """
    # empty lists to store adj. R-squared and RMSE values
    adj_r2 = []
    rmse = []

    # build ols models by dropping one of the high VIF columns at a time
    # store the adjusted R-squared and RMSE in the lists defined previously
    for cols in high_vif_columns:
        # defining the new train set
        train = predictors.loc[:, ~predictors.columns.str.startswith(cols)]

        # create the model
        olsmodel = sm.OLS(target, train).fit()

        # adding adj. R-squared and RMSE to the lists
        adj_r2.append(olsmodel.rsquared_adj)
        rmse.append(np.sqrt(olsmodel.mse_resid))

    # creating a dataframe for the results
    temp = pd.DataFrame(
        {
            "col": high_vif_columns,
            "Adj. R-squared after_dropping col": adj_r2,
            "RMSE after dropping col": rmse,
        }
    ).sort_values(by="Adj. R-squared after_dropping col", ascending=False)
    temp.reset_index(drop=True, inplace=True)

    return temp

col_list = ["Month_April", "Product_Line_Vapor Cricket","Price_Tier_Mid-Range"]

res = treating_multicollinearity(X_train, y_train, col_list)
res

col_to_drop = "Month_April", "Product_Line_Vapor Cricket","Price_Tier_Mid-Range","Product_Line_Tech Fleece","Region_America","Main_Category_Apparel","Sub_Category_Accessories","Product_Line_Air Force 1","Main_Category_Equipment	","Sub_Category_Cricket","Sub_Category_Lifestyle","Sub_Category_Performance"
x_train2 = X_train.loc[:, ~X_train.columns.str.startswith(col_to_drop)]
x_test2 = X_test.loc[:, ~X_test.columns.str.startswith(col_to_drop)]

# Check VIF now
vif = checking_vif(x_train2)
print("VIF after dropping ", col_to_drop)
vif

"""- all feature has <5. inf is showing because of dummy datapoint."""

olsmod1 = sm.OLS(y_train, x_train2).fit()
print(olsmod1.summary())

# initial list of columns
predictors = x_train2.copy()
cols = predictors.columns.tolist()

# setting an initial max p-value
max_p_value = 1

while len(cols) > 0:
    # defining the train set
    x_train_aux = predictors[cols]

    # fitting the model
    model = sm.OLS(y_train, x_train_aux).fit()

    # getting the p-values and the maximum p-value
    p_values = model.pvalues
    max_p_value = max(p_values)

    # name of the variable with maximum p-value
    feature_with_p_max = p_values.idxmax()

    if max_p_value > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break

selected_features = cols
print(selected_features)

x_train3 = x_train2[selected_features]
x_test3 = x_test2[selected_features]

olsmod2 = sm.OLS(y_train, x_train3).fit()
print(olsmod2.summary())

# checking model performance on train set (seen 70% data)
print("Training Performance\n")
olsmod2_train_perf = model_performance_regression(olsmod2, x_train3, y_train)
olsmod2_train_perf

# checking model performance on test set (seen 30% data)
print("Test Performance\n")
olsmod2_test_perf = model_performance_regression(olsmod2, x_test3, y_test)
olsmod2_test_perf

"""#### Test of linearity and independent"""

# let us create a dataframe with actual, fitted and residual values
df_pred = pd.DataFrame()

df_pred["Actual Values"] = y_train  # actual values
df_pred["Fitted Values"] = olsmod2.fittedvalues  # predicted values
df_pred["Residuals"] = olsmod2.resid  # residuals

df_pred.head()

# let's plot the fitted values vs residuals

sns.residplot(
    data=df_pred, x="Fitted Values", y="Residuals", color="purple", lowess=True
)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Fitted vs Residual plot")
plt.show()

"""#### Test of normality"""

sns.histplot(data=df_pred, x="Residuals", kde=True)
plt.title("Normality of residuals")
plt.show()

import pylab
import scipy.stats as stats

stats.probplot(df_pred["Residuals"], dist="norm", plot=pylab)
plt.show()

stats.shapiro(df_pred["Residuals"])

"""- p_value is less then 0.05. datapoint is not normal.but assuming it is close to normal.

#### Test of homocedasticity
"""

import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ["F statistic", "p-value"]
test = sms.het_goldfeldquandt(df_pred["Residuals"], x_train3)
lzip(name, test)

"""- **Since p-value > 0.05, we can say that the residuals are homoscedastic. So, this assumption is satisfied.**

## Predicton of Test set
"""

# predictions on the test set
pred = olsmod2.predict(x_test3)

df_pred_test = pd.DataFrame({"Actual": y_test, "Predicted": pred})
df_pred_test.sample(10, random_state=1)

"""- We can observe here that our model has returned  not pretty good prediction results, and the actual and predicted values are not comparable"""

x_train_final = x_train3.copy()
x_test_final = x_test3.copy()

olsmodel_final = sm.OLS(y_train, x_train_final).fit()
print(olsmodel_final.summary())

# checking model performance on train set (seen 70% data)
print("Training Performance\n")
olsmodel_final_train_perf = model_performance_regression(
    olsmodel_final, x_train_final, y_train
)
olsmodel_final_train_perf

# checking model performance on test set (seen 30% data)
print("Test Performance\n")
olsmodel_final_test_perf = model_performance_regression(
    olsmodel_final, x_test_final, y_test
)
olsmodel_final_test_perf

"""## Conclusion

**The model explains 91.3% of the variance in revenue (R-squared = 0.913), indicating strong predictive power. Most predictors, including Units_Sold and Retail_Price, are highly significant (p < 0.05), while some categories like Product_Line_Dri-FIT negatively impact revenue. However, the condition number and smallest eigenvalue suggest severe multicollinearity, potentially affecting coefficient reliability.**

# Lets check it with other algorithams
"""

# copying data
df3=df.copy()

# lets seprate independent and dependent variables
X1=df3.drop(['Revenue_USD'],axis=1)
y1=df3['Revenue_USD']
X1=pd.get_dummies(X1,drop_first=True)

# replace true as 1 and false as 0 in X
X1=X1.replace({True: 1, False: 0})

X1.head()

# lets seprate train and test set
X_train1,X_test1,y_train1,y_test1=train_test_split(X1,y1,test_size=0.3,random_state=1)

print(X_train1.shape)
print(X_test1.shape)
print(y_train1.shape)
print(y_test1.shape)

import sklearn
import xgboost
print("scikit-learn version:", sklearn.__version__)
print("xgboost version:", xgboost.__version__)

# pip install --upgrade scikit-learn xgboost

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Define models
models = {
    "AdaBoost": AdaBoostRegressor(random_state=1),
    "GradientBoosting": GradientBoostingRegressor(random_state=1),
    "Decision_tree": DecisionTreeRegressor(random_state=1),
    "Bagging": BaggingRegressor(random_state=1),
}

# Use k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

for name, model in models.items():
    # Cross-validation with 'neg_root_mean_squared_error' scoring
    cv_scores = cross_val_score(
        model, X_train1, y_train1, cv=kfold, scoring="neg_root_mean_squared_error"
    )
    print(f"{name}: RMSE scores - {-cv_scores}")  # Convert to positive RMSE
    print(f"{name}: Mean RMSE - {np.mean(-cv_scores)}")
    print(f"{name}: Standard Deviation RMSE - {np.std(-cv_scores)}")

    # Fit the model and evaluate on the test set
    model.fit(X_train1, y_train1)
    y_pred = model.predict(X_test1)
    test_rmse = np.sqrt(mean_squared_error(y_test1, y_pred))
    print(f"{name}: Test RMSE - {test_rmse}")
    print("-" * 30)

"""#### tune Gredient boosting"""

# Import necessary libraries (already imported in the provided code)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10,50, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
}

# Initialize GradientBoostingRegressor
gb_model = GradientBoostingRegressor(random_state=1)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train1, y_train1)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Train the model with the best hyperparameters
best_gb_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_gb_model.predict(X_test1)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test1, y_pred))
print(f"Test RMSE: {rmse}")

"""- lets train model"""

# Provided hyperparameters
params = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_split': 10, 'n_estimators': 200}

# Initialize and train Gradient Boosting Regressor with provided parameters
gb_model = GradientBoostingRegressor(**params, random_state=1)
gb_model.fit(X_train1, y_train1)

# Make predictions
y_pred = gb_model.predict(X_test1)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test1, y_pred))
print(f"Test RMSE: {rmse}")

# checking model performance on train set (seen 70% data)
print("Training Performance\n")
gb_final_train_perf = model_performance_regression(
    gb_model, X_train1, y_train1
)
gb_final_train_perf

# checking model performance on test set (seen 30% data)
print("Test Performance\n")
gb_final_test_perf = model_performance_regression(
    gb_model, X_test1, y_test1
)
gb_final_test_perf

"""## Observations -

- r2 is 99% so gb_model is best performing model.
- rmse is decreasing frequently which is good .
- model is not overfit or underfit so its generilized model.
"""
