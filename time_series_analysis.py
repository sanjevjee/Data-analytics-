# importing necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# giving access to google drive
from google.colab import drive
drive.mount('/content/drive')

# importing data from google drive
df = pd.read_csv('/content/drive/My Drive/financial_portfolio_data.csv')

# copying data
data = df.copy()

data

# checking shape of data
data.shape

# checking data types
data.info()

"""- data type in not correct. lets fix it
- rest of feature datatypes is correct.
"""

# lets fix date dtypes
data['Date'] = pd.to_datetime(data['Date'])

# checking discriptive statics
data.describe()

# checking missing values and duplicates
print(data.isnull().sum())
print("there are",data.duplicated().sum(),"duplicates")

# Plot the time series
plt.figure(figsize=(22, 10))
plt.plot(data['Price'])  # Replace 'Value' with your target column name
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid()
plt.show()

"""#### check stationality"""

from statsmodels.tsa.stattools import adfuller

# Perform the ADF test
result = adfuller(data['Price'])  # Replace 'Value' with your target column name
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] <= 0.05:
    print("The data is stationary.")
else:
    print("The data is not stationary. Differencing may be needed.")

"""## Differencing"""

# # Apply first differencing
# data['Value_diff'] = data['Price'].diff()  # Replace 'Value' with your column name
# data.dropna(inplace=True)

# # Plot the differenced data
# plt.figure(figsize=(10, 6))
# plt.plot(data['Value_diff'], label='Differenced Time Series')
# plt.title('Differenced Time Series')
# plt.xlabel('Date')
# plt.ylabel('Differenced Value')
# plt.legend()
# plt.grid()
# plt.show()

"""## Decomposition"""

# Decompose the time series
decomposition = seasonal_decompose(data['Price'], model='additive', period=12)  # Replace 'Value' with your column

# Plot the decomposition
decomposition.plot()
plt.show()

"""#### spliting data"""

# Split the data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

# Print the sizes
print(f"Train size: {len(train)}, Test size: {len(test)}")

"""## Best pdq"""

import itertools
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Function to check stationarity using Augmented Dickey-Fuller test
def check_stationarity(data):
    result = adfuller(data)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    if result[1] <= 0.05:
        print("Data is stationary.")
        return True  # Data is stationary
    else:
        print("Data is not stationary. Consider differencing.")
        return False  # Data is not stationary

# Grid search to find optimal p, d, q
def find_pdq(data, max_p=4, max_d=4, max_q=4):
    warnings.filterwarnings("ignore")  # Ignore convergence warnings
    best_aic = float("inf")
    best_pdq = None

    # Define the range for p, d, q
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)

    pdq_combinations = list(itertools.product(p, d, q))

    for pdq in pdq_combinations:
        try:
            model = SARIMAX(data, order=pdq, enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = pdq
        except Exception as e:
            print(f"Error with {pdq}: {e}")
            continue

    print(f"Best PDQ: {best_pdq} with AIC: {best_aic}")
    return best_pdq

# Example usage:
# Assuming train['Price'] is your time series data

# Check stationarity first
if not check_stationarity(train['Price']):
    # If not stationary, difference your data (first differencing if necessary)
    train_diff = train['Price'].diff().dropna()  # First differencing
    print("Differenced data")

    # After differencing, check stationarity again
    check_stationarity(train_diff)

# Find optimal p, d, q (use differenced data if original is non-stationary)
optimal_pdq = find_pdq(train['Price'])  # Use train_diff if differenced data was used

# You can use the optimal PDQ in the final SARIMA model:
# final_model = SARIMAX(train['Price'], order=optimal_pdq).fit(disp=False)
# print(final_model.summary())

"""### Build Sarima model"""

# Define the SARIMA model
model = SARIMAX(train['Price'], seasonal_order=(0 , 1, 4, 12))  # Adjust parameters as needed
sarima_model = model.fit(disp=False)

# Print the summary
print(sarima_model.summary())

"""#### model evaluation and forcasting"""

# Forecast the test set
forecast = sarima_model.get_forecast(steps=len(test))
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(train['Price'], label='Train')  # Replace 'Value' with your column name
plt.plot(test['Price'], label='Test')
plt.plot(forecast_values, label='Forecast')
plt.fill_between(forecast_values.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1],
                 color='pink', alpha=0.3)
plt.title('SARIMA Forecast')
plt.legend()
plt.grid()
plt.show()

# Calculate Mean Squared Error
mse = mean_squared_error(test['Price'], forecast_values)
print(f"Mean Squared Error: {mse}")

import joblib

# Save the model
joblib.dump(sarima_model, 'sarima_model.pkl')

# Load the model
loaded_model = joblib.load('sarima_model.pkl')

