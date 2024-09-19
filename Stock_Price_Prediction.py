import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score

# Download Disney (DIS) OHLC data from 2000 to 2024
disney_data = yf.download('DIS', start='2014-01-01', end='2024-01-01')
# Convert to DataFrame 
disney_df = pd.DataFrame(disney_data)
# Display the first few rows of the DataFrame
print(disney_df.head())

# Check for missing values in the DataFrame
missing_values = disney_df.isnull().sum()
# Display the result
print(missing_values)

# Calculate Simple Moving Averages (SMA)
disney_data['SMA_30'] = disney_data['Close'].rolling(window=30).mean()
disney_data['SMA_100'] = disney_data['Close'].rolling(window=100).mean()

# Calculate Exponential Moving Averages (EMA)
disney_data['EMA_30'] = disney_data['Close'].ewm(span=30, adjust=False).mean()
disney_data['EMA_100'] = disney_data['Close'].ewm(span=100, adjust=False).mean()

# Calculate Relative Strength Index (RSI)
delta = disney_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
disney_data['RSI'] = 100 - (100 / (1 + rs))

# Calculate Moving Average Convergence Divergence (MACD)
disney_data['MACD_Line'] = disney_data['Close'].ewm(span=12, adjust=False).mean() - disney_data['Close'].ewm(span=26, adjust=False).mean()
disney_data['Signal_Line'] = disney_data['MACD_Line'].ewm(span=9, adjust=False).mean()
disney_data['MACD_Histogram'] = disney_data['MACD_Line'] - disney_data['Signal_Line']

# Calculate Bollinger Bands
disney_data['SMA_20'] = disney_data['Close'].rolling(window=20).mean()
disney_data['Bollinger_Upper'] = disney_data['SMA_20'] + (disney_data['Close'].rolling(window=20).std() * 2)
disney_data['Bollinger_Lower'] = disney_data['SMA_20'] - (disney_data['Close'].rolling(window=20).std() * 2)

# Drop rows with NaN values resulting from rolling calculations
cleaned_data = disney_data.dropna()

# Define the time period
start = datetime(2014, 1, 1)
end = datetime(2024, 1, 1)

# Fetch VIX data
vix_data = yf.download('^VIX', start=start, end=end, interval='1d')
vix_data.dropna(inplace=True)
vix_data.reset_index(inplace=True)
vix_data.rename(columns={'Adj Close': 'VIX'}, inplace=True)
vix_data.set_index('Date', inplace=True)
vix_data['VIX_Change'] = vix_data['VIX'].pct_change() * 100
vix_data['VIX_Change'].fillna(vix_data['VIX_Change'].mean(), inplace=True)
vix_data['Standardized_VIX_Change'] = (vix_data['VIX_Change'] - vix_data['VIX_Change'].mean()) / vix_data['VIX_Change'].std()

# Fetch Gold price data
gold_data = yf.download('GC=F', start=start, end=end, interval='1d')
gold_data.dropna(inplace=True)
gold_data.reset_index(inplace=True)
gold_data.rename(columns={'Adj Close': 'Gold_Price'}, inplace=True)
gold_data.set_index('Date', inplace=True)
gold_data['Gold_Change'] = gold_data['Gold_Price'].pct_change() * 100
gold_data['Gold_Change'].fillna(gold_data['Gold_Change'].mean(), inplace=True)
gold_data['Standardized_Gold_Change'] = (gold_data['Gold_Change'] - gold_data['Gold_Change'].mean()) / gold_data['Gold_Change'].std()

# Fetch Crude Oil price data
crude_oil_data = yf.download('CL=F', start=start, end=end, interval='1d')
crude_oil_data.dropna(inplace=True)
crude_oil_data.reset_index(inplace=True)
crude_oil_data.rename(columns={'Adj Close': 'Crude_Oil_Price'}, inplace=True)
crude_oil_data.set_index('Date', inplace=True)
crude_oil_data['Crude_Oil_Change'] = crude_oil_data['Crude_Oil_Price'].pct_change() * 100
crude_oil_data['Crude_Oil_Change'].fillna(crude_oil_data['Crude_Oil_Change'].mean(), inplace=True)
crude_oil_data['Standardized_Crude_Oil_Change'] = (crude_oil_data['Crude_Oil_Change'] - crude_oil_data['Crude_Oil_Change'].mean()) / crude_oil_data['Crude_Oil_Change'].std()

# Merge all standardized dataframes into market_data
market_data = pd.merge(vix_data[['Standardized_VIX_Change']], 
                       gold_data[['Standardized_Gold_Change']], 
                       left_index=True, 
                       right_index=True)

market_data = pd.merge(market_data, 
                       crude_oil_data[['Standardized_Crude_Oil_Change']], 
                       left_index=True, 
                       right_index=True)

# Ensure the Date column is the index in all dataframes and is in datetime format
cleaned_data.index = pd.to_datetime(cleaned_data.index)
market_data.index = pd.to_datetime(market_data.index)

# Merge cleaned_data with market_data on the Date index
data = pd.merge(cleaned_data, 
                         market_data, 
                         left_index=True, 
                         right_index=True, 
                         how='left')

# Check for missing values
missing_values = data.isnull().sum()

# Calculate the percentage of missing data in each column
missing_percentage = (missing_values / len(data)) * 100

# Display the missing data statistics
print("Missing Values in Each Column:\n", missing_values)
print("\nPercentage of Missing Data:\n", missing_percentage)

# Drop any rows with missing values after merging
data.dropna(inplace=True)

# List of columns to drop (remove 'Adj close' related columns)
columns_to_drop = [
    'Open', 'High', 'Low', 'Close',                            # Raw price data
    'SMA_30', 'SMA_100',                                       # EMA is preferred over SMA
    'MACD_Line', 'Signal_Line',                                # Only using MACD_Histogram
]

# Drop the selected columns
data.drop(columns=columns_to_drop, inplace=True)

# Rename columns by removing "Standardized_" prefix
data.rename(columns={
    'Standardized_VIX_Change': 'VIX_Change',
    'Standardized_Gold_Change': 'Gold_Change',
    'Standardized_Crude_Oil_Change': 'Crude_Oil_Change'
}, inplace=True)

# Move 'Close' column to the end of the DataFrame
columns = [col for col in data.columns if col != 'Adj Close']  # All columns except 'Close'
columns.append('Adj Close')  # Append 'Adj Close' at the end
# Reorder the DataFrame columns
data = data[columns]

# Produce correlation heatmap for independent variables
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="YlOrRd", fmt=".2f", annot_kws={"size": 10, "color": "black"})
plt.title('Correlation Heatmap of Stock Price Data', fontsize=16, color='white')
# Change the color of the axis labels and ticks
plt.gca().tick_params(axis='both', colors='white')
# Change the background color of the plot area
plt.gca().patch.set_facecolor('#1e1e1e')
# Change the background color of the figure area
plt.gcf().patch.set_facecolor('#1e1e1e')
plt.show()

data = data.drop("EMA_30", axis = 1)
data = data.drop("Bollinger_Upper", axis = 1)
data = data.drop("Bollinger_Lower", axis = 1)
print(data.columns)

# Produce correlation heatmap for independent variables
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="Blues", fmt=".2f", annot_kws={"size": 10, "color": "black"})
plt.title('Correlation Heatmap of Stock Price Data with corellated variables removed', fontsize=16,color = 'white')
# Change the color of the axis labels and ticks
plt.gca().tick_params(axis='both', colors='white')
# Change the background color of the plot area
plt.gca().patch.set_facecolor('#1e1e1e')
# Change the background color of the figure area
plt.gcf().patch.set_facecolor('#1e1e1e')
plt.show()

print(data.head(10))

# Define features (independent variables) and target (dependent variable)
features = [col for col in data.columns if col != 'Adj Close']
target = 'Adj Close'
x = data[features]
y = data[target]

# Split the data into a training set and a testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1111)

# Check the size of the splits
print(f'Training set size: {x_train.shape[0]} samples')
print(f'Test set size: {x_test.shape[0]} samples')

# Add a constant to the predictors because statsmodels' OLS doesn't include it by default
x_train_const = sm.add_constant(x_train)

# Fit the OLS model
model_fitted = sm.OLS(y_train, x_train_const).fit()
# Print Summary
print(model_fitted.summary())

# Adding a constant to the test predictors
x_test_const = sm.add_constant(x_test)

# Making predictions on the test set
ols_predictions = model_fitted.predict(x_test_const)

# Scatter plot for observed vs predicted values on test data
plt.figure(figsize=(10, 6), facecolor='#1e1e1e')  # Set figure background color

plt.scatter(y_test, ols_predictions, color="orange")
plt.xlabel('Observed Values', color='white')  # Set xlabel color
plt.ylabel('Predicted Values', color='white')  # Set ylabel color
plt.title('Observed vs Predicted Values on Test Data', color='white')  # Set title color
plt.plot(y_test, y_test, color='red')  # Line for perfect prediction (true values)
# Enable grid lines and set their color
plt.grid(True, color='white', linestyle='--', linewidth=0.5)
# Change the color of the axis labels and ticks
plt.gca().tick_params(axis='both', colors='white')
# Change the background color of the plot area
plt.gca().patch.set_facecolor('#1e1e1e')
plt.show()


# Define a threshold (for example, the median of the observed values)
threshold = y_test.median()

# Convert predictions and actual values to binary categories based on the threshold
y_test_binary = (y_test > threshold).astype(int)
ols_predictions_binary = (ols_predictions > threshold).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_test_binary, ols_predictions_binary)
precision = precision_score(y_test_binary, ols_predictions_binary)
recall = recall_score(y_test_binary, ols_predictions_binary)
f1 = f1_score(y_test_binary, ols_predictions_binary)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Prepare training and testing data for Lasso
x_train_lasso, x_test_lasso, y_train_lasso, y_test_lasso = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize Lasso model with a chosen alpha (regularization strength)
lasso_model = Lasso(alpha=0.1)  # Might need to tune alpha using cross-validation
# Fit the model
lasso_model.fit(x_train_lasso, y_train_lasso)
# Make predictions
y_pred_lasso = lasso_model.predict(x_test_lasso)
# Evaluate the model
mse_lasso = mean_squared_error(y_test_lasso, y_pred_lasso)
print("Mean Squared Error for Lasso Regression:", mse_lasso)
# Coefficients
print("Lasso Coefficients:", lasso_model.coef_)

# Prepare training and testing data for Ridge
x_train_ridge, x_test_ridge, y_train_ridge, y_test_ridge = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize Ridge model with a chosen alpha (regularization strength)
ridge_model = Ridge(alpha=1.0)  # Might need to tune alpha using cross-validation
# Fit the model
ridge_model.fit(x_train_ridge, y_train_ridge)
# Make predictions
y_pred_ridge = ridge_model.predict(x_test_ridge)
# Evaluate the model
mse_ridge = mean_squared_error(y_test_ridge, y_pred_ridge)
print("Mean Squared Error for Ridge Regression:", mse_ridge)
# Coefficients
print("Ridge Coefficients:", ridge_model.coef_)

# Plot Observed vs Predicted for OLS, Lasso, and Ridge
plt.figure(figsize=(18, 6), facecolor='#1e1e1e')

# OLS Regression Plot
plt.subplot(1, 3, 1)
plt.scatter(y_test, ols_predictions, color='orange')
plt.xlabel('Observed Values', color='white')
plt.ylabel('Predicted Values', color='white')
plt.title('OLS Regression', color='white')
# Add white gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')
plt.gca().tick_params(axis='both', colors='white')
plt.gca().patch.set_facecolor('#1e1e1e')  # Background color of the plot area

# Lasso Regression Plot
plt.subplot(1, 3, 2)
plt.scatter(y_test_lasso, y_pred_lasso, color='blue')
plt.xlabel('Observed Values', color='white')
plt.ylabel('Predicted Values', color='white')
plt.title('Lasso Regression', color='white')
# Add white gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')
plt.gca().tick_params(axis='both', colors='white')
plt.gca().patch.set_facecolor('#1e1e1e')  # Background color of the plot area

# Ridge Regression Plot
plt.subplot(1, 3, 3)
plt.scatter(y_test_ridge, y_pred_ridge, color='cyan')
plt.xlabel('Observed Values', color='white')
plt.ylabel('Predicted Values', color='white')
plt.title('Ridge Regression', color='white')
# Add white gridlines
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='white')
plt.gca().tick_params(axis='both', colors='white')
plt.gca().patch.set_facecolor('#1e1e1e')  # Background color of the plot area

plt.tight_layout()
plt.show()



