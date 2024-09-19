# Stock Market Price Prediction using Machine Learning

## Overview
This project aims to predict stock prices using various technical indicators and machine learning models like Ordinary Least Squares (OLS), Lasso, and Ridge regression. It includes preprocessing financial data, feature engineering, and evaluation of model performance using regression metrics and classification metrics.

## Table of Contents
1. Installation
2. Usage
4. Examples
5. Contributing

## Installation

Use the package manager pip to install the required libraries:
```bash
pip install yfinance pandas numpy matplotlib seaborn scipy statsmodels scikit-learn
```
## Usage

The python file utilises the yfinance API to source historical stock price data for Disney (DIS), along with data for other financial instruments like the VIX, gold, and crude oil over a 10 year period. This can be altered to suit any company with public trading data by altering the first argument of yfinance's '.download' function:

```python
# Download Disney (DIS) OHLC data from 2000 to 2024
disney_data = yf.download('DIS', start='2014-01-01', end='2024-01-01')
# Convert to DataFrame 
disney_df = pd.DataFrame(disney_data)
# Display the first few rows of the DataFrame
print(disney_df.head())
```
```python
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
```

The project then calculates various technical indicators for the stock data, such as:

• Simple and Exponential Moving Averages (SMA, EMA) <br>
• Relative Strength Index (RSI) <br>
• Moving Average Convergence Divergence (MACD) <br>
• Bollinger Bands <br>
• External factors like VIX, Gold prices, and Crude Oil prices are also standardized and incorporated as features <br>

For example, Simple moving averages:

```python
# Calculate Simple Moving Averages (SMA)
disney_data['SMA_30'] = disney_data['Close'].rolling(window=30).mean()
disney_data['SMA_100'] = disney_data['Close'].rolling(window=100).mean()
```
Missing data and NaN values from rolling calculations are handled by cleaning the dataset. Correlated features are identified using heatmaps and removed to avoid multicollinearity.
```python
# Drop rows with NaN values resulting from rolling calculations
cleaned_data = disney_data.dropna()

# Define the time period
start = datetime(2014, 1, 1)
end = datetime(2024, 1, 1)
```

```python
# Produce correlation heatmap for independent variables
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="YlOrRd", fmt=".2f", annot_kws={"size": 10, "color": "black"})
plt.show()
```

Several regression models are built, including:
• OLS (Ordinary Least Squares) <br>
• Lasso Regression <br>
• Ridge Regression <br>
These models are trained on 80% of the data and evaluated using metrics like Mean Squared Error (MSE) and R-squared.

```python
# Fit OLS model
model_fitted = sm.OLS(y_train, x_train_const).fit()
print(model_fitted.summary())
```
The program also converts predictions into binary categories (based on a threshold) and evaluates model performance using Accuracy, Precision, Recall, and F1-Score.

```python
accuracy = accuracy_score(y_test_binary, ols_predictions_binary)
print(f'Accuracy: {accuracy}')
```
Finally visualizations are produced to help interpret the results and evaluate the preformance of the regression models:

• Scatter plots for observed vs. predicted values <br>
• Model comparison plots for OLS, Lasso, and Ridge regression <br>

```python
plt.scatter(y_test, ols_predictions, color="orange")
plt.plot(y_test, y_test, color='red')
plt.show()
```
## Examples

The following are examples of plots produced by the program, starting with the correlation heatmap for the initial set of features (independent variables).
![Figure 2024-09-19 222819](https://github.com/user-attachments/assets/085b2b48-202e-4bee-beb0-7c1de32c1423)

Heatmap after removal of several features exhibiting multicolinearity:
![Figure 2024-09-19 222954](https://github.com/user-attachments/assets/eb8b968d-0c5f-4615-8653-b1409d609ab5)

Plot for observed values against predicted values for the test data (OLS):
![Figure 2024-09-19 223111](https://github.com/user-attachments/assets/3efa4229-91e3-472c-9487-1eacb7b2383e)

Subplots to compare the results for all three regression models:
![Figure 2024-09-19 223120](https://github.com/user-attachments/assets/0e0d8cfb-1f06-4fd5-95c8-ae8e22518827)

## Contributing 
Suggestions for Improvement:

Feature Engineering:
Consider adding lag features (e.g., price movements from the past 5 or 10 days). These can help models capture trends better.
Interaction Terms: Feature interaction (e.g., product of RSI and MACD) could capture non-linear relationships between the indicators.
Add seasonality features, such as month or weekday, to see if there are any recurring patterns over time (e.g., earnings season impact on stock prices).
Hyperparameter Tuning:
In Lasso and Ridge regressions, it's crucial to tune the alpha parameter using techniques like cross-validation. You might want to use GridSearchCV from sklearn to find the best alpha values.
Model Comparison:
Apart from linear models, try using non-linear models like Random Forest or XGBoost. These models are often better suited to capturing complex relationships in stock market data.
You could also explore ensemble methods, where multiple models' predictions are averaged or combined for improved performance.
More Metrics for Stock Prediction:
Incorporating financial ratios (e.g., Price/Earnings (P/E) ratio, Debt/Equity ratio) can enhance prediction accuracy.
You could explore Sentiment Analysis by scraping news headlines or social media to gauge public sentiment about Disney and other key factors affecting the stock.
More Time Series Specific Models:
Stock market data is inherently time-dependent, so using time-series models like ARIMA (AutoRegressive Integrated Moving Average), Prophet, or LSTM (Long Short-Term Memory Networks) could be valuable for better handling the temporal aspects of the data.
Backtesting:
Consider running a backtest where you simulate trading strategies based on your predictions and check if your models lead to profitable trades.
Backtesting against real historical data can show how your models perform in real-world scenarios.
Explainability:
Use techniques like SHAP or LIME to explain the model’s predictions, especially with Lasso and Ridge. This will help identify which features (indicators) are most important for the prediction.





