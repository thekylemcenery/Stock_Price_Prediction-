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

The python file utilises the yfinance API to source historical stock price data for Disney (DIS) over a 10 year period. This can be altered to suit any company with public trading data by altering the first argument of yfinance's '.download' function:

```python
# Download Disney (DIS) OHLC data from 2000 to 2024
disney_data = yf.download('DIS', start='2014-01-01', end='2024-01-01')
# Convert to DataFrame 
disney_df = pd.DataFrame(disney_data)
# Display the first few rows of the DataFrame
print(disney_df.head())
```
