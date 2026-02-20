import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima

class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# region Functions
def plot_time_series(title, x, y, y_axis=None):
   """
   plot_time_series: plots the linear graph of a time series or differentiated time series.
   """
   plt.figure(figsize=(10, 5)) 
   plt.plot(x, y)
   plt.title(title)
   plt.xlabel('Date')
   if y_axis is not None:
      plt.ylabel(y_axis)
   plt.show()
def adf_test(col):
   """
   adf_test: calculates the Augmented Dickey Fuller test for the passed column
   """
   # adf test [In-Text Citation: (GeeksforGeeks, 2022)]
   adf_results = adfuller(col)
   alpha = 0.05
   print("Augmented Dickey Fuller Test")
   print(f"ADF Stat: {adf_results[0]}")
   print(f"p-value: {adf_results[1]}")
   print('Critical Values:')
   for k, v in adf_results[4].items():
      print('\t%s: %.3f' % (k, v))
   if adf_results[1] <= alpha:
      print("\nThe time series is " + text.BOLD +  "stationary" + text.END + " since the p-value is less than or equal to 0.05")
   else:
      print("\nThe time series is " + text.BOLD +  "nonstationary" + text.END + " since the p-value is greater than 0.05")
# endregion

# specifying the n/a values will allow us to remove empty rows
df = pd.read_csv('teleco_time_series .csv')
# aligns the df index to match the index of Day so df[0] doesn't exist
df.index = df.index + 1

# region Data Overview
# set date range [In-Text Citation: (NumFOCUS Inc, n.d.)]
df['Date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
df.drop(columns='Day', inplace=True)
df = df.reindex(columns=['Date', 'Revenue'])

print (text.BOLD + "\n-- Begin Data Analytics --\n" + text.END)
# plot the realized time series
plot_time_series('Realized Time Series', df['Date'], df['Revenue'], 'Daily Revenue (Millions)')
# adf for the original data
adf_test(df['Revenue'])
# endregion

# region Prepare the Data
print(text.UNDERLINE + "\nDifferentiating Data" + text.END)
stationary_df = pd.DataFrame(data={
   'Date': df['Date'],
   'Revenue': df['Revenue'].diff()
   })
# set index/frequency for seasonal_decompose function [In-Text Citation: (Snow & jezrael, 2019)]
stationary_df = stationary_df.set_index(df['Date']).asfreq('d')
# remove 1st row
stationary_df = stationary_df.iloc[1:, :]

# check stationarity again
plot_time_series('Differentiated (d=1) Time Series', stationary_df['Date'], stationary_df['Revenue'], 'Difference in Daily Revenue')
# adf for the differentiated data
adf_test(stationary_df['Revenue'])
#split Data
train = df.iloc[:584]
test = df.iloc[584:]
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index= False)
stationary_df.to_csv('prepared_data.csv', index= False)
# endregion

# region Graph Analysis
# seasonal decomposition [In-Text Citation: (Hayes, 2021)]
decompose_result = seasonal_decompose(stationary_df['Revenue'])
plot_time_series('Seasonality', stationary_df['Date'], decompose_result.seasonal)
plot_time_series('Trends', stationary_df['Date'], decompose_result.trend)
# acf and pacf [In-Text Citation: (Brownlee, 2020)]
fig, axs = plt.subplots(2)
fig.set_size_inches(10, 6)
plot_acf(stationary_df['Revenue'], ax=axs[0], zero=False)
plot_pacf(stationary_df['Revenue'], ax=axs[1], zero=False)
plt.show()
# spectral density [In-Text Citation: (Matplotlib Development Team, n.d.)]
plt.figure(figsize=(10, 5)) 
plt.title('Power Spectral Density')
plt.psd(stationary_df['Revenue'])
plt.show()
# decomposed time series
decompose_result.plot()
plt.show()
# confirmation of lack of trends
plot_time_series('Residuals', stationary_df['Date'], decompose_result.resid)
# endregion

# region Build ARIMA
print(text.UNDERLINE + "\nTesting ARIMA Models" + text.END)
# use auto arima to test best model [In-Text Citation: (Pulagam, 2020)]
best_model = auto_arima(train['Revenue'], trace=True, suppress_warnings=True)
print('\n')
# trend is t because original data has an upwards trend [In-Text Citation: (Vidal & Alien, 2021)]
arima_model = ARIMA(train['Revenue'], order=(best_model.order), trend='t').fit()
print(arima_model.summary())
# endregion

# region Forecasting
print(text.UNDERLINE + "\nForecasting ARIMA Models" + text.END)
# compare against test data [In-Text Citation: (GeeksforGeeks, 2020)]
start = len(train) 
end = len(train) + len(test) - 1
test_predictions = arima_model.predict(start, end,)
# plot predictions and actual values 
plt.plot(test['Date'], test['Revenue'], label='Actual')
plt.plot(test['Date'], test_predictions, label='Predictions')
plt.title('Predicting Test Data (147 Days)')
plt.ylabel('Daily Revenue (Millions)')
plt.xlabel('Date')
plt.legend()
plt.show()
# compare for 90 days from end (237 from end of train)
predictions = pd.DataFrame({
   'Date': pd.date_range(start='2025-01-01', periods=90, freq='d'),
   'Revenue': arima_model.predict(end, end+89)
})
print("Final Values:")
print(predictions.tail(3))
plt.plot(df['Date'], df['Revenue'], label='Actual Revenue')
plt.plot(predictions['Date'], predictions['Revenue'], label='Predictions')
plt.title('Predicting Data of the Next Business Quarter (90 Days)')
plt.ylabel('Daily Revenue (Millions)')
plt.xlabel('Date')
plt.legend()
plt.show()
# endregion

# region Evaluate Model
print(text.UNDERLINE + "\nEvaluating ARIMA Models" + text.END)
test = test.copy()
test['forecast_error'] = test['Revenue'] - test_predictions
test['abs_percent_error'] = (test['forecast_error'].abs() / test['Revenue']) * 100
mape = test['abs_percent_error'].mean()
mae = mean_absolute_error(test['Revenue'], test_predictions)
rmse = root_mean_squared_error(test['Revenue'], test_predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape.round(3)}")
print(f"Mean Absolute Error (MAE): {mae.round(3)}")
print(f"Root Mean Squared Error (RMSE): {rmse.round(3)}")
# endregion

print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)