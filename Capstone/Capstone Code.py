import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
import scipy.stats as stats
from scipy.stats import mannwhitneyu
from scipy.stats.mstats import winsorize
import numpy as np
from datetime import timedelta
from functools import partial
from pmdarima.preprocessing import FourierFeaturizer
from pmdarima.arima import auto_arima
# makes tensorflow depreciation warnings quiet [In-Text Citation: (user1315789 & Freeman, 2020)]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from keras_tuner import RandomSearch
from keras.models import Sequential, load_model #type: ignore
from keras.layers import Dense, LSTM, Dropout #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore


class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
class arima_values:
   def __init__(self, fourier, seasonality, D, m):
      self.fourier = fourier
      self.season = seasonality
      self.D = D
      self.m = m
# region Functions
def describe_vars(list):
   """Describe Vars: 
      Used to describe variables and to visually check for outliers. 


      Compare min and max to each other.
      Compare if 25% is close to min and if 75% is close to max.
      Consider where the information was obtained.
   """
   split = int(len(list)/2)
   first_half = list[:split]
   second_half = list[split:]
   print(str(df[first_half].describe()))
   print(str(df[second_half].describe()))
def view_categorical_values(cols):
    """View Categorical Values: 
        Used to visually check unique values for categorical columns. 


        Ensure the values align with what is stated in the dictonary
    """
    for col in cols:
      temp = df[col].unique()[pd.notna(df[col].unique())]
      temp.sort()
      print(f"{col}: " + str(temp))
def treat_outliers(col):
   """Treat Outliers: 
      Used to treat outliers using IQR. 
   """
   q1 = df[col].quantile(0.25)
   q3 = df[col].quantile(0.75)
   iqr = q3 - q1
   outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
   print(f"Outlier Count for {col}: {len(outliers)}")
   if df[col].dtype == 'int64':
      df.loc[outliers.index, col] = int(df[col].median())
   else:
      df.loc[outliers.index, col] = df[col].median()
def find_daily_occupancy(data, date_range):
   """
   find_daily_occupancy: counts the amount of people at the hotel on a given day
   """
   # initialized with 0 so every day has a value
   occupancy_series = pd.Series(0, index=date_range)
   for date in date_range:
      occupancy = data.loc[(data['arrival_date'] <= date) & (data['leave_date'] >= date)]
      occupancy_series[date] = len(occupancy)
   return occupancy_series
def graph_forecast_results(training, testing, test_predictions, forecast, hotel_type, test_type):
   """
   graph_forecast_results: graphs the results of forecasting and training the data
   """
   plt.plot(training.index, training, label='Training')
   plt.plot(testing.index, testing, label='Testing')
   plt.plot(testing.index, test_predictions, label='Prediction of Test')
   plt.plot(forecast.index, forecast, label='Forecast')
   plt.xlabel('Date')
   plt.ylabel('Count')
   plt.legend(loc='upper left')
   plt.title(f'Forecast of {hotel_type} Hotel Occupancy ({test_type})')

def create_x_y(data, x_len, y_len):
   """
   create_x_y: creates x and y values for the time series.
      The x values are x_len lagged values of y where y_len is the amount of the values to be predicted.
   """
   x_list, y_list = [], []
   for i in range(len(data)): 
      x_end = i + x_len
      y_end = x_end + y_len
      if y_end > len(data):
         break
      # x_list is the amount of values used in prediction
      x_list.append(data[i:x_end])
      # y_list is the predicted values
      y_list.append(data[x_end: y_end])
   return np.array(x_list), np.array(y_list)
def build_model(hp, prediction_len):
   """
   build model: runs the designated amount of tests to find the most accurate neural network
   """
   extra_LSTM = hp.Int('lstm_layers', 1, 2)
   model = Sequential()
   if hp.Boolean('Dropout'):
      model.add(Dropout(rate=0.2, seed=101))
   for i in range(extra_LSTM):
      # last lstm does not return sequences
      if i + 1 != extra_LSTM:
         model.add(LSTM(hp.Int(f'LSTM_{i}',min_value=32,max_value=512,step=32), activation=hp.Choice(f'LSTM_activation_{i}', values=['relu', 'swish']), return_sequences=True))
      else:
         model.add(LSTM(hp.Int(f'LSTM_{i}',min_value=32,max_value=512,step=32), activation=hp.Choice(f'LSTM_activation_{i}', values=['relu', 'swish'])))
   for i in range(hp.Int('dense_layers', 0, 3)):
      model.add(Dense(hp.Int(f'dense_{i}',min_value=32, max_value=512,step=32), activation=hp.Choice(f'Dense_activation_{i}', values=['relu', 'swish'])))
   model.add(Dense(prediction_len, activation='sigmoid'))
   model.compile(optimizer=Adam(learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])),
         loss='mae',
         metrics=['mean_absolute_percentage_error'])
   return model
def neural_network(hotel_data, type_name, model_dir, model_filename, metrics_filename):
   """
   neural_network: Creates or loads a neural network for the passed data
   """
   scaler = MinMaxScaler()
   normal_data = scaler.fit_transform(hotel_data.values.reshape(-1, 1))
   # https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
   # https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
   split_location = int(len(normal_data)*0.8) # the index where the data is split for testing/training
   y_len = len(normal_data) - split_location # length of testing data
   x_len = 425 # amount of lagged values to predict the next set of values (y_len)
   feature_count = 1 # only one feature tested since it is a time series
   X_train, y_train = create_x_y(normal_data, x_len, y_len)
   # change shape to be [samples, features, time steps(x_len)]
   X_train = X_train.reshape(len(X_train), feature_count, x_len)
   early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
   model = None
   metrics = None

   # loads in an existent model
   if os.path.isfile(model_filename) and os.path.isfile(metrics_filename):
      model = load_model(model_filename)
      csv = pd.read_csv(metrics_filename)
      metrics = pd.Series(data=csv.iloc[:, 1].values, index=csv.iloc[:, 0])
   else: 
      # find best model
      tunning_model = partial(build_model, prediction_len=y_len)
      # 20 trials with 3 executions per trial took ~9 mins for 1 hotel type
      # 100 trials with 5 executions per trial took ~2 hr 25 mins for 1 hotel type
      tuner = RandomSearch(
         tunning_model,
         objective='mean_absolute_percentage_error',
         max_trials=250, 
         executions_per_trial=3,
         directory=model_dir,
         project_name='my_model'
      )
      tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
      print(tuner.results_summary(num_trials=3))
      # get best hyperparameters, not best model [In-Text Citation: (Kashyap, 2024)]
      best_hp = tuner.get_best_hyperparameters()[0]
      model = tuner.hypermodel.build(best_hp)
      print()
      history = model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])
      stopped_epoch = early_stopping.best_epoch
      metrics = pd.Series({
      'MAPE': history.history['mean_absolute_percentage_error'][stopped_epoch],
      'Loss': history.history['loss'][stopped_epoch],
      })
      metrics.to_csv(metrics_filename)
      model.save(model_filename)

   # inspect results
   print()
   print(model.summary())
   print("\nModel Evaluation")
   print(f'Mean Absolute Percentage Error Score: {round(metrics['MAPE'], 3)} %')
   print(f"Loss: {round(metrics['Loss'], 3)}")
   print()

   # make predictions
   # need the last x_len values before split to make a prediction of the test data
   X_test = normal_data[(split_location-x_len):split_location]
   # value is 1 because only one sequence is passed to predict testing values
   X_test = X_test.reshape(1, feature_count, x_len)
   y_pred = model.predict(X_test)
   y_pred = scaler.inverse_transform(y_pred)
   # need last x_len values to make forecast
   forecast_input = normal_data[-x_len:]
   forecast_input = forecast_input.reshape(1, feature_count, x_len)
   forecast_pred = model.predict(forecast_input)
   forecast_pred = scaler.inverse_transform(forecast_pred)
      
   # graph results
   training = hotel_data[:split_location]
   testing = hotel_data[split_location:]
   test_predictions = pd.Series(data=y_pred[0], index=hotel_data.index[split_location:(split_location+y_len)])
   forecast = pd.Series(data=forecast_pred[0], index=pd.date_range(start = hotel_data.index[-1], periods=y_len))
   graph_forecast_results(training, testing, test_predictions, forecast, type_name, "Neural Network")
   plt.show()

   return test_predictions, forecast, round(metrics['MAPE'], 3)
def adf_test(col):
   """
   adf_test: calculates the Augmented Dickey Fuller test for the passed column
   """
   # adf test [In-Text Citation: (GeeksforGeeks, 2022)]
   adf_results = adfuller(col)
   alpha = 0.05
   print(f"Augmented Dickey-Fuller Test")
   print(f"ADF Stat: {adf_results[0]}")
   print(f"p-value: {adf_results[1]}")
   print('Critical Values:')
   for k, v in adf_results[4].items():
      print('\t%s: %.3f' % (k, v))
   if adf_results[1] <= alpha:
      print("The time series is " + text.BOLD +  "stationary" + text.END + " since the p-value is <= 0.05")
      return True
   else:
      print("The time series is " + text.BOLD +  "nonstationary" + text.END + " since the p-value >= 0.05")
      return False
def arima(hotel_data, type_name, arima_values):
   """
   arima: does an arima forecast
   """
   stationary_data = hotel_data
   while not(adf_test(stationary_data.values)):
      stationary_data = stationary_data.diff()
      stationary_data = stationary_data.iloc[1:]
   print()
   # graphing
   decompose_result = seasonal_decompose(stationary_data, model='additive')
   decompose_result.plot()
   plt.show() 
   # acf and pacf [In-Text Citation: (Brownlee, 2020)]
   fig, axs = plt.subplots(2)
   fig.set_size_inches(10, 6)
   plot_acf(stationary_data, ax=axs[0], zero=False)
   plot_pacf(stationary_data, ax=axs[1], zero=False)
   plt.show()

   # make sarima model
   # https://medium.com/@tirthamutha/time-series-forecasting-using-sarima-in-python-8b75cd3366f2
   split_location = int(len(hotel_data)*0.8) # the index where the data is split for testing/training
   y_len = len(hotel_data) - split_location
   training = hotel_data[:split_location]
   testing = hotel_data[split_location:]
   time_length = np.arange(len(hotel_data)+y_len)

   exog = []
   for i in range(len(arima_values.fourier)):
      # https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.preprocessing.FourierFeaturizer.html#pmdarima.preprocessing.FourierFeaturizer
      fourier = FourierFeaturizer(arima_values.season[i], arima_values.fourier[i]).fit_transform(time_length)[1]
      if i == 0:
         exog = fourier.to_numpy()
      else:
         exog = np.concatenate((exog, fourier.to_numpy()), axis=1)

   # use auto arima to testing best model [In-Text Citation: (Pulagam, 2020)]
   best_model = auto_arima(training.values, start_p=1, start_q=1, test='adf', D=arima_values.D, m=arima_values.m,
               exogenous=exog[:split_location], seasonal=True, trace=True, suppress_warnings=True)
   best_model.fit(training.values, X=exog[:split_location])
   print(best_model.summary())

   # compare against testing data [In-Text Citation: (GeeksforGeeks, 2020)]
   # predict both test and forecast at once
   both_predictions = best_model.predict((y_len*2), X=exog[split_location:])
   # sort the predictions
   forecast = pd.Series(both_predictions[y_len:], index = pd.date_range(start = hotel_data.index[-1], periods=y_len))
   test_predictions = pd.Series(data=both_predictions[:y_len], index=testing.index)
   mape = mean_absolute_percentage_error(hotel_data[split_location:], test_predictions)
   mae = mean_absolute_error(hotel_data[split_location:], test_predictions)
   rmse = root_mean_squared_error(hotel_data[split_location:], test_predictions)
   print("\nModel Metrics:")
   print(f"Mean Absolute Percentage Error (MAPE): {round((mape * 100), 3)} %")
   print(f"Mean Absolute Error (MAE): {mae}")
   print(f"Root Mean Squared Error: {rmse}")
   print()

   # plot forecast and actual values 
   graph_forecast_results(training, testing, test_predictions, forecast, type_name, "SARIMAX")
   plt.show()

   return test_predictions, forecast, round((mape * 100), 3)
def compare_forecasts(hotel_data, type_name, arima_test, arima_forecast, arima_mape, neural_test, neural_forecast, neural_mape):
   """
   compare_forecasts: compares the results of the two forecasting tests
   """
   split  = int(len(hotel_data) * 0.8)
   training = hotel_data[:split]
   testing = hotel_data[split:]
   print()
   print(f"SARIMAX MAPE: {arima_mape} %")
   print(f"Neural Network MAPE: {neural_mape} % ")
   plt.figure(figsize=(10, 5))
   plt.subplot(2, 1, 1)
   graph_forecast_results(training, testing, arima_test, arima_forecast, type_name, "SARIMAX")
   plt.subplot(2, 1, 2)
   graph_forecast_results(training, testing, neural_test, neural_forecast, type_name, "Neural Network")
   plt.tight_layout()
   plt.show()
# endregion

df = pd.read_csv('hotel_bookings.csv')

print (text.BOLD + "\n-- Begin Data Analytics --\n" + text.END)

# region Explore data
print(text.UNDERLINE + "Explore Data" + text.END)
cell_count = df.shape[0] * df.shape[1]
missing_count = sum(df.isna().sum())
print(f"Shape: {df.shape}")
print(f"Cell Count: {"{:,}".format(cell_count)}")
print(f"Missing Cell Count: {"{:,}".format(missing_count)}")
print(f"Sparsity: {round(((missing_count/cell_count) * 100), 2)}%")

# Data types
print("\nData Types: ")
print(df.dtypes)
# Check Duplicates
print("\nExact row duplicates: ", len(df)-len(df.drop_duplicates()))
df.drop_duplicates()
# Missing Values
print("\nMissing Values:")
print(df.isna().sum())
print()
# Check for Outliers
numeric_list = ['lead_time', 'arrival_date_year', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
                 'previous_cancellations', 'booking_changes', 'days_in_waiting_list', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']
print("Checking for Outliers")
describe_vars(numeric_list);
# Check Categorical Columns Values
print("\nCategorical Values")
categorical_list = ['hotel', 'is_canceled', 'arrival_date_month', 'arrival_date_week_number', 'meal', 'country', 'market_segment', 'distribution_channel',
                     'is_repeated_guest', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'agent', 'company', 'customer_type']
view_categorical_values(categorical_list)
# endregion

# region Clean Data
print(text.UNDERLINE + "Clean Data" + text.END)
# Missing Values
df.fillna({'agent': 0}, inplace=True)
df.fillna({'company': 0}, inplace=True)
df.fillna({'children': 0}, inplace=True)
df.fillna({'country': 'UNK'}, inplace=True)
print("Missing Values:")
print(df.isna().sum())
# incorrect data types
print("\nData Types: ")
df = df.astype({'children': 'int64', 'agent': 'int64', 'company': 'int64'})
print(df[['children', 'agent', 'company']].dtypes)
# Outliers
print("\nOutliers:")
outlying_vars = ['lead_time', 'adr']
df.replace({'adults': 0}, 1, inplace=True)
df.loc[df["adr"] < 0] = 0
for var in outlying_vars:
   treat_outliers(var)
outlying_vars.append('adults')
describe_vars(outlying_vars)
df.to_csv('clean_df.csv', index=False)
# endregion

# region Prepare Data
print(text.UNDERLINE + "\nPrepare Data" + text.END)
# remove canceled reservations
prepared_df = df[df['is_canceled'] == 0]
# keep only relevant variables
prepared_df = prepared_df.filter(items=[
   'hotel', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 
   'stays_in_weekend_nights', 'stays_in_week_nights'])
# create dictionary for months
dictionary = {'January':1, 
     'February':2, 
     'March':3, 
     'April':4, 
     'May':5, 
     'June':6, 
     'July':7, 
     'August':8, 
     'September':9, 
     'October':10, 
     'November':11, 
     'December':12 }                 
arrival_date_df = pd.DataFrame({
   'year': prepared_df['arrival_date_year'],
   'month': prepared_df['arrival_date_month'].map(dictionary),
   'day': prepared_df['arrival_date_day_of_month']
})
# convert arrival dates to appropriate format
prepared_df['arrival_date'] = pd.to_datetime(arrival_date_df[['year', 'month', 'day']])
prepared_df['length_of_stay'] = prepared_df['stays_in_weekend_nights'] + prepared_df['stays_in_week_nights']
prepared_df['leave_date'] = prepared_df['arrival_date'] + prepared_df['length_of_stay'].map(timedelta)
print("Resulting Columns:")
print(prepared_df.columns)
# split the data by hotel type
city_df = prepared_df[prepared_df['hotel'] == 'City Hotel']
resort_df = prepared_df[prepared_df['hotel'] == 'Resort Hotel']
# time series starts 2 weeks after 1st date b/c reservations before db start are not reflected
starting_date = prepared_df['arrival_date'].min() + timedelta(days=14)
entire_date_range = pd.date_range(start = starting_date, end = prepared_df['arrival_date'].max())
# occupancy is split by hotel type for Wilcoxon Rank-Sum Test
city_occupancy = find_daily_occupancy(city_df, entire_date_range)
resort_occupancy = find_daily_occupancy(resort_df, entire_date_range)
print("\nDays without Guests")
print(f"City Hotel: {city_occupancy[city_occupancy == 0].count()}")
print(f"Resort Hotel: {resort_occupancy[resort_occupancy == 0].count()}")
print('\nCity Hotel Occupancy Detailed:')
print(city_occupancy.describe())
print('Resort Hotel Occupancy Detailed:')
print(resort_occupancy.describe())
# winsorize outliers
# means the top/bottom 2% are replaced with the 2nd/98th percentile
winsorize_limits = 0.02
winsorize(city_occupancy, limits=[winsorize_limits, winsorize_limits], inplace=True)
winsorize(resort_occupancy, limits=[winsorize_limits, winsorize_limits], inplace=True)
print("\nAfter Winsorization")
print('City Hotel Occupancy Detailed:')
print(city_occupancy.describe())
print('Resort Hotel Occupancy Detailed:')
print(resort_occupancy.describe())
city_occupancy.to_csv('city.csv', index_label='Date', header=['Hotel Occupancy'])
resort_occupancy.to_csv('resort.csv', index_label='Date', header=['Hotel Occupancy'])
prepared_df.to_csv('prep_df.csv', index=False)
# endregion

# region Explore Prepared Data
print(text.UNDERLINE + "\nExplore Prepared Data" + text.END)
# Wilcoxon Rank-Sum Test
print("Wilcoxon Rank-Sum Test (Mann-Whitney test):")
# https://www.geeksforgeeks.org/mann-and-whitney-u-test/
stat, p_val = mannwhitneyu(city_occupancy, resort_occupancy)
print(f"statistic: {stat}, p-value: {p_val}")
if p_val < 0.05:
   print(f"p < 0.05 so the alternative hypothesis is " + text.BOLD + "true" + text.END + ".")
else:
   print(f"p > 0.05 so the null hypothesis is " + text.BOLD + "true" + text.END + ". ")
print(f"The samples have values that are distributed {"differently" if p_val < 0.05 else "similarly"}.")
print()

# line graph occupancy of both hotel types
plt.figure(figsize=(10, 5)) 
plt.plot(city_occupancy.index, city_occupancy, label='City')
plt.plot(resort_occupancy.index, resort_occupancy, label='Resort')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.title('Occupancy of City and Resort Hotels')
plt.show()

# checking for normality with q-q plot
# https://www.geeksforgeeks.org/quantile-quantile-plots/
plt.figure(figsize=(10, 5)) 
plt.subplot(1, 2, 1)
stats.probplot(city_occupancy, dist="norm", plot=plt,)
plt.title('Normal Q-Q plot (City)')
plt.xlabel('Quantiles')
plt.ylabel('Ordered Values')
plt.subplot(1, 2, 2)
stats.probplot(resort_occupancy, dist="norm", plot=plt)
plt.title('Normal Q-Q plot (Resort)')
plt.xlabel('Quantiles')
plt.ylabel('Ordered Values')
plt.show()
# data is over dispersed https://www.datacamp.com/tutorial/qq-plot

# looking for similarities in shape by min max normalization
scaler = MinMaxScaler()
normal_city = scaler.fit_transform(city_occupancy.values.reshape(-1, 1))
normal_resort = scaler.fit_transform(resort_occupancy.values.reshape(-1, 1))
plt.figure(figsize=(10, 5)) 
plt.plot(city_occupancy.index, normal_city, label='City')
plt.plot(resort_occupancy.index, normal_resort, label='Resort')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.title('Reservations Shape Comparision Using Normalization')
plt.show()
# endregion

# region City Forecast
print(text.UNDERLINE + "City Forecast" + text.END)
print("SARIMAX Model:")
(c_arima_test, c_arima_forecast, 
 c_arima_mape) = arima(city_occupancy, 'City', arima_values([11], [365], 1, 7))
print("Neural Network:")
(c_neural_test, c_neural_forecast, 
 c_neural_mape) = neural_network(city_occupancy, 'City', 'city_model_dir', 'City_Model.keras', 'City_Metrics.csv')
compare_forecasts(city_occupancy, 'City', c_arima_test, c_arima_forecast, c_arima_mape, c_neural_test, c_neural_forecast, c_neural_mape)
# endregion

# region Resort Forecast
print(text.UNDERLINE + "\nResort Forecast" + text.END)
print("SARIMAX Model:")
(r_arima_test, r_arima_forecast, 
 r_arima_mape) = arima(resort_occupancy, 'Resort', arima_values([11], [365], 1, 7))
print("Neural Network:")
(r_neural_test, r_neural_forecast, 
 r_neural_mape) = neural_network(resort_occupancy, 'Resort', 'resort_model_dir', 'Resort_Model.keras', 'Resort_Metrics.csv')
compare_forecasts(resort_occupancy, 'Resort', r_arima_test, r_arima_forecast, r_arima_mape, r_neural_test, r_neural_forecast, r_neural_mape)
# endregion

print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)