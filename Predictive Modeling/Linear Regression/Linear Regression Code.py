import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def detect_duplicates(col_name):
   """Detect Duplicates: 
      Detects duplicate values in a single column.
   """
   print(f"{col_name}: ",  df.duplicated(subset = col_name).sum())
def describe_vars(list_name, list):
   """Describe Vars: 
      Used to describe variables and to visually check for outliers. 


      Compare min and max to each other.
      Compare if 25% is close to min and if 75% is close to max.
      Consider where the information was obtained.
   """
   print(f"{list_name} Values")
   print(str(df[list].describe()) + "\n")
def treat_outliers(col):
   """Treat Outliers: 
      Used to treat outliers using IQR. 
   """
   q1 = df[col].quantile(0.25)
   q3 = df[col].quantile(0.75)
   iqr = q3 - q1
   outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
   print(f"Outlier Count for {col}: {len(outliers)}")
   df.drop(outliers.index, inplace=True)
   print(f"New dataframe length: {len(df)}\n" )
   print(str(df['Income'].describe()))
def view_categorical_values(cols):
    """View Categorical Values: 
        Used to visually check unique values for categorical columns. 
    """
    for col in cols:  
      print(str(df[col].value_counts())+ "\n")
def plot_categorical_values(cols):
   """Plot Categorical Values:
      Univariate plots for a list of categorical values
   """
   print("\nClose the figure to continue...", end = " ")
   # calculate columns for the plot window
   total_cols = round(len(cols)/2)
   fig = plt.figure(figsize = (10,8))
   fig.suptitle('Independent Categorical Values', fontsize= 16)
   index = 1
   for col in cols:
      plt.subplot(2,total_cols, index)
      # makes a donut graph for categorical values
      df[col].value_counts().plot(kind='pie', autopct='%1.0f%%', colormap='Set3', title=col)
      circle = plt.Circle((0, 0), 0.75, fc='white')
      plt.gcf().gca().add_artist(circle)
      plt.ylabel(None)
      index += 1
   plt.tight_layout()
   plt.show()
def plot_numeric_values(cols):
   """Plot Numeric Values:
      Univariate plots for a list of numeric values
   """
   print("\nClose the figure to continue...", end = " ")
   # These are the bin ranges for each bar graph
   ranges = [range(0,10), range(15, 90, 5), range(0,260000,25000), range(60, 300, 20)]
   # calculate columns for the plot window
   total_cols = round(len(cols)/2)
   fig = plt.figure(figsize = (10,8))
   fig.suptitle('Independent Numeric Values', fontsize= 16)
   index = 1
   for col in cols:
      plt.subplot(2,total_cols, index)
      df[col].plot(
      kind='hist', bins=ranges[index-1], colormap='plasma', edgecolor='white', linewidth=1, title=col)
      index += 1
   plt.show()
def bivariate_categorical(cols, dependent, part):
   """Bivariate categorical:
      Bivariate plots with independent categorical variables and the dependent variable
      Plots in a 2x2 window, so this function runs twice b/c there are 8 categorical variables.
   """
   print("\nClose the figure to continue...", end = " ")
   current_col = 0
   current_row = 0
   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
   fig.suptitle(f"Bivariate Graphs with {dependent} and Categorical Variables ({part})", fontsize= 16)
   for i in range(4):
      # Creates bivariate histograms [In-Text Citation: (jhelpz, 2019)]
      df.groupby(cols[i])[dependent].plot.hist(alpha=0.5, bins=range(0,7500,500), ax=axes[current_row,current_col], edgecolor='white', linewidth=1, title=f"{cols[i]}")
      axes[current_row,current_col].legend(sorted(df[cols[i]].unique()))
      axes[current_row,current_col].set_xlabel(dependent)
      # The row index value is updated before the column
      if(current_col < 1):
         current_col += 1
      else:
         current_col = 0
         current_row += 1
   plt.tight_layout()
   plt.show()
def bivariate_numeric(cols, dependent):
   """Bivariate numeric:
      Bivariate plots with independent numeric variables and the dependent variable.
      Is also a 2x2 window
   """
   print("\nClose the figure to continue...", end = " ")
   current_col = 0
   current_row = 0
   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
   fig.suptitle(f"Bivariate Graphs with {dependent} and Numeric Variables", fontsize= 16)
   for col in cols:
      df.plot.scatter(x=dependent, y=col, ax=axes[current_row,current_col], c='#b82359')
      axes[current_row,current_col].set_xlabel(dependent)
       # The row index value is updated before the column
      if(current_col < 1):
         current_col += 1
      else:
         current_col = 0
         current_row += 1
   plt.show()
def regression_summary(indepen_columns, y, dropped):
   """Regression Summary:
      Creates a regression summary based on the passed values
   """
   print(f"\nDropped Columns: {dropped[1:]}")
   # Creates the regression model/summary [In-Text Citation: (Bobbitt, 2022)]
   x = sm.add_constant(indepen_columns)
   model = sm.OLS(y, x).fit()
   print(model.summary())
   return model

df = pd.read_csv('churn_clean.csv', keep_default_na = False, 
                 na_values = [' ', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'])
# Index values are the same as CaseOrder
df.index = df.index + 1

print (text.BOLD + "\n-- Begin Data Analytics --\n" + text.END)

# region Clean Data
# Duplicates
print(text.UNDERLINE + "Detecting Duplicates" + text.END)
print("Exact row: ", len(df)-len(df.drop_duplicates()))
detect_duplicates('CaseOrder');
detect_duplicates('Customer_id');
detect_duplicates('Interaction');
df.drop_duplicates()

# Missing Values
print(text.UNDERLINE + "\nDetecting Missing Values" + text.END)
print(df.isna().sum())

# Outliers
census_list = ['Population']
signup_list = ['Age', 'Children', 'Income']
business_list = ['Outage_sec_perweek', 'Email', 'Contacts', 'Yearly_equip_failure', 
                'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']
print(text.UNDERLINE + "\nDetecting Outliers" + text.END)
describe_vars("Census", census_list);
describe_vars("Sign-up", signup_list);
describe_vars("Business Generated", business_list);
print(text.UNDERLINE + "Treating Outliers" + text.END)
treat_outliers('Income')
# endregion

# region Describing Regression Independent Variables
print(text.UNDERLINE + "\nDescribing Variables for Analysis" + text.END)
independent_numeric =['Children', 'Age', 'Income', 'MonthlyCharge']
independent_categorical = ['Gender', 'Port_modem', 'Tablet', 'InternetService', 'Phone', 'Multiple', 'StreamingTV', 'StreamingMovies']
dependent_var = 'Bandwidth_GB_Year'

describe_vars("Numeric Variables", independent_numeric);
print("Categorical Variables")
view_categorical_values(independent_categorical);
print("Dependent Variable\n" + str(df[dependent_var].describe()) + "\n")
# graphs
print("\nClose the figure to continue...", end = " ")
df[dependent_var].plot(
   kind='hist', bins=range(0,7500,500), colormap='plasma', edgecolor='white', linewidth=1, title='Bandwidth Per Year (Dependent Variable)')
plt.xlabel('Bandwidth Usage in GB')
plt.show();
plot_categorical_values(independent_categorical);
plot_numeric_values(independent_numeric);
bivariate_categorical(independent_categorical[0:4], dependent_var, 1)
bivariate_categorical(independent_categorical[4:9], dependent_var, 2)
bivariate_numeric(independent_numeric, dependent_var)
# endregion

# region Data Transformations
print(text.UNDERLINE + "\n\nData Transformations" + text.END)
# remove excess columns
relevant_columns = independent_categorical + independent_numeric
relevant_columns.append(dependent_var)
df = df.filter(relevant_columns)
print("Excess Columns Removed. Remaining Columns:")
print(df.columns.values)
df = pd.get_dummies(df, drop_first=True, dtype=int)
print("With Dummy Columns:")
print(df.columns.values)
df.to_csv('churn_regression.csv')
print("Values saved to 'churn_regression.csv'")
# endregion

# region Multilinear Regression Model
print(text.UNDERLINE + "\nMultilinear Regression" + text.END)
# independent variables that are removed
dropped_columns = [dependent_var, 'StreamingMovies_Yes', 'Tablet_Yes', 'Income', 'Gender_Male', 'Phone_Yes', 
                  'Port_modem_Yes', 'Gender_Nonbinary', 'StreamingTV_Yes', 'Age']
# performs a regression summary increasing the amount of dropped columns for each summary
for i in range(1, len(dropped_columns)):
   regression_summary(df.drop(columns=dropped_columns[0:i]), df[dependent_var], dropped_columns[0:i]);
final_model = regression_summary(df.drop(columns=dropped_columns[0:]), df[dependent_var], dropped_columns[0:]);
# residual graphs
kept_columns = ['Children', 'MonthlyCharge', 'InternetService_Fiber Optic', 'InternetService_None', 'Multiple_Yes']
for col in kept_columns:
   print("\nClose the figure to continue...", end = " ")
   # Residual Plot [In-Text Citation: (Bobbitt, 2020)]
   resid_fig = plt.figure(figsize=(12,8))
   sm.graphics.plot_regress_exog(final_model, col, fig=resid_fig)
   plt.show()
# Residual Plot [In-Text Citation: (Straw, 2024)]
print(f"\nResidual Standard Error: {np.sqrt(final_model.mse_resid)}")
# endregion
print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)
