import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import (accuracy_score) 

class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

#region functions
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
   print(str(df[col].describe()))
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
   fig = plt.figure(figsize = (10,8))
   fig.suptitle('Independent Categorical Values', fontsize= 16)
   index = 1
   for col in cols:
      plt.subplot(2,3, index)
      # makes a donut graph for categorical values
      df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', colormap='Set3', title=col)
      circle = plt.Circle((0, 0), 0.75, fc='white')
      plt.gcf().gca().add_artist(circle)
      plt.ylabel(None)
      index += 1
   plt.tight_layout()
   plt.show()
def plot_numeric_values(cols, part):
   """Plot Numeric Values:
      Univariate plots for a list of numeric values.
      Runs twice because of the number of numeric values
   """
   print("\nClose the figure to continue...", end = " ")
   # These are the bin ranges for each bar graph
   ranges = []
   if part == 1:
      ranges = [range(0,10), range(15, 90, 5), range(0,260000,25000), range(0, 24, 2)]
   else: 
      ranges = [range(0,6), range(0, 75, 5), range(60, 300, 20)]
   # calculate columns for the plot window
   total_cols = round(len(cols)/2)
   fig = plt.figure(figsize = (10,8))
   fig.suptitle(f'Independent Numeric Values ({part})', fontsize= 16)
   index = 1
   for col in cols:
      plt.subplot(2,total_cols, index)
      df[col].plot(
      kind='hist', bins=ranges[index-1], colormap='plasma', edgecolor='white', linewidth=1, title=col)
      index += 1
   plt.show()
def bivariate_categorical(cols, dependent):
   """Bivariate categorical:
      Bivariate plots with independent categorical variables and the dependent variable
      They are plotted as count plots.
   """
   print("\nClose the figure to continue...", end = " ")
   current_col = 0
   current_row = 0
   fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,8))
   fig.suptitle(f"Bivariate Graphs with {dependent} and Categorical Variables", fontsize= 16)
   for col in cols:
      # Creates count subplots [In-Text Citation: (GeeksforGeeks, 2021)]
      sns.countplot(x=col, hue=dependent_var, data=df, ax=axes[current_row,current_col])
      axes[current_row,current_col].set_ylabel("Customer Count")
      if(current_col < 2):
         current_col += 1
      else:
         current_col = 0
         current_row += 1
   # Deletes extra graph space [In-Text Citation: (LeoC & DannyMoshe, 2015)]
   fig.delaxes(axes[1,2])
   plt.tight_layout()
   plt.show()
def bivariate_numeric(cols, dependent, part):
   """Bivariate numeric:
      Bivariate plots with independent numeric variables and the dependent variable.
   """
   print("\nClose the figure to continue...", end = " ")
   current_col = 0
   current_row = 0
   ranges = []
   if part == 1:
      ranges = [range(0,10), range(15, 90, 5), range(0,260000,25000), range(0, 24, 2)]
   else: 
      ranges = [range(0,6), range(0, 75, 5), range(60, 300, 20)]
   fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
   fig.suptitle(f"Bivariate Graphs with {dependent} and Numeric Variables ({part})", fontsize= 16)
   for i in range(len(cols)):
      df.groupby(dependent)[cols[i]].plot.hist(alpha=0.5, bins=ranges[i], ax=axes[current_row,current_col], edgecolor='white', linewidth=1)
      axes[current_row,current_col].legend(labels=sorted(df[dependent].unique()), title=dependent)
      axes[current_row,current_col].set_xlabel(cols[i])
       # The row index value is updated before the column
      if(current_col < 1):
         current_col += 1
      else:
         current_col = 0
         current_row += 1
   if part == 2:
      fig.delaxes(axes[1,1])
   plt.tight_layout()
   plt.show()
def regression_summary(cols, y, dropped):
   """Regression Summary:
      Generatees the logistic regression summary
   """
   #[In-Text Citation: (GeeksforGeeks, 2023)]
   print(f"\nDropped Columns: {dropped[1:]}")
   x = sm.add_constant(cols, prepend=False)
   log_reg = sm.Logit(y, x).fit()
   print(log_reg.summary())
   return log_reg
# endregion

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

# region Describing Regression Variables
print(text.UNDERLINE + "\nDescribing Variables for Analysis" + text.END)
independent_numeric =['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge']
independent_categorical = ['Gender', 'Techie', 'InternetService', 'Tablet', 'Phone']
dependent_var = 'TechSupport'

describe_vars("Numeric Variables", independent_numeric);
print("Categorical Variables")
view_categorical_values(independent_categorical);
print("Dependent Variable")
view_categorical_values([dependent_var]);
# graphs
print("\nClose the figure to continue...", end = " ")
df[dependent_var].value_counts().plot(kind='pie', autopct='%1.1f%%', colormap='Set3', title=f"{dependent_var} (Dependent Variable)")
circle = plt.Circle((0, 0), 0.75, fc='white')
plt.gcf().gca().add_artist(circle)
plt.ylabel(None)
plt.show()
plot_categorical_values(independent_categorical);
# Since there are so many numeric variables, they are split into two subplots
plot_numeric_values(independent_numeric[0:4], 1);
plot_numeric_values(independent_numeric[4:7], 2);
bivariate_categorical(independent_categorical, dependent_var)
bivariate_numeric(independent_numeric[0:4], dependent_var, 1)
bivariate_numeric(independent_numeric[4:7], dependent_var, 2)
# endregion

# region Data Transformation
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
print("\nValues saved to 'churn_regression.csv'")
# endregion

# region Logistic Regression
print(text.UNDERLINE + "\nLogistic Regression" + text.END)
dummy_dependent = "TechSupport_Yes"
kept_columns = ['MonthlyCharge', 'InternetService_Fiber Optic']
dropped_columns = [dummy_dependent, 'Income', 'Tenure', 'Children', 'Gender_Nonbinary','Yearly_equip_failure', 
                   'Phone_Yes', 'Techie_Yes', 'InternetService_None', 'Gender_Male', 'Tablet_Yes', 'Outage_sec_perweek', 'Age']
for i in range(1, len(dropped_columns)):
   regression_summary(df.drop(columns=dropped_columns[0:i]), df[dummy_dependent], dropped_columns[0:i])
final_model = regression_summary(df.drop(columns=dropped_columns[0:]), df[dummy_dependent], dropped_columns[0:]);
# confusion matrix
print(text.UNDERLINE + "\nConfusion Matrix" + text.END)
print("Close the figure to continue...", end = " ")
y_predict = final_model.predict(sm.add_constant(df[kept_columns]))
prediction = list(map(round,y_predict))
sns.heatmap(pd.crosstab(columns = df[dummy_dependent], colnames = ["Actual"], index = prediction, rownames = ["Predicted"]),
            annot=True, fmt='g', cbar=False, cmap='flare' )
plt.title("Confusion Matrix for TechSupport")
plt.show()
# accuracy score of the model 
print('\nLogistic Test accuracy: ', accuracy_score(df[dummy_dependent], prediction))
# odds ratio
# put e to the power of the coefficients for odds ratio [In-Text Citation: (Donbeo & lincolnfrias, 2016)]
odds_ratio=np.exp(final_model.params)
odds_ratio.drop(labels=['const'], inplace=True)
# create the change odd percents [In-Text Citation: (Bobbitt, 2021)]
change_odds= pd.Series(index=odds_ratio.index, dtype='object')
for i in range(len(odds_ratio)):
   result = (odds_ratio.iloc[i]-1)*100
   change_odds.iloc[i] = str(round(result, 2)) + ' %'
print("\nChange in Odds")
print(change_odds)
# endregion
print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)