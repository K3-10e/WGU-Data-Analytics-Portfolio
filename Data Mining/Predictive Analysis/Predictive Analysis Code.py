import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, root_mean_squared_error

class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# region functions
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
# endregion

df = pd.read_csv('churn_clean.csv', keep_default_na = False, 
                 na_values = [' ', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'])
# aligns the df index to match the index of CaseOrder so df[0] doesn't exist
df.index = df.index + 1

print (text.BOLD + "\n-- Begin Data Analytics --\n" + text.END)

# region Prepare Data
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

# Data Transformations
print(text.UNDERLINE + "\nTranforming Data" + text.END)
# filter columns
numeric_columns = ['Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek', 'Yearly_equip_failure', 'Tenure', 'Bandwidth_GB_Year']
categorical_columns = ['Area', 'Gender','Techie', 'InternetService', 'Multiple', 'DeviceProtection', 'TechSupport']
all_columns = numeric_columns + categorical_columns
prepared_df = df.filter(all_columns)
print("Excess Columns Removed. Remaining Columns:")
print(prepared_df.columns.values)
prepared_df = pd.get_dummies(prepared_df, dtype=int)
# Keeps only one column for binary variables
prepared_df.drop(['Techie_No', 'Multiple_No', 'DeviceProtection_No', 'TechSupport_No'], inplace=True, axis=1)
print("\nWith Dummy Columns:")
print(prepared_df.columns.values)
prepared_df.to_csv('churn_prepared.csv', index=False)
print("\nTransformations complete. File saved as 'churn_prepared.csv'")
# endregion

# show distribution of y variable [In-Text Citation: (stats writer, 2023)]
print(text.UNDERLINE + "\nY Variable Distribution" + text.END)
counts = prepared_df['Yearly_equip_failure'].value_counts()
percent = prepared_df['Yearly_equip_failure'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
print(pd.concat([counts,percent], axis=1, keys=['count', 'percentage']))

# region Decision Tree
print(text.UNDERLINE + "\nDecision Tree Analysis" + text.END)
# split data
X = prepared_df.drop(['Yearly_equip_failure'], axis=1)
y = prepared_df[['Yearly_equip_failure']]
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size=0.2, random_state=1001)
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index= False)
y_train.to_csv('ytrain.csv', index= False)
y_test.to_csv('ytest.csv', index= False)

# Tunning Tree [In-Text Citation: (Navlani, 2024)]
dt = DecisionTreeClassifier(random_state=1001)
# creating testing parameters [In-Text Citation: (scikit-learn, n.d.)]
testing_params = {
   "criterion": ['gini', 'entropy'], # methodology for determining split
   "splitter": ['best', 'random'], # strategy for splitting
   "max_depth": [2, 3, 4, 5, 6, 7, 8, None], # how deep the tree is allowed to be
   "min_samples_split": [2, 3, 4, 5, 6, 7, 8], # minimum count of samples to make a node split
   "max_features": [None, 1, 2, 3, 4, 5, 'sqrt', 'log2'] # features to consider when picking best split
}
# Hyper parameter tunning [In-Text Citation: (Saini, 2020)]
# With GridSearch [In-Text Citation: (scikit-learn, n.d.-b)]
# Using KFold [In-Text Citation: (MAdness & EEtch, 2022)]
tunning = GridSearchCV(dt, param_grid=testing_params, n_jobs=-1, cv=KFold(5))
tunning.fit(X_train, y_train)
print(f"Best Decision Tree Parameters: {tunning.best_params_}")

"""While I can just use the 'tunning' variable for prediction, I created another tree with 
the best parameters so I don't have to run a gridsearch, which takes a while, every time
"""
better_dt = DecisionTreeClassifier(
   random_state=1001,
   criterion = 'entropy',
   max_depth = 5,
   max_features = 3, 
   min_samples_split = 7, 
   splitter = 'random'
   )
better_dt.fit(X_train, y_train)
y_pred = better_dt.predict(X_test)

# accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy : {accuracy}, ({round(accuracy * 100, 1)} %)")
# MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# RMSE
rmse = root_mean_squared_error(y_test, y_pred)
print(f"Root Mean Squared Error: {rmse}")
# R Squared
r_squared = r2_score(y_test, y_pred)
print(f"R Squared: {r_squared}")
# Predicted values
print(text.UNDERLINE + "\nPrediction Distributions" + text.END)
prediction_df = pd.DataFrame(y_pred, columns=['Yearly_equip_failure_predict'])
print(prediction_df['Yearly_equip_failure_predict'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')
# endregion

print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)