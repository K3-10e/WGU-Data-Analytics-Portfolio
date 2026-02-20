import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler

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

# Data Tranformations
print(text.UNDERLINE + "\nTranforming Data" + text.END)
numeric_columns = ['Age', 'Income', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge']
categorical_columns = ['Gender', 'Techie', 'Contract', 'Port_modem', 'InternetService', 'OnlineSecurity', 
                       'DeviceProtection', 'TechSupport']
all_columns = numeric_columns + categorical_columns
prepared_df = df.filter(all_columns)
# Normalizes only numeric columns [In-Text Citation: (Jeremy & Yaron, 2015)]
prepared_df[numeric_columns] = StandardScaler().fit_transform(prepared_df[numeric_columns])
prepared_df = pd.get_dummies(prepared_df, dtype=int)
# Keeps only one column for the dependent variable, otherwise issues are posed with KNN and ROC
prepared_df.drop(['OnlineSecurity_No'], inplace=True, axis=1)
prepared_df.to_csv('churn_prepared.csv', index=False)
print("Transformations complete. File saved as 'churn_prepared.csv'")
# # endregion

# region KNN Analysis
print(text.UNDERLINE + "\nKNN Analysis" + text.END)
# split dataset [In-Text Citation: (GeeksforGeeks, 2023)]
X = prepared_df.drop(['OnlineSecurity_Yes'], axis=1)
y = prepared_df[['OnlineSecurity_Yes']]
X_train, X_test, y_train, y_test = train_test_split( 
             X, y, test_size=0.2, random_state=1001)
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index= False)
y_train.to_csv('ytrain.csv', index= False)
y_test.to_csv('ytest.csv', index= False)

# transforms the y test and train values to an array for knn and roc calculations
y_test_val = y_test['OnlineSecurity_Yes'].values
y_train_val = y_train['OnlineSecurity_Yes'].values

# find best k-value [In-Text Citation: (Korstanje, n.d.)]
parameters = {"n_neighbors": range(1, 50)}
k_score = GridSearchCV(KNeighborsClassifier(), parameters)
k_score.fit(X_train, y_train_val)
print(f"{k_score.best_params_}\n")

# perform knn [In-Text Citation: (GeeksforGeeks, 2023)]
knn = KNeighborsClassifier(n_neighbors= 48)
knn.fit(X_train, y_train_val)

# confusion matrix
print("Confusion Matrix")
y_preds = knn.predict(X_test)
y_confusion = pd.crosstab(columns = y_test_val, colnames = ["Actual"], 
                          index = y_preds, rownames = ["Predicted"], margins=True, margins_name="Total")
print(y_confusion)

# classification report [In-Text Citation: (Bobbitt, 2022)]
print("\nClassification Report")
print(classification_report(y_test, y_preds))

# test/training accuracy
training = knn.score(X_train, y_train)
testing = knn.score(X_test, y_test)
print(f"Training Accuracy: {training}, ({round(training*100, 1)} %)")
print(f"Test Accuracy: {testing}, ({round(testing*100, 1)} %)")

# AUC score [In-Text Citation: (GeeksforGeeks, 2024)]
auc_score = roc_auc_score(y_test, y_preds)
print(f"Area Under Curve Score: {auc_score}, ({round(auc_score*100, 1)} %)")

# ROC curve/graph [In-Text Citation: (GeeksforGeeks, 2024a)]
y_pred_proba = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_val, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()  
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.title('ROC Curve for KNN')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
# endregion

print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)