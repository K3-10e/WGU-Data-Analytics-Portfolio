import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 

class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# region Functions
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
def generate_PCA(component_count):
   """Generate PCA:
      Creates the PCA, the loadings matrix and variance dataframe.
   """
   # generates pca names for each component
   pca_names = []
   for i in range(component_count):
      pca_names.append(f"PC{str(i+1)}")
   
   # actual pca
   pca = PCA(n_components=component_count)
   pca.fit_transform(standard_df)

   # loadings matrix (weight of each variable on component) [In-Text Citation: (Centellegher, 2020)]
   loadings_matrix = pd.DataFrame(pca.components_.T, columns = pca_names, index=standard_df.columns)
   
   # variance df
   eigenvalues = pca.explained_variance_
   variance_ratio = pca.explained_variance_ratio_ * 100
   variance_df = pd.DataFrame({
      "Eigenvalues": eigenvalues,
      "Variance_Percentage" : variance_ratio,
      "Cumulative_Percentage" : variance_ratio.cumsum(),
   })
   # makes the data frame start at 1 so it aligns with the pca it represents
   variance_df.index += 1

   return pca, loadings_matrix, variance_df
# endregion

# specifying the n/a values will allow us to keep the 'None' value in InternetService
df = pd.read_csv('churn_clean.csv', keep_default_na = False, 
                 na_values = [' ', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'])
# aligns the df index to match the index of CaseOrder so df[0] doesn't exist
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

# region Prepare Data
print(text.UNDERLINE + "\nFilter & Format Data" + text.END)
# standardize numeric data [In-Text Citation: (GeeksforGeeks, 2021)]
cluster_num = ['Lat', 'Lng', 'Population', 'Children', 'Age', 'Income', 'Outage_sec_perweek',
            'Email', 'Contacts', 'Yearly_equip_failure', 'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year']
scaler = StandardScaler().set_output(transform='pandas')
standard_df = scaler.fit_transform(df.filter(cluster_num))
print("Standardized Data:")
print(standard_df.head(3))

# save to file
standard_df.to_csv('churn_prepared.csv', index=False)
print("\nTransformations complete. File saved as 'churn_prepared.csv'")
# endregion

# region PCA
print(text.UNDERLINE + "\nPrincipal Component Analysis" + text.END)
# gets the count of pca
pca, loading_df, variance_df = generate_PCA(standard_df.shape[1])

# removes scientific notation [In-Text Citation: (Saturn Cloud, 2023)]
pd.options.display.float_format = '{:.3f}'.format
print("Loading Matrix:")
print(loading_df)
print("\nPCA Variance Data Frame:")
print(variance_df)

# start graph at 1 so it aligns with the pca it represents
graph_range = range(1, len(variance_df) + 1)

# eigen plot
# plt.plot(graph_range, variance_df["Eigenvalues"], marker = "o")
# plt.xticks(graph_range)
# plt.xlabel('Components')
# plt.ylabel('Eigenvalue')
# plt.title("Scree Plot (Kaiser Criterion)")
# plt.axhline(y=1, color='red')
# plt.show()

# variance elbow plot
plt.plot(graph_range, variance_df['Variance_Percentage'], marker = "o")
plt.xticks(graph_range)
plt.xlabel('Component')
plt.ylabel('Explained Variance by Component')
plt.title("Explained Variance Scree Plot")
plt.show()

# endregion
print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)