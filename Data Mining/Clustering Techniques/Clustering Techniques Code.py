import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
 

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
def cluster_histogram(col, bin_count, bin_ranges, y_range):
   """Cluster histogram: 
      Creates histograms that use percentages to measure different characteristics
        of clusters. Uses numeric valuees
   """
   # all clusters are displayed in 1 plot for comparison
   print(f"\n{col}")
   plt.figure(figsize=(10, 7))
   plt.suptitle(f"{col} by Cluster")
   for i in range (0,4):
      values = filtered_df.loc[filtered_df['Cluster'] == i, col]
      plt.subplot(2, 2 ,i+1)
      plt.ylim(y_range)
      sns.histplot(data=filtered_df , x=values, 
                     bins=bin_count, binrange=bin_ranges, 
                     stat='percent', color=cluster_colors[i], alpha=plot_alpha).set_title(f"Cluster {i}")
      print(f"Cluster {i} Average: {values.mean().round(2)}")
   plt.tight_layout()
   plt.show()
def cluster_countplot(col, ordering, y_range):
   """Cluster Countplot:
      Like cluster histogram, but creates a countplot for the categorical variables
   """
   # all clusters are displayed in 1 plot for comparison, with maintained ordering of categories
   plt.figure(figsize=(10, 7))
   plt.suptitle(f"{col} by Cluster")
   for i in range (0,4):
      plt.subplot(2, 2 ,i+1)
      plt.ylim(y_range)
      sns.countplot(data=filtered_df , x=filtered_df.loc[filtered_df['Cluster'] == i, col],
                     order=ordering, stat='percent', color=cluster_colors[i],
                       alpha=plot_alpha).set_title(f"Cluster {i}")
   plt.tight_layout()
   plt.show()
# endregion

# specifying the n/a values will allow us to keep the 'None' value in InternetService
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

# filter data 
print(text.UNDERLINE + "\nFilter Data" + text.END)
cluster_cat = ['Marital', 'Gender', 'Techie']
cluster_num = ['Children', 'Age', 'Income', 'MonthlyCharge']
filtered_df = df.filter(cluster_cat + cluster_num)
print("Excess Columns Removed. Remaining Columns:")
print(filtered_df.columns.values)

# standardize numeric data [In-Text Citation: (GeeksforGeeks, 2021)]
scaler = StandardScaler()
# Convert array back into dataframe [In-Text Citation: (dmmmmd & FBruzzesi, 2020)]
standard_df = pd.DataFrame(scaler.fit_transform(df.filter(cluster_num)), columns = cluster_num)
# Recombine numeric data with categorical; make row count stay the same [In-Text Citation: (Rocketq & Lucky Suman, 2018)]
standard_df = pd.concat([standard_df.reset_index(drop=True), df.filter(cluster_cat).reset_index(drop=True)], axis=1)

# hot encode categorical data
encoded_df = pd.get_dummies(standard_df, dtype=int)
encoded_df.drop(['Techie_No' ], inplace=True, axis=1)
print("\nColumns Hot Encoded:")
print(encoded_df.columns.values)

# save to file
encoded_df.to_csv('churn_prepared.csv', index=False)
print("\nTransformations complete. File saved as 'churn_prepared.csv'")
# endregion

# region Hiearchical Cluster
print(text.UNDERLINE + "\nHierarchical Cluster Analysis" + text.END)
print("Plotting dendrogram...")
plt.figure(figsize=(14, 7))
plt.title("Hierarchical Cluster Dendrogram")
# create the cluster/dendrogram [In-Text Citation: (Sampaio, 2023)]
h_cluster = shc.linkage(encoded_df, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=h_cluster)
plt.axhline(y = 85, color = 'r', linestyle = '-')
plt.show()

# Labels the cluster that the data points are in [In-Text Citation: (scikit-learn, n.d.)]
clustering_model = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
clustering_model.fit(encoded_df)
filtered_df['Cluster'] = clustering_model.labels_.tolist()
print("\nCustomer Distribution in the Clusters:")
# Print the count/percent of each cluster [In-Text Citation: (tenebris silentio & Dabas, 2021)]
print(pd.concat([filtered_df["Cluster"].value_counts(), 
                 filtered_df["Cluster"].value_counts(normalize=True).mul(100).round(1).astype(str)+'%'], 
                 axis=1, keys=["Count", "Percentage"]))
# silhouette score [In-Text Citation: (Bhardwaj, 2020)]
print(f"\nSilhouette Score: {silhouette_score(encoded_df, filtered_df['Cluster'])}")
# endregion

# region Histograms
#  set transparency of histograms [In-Text Citation: (JohanC & Hager, 2021)]
plot_alpha = 0.7
cluster_colors = ['palevioletred', 'lightseagreen', 'lightslategray', 'burlywood']
cluster_histogram('Income', 20, (0, 110000), (0, 15))
cluster_histogram('Age', 20, (15, 90), (0, 10))
cluster_histogram('Children', 10, (0, 10), (0, 35))
cluster_histogram('MonthlyCharge', 25, (50, 300), (0, 16))

cluster_countplot('Marital', ['Divorced', 'Married', 'Never Married', 'Separated', 'Widowed'], (0, 30))
cluster_countplot('Gender', ['Female', 'Male', 'Nonbinary'], (0, 60))
cluster_countplot('Techie', ['Yes', 'No'], (0, 90))
# endregion
print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)