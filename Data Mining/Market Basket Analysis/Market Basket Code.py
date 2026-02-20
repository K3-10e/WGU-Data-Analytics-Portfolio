import pandas as pd
from apyori import apriori
 

class text:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

# specifying the n/a values will allow us to remove empty rows
df = pd.read_csv('teleco_market_basket.csv', keep_default_na = False, 
                 na_values = [' ', '', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                             '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'])

print (text.BOLD + "\n-- Begin Data Analytics --\n" + text.END)
# region Clean Data
print(text.UNDERLINE + "Clean & Preparing Data" + text.END)
# Missing values [In-Text Citation: (NumFOCUS, Inc., n.d.)]
print(f"Purchase Shape Before: {df.shape}")
df.dropna(how='all', inplace=True, ignore_index=True)
print(f"Purchase Shape After: {df.shape}")
print(df.head(3))

# turn purchases into list of list, the inner list will be each transaction
transactions_list = []
column_count = df.count(1)
for i in range (0, df.shape[0]):
   single_transaction = []
   for j in range (0, column_count[i]):
    single_transaction.append(df.values[i,j])
   transactions_list.append(single_transaction)

transactions_df = pd.DataFrame({'Transactions': transactions_list})
print("\nData in one column:")
print(transactions_df)

# save to file
transactions_df.to_csv('transactions_prepared.csv', index=False)
print("\nTransformations complete. File saved as 'transactions_prepared.csv'")
# endregion

# region Apriori Algorithm [In-Text Citation: (Amruta, 2024)]
print(text.UNDERLINE + "\nApriori Algorithm" + text.END)
association_rules = apriori(transactions_df['Transactions'], min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

# organize results into a dataframe
items = []
antecedents = []
consequents = []
support = []
confidence = []
lift = []
for i in range(0, len(association_results)):
   #  appends the products as an array
    items.append([item for item  in  association_results[i].items])
    antecedents.append([item for item  in  association_results[i].ordered_statistics[0].items_base])
    consequents.append([item for item  in  association_results[i].ordered_statistics[0].items_add])
    support.append(association_results[i].support)
    confidence.append(association_results[i][2][0].confidence)
    lift.append(association_results[i][2][0].lift)

df_results = pd.DataFrame({
      "Items": items,
      "Antecedents": antecedents,
      "Consequents": consequents,
      "Support": support,
      "Confidence" : confidence,
      "Lift" : lift,
   })

df_results.sort_values(by='Lift', ascending=False, inplace=True,)

print(df_results.head(3))
df_results.to_csv('apriori.csv', index=False)
# endregion


print (text.BOLD + "\n-- End Data Analytics --\n" + text.END)