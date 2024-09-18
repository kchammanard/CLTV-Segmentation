# Databricks notebook source
# MAGIC %md
# MAGIC <h1>Merge Relevant Datasets
# MAGIC

# COMMAND ----------

import mlflow
import shap
import sys
import warnings
import pandas as pd

# COMMAND ----------

# MAGIC %md 
# MAGIC <h2>Modify "what_is" dataset

# COMMAND ----------

cltv_df = spark.table("hive_metastore.default.cltv")
cltv_df = cltv_df.toPandas()
display(cltv_df)

# COMMAND ----------

cltv_df = cltv_df.drop(["9 Year Revenue", "RevenueCluster", "RecencyCluster","FrequencyCluster","Segment"], axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Modify "users" Dataset

# COMMAND ----------

users_df = spark.table("hive_metastore.default.sd_254_users")
display(users_df)

# COMMAND ----------

users_df = users_df.toPandas().reset_index()
display(users_df)

# COMMAND ----------

users_df = users_df.drop(["Person","Address","Apartment","City","State","Zipcode","Latitude", "Longitude", "Birth Year", "Birth Month", "Gender"],axis = 1)
users_df.head()

# COMMAND ----------

dollar_columns = ["Per Capita Income - Zipcode", "Yearly Income - Person", "Total Debt"]
for column in dollar_columns:
    users_df[column] = users_df[column].str.replace('$', '').str.replace(',', '').astype(float)
users_df.head()

# COMMAND ----------

users_df.dtypes

# COMMAND ----------

cards_df = spark.table("hive_metastore.default.sd_254_cards")
display(cards_df)

# COMMAND ----------

cards_df = cards_df.toPandas().drop(columns=['Card Number', 'Expires', 'CVV', 'Acct Open Date', 'Year PIN last Changed'])

cards_df = cards_df.groupby('User').agg({
    'CARD INDEX': 'count',
    'Card Brand': lambda x: list(x.unique()),
    'Card Type': lambda x: list(x.unique()),
    'Has Chip': lambda x: sum(x == 'YES'),
    'Cards Issued': 'sum',
    'Credit Limit': lambda x: x.str.replace('$', '').astype(float).mean(),
    'Card on Dark Web': lambda x: sum(x == 'Yes')
}).reset_index()

cards_df.columns = [
    'User', 'Total Cards', 'Unique Card Brands', 'Unique Card Types',
    'Total Cards with Chips', 'Total Cards Issued', 'Average Credit Limit', 'Cards on Dark Web'
]

cards_df.head()

# COMMAND ----------

card_brands_encoded = pd.get_dummies(cards_df['Unique Card Brands'].apply(pd.Series).stack()).sum(level=0)
card_types_encoded = pd.get_dummies(cards_df['Unique Card Types'].apply(pd.Series).stack()).sum(level=0)

cards_df = pd.concat([cards_df, card_brands_encoded, card_types_encoded], axis=1)
cards_df = cards_df.drop(columns=['Unique Card Brands', 'Unique Card Types'])

cards_df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Merge Datasets

# COMMAND ----------

merged_df = pd.merge(cltv_df, users_df, left_on = "User", right_on = "index")
merged_df.head()

# COMMAND ----------

merged_df = pd.merge(merged_df, cards_df, left_on = "User", right_on = "User")
merged_df.head()

# COMMAND ----------

merged_df = merged_df.drop("index",axis = 1)

# COMMAND ----------

merged_df = merged_df.drop("Total Cards", axis = 1)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Statistical Relevancy Analyses using ANOVA and Chi-Square Tests

# COMMAND ----------

from scipy.stats import chi2_contingency, f_oneway

continuous_cols = ['Num Credit Cards', 'Total Cards with Chips', 'Total Cards Issued', 'Average Credit Limit',
                   'Revenue', 'Recency', 'Frequency', 'overall_score',"Current Age", "Retirement Age", "Per Capita Income - Zipcode",
                   "Yearly Income - Person", "Total Debt", "FICO Score"]
categorical_cols = ['Cards on Dark Web', 'Amex', 'Discover', 'Mastercard', 'Visa', 'Credit', 'Debit', 'Debit (Prepaid)']

anova_results = {}
chi2_results = {}

for col in continuous_cols:
    groups = [merged_df['LTV Cluster'] == i for i in merged_df['LTV Cluster'].unique()]
    f_stat, p_val = f_oneway(*(merged_df[col][group] for group in groups))
    anova_results[col] = p_val

for col in categorical_cols:
    contingency_table = pd.crosstab(merged_df[col], merged_df['LTV Cluster'])
    chi2, p_val, _, _ = chi2_contingency(contingency_table)
    chi2_results[col] = p_val

anova_results, chi2_results


# COMMAND ----------

summary_df = pd.DataFrame({
    'Variable': list(anova_results.keys()) + list(chi2_results.keys()),
    'Test Used': ['ANOVA'] * len(anova_results) + ['Chi-Square'] * len(chi2_results),
    'P-Value': list(anova_results.values()) + list(chi2_results.values())
})

summary_df['Significance'] = summary_df['P-Value'].apply(lambda x: 'Significant' if x < 0.05 else 'Not Significant')

summary_df.sort_values(by='P-Value', inplace=True)
summary_df.reset_index(drop=True, inplace=True)

summary_df


# COMMAND ----------

significant_vars = summary_df[summary_df['Significance'] == 'Significant']['Variable'].tolist()
card_related_columns = ['Amex', 'Discover', 'Mastercard', 'Credit', 'Debit (Prepaid)']

merged_df = merged_df[['User', 'LTV Cluster'] + significant_vars + card_related_columns]

display(merged_df)


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

corr_matrix = merged_df.corr()

plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.show()

