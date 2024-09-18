# Databricks notebook source
# MAGIC %md
# MAGIC <h1>Project 2 : Customer Lifetime Value (CLV)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Configure Environment & Mount Files

# COMMAND ----------

container_name = "case1"
account_name = "caseone"
mount_name = "clv_analysis"
mount_point = f"/mnt/{mount_name}"

# COMMAND ----------

# dbutils.fs.ls(mount_point)

# COMMAND ----------

cards = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(mount_point + "/sd254_cards.csv")
users = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(mount_point + "/sd254_users.csv")
what_is = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(mount_point + "/what_is.csv")

# COMMAND ----------

display(what_is)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Merge Datasets

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
users = users.withColumn("User", monotonically_increasing_id())
display(users)

# COMMAND ----------

df = what_is.join(users, "User")

# COMMAND ----------

# MAGIC %md
# MAGIC <h1>Experiment 1: Customer Segmentation using RFM</h1>
# MAGIC <br>
# MAGIC <a href="https://medium.com/analytics-vidhya/integrated-approach-of-rfm-clustering-cltv-machine-learning-algorithm-15f9cb3653b0">Reference</a>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Create Utility Class

# COMMAND ----------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
class Exp1Utility:
    @staticmethod
    def elbow(df, col):
        sse = {}
        frequency = df[[col]]

        for k in range(1,10):
            kmeans = KMeans(n_clusters = k, max_iter = 1000).fit(frequency)
            frequency["clusters"]= kmeans.labels_ 
            sse[k] = kmeans.inertia_

        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.show()

    @staticmethod
    def order_cluster(cluster_field_name, target_field_name, df, ascending):
        new_cluster_field_name = 'new_' + cluster_field_name
        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
        df_new['index'] = df_new.index
        df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
        df_final = df_final.drop([cluster_field_name], axis=1)
        df_final = df_final.rename(columns={"index": cluster_field_name})
        return df_final
    
    @staticmethod
    def segment(score):
        if score <= 3:
            return "Low-Value"
        elif score > 3 and score <= 6:
            return "Mid-Value"
        elif score > 6:
            return "High-Value"
    

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Exploratory Data Analysis

# COMMAND ----------

what_is_pd = what_is.toPandas()
what_is_sample = what_is_pd.sample(100000)
display(what_is_sample)
#transactions per year, month, day (ARIMA)
#cholopleth map for transaction density

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Data Cleansing

# COMMAND ----------

what_is_pd["Amount"] = what_is_pd["Amount"].str.replace(r'[^-+\d.]', '').astype(float)

# COMMAND ----------

display(what_is_pd)

# COMMAND ----------

# MAGIC %md 
# MAGIC <h3>Clustering by Recency

# COMMAND ----------

what_is_pd[["Year", "Month", "Day"]] = what_is_pd[["Year", "Month", "Day"]].astype(int)
what_is_pd["Date"] = pd.to_datetime(what_is_pd[["Year", "Month", "Day"]])

user_rfm = what_is_pd.groupby("User").Date.max().reset_index()
user_rfm["Recency"] = (user_rfm["Date"].max() - user_rfm["Date"]).dt.days
user_rfm = user_rfm.drop("Date", axis=1)
display(user_rfm)


# COMMAND ----------

what_is_pd.groupby("User").Date.max().reset_index()

# COMMAND ----------


sse = {}
recency = user_rfm[['Recency']]

for k in range(1,10):
    kmeans = KMeans(n_clusters = k, max_iter = 1000).fit(recency)
    recency["clusters"]= kmeans.labels_ 
    sse[k] = kmeans.inertia_

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()

# COMMAND ----------

kmeans = KMeans(n_clusters=6)
user_rfm['RecencyCluster'] = kmeans.fit_predict(user_rfm[['Recency']])
display(user_rfm)

# COMMAND ----------

user_rfm.groupby('RecencyCluster')['Recency'].describe()

# COMMAND ----------

# MAGIC %md 
# MAGIC <h3>Order Clusters with Utility Function

# COMMAND ----------

user_rfm = Exp1Utility.order_cluster('RecencyCluster', 'Recency', user_rfm,False)
user_rfm.groupby('RecencyCluster')['Recency'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Clustering by Frequency

# COMMAND ----------

by_freq = what_is_pd.groupby('User').Year.count().reset_index().rename(columns = {'Year': 'Frequency'})
display(by_freq)


# COMMAND ----------


# sse = {}
# frequency = by_freq[['Frequency']]

# for k in range(1,10):
#     kmeans = KMeans(n_clusters = k, max_iter = 1000).fit(frequency)
#     frequency["clusters"]= kmeans.labels_ 
#     sse[k] = kmeans.inertia_

# plt.figure()
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel("Number of cluster")
# plt.show()

# COMMAND ----------

kmeans = KMeans(n_clusters = 6)
by_freq['FrequencyCluster'] = kmeans.fit_predict(by_freq[['Frequency']])

by_freq = Exp1Utility().order_cluster('FrequencyCluster', 'Frequency', by_freq,True)
display(by_freq)
display(by_freq.groupby("FrequencyCluster")['Frequency'].describe())


# COMMAND ----------

user_rfm = pd.merge(user_rfm, by_freq)

# COMMAND ----------

display(user_rfm)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Clustering by Revenue

# COMMAND ----------

by_revenue = what_is_pd.groupby('User').Amount.sum().reset_index().rename(columns = {'Amount': 'Revenue'})
display(by_revenue)


# COMMAND ----------

Exp1Utility().elbow(by_revenue, "Revenue")

# COMMAND ----------

kmeans = KMeans(n_clusters = 6)
by_revenue['RevenueCluster'] = kmeans.fit_predict(by_revenue[['Revenue']])

by_revenue = Exp1Utility().order_cluster('RevenueCluster', 'Revenue', by_revenue,True)
display(by_revenue)
display(by_revenue.groupby("RevenueCluster")['Revenue'].describe())


# COMMAND ----------

user_rfm = pd.merge(by_revenue, user_rfm, on = "User")

# COMMAND ----------

display(user_rfm)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Calculate Overall Score

# COMMAND ----------

user_rfm["overall_score"] = user_rfm["RecencyCluster"] + user_rfm["FrequencyCluster"] + user_rfm["RevenueCluster"]
display(user_rfm)


# COMMAND ----------



# COMMAND ----------

user_rfm.groupby('overall_score')[['Recency','Frequency','Revenue']].mean()

# COMMAND ----------

user_rfm["Segment"] = user_rfm["overall_score"].apply(Exp1Utility.segment)


# COMMAND ----------

display(user_rfm)

# COMMAND ----------

user_rfm.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>LTV Prediction</h2>
# MAGIC <p>We will predict a customer's lifetime value over the span of 6 months, using segmented insights to produce more actionable results