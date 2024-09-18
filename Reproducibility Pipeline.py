# Databricks notebook source
# MAGIC %md
# MAGIC <h1>CLTV Segmentation Evaluation using Databricks AutoML and SHAP Interpretation</h1>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Import Dataset & Models

# COMMAND ----------

import mlflow
import shap
import sys
import warnings
import pandas as pd

# COMMAND ----------

cltv_df = spark.table("hive_metastore.default.cltv")
cltv_df = cltv_df.toPandas()
display(cltv_df)

# COMMAND ----------

users_df = spark.table("hive_metastore.default.sd_254_users")
display(users_df)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Re-Train Models - Demonstration on Debt

# COMMAND ----------

users_df = users_df.toPandas().reset_index()
display(users_df)

# COMMAND ----------

users_df[["Yearly Income - Person","Total Debt"]] = users_df[["Yearly Income - Person","Total Debt"]].replace('[\$,]', '', regex=True).astype(int)
users_df.head()

# COMMAND ----------

users_df[["Yearly Income - Person","Total Debt"]] = users_df[["Yearly Income - Person","Total Debt"]].replace('[\$,]', '', regex=True).astype(int)
users_df.head()

# COMMAND ----------

cltv_df = cltv_df.merge(users_df[["index","Yearly Income - Person", "Total Debt"]], left_on = "User", right_on = "index")
display(cltv_df)

# COMMAND ----------

