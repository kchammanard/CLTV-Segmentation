# Databricks notebook source
# MAGIC %md
# MAGIC <h1>Customer Lifetime Value Analysis - Based on segments</h1>

# COMMAND ----------

# MAGIC %md
# MAGIC <h2>Configure Environment

# COMMAND ----------

container_name = "case1"
account_name = "caseone"
mount_name = "clv_analysis"
mount_point = f"/mnt/{mount_name}"

# COMMAND ----------

dbutils.fs.ls(mount_point)

# COMMAND ----------

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyspark.sql import SparkSession

class Exp1Utility:
    @staticmethod
    def sparktopandas(df):
        if not isinstance(df, pd.DataFrame):
            return df.toPandas()
        
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
    def kmeans_clustering(df, column_name, cluster_name, num_clusters=3):
        if isinstance(df,pd.DataFrame):
            data = df[[column_name]].copy()

            scaler = StandardScaler()
            data[column_name] = scaler.fit_transform(data[[column_name]])

            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df[cluster_name] = kmeans.fit_predict(data[[column_name]])

        else:
            df = df.toPandas()
            return kmeans_clustering(df,column_name,num_clusters=3)

        return df
    
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
    def scale_features(df, columns):
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df
        
    @staticmethod
    def one_hot_encode(df, columns):
        return pd.get_dummies(df, columns = columns)
    
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
# MAGIC <h2>Build Pipeline</h2>
# MAGIC <p>Based on the visualization above, we will select the timeframe with most active users, which is 2010-2019

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import *

class Pipeline:
    def __init__(self,mount_point, path1,path2, start_year, end_year):
        self.mount_point = mount_point
        self.path1 = path1
        self.path2 = path2
        self.start_year = start_year
        self.end_year = end_year
        self.what_is = None
        self.segmented = None
    
    def load_data(self):
        self.what_is = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(self.mount_point + self.path1)
        self.segmented = spark.read.option("header","true").option("inferSchema","true").option("delimiter",",").csv(self.mount_point + self.path2)
    
    def data_cleansing(self):
        self.what_is = self.what_is.withColumn('Time', date_format('Time', 'HH:mm:ss'))
        self.what_is = self.what_is.withColumn("Month",format_string("%02d","Month"))
        self.what_is = self.what_is.withColumn("Day",format_string("%02d","Day"))
        self.what_is = self.what_is.withColumn('DateTime', concat(col('Year'), col('Month'), col('Day'), col('Time')))
        self.what_is = self.what_is.withColumn('DateTime', to_timestamp('DateTime', 'yyyyMMddHH:mm:ss'))

        self.what_is = self.what_is.withColumn('Amount', substring(col('Amount'), 2, 10).cast("float"))

        display(self.what_is)
        
    def validation(self):
        self.valid_what_is = self.what_is.filter(col("Is Fraud?") == "No").filter(col("Errors?").isNull())
        self.fraud_what_is = self.what_is.filter(col("Is Fraud?") == "Yes")
        self.error_what_is = self.what_is.filter(col("Errors?").isNotNull())
        self.what_is = self.valid_what_is
    
    def select_timeframe(self):
        self.what_is = self.what_is.filter((year(col('DateTime')) >= self.start_year) & (year(col('DateTime')) <= self.end_year))
        self.what_is = self.what_is.select('User', 'Amount').groupBy('User').sum().select('User', round('sum(Amount)', 2).alias('9 Year Revenue'))

    def merge_datasets(self):
        self.what_is = self.what_is.join(self.segmented, on = "User", how = "inner")  

    def run_pipeline(self):
        if self.what_is is None:
            self.load_data()
            self.data_cleansing()
            self.validation()
            self.select_timeframe()
            self.merge_datasets()
        else:
            print("Not None")
        return self.what_is

SEGMENTED_PATH = "/segmented_clv.csv"
WHAT_IS_PATH = "/what_is.csv"

pipeline = Pipeline(mount_point,WHAT_IS_PATH,SEGMENTED_PATH,2010,2019)

if pipeline.what_is is None:
    print("None")
    what_is = pipeline.run_pipeline()
else:
    print("Exists")
    what_is = pipeline.what_is

display(what_is)


# COMMAND ----------

# MAGIC %md 
# MAGIC <h2>Cluster Revenue into Segments

# COMMAND ----------

what_is = what_is.toPandas()
print(type(what_is))
display(what_is)

# COMMAND ----------



# COMMAND ----------


def preprocessing(df):
    df['Segment'] = df['Segment'].astype(str)
    df = Exp1Utility.kmeans_clustering(df, "9 Year Revenue", "LTV Cluster", num_clusters=3)
    df = Exp1Utility.order_cluster("LTV Cluster", "9 Year Revenue", df,True)
    display(what_is)
    df = Exp1Utility.scale_features(df, ["9 Year Revenue", "Revenue","Recency","Frequency"])
    df = Exp1Utility.one_hot_encode(df,["Segment"])
    return df
what_is = preprocessing(what_is)

# COMMAND ----------

display(what_is)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt
corr_matrix = what_is.corr()

print(corr_matrix['LTV Cluster'].sort_values(ascending=False))
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC <h2> Train Test Split
# MAGIC

# COMMAND ----------

from sklearn.model_selection import train_test_split

X = what_is.drop(['LTV Cluster','9 Year Revenue'],axis=1)
y = what_is['LTV Cluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>XGBoost

# COMMAND ----------

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from xgboost import XGBClassifier

# Define the StratifiedKFold object
skf = StratifiedKFold(n_splits=5)  # 5-fold stratified cross validation

accuracies = []
f1_scores = []
precision_scores = []
recall_scores = []

# Iterate over each split
for train_index, val_index in skf.split(X, y):
    # Select the data for this split
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Train the classifier
    space = {
    "colsample_bytree": 0.6510163553874492,
    "learning_rate": 1.3405182654493213,
    "max_depth": 6,
    "min_child_weight": 18,
    "n_estimators": 56,
    "n_jobs": 100,
    "subsample": 0.6114907939145026,
    "verbosity": 0,
    "random_state": 280790652,
    }
    clf = XGBClassifier(space)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_val)

    # Calculate and store the metrics for this split
    accuracies.append(accuracy_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred, average='weighted'))
    precision_scores.append(precision_score(y_val, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_val, y_pred, average='weighted'))

# Compute the mean of each metric over all splits
mean_accuracy = np.mean(accuracies)
mean_f1_score = np.mean(f1_scores)
mean_precision_score = np.mean(precision_scores)
mean_recall_score = np.mean(recall_scores)

print("Mean cross-validation accuracy: ", mean_accuracy)
print("Mean cross-validation F1 score: ", mean_f1_score)
print("Mean cross-validation precision score: ", mean_precision_score)
print("Mean cross-validation recall score: ", mean_recall_score)


# COMMAND ----------

import xgboost as xgb

ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,n_jobs=-1).fit(X_train, y_train)

print('Accuracy of XGB classifier on training set: {:.2f}'
       .format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'
       .format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))

y_pred = ltv_xgb_model.predict(X_test)

# COMMAND ----------

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Logistic Regression

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

log_reg = LogisticRegression(random_state=42)

log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

metrics = {
    'Logistic Regression': {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
}

metrics['Logistic Regression']


# COMMAND ----------

print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Decision Tree Classifier

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=42)

tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

metrics['Decision Tree'] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

metrics['Decision Tree']


# COMMAND ----------

print(classification_report(y_test,y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC <h3>Using H2o AutoML

# COMMAND ----------

!pip install h2o

# COMMAND ----------

import h2o
from h2o.automl import H2OAutoML

h2o.init()
# Import the data into H2O
df_h2o = h2o.H2OFrame(what_is)

# Define the features and the target
features = df_h2o.columns
target = 'LTV Cluster'
features.remove(target)

# Split the data into a training set and a validation set
train, valid = df_h2o.split_frame(ratios=[0.8], seed=42)


# COMMAND ----------

# Run H2O AutoML
aml = H2OAutoML(max_models=10, seed=42)
aml.train(x=features, y=target, training_frame=train, validation_frame=valid)

# COMMAND ----------

# View the AutoML leaderboard
lb = aml.leaderboard
print(lb)


# COMMAND ----------

# Use the "leader" model to make predictions
preds = aml.leader.predict(valid)
preds
