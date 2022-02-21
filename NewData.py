# Databricks notebook source
# File location and type
file_location = "/FileStore/tables/diabetes.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "false"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

temp_table_name = "diabetes_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `diabetes_csv`

# COMMAND ----------

permanent_table_name = "diabetes_csv"

# COMMAND ----------

pip install pandas

# COMMAND ----------

pip install numpy

# COMMAND ----------

pip install flask

# COMMAND ----------

pip install mlflow

# COMMAND ----------

import pickle

# COMMAND ----------

from flask import Flask,render_template,request,jsonify

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
from numpy import savetxt
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# COMMAND ----------

df=spark.read.csv('/FileStore/tables/diabetes.csv')

# COMMAND ----------

df = pd.read_csv("/dbfs/FileStore/tables/diabetes.csv")

# COMMAND ----------

df.head(5)

# COMMAND ----------

df.columns

# COMMAND ----------

X = df.drop(columns='Outcome')
y=df[['Outcome']]

# COMMAND ----------

print(y.columns)
X.columns

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=0)

# COMMAND ----------

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()
 
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 190
  max_depth = 6
  max_features =7
  
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  # Use the model to make predictions on the test dataset.
  predictions = rf.predict(X_test)

# COMMAND ----------


