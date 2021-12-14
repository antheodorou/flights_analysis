#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
import pyspark.ml.evaluation
from pyspark.ml.feature import IndexToString, MinMaxScaler, StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator


# In[2]:


conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


# In[3]:


flight_data = spark.read.option("inferSchema", "true").option("header", "true").csv("671009038_T_ONTIME_REPORTING.csv")
flight_data.count()


# In[4]:


flight_data.show(3)


# In[5]:


flight_data.createOrReplaceTempView("flight_data")


# In[6]:


#convert the NULL values with 0 number
flight_data = spark.sql("""
SELECT *, ifnull(DEP_DELAY, '0') as DEP_DELAY1, ifnull(ARR_DELAY, '0') as ARR_DELAY1
FROM flight_data 
""")
flight_data.show(3)


# In[7]:


#drop the previous columns referring to the delays
flight_data = flight_data.drop("DEP_DELAY", "ARR_DELAY")
flight_data.createOrReplaceTempView("flight_data")


# In[8]:


flight_data.show(3)


# In[9]:


flight_data.count()*0.01


# In[10]:


#remove outliers
flights = spark.sql("""
SELECT ORIGIN AS airports, count(ORIGIN) as flight_numbers
FROM flight_data 
GROUP BY ORIGIN
HAVING count(ORIGIN) > (SELECT count(ORIGIN)*0.01 FROM flight_data)
""")
flights.show()


# In[11]:


#create the dataset of the airways without outliers
airways = spark.sql("""
SELECT CARRIER AS airways, count(CARRIER) as flight_numbers
FROM flight_data 
GROUP BY CARRIER
HAVING count(CARRIER) > (SELECT count(CARRIER)*0.01 FROM flight_data)
""")
airways.show()


# In[12]:


airways.createOrReplaceTempView("airways")
flights.createOrReplaceTempView("flights")


# In[13]:


#combine the above tables to one
data = spark.sql("""
SELECT ORIGIN AS airports, CARRIER AS airways, DEP_TIME AS dep_time, DEP_DELAY1 AS delay
FROM flight_data, flights, airways
WHERE CARRIER = airways AND ORIGIN = airports
""")
data.show()


# In[14]:


data.createOrReplaceTempView("data")


# In[15]:


#prepare feature vectors


# In[16]:


#place 0 when the time had 3 digits and select only the hour
data = spark.sql("""
SELECT airports,
airways, CAST(SUBSTRING(IF(0<(dep_time DIV 1000) AND (dep_time DIV 1000)<=9, dep_time, CONCAT(0,dep_time)),1,2) AS int) AS dep_time,
delay
FROM data
""")
data.show()
data.createOrReplaceTempView("data")


# In[17]:


#create the "df" for the one-hot encoding
df = data.na.fill(0)

df.select("dep_time").distinct().orderBy("dep_time").show(5)


# In[18]:


#one hot encoding of all the string columns
categorical_columns= ['airports', 'airways', 'dep_time']

# The index of string vlaues multiple columns
indexers = [
    StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
    for c in categorical_columns
]

# The encode of indexed vlaues multiple columns
encoders = [OneHotEncoder(dropLast=False,inputCol=indexer.getOutputCol(),
            outputCol="{0}_encoded".format(indexer.getOutputCol())) 
    for indexer in indexers
]

# Vectorizing encoded values
assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders],outputCol="features")

pipeline = Pipeline(stages=indexers + encoders+[assembler])
model=pipeline.fit(df)
newdf = model.transform(df)
newdf.show(5)


# In[19]:


newdf.printSchema()


# In[20]:


newdf.count()


# In[21]:


#training - test sets


# In[22]:


#we are going to need features and delay columns
lrdata = newdf.select('airports', 'airways', 'dep_time','features', newdf.delay.cast("float"))
lrdata.show(5)


# In[23]:


#split to training and test sets
training, test = lrdata.randomSplit(weights = [0.70, 0.30], seed = 1)
print("Size of training set: " + str(training.count()))
print("Size of test set: " + str(test.count()))


# In[24]:


#create the linear regression model
lr = LinearRegression(featuresCol = 'features', labelCol='delay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(training)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))


# In[25]:


#summary and goodness of fit
training.select('delay','dep_time').describe().show()


# In[26]:


trainingSummary = lr_model.summary
print("R Squared (R2) on training data = %f" % trainingSummary.r2)
print("Root Mean Squared Error (RMSE) on training data = %f" % trainingSummary.rootMeanSquaredError)


# In[27]:


#predictions with test set
lr_predictions = lr_model.transform(test)
lr_predictions.select("prediction","delay",'airports', 'airways', 'dep_time').distinct().show(10)


# In[28]:


#model evaluation
lr_evaluator = RegressionEvaluator(predictionCol="prediction",                  labelCol="delay",metricName="r2")
test_eval = lr_model.evaluate(test)
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
print("Root Mean Squared Error (RMSE) on test data = %g" % test_eval.rootMeanSquaredError)


# In[29]:


test.select('delay','dep_time').describe().show()


# In[30]:


sc.stop()

