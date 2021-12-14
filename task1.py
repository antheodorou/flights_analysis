#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pyspark


# In[3]:


from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


# In[4]:


#read the whole dataset
flight_data = spark.read.option("inferSchema", "true").option("header", "true").csv("671009038_T_ONTIME_REPORTING.csv")
flight_data.count()


# In[5]:


flight_data.show(3)


# In[6]:


flight_data.createOrReplaceTempView("flight_data")


# In[7]:


delay = spark.sql("""
SELECT ifnull(DEP_DELAY, '0') as DEP_DELAY, ifnull(ARR_DELAY, '0') as ARR_DELAY
FROM flight_data 
""")
delay.show()


# In[8]:


delay.createOrReplaceTempView("delay")


# In[9]:


delay = spark.sql("""
SELECT avg(DEP_DELAY) as departures_delay, avg(ARR_DELAY) as arrivals_delay
FROM delay 
""")
delay.show()


# In[10]:


sc.stop()

