#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark


# In[2]:


from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


# In[3]:


#read the whole dataset
flight_data = spark.read.option("inferSchema", "true").option("header", "true").csv("671009038_T_ONTIME_REPORTING.csv")
flight_data.count()


# In[4]:


flight_data.show(3)


# In[5]:


flight_data.createOrReplaceTempView("flight_data")


# In[23]:


#convert the NULL values with 0 number
flight_data = spark.sql("""
SELECT *, ifnull(DEP_DELAY, '0') as DEP_DELAY1, ifnull(ARR_DELAY, '0') as ARR_DELAY1
FROM flight_data 
""")
flight_data.show(3)


# In[24]:


#drop the previous columns referring to the delays
flight_data = flight_data.drop("DEP_DELAY", "ARR_DELAY")
flight_data.createOrReplaceTempView("flight_data")


# In[25]:


flight_data.show(3)


# In[26]:


#create the dataset of the airports (in the DEST column the airports were
#the same as in the origin column)
flights = spark.sql("""
SELECT ORIGIN AS airports, count(ORIGIN) as flight_numbers
FROM flight_data 
GROUP BY ORIGIN
HAVING count(ORIGIN) > (SELECT count(ORIGIN)*0.01 FROM flight_data)
""")
flights.show() #the number of flights for each airport


# In[27]:


print(flights.count()) #the size of the unique airports


# In[28]:


#create the dataset of the airways
airways = spark.sql("""
SELECT CARRIER AS airways, count(CARRIER) as flight_numbers
FROM flight_data 
GROUP BY CARRIER
HAVING count(CARRIER) > (SELECT count(CARRIER)*0.01 FROM flight_data)
""")
airways.show()


# In[29]:


print(airways.count()) #the size of the unique airways


# In[30]:


#make those datasets views in order to use them in the below queries
flights.createOrReplaceTempView("flights")
airways.createOrReplaceTempView("airways")


# In[31]:


#Report 1
#we want the avg for the departures delays for every airport (origin)
avg_airports = spark.sql("""
SELECT fd.ORIGIN as Airports, round(avg(fd.DEP_DELAY1), 3) as Departures_Delay
FROM flight_data as fd, flights as f
WHERE fd.ORIGIN = f.airports
GROUP BY fd.ORIGIN
ORDER BY avg(fd.DEP_DELAY1) DESC""")
avg_airports.show()


# In[32]:


avg_airports.toPandas().to_csv('task2-ap-avg.csv', header=False)


# In[33]:


#Report 2
#place an iterator
c_airports = spark.sql("""
SELECT ROW_NUMBER() OVER(PARTITION BY ORIGIN 
                            ORDER BY DEP_DELAY1 ASC) AS NUM_ROW,
        ORIGIN, DEP_DELAY1 AS DEP_DELAY
FROM flight_data, flights as f
WHERE ORIGIN = f.airports
""")
c_airports.show()


# In[34]:


#save the length of the airport's flights
count_airports = spark.sql("""
SELECT ORIGIN, count(ORIGIN) as COR
FROM flight_data, flights as f
WHERE ORIGIN = f.airports
GROUP BY ORIGIN
""")
count_airports.show()


# In[35]:


count_airports.createOrReplaceTempView("count_airports")


# In[36]:


#find the lines where the medians are
count_airports = spark.sql("""
SELECT ORIGIN, COR, CAST(IF(MOD(COR, 2)=0, COR/2, (COR+1)/2) AS int) AS P,
        CAST(IF(MOD(COR, 2)=0, COR/2+1, 0) AS int) AS P1
FROM count_airports
""")
count_airports.show()


# In[37]:


c_airports.createOrReplaceTempView("c_airports")
count_airports.createOrReplaceTempView("count_airports")


# In[38]:


#match to each P value the recording row of the delay
p_set = spark.sql("""
SELECT air.ORIGIN as AIRPORT, DEP_DELAY as DELAY1
FROM c_airports as air, count_airports as c
WHERE air.ORIGIN = c.ORIGIN AND NUM_ROW = P
""")
p_set.show()


# In[39]:


#match to each P1 value the recording row of the delay
p1_set = spark.sql("""
SELECT air.ORIGIN as AIRPORT, DEP_DELAY as DELAY2
FROM c_airports as air, count_airports as c
WHERE air.ORIGIN = c.ORIGIN AND NUM_ROW = P1
""")
p1_set.show()


# In[40]:


p_set.createOrReplaceTempView("p_set")
p1_set.createOrReplaceTempView("p1_set")


# In[41]:


#join the above tables to a final one
med_air = spark.sql("""
SELECT p_set.AIRPORT, DELAY1, DELAY2
FROM p_set
LEFT JOIN p1_set
ON p_set.AIRPORT = p1_set.AIRPORT
""")
med_air.show()


# In[42]:


med_air.createOrReplaceTempView("med_air")


# In[43]:


#find the median and sort it
med_air = spark.sql("""
SELECT AIRPORT, IF((DELAY2 IS NULL), DELAY1, (DELAY1 + DELAY2)/2) AS MEDIAN
FROM med_air
ORDER BY MEDIAN DESC
""")
med_air.show()


# In[44]:


med_air.toPandas().to_csv('task2-ap-med.csv', header=False)


# In[45]:


#Report 3
#we want the avg for the departures delays for every airway (carrier)
avg_airways = spark.sql("""
SELECT fd.CARRIER as Airways, round(avg(fd.DEP_DELAY1), 3) as Departures_Delay
FROM flight_data as fd, airways as a
WHERE fd.CARRIER = a.airways
GROUP BY fd.CARRIER
ORDER BY avg(fd.DEP_DELAY1) DESC""")
avg_airways.show()


# In[46]:


avg_airways.toPandas().to_csv('task2-aw-avg.csv', header=False)


# In[47]:


#Report 4
#The same steps as the "Report 2"
c_airways = spark.sql("""
SELECT ROW_NUMBER() OVER(PARTITION BY CARRIER 
                            ORDER BY DEP_DELAY1 ASC) AS NUM_ROW,
        CARRIER, DEP_DELAY1 AS DEP_DELAY
FROM flight_data, airways as a
WHERE CARRIER = a.airways 
""")
c_airways.show()


# In[49]:


count_airways = spark.sql("""
SELECT CARRIER, count(CARRIER) as CAR
FROM flight_data, airways as a
WHERE CARRIER = a.airways
GROUP BY CARRIER
""")
count_airways.show()


# In[50]:


count_airways.createOrReplaceTempView("count_airways")


# In[51]:


count_airways = spark.sql("""
SELECT CARRIER, CAR, CAST(IF(MOD(CAR, 2)=0, CAR/2, (CAR+1)/2) AS int) AS P,
        CAST(IF(MOD(CAR, 2)=0, CAR/2+1, 0) AS int) AS P1
FROM count_airways
""")
count_airways.show()


# In[52]:


c_airways.createOrReplaceTempView("c_airways")
count_airways.createOrReplaceTempView("count_airways")


# In[53]:


p_set = spark.sql("""
SELECT air.CARRIER as AIRWAY, DEP_DELAY as DELAY1
FROM c_airways as air, count_airways as c
WHERE air.CARRIER = c.CARRIER AND NUM_ROW = P
""")
p_set.show()


# In[54]:


p1_set = spark.sql("""
SELECT air.CARRIER as AIRWAY, DEP_DELAY as DELAY2
FROM c_airways as air, count_airways as c
WHERE air.CARRIER = c.CARRIER AND NUM_ROW = P1
""")
p1_set.show()


# In[55]:


p_set.createOrReplaceTempView("p_set")
p1_set.createOrReplaceTempView("p1_set")


# In[56]:


med_air = spark.sql("""
SELECT p_set.AIRWAY, DELAY1, DELAY2
FROM p_set
LEFT JOIN p1_set
ON p_set.AIRWAY = p1_set.AIRWAY
""")
med_air.show()


# In[57]:


med_air.createOrReplaceTempView("med_air")


# In[ ]:


med_air = spark.sql("""
SELECT AIRWAY, IF((DELAY2 IS NULL), DELAY1, (DELAY1 + DELAY2)/2) AS MEDIAN
FROM med_air
ORDER BY MEDIAN DESC
""")
med_air.show()


# In[ ]:


med_air.toPandas().to_csv('task2-aw-med.csv', header=False)


# In[ ]:


sc.stop()

