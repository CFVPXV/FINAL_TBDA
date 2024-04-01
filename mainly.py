from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql as spark

sc = SparkContext()

text_file = sc.textFile("String_data.txt")

"""
with open("String_data.txt", "r") as f:
    inter = f.read()
    stringly = ''.join(inter.splitlines())

print(stringly)

dat = [stringly]

rdd = spark.sparkContext.parallelize(dat)

length = len(stringly)

print(length)
"""

out = text_file.flatMap(lambda x: [x[i:i+3] for i in range(len(x)-2)]).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).collect()

print(out)