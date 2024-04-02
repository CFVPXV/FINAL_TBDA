"""
Program: Assignment 5 Part 1

Author: Nicholas Porter

Description: A K Pyspark program to calculate all of the three-mers in a DNA strand
"""
from pyspark.sql import SparkSession
import pyspark.sql as spark

spark = SparkSession.builder.appName('cluster').getOrCreate()

sc = spark.sparkContext

# Ingest data
text_file = sc.textFile("String_data.txt")

# First we will get all possible 3-mers with a list comprehension anonymous function
# essentially, acting as a sliding window across our data
out = text_file.flatMap(
    lambda x: [x[i:i+3] for i in range(len(x)-2)]
    # Then we will map each three-mer found with the anonymous function
    ).map(
        lambda x: (x, 1)
        # And finally, reduce and sort by the mapped key.
        ).reduceByKey(
            lambda x, y: x + y
            ).sortByKey(True, 1)

# Print out as both a DF and rdd
out_df = out.toDF()

out_rdd = out.collect()

out_df.show()

print(out_rdd)
