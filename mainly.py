from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql as spark

spark = SparkSession.builder.appName('cluster').getOrCreate()

sc = spark.sparkContext

text_file = sc.textFile("String_data.txt")

out = text_file.flatMap(
    lambda x: [x[i:i+3] for i in range(len(x)-2)]
    ).map(
        lambda x: (x, 1)
        ).reduceByKey(
            lambda x, y: x + y
            ).sortByKey(True, 1)

out_df = out.toDF()

out_rdd = out.collect()

out_df.show()

print(out_rdd)
