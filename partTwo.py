"""
Program: Assignment 5 Part 2
Author: Nicholas Porter
Description: A K means clustering program to group patients found in a csv. Primarily focusing
on using the pyspark library to distribute the tasks at hand.
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.clustering import KMeans

spark = SparkSession.builder.appName('cluster').getOrCreate()

#Read in CSV
df = spark.read.csv("patient.csv",header=True,inferSchema=True)

#Create a vector assembler which will group all of the input columns of our training task and put them in a new column entirely (excluding Pid)
vec_assembler = VectorAssembler(inputCols = ["HeartDisease", "BMI", "Gender", "Race", "PhysicalActivity", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer"], outputCol='features')

final_df = vec_assembler.transform(df)

# Perform K means with the features column with the proposed 15 K's in the instructions, and a maxIterations of 100
kmeans=KMeans(featuresCol='features', k=15, maxIter=100, seed=42)
model=kmeans.fit(final_df)
predictions=model.transform(final_df)

# Since computeCost is depreceated for getting WSSSE, we are told to use the following Clustering Evaluator
# by the Spark documentation...
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette', distanceMeasure='squaredEuclidean')

wssse = evaluator.evaluate(predictions)
print(f"Within Set Sum of Squared Errors (WSSSE) = {wssse}")


# Find clusters
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# And show the predictions
predictions.select('Pid', 'prediction').show()
