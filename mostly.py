from math import sqrt
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler

spark = SparkSession.builder.appName("Read CSV").getOrCreate()

df = spark.read.csv("patient.csv", header=True, inferSchema=True)

df = df.drop("Pid")
"""
clusters = KMeans.train(df, 15, maxIterations=100, initializationMode="random")
"""

df.show()

assembler = VectorAssembler(inputCols=["HeartDisease", "BMI", "Gender", "Race", "PhysicalActivity", "SleepTime", "Asthma", "KidneyDisease", "SkinCancer"], outputCol="features")

assembled_data=assembler.transform(df)

assembled_data.show()

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

scaler_model = scaler.fit(assembled_data)
df = scaler_model.transform(assembled_data)

df.show()

clusters = KMeans(featuresCol="scaled_features", k=15).train(df, 2, maxIterations=10, runs=10, initializationMode="random")

def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))
"""
evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='scaled_features', metricName='silhouette', distanceMeasure='squaredEuclidean')

KMeans_mod = KMeans(featuresCol='scaled_features', k=15)  
KMeans_fit = KMeans_mod.fit(df)  
output = KMeans_fit.transform(df)   
score = evaluator.evaluate(output)   
print("Silhouette Score:",score)

cost = KMeans_fit.computeCost(df)
print("Within Set Sum of Squared Errors = " + str(cost))

"""
kmeans = KMeans().setK(15).setSeed(1)
model = kmeans.fit(df)

predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)
