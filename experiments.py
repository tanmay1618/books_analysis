from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Tokenizer, HashingTF
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Book Impact Prediction") \
    .getOrCreate()

# Load data
df = spark.read.option("header", "true").option("delimiter", ",").option("quote", "\"").option("escape", "\"").csv("books_features.csv")

# Split data into train and test sets
train, test = df.randomSplit([0.8, 0.2], seed=42)

#-------------------Experiement 1------------------------------------------
exp_name = "1"
# Make predictions
# Train linear regression model
lr = LinearRegression(featuresCol="features", labelCol="Impact")
model = lr.fit(train)

# Make predictions
predictions = model.transform(test)

# Evaluate model
evaluator = RegressionEvaluator(labelCol="Impact", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
print("Experiment ",exp_name)
print("Mean Squared Error:", mse)
print("*"*100)

#---------------Experiment 2---------------------------------------------
exp_name = "2"
# Define RandomForestRegressor
rf = RandomForestRegressor(featuresCol="features", labelCol="Impact")

# Define parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# Define evaluator
evaluator = RegressionEvaluator(labelCol="Impact", predictionCol="prediction", metricName="mape")

# Cross-validation with 3 folds
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# Time start
start_time = datetime.now()
# Fit model
cvModel = crossval.fit(train)
# Time end
end_time = datetime.now()
# Total training time
total_time = end_time - start_time
# Make predictions
predictions = cvModel.transform(test)

# Calculate MAPE
mape = evaluator.evaluate(predictions)
print("Experiement ",exp_name)
print("Mean Absolute Percentage Error (MAPE):", mape)
print("Total Training Time:", total_time)
print("*"*100)


#-----------------Experiment-3------------------------------------------------
exp_name = "3"
input_size = 10000  # Example size of vocabulary
hidden_size1 = 128
hidden_size2 = 64

model = BookImpactPredictor(input_size, hidden_size1, hidden_size2)
predicted_impact = model(title, description, published_year, other_features)
print(predicted_impact)
