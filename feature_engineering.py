from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Tokenizer, HashingTF
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType
from gensim.models import KeyedVectors
import numpy as np

# Load the pre-trained word vectors model
model_path = '/home/tanmay/Downloads/GoogleNews-vectors-negative300.bin'  # Replace this with the path to your pre-trained model
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from datetime import datetime

# Function to compute sentence embeddings
def compute_sentence_embedding(sentence):
    words = sentence.split()
    vectors = []
    for word in words:
        if word in word_vectors:
            vectors.append(word_vectors[word])
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(word_vectors.vector_size)


# Initialize Spark session
spark = SparkSession.builder \
    .appName("Book Impact Prediction") \
    .getOrCreate()

# Load data
df = spark.read.option("header", "true").option("delimiter", ",").option("quote", "\"").option("escape", "\"").csv("../csv/books_task.csv")

embedding_udf = udf(compute_sentence_embedding, StringType())

df = df.withColumn("title_embedding", uppercase_udf(df["Title"]))
df = df.withColumn("description_embedding", uppercase_udf(df["description"]))

tokenizer = Tokenizer(inputCol="Title", outputCol="tileWords")
df = tokenizer.transform(df)

df = df.withColumn("Impact", col("Impact").cast("float"))
# Data Cleaning and Feature Engineering
# Convert publishedDate to year and month
df = df.withColumn("publishedYear", df["publishedDate"].substr(1, 4).cast(IntegerType()))
df = df.withColumn("publishedMonth", df["publishedDate"].substr(6, 2).cast(IntegerType()))

df = df.fillna(0)
# Handle missing values
df = df.fillna({'description': ''})
df = df.fillna({'authors': ''})
df = df.fillna({'publisher': ''})

# Feature Engineering for 'categories'
# One-hot encode 'categories'
indexer = StringIndexer(inputCol="categories", outputCol="categoryIndex")
encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
pipeline = Pipeline(stages=[indexer, encoder])
model = pipeline.fit(df)
df = model.transform(df)

# Feature Engineering for 'authors', 'publisher', and 'description'
# Tokenize 'authors', 'publisher', and 'description'
tokenizer = Tokenizer(inputCol="authors", outputCol="authorWords")
df = tokenizer.transform(df)

tokenizer = Tokenizer(inputCol="publisher", outputCol="publisherWords")
df = tokenizer.transform(df)

tokenizer = Tokenizer(inputCol="description", outputCol="descWords")
df = tokenizer.transform(df)

# Convert 'description' to TF-IDF features
hashingTF = HashingTF(inputCol="descWords", outputCol="descFeatures", numFeatures=100)
df = hashingTF.transform(df)

# Convert 'authors' and 'publisher' to TF-IDF features
hashingTF = HashingTF(inputCol="authorWords", outputCol="authorFeatures", numFeatures=100)
df = hashingTF.transform(df)

hashingTF = HashingTF(inputCol="publisherWords", outputCol="publisherFeatures", numFeatures=100)
df = hashingTF.transform(df)


df = df.fillna(0)

# Regression
# Prepare feature vector
assembler = VectorAssembler(inputCols=["publishedYear", "publishedMonth", "descFeatures", "authorFeatures", "publisherFeatures", "categoryVec"], outputCol="features")
df = assembler.transform(df)

df = df.fillna(0)

df.toPandas().to_csv("feature_data.csv")
