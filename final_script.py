{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7194652a-dd60-4c23-8e5f-db891be0b2d5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.sql import SparkSession\n",
    "from pyspark.ml.feature import VectorAssembler, Tokenizer, HashingTF\n",
    "from pyspark.ml.regression import LinearRegression\n",
    "from pyspark.ml.evaluation import RegressionEvaluator\n",
    "from pyspark.ml.feature import StringIndexer, OneHotEncoder\n",
    "from pyspark.ml import Pipeline\n",
    "from pyspark.sql.functions import col, udf\n",
    "from pyspark.sql.types import IntegerType\n",
    "\n",
    "from pyspark.ml.regression import RandomForestRegressor\n",
    "from pyspark.ml.tuning import CrossValidator, ParamGridBuilder\n",
    "from datetime import datetime\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "46999bfd-5dcf-4b6e-b382-c06bec42863b",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "24/04/25 17:49:43 WARN Utils: Your hostname, tanmay-Inspiron-15-7000-Gaming resolves to a loopback address: 127.0.1.1; using 192.168.0.240 instead (on interface wlp3s0)\n",
      "24/04/25 17:49:43 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address\n",
      "WARNING: An illegal reflective access operation has occurred\n",
      "WARNING: Illegal reflective access by org.apache.spark.unsafe.Platform (file:/opt/spark/jars/spark-unsafe_2.12-3.2.0.jar) to constructor java.nio.DirectByteBuffer(long,int)\n",
      "WARNING: Please consider reporting this to the maintainers of org.apache.spark.unsafe.Platform\n",
      "WARNING: Use --illegal-access=warn to enable warnings of further illegal reflective access operations\n",
      "WARNING: All illegal access operations will be denied in a future release\n",
      "Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties\n",
      "Setting default log level to \"WARN\".\n",
      "To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).\n",
      "24/04/25 17:49:43 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable\n",
      "                                                                                \r"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "# Initialize Spark session\n",
    "spark = SparkSession.builder \\\n",
    "    .appName(\"Book Impact Prediction\") \\\n",
    "    .getOrCreate()\n",
    "\n",
    "# Load data\n",
    "df = spark.read.option(\"header\", \"true\").option(\"delimiter\", \",\").option(\"quote\", \"\\\"\").option(\"escape\", \"\\\"\").csv(\"../csv/books_task.csv\")\n",
    "\n",
    "\n",
    "df = df.withColumn(\"Impact\", col(\"Impact\").cast(\"float\"))\n",
    "# Data Cleaning and Feature Engineering\n",
    "# Convert publishedDate to year and month\n",
    "df = df.withColumn(\"publishedYear\", df[\"publishedDate\"].substr(1, 4).cast(IntegerType()))\n",
    "df = df.withColumn(\"publishedMonth\", df[\"publishedDate\"].substr(6, 2).cast(IntegerType()))\n",
    "\n",
    "df = df.fillna(0)\n",
    "# Handle missing values\n",
    "df = df.fillna({'description': ''})\n",
    "df = df.fillna({'authors': ''})\n",
    "df = df.fillna({'publisher': ''})\n",
    "\n",
    "# Feature Engineering for 'categories'\n",
    "# One-hot encode 'categories'\n",
    "indexer = StringIndexer(inputCol=\"categories\", outputCol=\"categoryIndex\")\n",
    "encoder = OneHotEncoder(inputCol=\"categoryIndex\", outputCol=\"categoryVec\")\n",
    "pipeline = Pipeline(stages=[indexer, encoder])\n",
    "model = pipeline.fit(df)\n",
    "df = model.transform(df)\n",
    "\n",
    "# Feature Engineering for 'authors', 'publisher', and 'description'\n",
    "# Tokenize 'authors', 'publisher', and 'description'\n",
    "tokenizer = Tokenizer(inputCol=\"authors\", outputCol=\"authorWords\")\n",
    "df = tokenizer.transform(df)\n",
    "\n",
    "tokenizer = Tokenizer(inputCol=\"publisher\", outputCol=\"publisherWords\")\n",
    "df = tokenizer.transform(df)\n",
    "\n",
    "tokenizer = Tokenizer(inputCol=\"description\", outputCol=\"descWords\")\n",
    "df = tokenizer.transform(df)\n",
    "\n",
    "# Convert 'description' to TF-IDF features\n",
    "hashingTF = HashingTF(inputCol=\"descWords\", outputCol=\"descFeatures\", numFeatures=100)\n",
    "df = hashingTF.transform(df)\n",
    "\n",
    "# Convert 'authors' and 'publisher' to TF-IDF features\n",
    "hashingTF = HashingTF(inputCol=\"authorWords\", outputCol=\"authorFeatures\", numFeatures=100)\n",
    "df = hashingTF.transform(df)\n",
    "\n",
    "hashingTF = HashingTF(inputCol=\"publisherWords\", outputCol=\"publisherFeatures\", numFeatures=100)\n",
    "df = hashingTF.transform(df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8e67930e-9a3f-4988-a6f9-ca8d6fe3143a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8fae655c-3abd-4b64-b936-c45ef3cfcfc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Regression\n",
    "# Prepare feature vector\n",
    "assembler = VectorAssembler(inputCols=[\"publishedYear\", \"publishedMonth\", \"descFeatures\", \"authorFeatures\", \"publisherFeatures\", \"categoryVec\"], outputCol=\"features\")\n",
    "df = assembler.transform(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e5fbb6fa-6c6f-427e-8003-f3dbff48f491",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c3ba7669-e7f7-4eed-a2d9-8ac861b8f524",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split data into train and test sets\n",
    "train, test = df.randomSplit([0.8, 0.2], seed=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "2c5cc0a0-8e9c-4e4a-a345-49233ec13c5e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "24/04/25 17:49:52 WARN Instrumentation: [c4cab3ab] regParam is zero, which might cause numerical instability and overfitting.\n",
      "[Stage 8:===================================================>       (7 + 1) / 8]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 3942.9142309505382\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    }
   ],
   "source": [
    "# Make predictions\n",
    "# Train linear regression model\n",
    "lr = LinearRegression(featuresCol=\"features\", labelCol=\"Impact\")\n",
    "model = lr.fit(train)\n",
    "\n",
    "# Make predictions\n",
    "predictions = model.transform(test)\n",
    "\n",
    "# Evaluate model\n",
    "evaluator = RegressionEvaluator(labelCol=\"Impact\", predictionCol=\"prediction\", metricName=\"mse\")\n",
    "mse = evaluator.evaluate(predictions)\n",
    "print(\"Mean Squared Error:\", mse)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ed051b32-542e-442a-8776-92604173bbb7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "ename": "IllegalArgumentException",
     "evalue": "RegressionEvaluator_5950b43eb6be parameter metricName given invalid value mape.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mIllegalArgumentException\u001b[0m                  Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[8], line 23\u001b[0m\n\u001b[1;32m     20\u001b[0m start_time \u001b[38;5;241m=\u001b[39m datetime\u001b[38;5;241m.\u001b[39mnow()\n\u001b[1;32m     22\u001b[0m \u001b[38;5;66;03m# Fit model\u001b[39;00m\n\u001b[0;32m---> 23\u001b[0m cvModel \u001b[38;5;241m=\u001b[39m \u001b[43mcrossval\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mtrain\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     25\u001b[0m \u001b[38;5;66;03m# Time end\u001b[39;00m\n\u001b[1;32m     26\u001b[0m end_time \u001b[38;5;241m=\u001b[39m datetime\u001b[38;5;241m.\u001b[39mnow()\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/base.py:161\u001b[0m, in \u001b[0;36mEstimator.fit\u001b[0;34m(self, dataset, params)\u001b[0m\n\u001b[1;32m    159\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcopy(params)\u001b[38;5;241m.\u001b[39m_fit(dataset)\n\u001b[1;32m    160\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m--> 161\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_fit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdataset\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    162\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    163\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mParams must be either a param map or a list/tuple of param maps, \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    164\u001b[0m                     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbut got \u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m.\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m \u001b[38;5;28mtype\u001b[39m(params))\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/tuning.py:689\u001b[0m, in \u001b[0;36mCrossValidator._fit\u001b[0;34m(self, dataset)\u001b[0m\n\u001b[1;32m    684\u001b[0m train \u001b[38;5;241m=\u001b[39m datasets[i][\u001b[38;5;241m0\u001b[39m]\u001b[38;5;241m.\u001b[39mcache()\n\u001b[1;32m    686\u001b[0m tasks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mmap\u001b[39m(\n\u001b[1;32m    687\u001b[0m     inheritable_thread_target,\n\u001b[1;32m    688\u001b[0m     _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam))\n\u001b[0;32m--> 689\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m j, metric, subModel \u001b[38;5;129;01min\u001b[39;00m pool\u001b[38;5;241m.\u001b[39mimap_unordered(\u001b[38;5;28;01mlambda\u001b[39;00m f: f(), tasks):\n\u001b[1;32m    690\u001b[0m     metrics[j] \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m (metric \u001b[38;5;241m/\u001b[39m nFolds)\n\u001b[1;32m    691\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m collectSubModelsParam:\n",
      "File \u001b[0;32m/usr/lib/python3.10/multiprocessing/pool.py:873\u001b[0m, in \u001b[0;36mIMapIterator.next\u001b[0;34m(self, timeout)\u001b[0m\n\u001b[1;32m    871\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m success:\n\u001b[1;32m    872\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m value\n\u001b[0;32m--> 873\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m value\n",
      "File \u001b[0;32m/usr/lib/python3.10/multiprocessing/pool.py:125\u001b[0m, in \u001b[0;36mworker\u001b[0;34m(inqueue, outqueue, initializer, initargs, maxtasks, wrap_exception)\u001b[0m\n\u001b[1;32m    123\u001b[0m job, i, func, args, kwds \u001b[38;5;241m=\u001b[39m task\n\u001b[1;32m    124\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 125\u001b[0m     result \u001b[38;5;241m=\u001b[39m (\u001b[38;5;28;01mTrue\u001b[39;00m, \u001b[43mfunc\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwds\u001b[49m\u001b[43m)\u001b[49m)\n\u001b[1;32m    126\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m    127\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m wrap_exception \u001b[38;5;129;01mand\u001b[39;00m func \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m _helper_reraises_exception:\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/tuning.py:689\u001b[0m, in \u001b[0;36mCrossValidator._fit.<locals>.<lambda>\u001b[0;34m(f)\u001b[0m\n\u001b[1;32m    684\u001b[0m train \u001b[38;5;241m=\u001b[39m datasets[i][\u001b[38;5;241m0\u001b[39m]\u001b[38;5;241m.\u001b[39mcache()\n\u001b[1;32m    686\u001b[0m tasks \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mmap\u001b[39m(\n\u001b[1;32m    687\u001b[0m     inheritable_thread_target,\n\u001b[1;32m    688\u001b[0m     _parallelFitTasks(est, train, eva, validation, epm, collectSubModelsParam))\n\u001b[0;32m--> 689\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m j, metric, subModel \u001b[38;5;129;01min\u001b[39;00m pool\u001b[38;5;241m.\u001b[39mimap_unordered(\u001b[38;5;28;01mlambda\u001b[39;00m f: \u001b[43mf\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m, tasks):\n\u001b[1;32m    690\u001b[0m     metrics[j] \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m (metric \u001b[38;5;241m/\u001b[39m nFolds)\n\u001b[1;32m    691\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m collectSubModelsParam:\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/util.py:326\u001b[0m, in \u001b[0;36minheritable_thread_target.<locals>.wrapped\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    323\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m    324\u001b[0m     \u001b[38;5;66;03m# Set local properties in child thread.\u001b[39;00m\n\u001b[1;32m    325\u001b[0m     SparkContext\u001b[38;5;241m.\u001b[39m_active_spark_context\u001b[38;5;241m.\u001b[39m_jsc\u001b[38;5;241m.\u001b[39msc()\u001b[38;5;241m.\u001b[39msetLocalProperties(properties)\n\u001b[0;32m--> 326\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mf\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    327\u001b[0m \u001b[38;5;28;01mfinally\u001b[39;00m:\n\u001b[1;32m    328\u001b[0m     InheritableThread\u001b[38;5;241m.\u001b[39m_clean_py4j_conn_for_current_thread()\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/tuning.py:74\u001b[0m, in \u001b[0;36m_parallelFitTasks.<locals>.singleTask\u001b[0;34m()\u001b[0m\n\u001b[1;32m     69\u001b[0m index, model \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mnext\u001b[39m(modelIter)\n\u001b[1;32m     70\u001b[0m \u001b[38;5;66;03m# TODO: duplicate evaluator to take extra params from input\u001b[39;00m\n\u001b[1;32m     71\u001b[0m \u001b[38;5;66;03m#  Note: Supporting tuning params in evaluator need update method\u001b[39;00m\n\u001b[1;32m     72\u001b[0m \u001b[38;5;66;03m#  `MetaAlgorithmReadWrite.getAllNestedStages`, make it return\u001b[39;00m\n\u001b[1;32m     73\u001b[0m \u001b[38;5;66;03m#  all nested stages and evaluators\u001b[39;00m\n\u001b[0;32m---> 74\u001b[0m metric \u001b[38;5;241m=\u001b[39m \u001b[43meva\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mevaluate\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtransform\u001b[49m\u001b[43m(\u001b[49m\u001b[43mvalidation\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mepm\u001b[49m\u001b[43m[\u001b[49m\u001b[43mindex\u001b[49m\u001b[43m]\u001b[49m\u001b[43m)\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     75\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m index, metric, model \u001b[38;5;28;01mif\u001b[39;00m collectSubModel \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/evaluation.py:84\u001b[0m, in \u001b[0;36mEvaluator.evaluate\u001b[0;34m(self, dataset, params)\u001b[0m\n\u001b[1;32m     82\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcopy(params)\u001b[38;5;241m.\u001b[39m_evaluate(dataset)\n\u001b[1;32m     83\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m---> 84\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_evaluate\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdataset\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     85\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m     86\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mTypeError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mParams must be a param map but got \u001b[39m\u001b[38;5;132;01m%s\u001b[39;00m\u001b[38;5;124m.\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;241m%\u001b[39m \u001b[38;5;28mtype\u001b[39m(params))\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/evaluation.py:119\u001b[0m, in \u001b[0;36mJavaEvaluator._evaluate\u001b[0;34m(self, dataset)\u001b[0m\n\u001b[1;32m    105\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21m_evaluate\u001b[39m(\u001b[38;5;28mself\u001b[39m, dataset):\n\u001b[1;32m    106\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    107\u001b[0m \u001b[38;5;124;03m    Evaluates the output.\u001b[39;00m\n\u001b[1;32m    108\u001b[0m \n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    117\u001b[0m \u001b[38;5;124;03m        evaluation metric\u001b[39;00m\n\u001b[1;32m    118\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m--> 119\u001b[0m     \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_transfer_params_to_java\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    120\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_java_obj\u001b[38;5;241m.\u001b[39mevaluate(dataset\u001b[38;5;241m.\u001b[39m_jdf)\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/wrapper.py:143\u001b[0m, in \u001b[0;36mJavaParams._transfer_params_to_java\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m    141\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m param \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mparams:\n\u001b[1;32m    142\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39misSet(param):\n\u001b[0;32m--> 143\u001b[0m         pair \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_make_java_param_pair\u001b[49m\u001b[43m(\u001b[49m\u001b[43mparam\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_paramMap\u001b[49m\u001b[43m[\u001b[49m\u001b[43mparam\u001b[49m\u001b[43m]\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    144\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_java_obj\u001b[38;5;241m.\u001b[39mset(pair)\n\u001b[1;32m    145\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhasDefault(param):\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/ml/wrapper.py:134\u001b[0m, in \u001b[0;36mJavaParams._make_java_param_pair\u001b[0;34m(self, param, value)\u001b[0m\n\u001b[1;32m    132\u001b[0m java_param \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_java_obj\u001b[38;5;241m.\u001b[39mgetParam(param\u001b[38;5;241m.\u001b[39mname)\n\u001b[1;32m    133\u001b[0m java_value \u001b[38;5;241m=\u001b[39m _py2java(sc, value)\n\u001b[0;32m--> 134\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mjava_param\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mw\u001b[49m\u001b[43m(\u001b[49m\u001b[43mjava_value\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/py4j/java_gateway.py:1309\u001b[0m, in \u001b[0;36mJavaMember.__call__\u001b[0;34m(self, *args)\u001b[0m\n\u001b[1;32m   1303\u001b[0m command \u001b[38;5;241m=\u001b[39m proto\u001b[38;5;241m.\u001b[39mCALL_COMMAND_NAME \u001b[38;5;241m+\u001b[39m\\\n\u001b[1;32m   1304\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mcommand_header \u001b[38;5;241m+\u001b[39m\\\n\u001b[1;32m   1305\u001b[0m     args_command \u001b[38;5;241m+\u001b[39m\\\n\u001b[1;32m   1306\u001b[0m     proto\u001b[38;5;241m.\u001b[39mEND_COMMAND_PART\n\u001b[1;32m   1308\u001b[0m answer \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mgateway_client\u001b[38;5;241m.\u001b[39msend_command(command)\n\u001b[0;32m-> 1309\u001b[0m return_value \u001b[38;5;241m=\u001b[39m \u001b[43mget_return_value\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1310\u001b[0m \u001b[43m    \u001b[49m\u001b[43manswer\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mgateway_client\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtarget_id\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mname\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1312\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m temp_arg \u001b[38;5;129;01min\u001b[39;00m temp_args:\n\u001b[1;32m   1313\u001b[0m     temp_arg\u001b[38;5;241m.\u001b[39m_detach()\n",
      "File \u001b[0;32m~/Documents/highlevel/venv/lib/python3.10/site-packages/pyspark/sql/utils.py:117\u001b[0m, in \u001b[0;36mcapture_sql_exception.<locals>.deco\u001b[0;34m(*a, **kw)\u001b[0m\n\u001b[1;32m    113\u001b[0m converted \u001b[38;5;241m=\u001b[39m convert_exception(e\u001b[38;5;241m.\u001b[39mjava_exception)\n\u001b[1;32m    114\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(converted, UnknownException):\n\u001b[1;32m    115\u001b[0m     \u001b[38;5;66;03m# Hide where the exception came from that shows a non-Pythonic\u001b[39;00m\n\u001b[1;32m    116\u001b[0m     \u001b[38;5;66;03m# JVM exception message.\u001b[39;00m\n\u001b[0;32m--> 117\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m converted \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    118\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    119\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m\n",
      "\u001b[0;31mIllegalArgumentException\u001b[0m: RegressionEvaluator_5950b43eb6be parameter metricName given invalid value mape."
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "ERROR:root:Exception while sending command.\n",
      "Traceback (most recent call last):\n",
      "  File \"/home/tanmay/Documents/highlevel/venv/lib/python3.10/site-packages/py4j/clientserver.py\", line 480, in send_command\n",
      "    raise Py4JNetworkError(\"Answer from Java side is empty\")\n",
      "py4j.protocol.Py4JNetworkError: Answer from Java side is empty\n",
      "\n",
      "During handling of the above exception, another exception occurred:\n",
      "\n",
      "Traceback (most recent call last):\n",
      "  File \"/home/tanmay/Documents/highlevel/venv/lib/python3.10/site-packages/py4j/java_gateway.py\", line 1038, in send_command\n",
      "    response = connection.send_command(command)\n",
      "  File \"/home/tanmay/Documents/highlevel/venv/lib/python3.10/site-packages/py4j/clientserver.py\", line 503, in send_command\n",
      "    raise Py4JNetworkError(\n",
      "py4j.protocol.Py4JNetworkError: Error while sending or receiving\n"
     ]
    }
   ],
   "source": [
    "# Define RandomForestRegressor\n",
    "rf = RandomForestRegressor(featuresCol=\"features\", labelCol=\"Impact\")\n",
    "\n",
    "# Define parameter grid\n",
    "paramGrid = ParamGridBuilder() \\\n",
    "    .addGrid(rf.numTrees, [10, 20, 30]) \\\n",
    "    .addGrid(rf.maxDepth, [5, 10, 15]) \\\n",
    "    .build()\n",
    "\n",
    "# Define evaluator\n",
    "evaluator = RegressionEvaluator(labelCol=\"Impact\", predictionCol=\"prediction\", metricName=\"mape\")\n",
    "\n",
    "# Cross-validation with 3 folds\n",
    "crossval = CrossValidator(estimator=rf,\n",
    "                          estimatorParamMaps=paramGrid,\n",
    "                          evaluator=evaluator,\n",
    "                          numFolds=3)\n",
    "\n",
    "# Time start\n",
    "start_time = datetime.now()\n",
    "\n",
    "# Fit model\n",
    "cvModel = crossval.fit(train)\n",
    "\n",
    "# Time end\n",
    "end_time = datetime.now()\n",
    "\n",
    "# Total training time\n",
    "total_time = end_time - start_time\n",
    "\n",
    "# Make predictions\n",
    "predictions = cvModel.transform(test)\n",
    "\n",
    "# Calculate MAPE\n",
    "mape = evaluator.evaluate(predictions)\n",
    "\n",
    "print(\"Mean Absolute Percentage Error (MAPE):\", mape)\n",
    "print(\"Total Training Time:\", total_time)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7921d922-8ccd-4ee9-a1e6-507acf19948e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
