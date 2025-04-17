from pyspark.sql import SparkSession
from pyspark.sql.functions import count, col, sum
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

spark = SparkSession.builder.appName("Detecting Malware in Network Traffic").getOrCreate()

pd_df = pd.read_csv("/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-9-1conn.log.labeled.csv", sep=";")
df = spark.read.csv("/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-9-1conn.log.labeled.csv", header=True, sep="|", inferSchema=True)

schema = StructType([
    StructField("ts", DoubleType(), True),
    StructField("uid", StringType(), True),
    StructField("id.orig_h", StringType(), True),
    StructField("id.orig_p", IntegerType(), True),
    StructField("id.resp_h", StringType(), True),
    StructField("id.resp_p", IntegerType(), True),
    StructField("proto", StringType(), True),
    StructField("service", StringType(), True),
    StructField("duration", DoubleType(), True),
    StructField("orig_bytes", IntegerType(), True),
    StructField("resp_bytes", IntegerType(), True),
    StructField("conn_state", StringType(), True),
    StructField("local_orig", StringType(), True),
    StructField("local_resp", StringType(), True),
    StructField("missed_bytes", DoubleType(), True),
    StructField("history", StringType(), True),
    StructField("orig_pkts", DoubleType(), True),
    StructField("orig_ip_bytes", DoubleType(), True),
    StructField("resp_pkts", DoubleType(), True),
    StructField("resp_ip_bytes", DoubleType(), True),
    StructField("tunnel_parents", StringType(), True),
    StructField("label", StringType(), True),
    StructField("detailed_label", StringType(), True)
])

df = spark.read.csv("/kaggle/input/network-malware-detection-connection-analysis/CTU-IoT-Malware-Capture-9-1conn.log.labeled.csv", header=True, sep="|", schema=schema)

df.limit(5).toPandas()

df = df.withColumnsRenamed({"id.orig_h": "id_orig_h",
                      "id.orig_p": "id_orig_p",
                      "id.resp_h": "id_resp_h",
                      "id.resp_p": "id_resp_p"})

df.printSchema()

print("Number of Rows:", df.count())
print("Number of Columns:", len(df.columns))

double_cols = [col for col, dtype in df.dtypes if dtype == "double"]

descriptive_stats = df.select(double_cols).describe().limit(5).toPandas()

descriptive_stats.set_index("summary").T

for col in df.columns:
    df.groupBy(col).count().show(5)
    print("-"*100)
    
cols_to_drop = ["uid", "id_orig_h", "id_orig_p", "id_resp_h", "id_resp_p", "local_orig", 
                "local_resp", "service", "history","tunnel_parents", "detailed_label"]

df = df.drop(*cols_to_drop)

df.limit(5).toPandas()

null_values = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).limit(5).toPandas()

null_values.T.rename(columns={0: "Number of Null Values"})

cols_with_higher_na = ["duration", "orig_bytes", "resp_bytes"]
df = df.drop(*cols_with_higher_na)

null_values = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns]).limit(5).toPandas()

null_values.T.rename(columns={0: "Number of Null Values"})

df.limit(5).toPandas()

# def obtain_mode(df, col):
#     df.createOrReplaceTempView("df")
#     mode = spark.sql(f"SELECT {col}, COUNT(*) AS Count FROM df WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY Count DESC").first()[0]
#     return mode

# def calculate_median(df, col):
#     non_null_values = spark.sql("SELECT {col} FROM {df} WHERE {col} IS NOT NULL", df=df, col=col).collect()
    
#     if len(non_null_values)%2 == 0:
#         median = df.approxQuantile(col, [0.5], relativeError=0)[0]

        # Filter values greater than median
#         df_above_median = df.filter(df[col] > median)

        # Find the minimum value in the filtered DataFrame
#         value_above_median = df_above_median.agg({col: "min"}).collect()[0][0]
#         return (median + value_above_median)/2
#     else:
#         return df.approxQuantile(col, [0.5], 0.0)

# df = df.na.fill(mode, subset=["ColumnName"])

# Data processing

train, test = df.randomSplit([0.8, 0.2], seed=42)

indexer = StringIndexer(inputCols=["proto", "conn_state"], outputCols=["proto_indexed", "conn_state_indexed"])
indexer = indexer.fit(train)
train = indexer.transform(train)
test = indexer.transform(test)

oh_encoder = OneHotEncoder(inputCols=["proto_indexed", "conn_state_indexed"], outputCols=["proto_oh_encoded", "conn_state_oh_encoded"])
oh_encoder = oh_encoder.fit(train)
train = oh_encoder.transform(train)
test = oh_encoder.transform(test)


# --- --- --- --- --- --- Encoding the Target Variable --- --- --- --- --- ---

label_indexer = StringIndexer(inputCol="label", outputCol="encoded_label")
label_indexer = label_indexer.fit(train)
train = label_indexer.transform(train)
test = label_indexer.transform(test)

train.limit(5).toPandas()

test.limit(5).toPandas()

cols_to_remove = ["label", "encoded_label","proto", "proto_indexed", "conn_state", "conn_state_indexed"]

X_cols = [col for col in train.columns if col not in cols_to_remove]

vector_assembler = VectorAssembler(inputCols=X_cols, outputCol="feature_vector")

train = vector_assembler.transform(train)
test = vector_assembler.transform(test)

train.limit(5).toPandas()

train.select("feature_vector").limit(5).toPandas()

test.select("feature_vector").limit(5).toPandas()

normalizer = MinMaxScaler(inputCol="feature_vector", outputCol="normalized_feature_vector")

normalizer = normalizer.fit(train)
train = normalizer.transform(train)
test = normalizer.transform(test)

train.select("feature_vector", "normalized_feature_vector").limit(5).toPandas()

test.select("feature_vector", "normalized_feature_vector").limit(5).toPandas()

model = RandomForestClassifier(featuresCol="normalized_feature_vector", labelCol="encoded_label")

model = model.fit(train)

test_prediction = model.transform(train)

test_prediction.select("encoded_label", "prediction", "probability").limit(5).toPandas()

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="encoded_label")

accuracy = evaluator.evaluate(test_prediction, {evaluator.metricName: "accuracy"})
print("Accuracy:", accuracy)

precision = evaluator.evaluate(test_prediction, {evaluator.metricName: "weightedPrecision"})
print("Precision:", precision)

recall = evaluator.evaluate(test_prediction, {evaluator.metricName: "weightedRecall"})
print("Recall:", recall)

f1 = evaluator.evaluate(test_prediction, {evaluator.metricName: "f1"})
print("F1 Score:", f1)