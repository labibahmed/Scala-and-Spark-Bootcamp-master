import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("titanic.csv")

val logregdataall = (data.select(data("Survived").as("label"),
                    $"Pclass", $"Name", $"Sex", $"Age", $"SibSp", $"Parch",
                    $"Fare", $"Embarked"))

val logregdata = logregdataall.na.drop()


import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

val assembler = new VectorAssembler()setInputCols(Array("Pclass","SexVec","SibSp","Parch","Fare","EmbarkVec"))
                .setOutputCol("features")

val Array(training,test) = logregdata.randomSplit(Array(0.7,0.3),seed=12345)


import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkIndexer,genderEncoder,embarkEncoder, assembler,lr))

val model = pipeline.fit(training)

val results = model.transform(test)

import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"predictions",$"label").as[(Double,Double)].rdd

val metrics = new MulticlassMetrics(predictionAndLabels)

println(metrics.confusionMatrix)