import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
val df = spark.read.option("header","true").option("inferSchema","true").csv("CitiGroup2006_2008")

val df2 = df.withColumn("Year",year(df("Date")))
val dfmins = df2.groupBy("Year").min()

dfmins.select($"Year",$"min(Open)").show()

