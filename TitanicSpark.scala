import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object TitanicSpark {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Titanic Assignment 2")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // -------------------------------
    // Load Training Data
    // -------------------------------

    val train = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("train.csv")

    train.show()
    train.printSchema()

    // -------------------------------
    // Missing Values Analysis
    // -------------------------------

    train.select(
      count(when(col("Age").isNull, true)).alias("Missing Age"),
      count(when(col("Cabin").isNull, true)).alias("Missing Cabin"),
      count(when(col("Embarked").isNull, true)).alias("Missing Embarked")
    ).show()

    // -------------------------------
    // Survival Statistics
    // -------------------------------

    train.groupBy("Survived").count().show()
    train.groupBy("Sex", "Survived").count().show()
    train.groupBy("Pclass", "Survived").count().show()

    train.select(avg("Age")).show()
    train.select(avg("Fare")).show()

    // -------------------------------
    // Handle Missing Values
    // -------------------------------

    val avgAge = train.select(avg("Age")).first().getDouble(0)
    val avgFare = train.select(avg("Fare")).first().getDouble(0)

    val trainClean = train.na.fill(Map(
      "Age" -> avgAge,
      "Fare" -> avgFare
    ))

    // -------------------------------
    // Feature Engineering
    // -------------------------------

    val trainFeatures = trainClean
      .withColumn("SexNumeric", when(col("Sex") === "male", 1).otherwise(0))
      .withColumn("FamilySize", col("SibSp") + col("Parch"))

    // -------------------------------
    // Assemble Features
    // -------------------------------

    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass","Age","Fare","FamilySize","SexNumeric"))
      .setOutputCol("features")

    val trainFinal = assembler.transform(trainFeatures)

    // -------------------------------
    // Train Model
    // -------------------------------

    val lr = new LogisticRegression()
      .setLabelCol("Survived")
      .setFeaturesCol("features")

    val model = lr.fit(trainFinal)

    val trainPredictions = model.transform(trainFinal)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("Survived")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(trainPredictions)

    println("Training Accuracy = " + accuracy)

    println("Model Training Completed")

    // -------------------------------
    // Load Test Dataset
    // -------------------------------

    val test = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("test.csv")

    // Handle missing values in test dataset

    val testClean = test.na.fill(Map(
      "Age" -> avgAge,
      "Fare" -> avgFare
    ))

    // Feature engineering on test data

    val testFeatures = testClean
      .withColumn("SexNumeric", when(col("Sex") === "male", 1).otherwise(0))
      .withColumn("FamilySize", col("SibSp") + col("Parch"))

    // Assemble features

    val testFinal = assembler.transform(testFeatures)

    // -------------------------------
    // Prediction
    // -------------------------------

    val predictions = model.transform(testFinal)

    predictions.select(
      "PassengerId",
      "prediction",
      "probability"
    ).show()

    spark.stop()
  }
}