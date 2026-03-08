import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TitanicSpark {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Titanic Assignment 2")
      .master("local[*]")
      .getOrCreate()

    val df = spark.read
      .option("header","true")
      .option("inferSchema","true")
      .csv("train.csv")

    df.show()
    df.printSchema()

    // Missing values
    df.select(
      count(when(col("Age").isNull, true)).alias("Missing Age"),
      count(when(col("Cabin").isNull, true)).alias("Missing Cabin"),
      count(when(col("Embarked").isNull, true)).alias("Missing Embarked")
    ).show()

    // Survival count
    df.groupBy("Survived").count().show()

    // Survival by gender
    df.groupBy("Sex","Survived").count().show()

    // Survival by class
    df.groupBy("Pclass","Survived").count().show()

    // Average age
    df.select(avg("Age")).show()

    // Average fare
    df.select(avg("Fare")).show()

    spark.stop()
  }
}