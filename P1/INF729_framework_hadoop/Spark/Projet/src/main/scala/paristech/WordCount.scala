package paristech

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{sum, lower, split, explode}
import org.apache.spark.sql.{DataFrame, SparkSession}

object WordCount {

  /**
    * TP 1 : lecture de données, word count, map reduce
    */

  // fonction main <=> fonction qui sera exécutée par Spark
  def main(args: Array[String]): Unit = {

    // la conf qui sera utilisée par Spark lorsqu'on exécutera cette fonction
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // création du SparkSession, la base de tout programme Spark
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Word Count")
      .getOrCreate()

    // on récupère le SparkContext à partir du SparkSession
    val sc = spark.sparkContext

    /**
      * Word Count via des RDDs
      */

    // utilisez votre path
    val filepath: String = "/Users/flo/Documents/packages/spark-2.3.4-bin-hadoop2.7/README.md"

    val rdd: RDD[String] = sc.textFile(filepath)

    println("Les 10 premières lignes du RDD")
    rdd.take(10).foreach(println)

    println("word count basique")
    val wordCount: RDD[(String, Int)] = rdd
      .flatMap(line => line.split(" "))
      .map(word => (word, 1))
      .reduceByKey((i, j) => i + j)

    println("Avec les counts affichés de façon décroissante")
    wordCount
      .sortBy(wordAndCount => wordAndCount._2, ascending = false)
      .take(10)
      .foreach(println)

    println("On passe tout en lowercase")
    wordCount
      .map(wordAndCount => (wordAndCount._1.toLowerCase, wordAndCount._2))
      .reduceByKey((i, j) => i + j)
      .sortBy(wordAndCount => wordAndCount._2, ascending = false)
      .take(10)
      .foreach(println)

    /**
      * Word count via des DataFrames
      */

    val df: DataFrame = spark.read.text(filepath)

    println("Les 10 premières lignes du DataFrame")
    df.show(10)

    println("Sans troncature")
    df.show(10, truncate = false)

    // on import les implicites de notre SparkSession pour utiliser la notation $
    import spark.implicits._

    println("word count basique")
    val wordCountDF: DataFrame = df
      .withColumn("words", split($"value", " "))
      .withColumn("word", explode($"words"))
      .groupBy("word")
      .count

    println("Avec les counts affichés de façon décroissante")
    wordCountDF
      .orderBy($"count".desc)
      .show(10)

    println("Passons tout en lowercase et affichons de nouveau les résultats de façon décroissante")
    wordCountDF
      .withColumn("word", lower($"word"))
      .groupBy("word")
      .agg(sum($"count") as "count")
      .orderBy($"count".desc)
      .show(10)

    /**
      * Plusieurs exemples de syntaxes, de la plus lourde à la plus légère.
      *
      * En termes de lisibilité et de compréhension du code, la plus lourde n'est quasi jamais la meilleure, la plus
      * légère pas toujours appropriée.
      *
      * Ici la seconde syntaxe est plus appropriée car :
      * - les types ainsi que les noms de variables permettent de savoir avec quel type de données on travaille
      * - les types assurent la consistence des données
      * - les noms de variables permettent de comprendre le code plus facilement
      */

    val dfWordCount: DataFrame = sc.textFile(filepath)
      .flatMap { case (line: String) => line.split(" ") }
      .map { case (word: String) => (word, 1) }
      .reduceByKey { case (i: Int, j: Int) => i + j }
      // permet de passer d'un RDD à un DataFrame. ATTENTION : on peut faire ça car on a importé plus haut les
      // implicites du SparkSession via : import spark.implicits._
      .toDF("word", "count")

    dfWordCount.orderBy($"count".desc).show // par défaut ça affiche 20 lignes

    val dfWordCountLight: DataFrame = sc.textFile(filepath)
      .flatMap { line: String => line.split(" ") }
      .map { word: String => (word, 1) }
      .reduceByKey { (i: Int, j: Int) => i + j }
      .toDF("word", "count")

    dfWordCountLight.orderBy($"count".desc).show

    val dfWordCountLighter: DataFrame = sc.textFile(filepath)
      .flatMap(line => line.split(" "))
      .map(word => (word, 1))
      .reduceByKey((i, j) => i + j)
      .toDF("word", "count")

    dfWordCountLighter.orderBy($"count".desc).show

    val dfWordCountLightest: DataFrame = sc.textFile(filepath)
      .flatMap(_.split(" "))
      .map((_, 1))
      .reduceByKey(_ + _)
      .toDF("word", "count")

    dfWordCountLightest.orderBy($"count".desc).show
  }
}
