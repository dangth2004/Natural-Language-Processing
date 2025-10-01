package com.dangth2004.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
// import com.dangth2004.spark.Utils._


object Lab17_NLPPipeline_Word2Vec {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example - Word2Vec")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // 1. --- Read Dataset ---
    val dataPath = "../data/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(1000) // Limit for faster processing during lab
    println(s"Successfully read ${initialDF.count()} records.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false)

    // --- Pipeline Stages Definition ---

    // 2. --- Tokenization ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // 3. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens") // The input for Word2Vec

    // 4. --- Word2Vec (New Vectorization Stage) ---
    // Word2Vec learns a vector representation (embedding) for each word.
    // It then calculates the average of all word vectors in a document to get the document vector.
    // setVectorSize defines the dimension of the resulting feature vector. 100 is a common value.
    val word2Vec = new Word2Vec()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("features") // Use "features" to keep downstream compatibility
      .setVectorSize(100)       // Set embedding dimension (e.g., 100)
      .setMinCount(5)           // Ignore words that appear less than 5 times

    /*
    // 4. --- Vectorization (Term Frequency) ---
    // val hashingTF = new HashingTF()
    //   .setInputCol(stopWordsRemover.getOutputCol)
    //   .setOutputCol("raw_features")
    //   .setNumFeatures(20000)

    // 5. --- Vectorization (Inverse Document Frequency) ---
    // val idf = new IDF()
    //   .setInputCol(hashingTF.getOutputCol)
    //   .setOutputCol("features")
    */

    // 6. --- Assemble the Pipeline ---
    // The pipeline now uses word2Vec instead of hashingTF and idf
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, word2Vec)) // <-- Updated stages

    // --- Time the main operations ---

    println("\nFitting the NLP pipeline (with Word2Vec)...")
    val fitStartTime = System.nanoTime()
    // Word2Vec is a transformation that is trained (fitted) on the tokenized data
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Calculate actual vocabulary size after tokenization and stop word removal
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms.")

    // --- Show and Save Results ---
    println("\nSample of transformed data (Word2Vec document vectors):")
    // Show the document vector and its dimension
    transformedDF.select("text", "features").show(5, truncate = 50)
    transformedDF.select("features").limit(1).collect().foreach(row =>
        println(s"Document vector size: ${row.getAs[org.apache.spark.ml.linalg.Vector]("features").size}")
    )


    val n_results = 20
    val results = transformedDF.select("text", "features").take(n_results)

    // 7. --- Write Metrics and Results to Separate Files ---

    // Write metrics to the log folder
    val log_path = "../log/lab17_metrics_word2vec.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics (Word2Vec) ---")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms.")
      logWriter.println(s"Word2Vec Vector Size set to: 100")
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // Write data results to the results folder
    val result_path = "../results/lab17_pipeline_word2vec_output.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Word2Vec Output (First $n_results results) ---")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      results.foreach { row =>
        val text = row.getAs[String]("text")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Word2Vec Document Vector (Size 100): ${features.toString.substring(0, Math.min(features.toString.length, 150))}...")
        resultWriter.println("="*80)
        resultWriter.println()
      }
      println(s"Successfully wrote $n_results results to $result_path")
    } finally {
      resultWriter.close()
    }

    spark.stop()
    println("Spark Session stopped.")
  }
}
