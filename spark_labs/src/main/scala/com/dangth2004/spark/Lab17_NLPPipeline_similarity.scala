package com.dangth2004.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Normalizer}
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}
import org.apache.spark.ml.linalg.{Vector, Vectors, SparseVector}

object Lab17_NLPPipeline_similarity {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    println(s"Spark UI available at http://localhost:4040")
    println("Pausing for 10 seconds to allow you to open the Spark UI...")
    Thread.sleep(10000)

    // 1. --- Read Document Limit from User Input ---
    val limitDocuments = if (args.length > 0 && args(0).toInt > 0) {
      args(0).toInt
    } else {
      println("No valid document limit provided. Using default: 1000")
      1000
    }
    println(s"Processing $limitDocuments documents.")

    // 2. --- Read Dataset ---
    val dataPath = "../data/c4-train.00000-of-01024-30K.json.gz"
    val readStartTime = System.nanoTime()
    val initialDF = spark.read.json(dataPath).limit(limitDocuments)
    val readDuration = (System.nanoTime() - readStartTime) / 1e9d
    println(s"Successfully read ${initialDF.count()} records in $readDuration%.2f seconds.")
    initialDF.printSchema()
    println("\nSample of initial DataFrame:")
    initialDF.show(5, truncate = false)

    // --- Pipeline Stages Definition ---

    // 3. --- Tokenization ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // 4. --- Stop Words Removal ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 5. --- Vectorization (Term Frequency) ---
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000)

    // 6. --- Vectorization (Inverse Document Frequency) ---
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("tfidf_features")

    // 7. --- Normalization ---
    val normalizer = new Normalizer()
      .setInputCol(idf.getOutputCol)
      .setOutputCol("features")
      .setP(2.0) // L2 norm for unit length vectors

    // 8. --- Assemble the Pipeline ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, normalizer))

    // --- Time the Pipeline Fitting ---
    println("\nFitting the NLP pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(initialDF)
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    // --- Transform Data ---
    println("\nTransforming data with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    val transformedDF = pipelineModel.transform(initialDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation of $transformCount records took $transformDuration%.2f seconds.")

    // Calculate actual vocabulary size
    val vocabStartTime = System.nanoTime()
    val actualVocabSize = transformedDF
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1)
      .distinct()
      .count()
    val vocabDuration = (System.nanoTime() - vocabStartTime) / 1e9d
    println(s"--> Actual vocabulary size after tokenization and stop word removal: $actualVocabSize unique terms in $vocabDuration%.2f seconds.")

    // --- Cosine Similarity Calculation ---
    // UDF for cosine similarity
    val cosine = udf { (v1: SparseVector, v2: SparseVector) =>
      val norm1 = Vectors.norm(v1, 2)
      val norm2 = Vectors.norm(v2, 2)
      val dot = v1.dot(v2)
      if (norm1 == 0 || norm2 == 0)
        0.0
      else
        dot / (norm1 * norm2)
    }

    // Select a random document
    val similarityStartTime = System.nanoTime()
    val randomDoc = transformedDF.select("text", "features").take(1)(0)
    val randomDocText = randomDoc.getAs[String]("text")
    val randomDocVector = randomDoc.getAs[Vector]("features")

    // Create a single-row DataFrame for the random document
    val randomDocDF = Seq((randomDocText, randomDocVector)).toDF("random_text", "random_features")

    // Cross-join with transformedDF to compute similarities
    val similarities = transformedDF
      .crossJoin(randomDocDF)
      .withColumn("similarity", cosine($"features", $"random_features"))
      .select($"text", $"similarity")
      .orderBy(desc("similarity"))
      .limit(6) // Top 5 + the document itself
      .filter($"similarity" < 0.999) // Exclude the document itself
      .take(5)
    val similarityDuration = (System.nanoTime() - similarityStartTime) / 1e9d
    println(f"--> Cosine similarity calculation took $similarityDuration%.2f seconds.")

    // --- Show and Save Results ---
    println("\nSample of transformed data:")
    transformedDF.select("text", "features").show(5, truncate = 50)

    val n_results = 20
    val results = transformedDF.select("text", "features").take(n_results)

    // --- Write Metrics to Log File ---
    val log_path = "../log/lab17_metrics_similarity.log"
    new File(log_path).getParentFile.mkdirs()
    val writeMetricsStartTime = System.nanoTime()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      logWriter.println("--- Performance Metrics ---")
      logWriter.println(f"Data reading duration: $readDuration%.2f seconds")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      logWriter.println(f"Vocabulary size calculation duration: $vocabDuration%.2f seconds")
      logWriter.println(f"Cosine similarity calculation duration: $similarityDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures set to: 20000")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize). Hash collisions are expected.")
      }
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      logWriter.println("\nFor detailed stage-level metrics, view the Spark UI at http://localhost:4040 during execution.")
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }
    val writeMetricsDuration = (System.nanoTime() - writeMetricsStartTime) / 1e9d
    println(f"--> Metrics file writing took $writeMetricsDuration%.2f seconds.")

    // --- Write Results and Metrics to Results File ---
    val result_path = "../results/lab17_pipeline_output_similarity.txt"
    new File(result_path).getParentFile.mkdirs()
    val writeResultsStartTime = System.nanoTime()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline Output (First $n_results results) ---")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")
      resultWriter.println("--- Performance Metrics ---")
      resultWriter.println(f"Data reading duration: $readDuration%.2f seconds")
      resultWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      resultWriter.println(f"Data transformation duration: $transformDuration%.2f seconds")
      resultWriter.println(f"Vocabulary size calculation duration: $vocabDuration%.2f seconds")
      resultWriter.println(f"Cosine similarity calculation duration: $similarityDuration%.2f seconds")
      resultWriter.println(f"Metrics file writing duration: $writeMetricsDuration%.2f seconds")
      resultWriter.println("\n--- Top 5 Similar Documents ---")
      resultWriter.println(s"Reference Document: ${randomDocText.substring(0, Math.min(randomDocText.length, 100))}...")
      resultWriter.println("\nMost Similar Documents:")
      similarities.foreach { row =>
        val text = row.getAs[String]("text")
        val similarity = row.getAs[Double]("similarity")
        resultWriter.println("-"*80)
        resultWriter.println(s"Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(f"Cosine Similarity: $similarity%.4f")
        resultWriter.println("-"*80)
        resultWriter.println()
      }
      println(s"Successfully wrote $n_results results, performance metrics, and similarity analysis to $result_path")
    } finally {
      resultWriter.close()
    }
    val writeResultsDuration = (System.nanoTime() - writeResultsStartTime) / 1e9d
    println(f"--> Results file writing took $writeResultsDuration%.2f seconds.")

    spark.stop()
    println("Spark Session stopped.")
  }
}
