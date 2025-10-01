package com.dangth2004.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.ml.classification.LogisticRegression // <-- IMPORT MỚI
import org.apache.spark.sql.functions._
import java.io.{File, PrintWriter}

// Đổi tên object để tránh lỗi xung đột
object Lab17_NLPPipeline_ML {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example with ML and Logging")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")

    // 1. --- Read Dataset ---
    val dataPath = "../data/c4-train.00000-of-01024-30K.json.gz"
    val initialDF = spark.read.json(dataPath).limit(1000)

    // 1.5. --- THÊM CỘT LABEL (BẮT BUỘC CHO ML) ---
    // Tạo một cột nhãn (label) giả định dựa trên độ dài văn bản để mô phỏng tác vụ phân loại nhị phân.
    val labeledDF = initialDF.withColumn(
        "label",
        when(length(col("text")) > 500, 1.0).otherwise(0.0) // Label = 1.0 nếu độ dài > 500, ngược lại là 0.0
    ).select("text", "label")

    println(s"Successfully read ${labeledDF.count()} records and added 'label' column.")
    labeledDF.printSchema()

    // --- Định nghĩa các giai đoạn Pipeline (Stages) ---

    // 2. --- Tokenization (Tách từ) ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']")

    // 3. --- Stop Words Removal (Loại bỏ từ dừng) ---
    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // 4. --- HashingTF (Vector hóa tần suất) ---
    val hashingTF = new HashingTF()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("raw_features")
      .setNumFeatures(20000)

    // 5. --- IDF (Tính toán Inverse Document Frequency) ---
    val idf = new IDF()
      .setInputCol(hashingTF.getOutputCol)
      .setOutputCol("features") // Vector đặc trưng cuối cùng

    // 6. --- THÊM GIAI ĐOẠN PHÂN LOẠI (Logistic Regression) ---
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.01)
      .setFeaturesCol("features") // Lấy từ output của IDF
      .setLabelCol("label")     // Lấy từ cột label đã tạo
      .setPredictionCol("prediction") // Cột output cho nhãn dự đoán

    // 7. --- Xây dựng Pipeline Hoàn chỉnh ---
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf, lr)) // <-- Đã thêm 'lr'

    // --- Thực hiện và đo thời gian ---

    println("\nFitting the Full NLP + LR pipeline...")
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(labeledDF) // Model train trên dữ liệu có label
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting took $fitDuration%.2f seconds.")

    println("\nTransforming data and predicting with the fitted pipeline...")
    val transformStartTime = System.nanoTime()
    // Transform giờ tạo ra các cột prediction và probability
    val transformedDF = pipelineModel.transform(labeledDF)
    transformedDF.cache()
    val transformCount = transformedDF.count()
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation and prediction of $transformCount records took $transformDuration%.2f seconds.")

    // Tính toán Accuracy đơn giản
    val correctPredictions = transformedDF.filter($"label" === $"prediction").count()
    val accuracy = correctPredictions.toDouble / transformCount
    println(f"\nSimple Classification Accuracy (on training data): $accuracy%.4f")

    // Lấy 20 kết quả đầu tiên
    val n_results = 20
    // Lấy các cột cần thiết cho việc ghi file
    val results = transformedDF.select("text", "label", "features", "prediction").take(n_results)

    // 8. --- Ghi Metrics (Số liệu) ra file LOG ---

    val log_path = "../log/lab17_metrics_ml.log"
    new File(log_path).getParentFile.mkdirs()
    val logWriter = new PrintWriter(new File(log_path))
    try {
      // Tính toán kích thước từ vựng thực tế
      val actualVocabSize = transformedDF
        .select(explode($"filtered_tokens").as("word"))
        .filter(length($"word") > 1)
        .distinct()
        .count()

      logWriter.println("--- Performance Metrics and ML Results ---")
      logWriter.println(f"Pipeline fitting duration: $fitDuration%.2f seconds")
      logWriter.println(f"Data transformation + prediction duration: $transformDuration%.2f seconds")
      logWriter.println(s"Actual vocabulary size (after preprocessing): $actualVocabSize unique terms")
      logWriter.println(s"HashingTF numFeatures set to: 20000")
      logWriter.println(f"Classification Accuracy (Training Data): $accuracy%.4f")
      if (20000 < actualVocabSize) {
        logWriter.println(s"Note: numFeatures (20000) is smaller than actual vocabulary size ($actualVocabSize). Hash collisions are expected.")
      }
      logWriter.println(s"Metrics file generated at: ${new File(log_path).getAbsolutePath}")
      println(s"\nSuccessfully wrote metrics to $log_path")
    } finally {
      logWriter.close()
    }

    // 9. --- Ghi Kết quả (Results) ra file TEXT ---

    val result_path = "../results/lab17_pipeline_ml_output.txt"
    new File(result_path).getParentFile.mkdirs()
    val resultWriter = new PrintWriter(new File(result_path))
    try {
      resultWriter.println(s"--- NLP Pipeline + LR Output (First $n_results results) ---")
      resultWriter.println(s"Output file generated at: ${new File(result_path).getAbsolutePath}\n")

      results.foreach { row =>
        val text = row.getAs[String]("text")
        val label = row.getAs[Double]("label")
        val features = row.getAs[org.apache.spark.ml.linalg.Vector]("features")
        val prediction = row.getAs[Double]("prediction")

        resultWriter.println("="*80)
        resultWriter.println(s"Original Text: ${text.substring(0, Math.min(text.length, 100))}...")
        resultWriter.println(s"Actual Label: $label")
        resultWriter.println(s"Predicted Label: $prediction")
        resultWriter.println(s"TF-IDF Vector: ${features.toString}")
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
