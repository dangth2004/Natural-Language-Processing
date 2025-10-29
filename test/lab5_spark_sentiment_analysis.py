import time
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col


def main():
    # 1. Khởi tạo Spark Session
    spark = SparkSession.builder.appName("SentimentAnalysis").getOrCreate()
    data_path = "/home/dangth2004/Programming/Natural-Language-Processing/data/sentiments.csv"

    try:
        # 2. Tải và chuẩn bị dữ liệu
        df = spark.read.csv(data_path, header=True, inferSchema=True)

        # Bỏ qua các hàng có giá trị sentiment bị rỗng
        df = df.dropna(subset=["sentiment"])

        # Chuyển đổi nhãn -1/1 thành 0/1 để phù hợp với LogisticRegression
        # (-1 -> 0, 1 -> 1)
        df = df.withColumn("label", (col("sentiment").cast("integer") + 1) / 2)

        # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
        (trainingData, testData) = df.randomSplit([0.8, 0.2], seed=1234)

        # 3. Xây dựng Pipeline tiền xử lý

        # Tách văn bản thành các từ (tokens)
        tokenizer = Tokenizer(inputCol="text", outputCol="words")

        # Loại bỏ các stop words (từ phổ biến nhưng ít ý nghĩa)
        stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

        # Chuyển đổi tokens thành vector đặc trưng bằng HashingTF
        # numFeatures: số lượng đặc trưng (kích thước vector)
        hashingTF = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)

        # Tính toán IDF để điều chỉnh trọng số của các đặc trưng
        # Giảm trọng số của các từ xuất hiện thường xuyên
        idf = IDF(inputCol="raw_features", outputCol="features")

        # 4. Huấn luyện mô hình (Logistic Regression)
        lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

        # Lắp ráp tất cả các bước vào một Pipeline
        pipeline = Pipeline(stages=[tokenizer, stopwordsRemover, hashingTF, idf, lr])

        # Huấn luyện mô hình trên dữ liệu trainingData
        print("Đang huấn luyện mô hình...")
        start_time = time.perf_counter()
        model = pipeline.fit(trainingData)
        end_time = time.perf_counter()
        print("Huấn luyện hoàn tất.")

        # 5. Đánh giá mô hình

        # Dự đoán trên dữ liệu testData
        predictions = model.transform(testData)

        # Tính toán độ chính xác (Accuracy)
        evaluator_acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator_acc.evaluate(predictions)
        print(f"Độ chính xác (Accuracy): {accuracy:.4f}")

        # Tính toán Precision (Weighted)
        evaluator_precision = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
        )
        precision = evaluator_precision.evaluate(predictions)
        print(f"Precision: {precision:.4f}")

        # Tính toán Recall (Weighted)
        evaluator_recall = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="weightedRecall"
        )
        recall = evaluator_recall.evaluate(predictions)
        print(f"Recall: {recall:.4f}")

        # Tính toán F1-score
        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        )
        f1_score = evaluator_f1.evaluate(predictions)
        print(f"F1-Score: {f1_score:.4f}")
        print(f"Training time: {end_time - start_time} seconds.")

        # Hiển thị một vài kết quả dự đoán
        print("\nMột vài kết quả dự đoán:")
        predictions.select("text", "sentiment", "label", "prediction").show(10, truncate=False)

    except Exception as e:
        print(f"Đã xảy ra lỗi: {e}")
        print(f"Hãy chắc chắn rằng đường dẫn '{data_path}' là chính xác và tệp CSV tồn tại.")

    finally:
        # Dừng Spark session
        spark.stop()


if __name__ == "__main__":
    main()
