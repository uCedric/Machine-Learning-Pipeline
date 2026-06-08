from pyspark.sql import SparkSession
import socket

def test_spark_cluster():
    print(f"🚀 開始測試... (目前執行主機: {socket.gethostname()})")
    
    # 建立 SparkSession，請確認你的 master 網址與 Port 是否正確
    # 在 Docker Compose 網路中，通常是 spark://<master-service-name>:7077
    master_url = "spark://spark-master:7077"
    
    try:
        spark = SparkSession.builder \
            .appName("ClusterConnectivityTest") \
            .master(master_url) \
            .getOrCreate()

        sc = spark.sparkContext
        sc.setLogLevel("ERROR") # 隱藏過多的 INFO 雜訊

        print("✅ 成功連線至 Spark Master!")
        
        # 建立一組測試資料
        data = [1, 2, 3, 4, 5]
        print(f"📦 準備將資料派發至 Worker 進行平行運算: {data}")

        # 將資料分散至 Worker 並執行 reduce 動作 (加總)
        # 這是觸發 Worker 實際運算的關鍵點！
        rdd = sc.parallelize(data)
        result = rdd.reduce(lambda a, b: a + b)

        print("-" * 40)
        if result == 15:
            print("🎉 大成功！Master 與 Worker 完美連線並完成協同運算。")
            print(f"📊 運算結果: {result}")
        else:
            print("⚠️ 運算完成，但結果不符預期。")
        print("-" * 40)

        spark.stop()

    except Exception as e:
        print("-" * 40)
        print("❌ 叢集測試失敗！請檢查以下可能原因：")
        print("1. Master URL 是否正確？")
        print("2. Worker 是否已經成功向 Master 註冊？")
        print("3. 防火牆或 Docker 網路是否阻擋了通訊？")
        print(f"詳細錯誤訊息:\n{e}")
        print("-" * 40)

if __name__ == "__main__":
    test_spark_cluster()