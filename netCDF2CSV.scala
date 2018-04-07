import scala.sys.process._
import java.io.{FileOutputStream => FileStream, OutputStreamWriter => StreamWriter}
import java.util.Arrays

import ucar.multiarray.ArrayMultiArray
import ucar.multiarray.IndexIterator
import ucar.multiarray.MultiArray
import ucar.multiarray.MultiArrayImpl
import ucar.netcdf.Attribute
import ucar.netcdf.Netcdf
import ucar.netcdf.NetcdfFile
import ucar.netcdf.Variable

import scala.collection.JavaConverters._
import collection.mutable._
import java.io._

import org.apache.spark.sql._
import org.apache.spark.SparkContext
//import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import scala.math.BigDecimal
import org.apache.spark.sql.SaveMode
import org.apache.spark.sql.types._

object SimpleApp {
  def main(args: Array[String]) {
    val file = "C:\\Users\\tatab\\OneDrive\\Programing\\scala\\scala-test\\0607 10-high_alfa_cellose_0415.CDF"
    val nc = new NetcdfFile(file, true)
    val mass_values = nc.get("mass_values")
    val time_values = nc.get("scan_acquisition_time")
    val point_values = nc.get("point_count")
    val intensity_values = nc.get("intensity_values")
    //mass
    val mass_ = mass_values.toArray()
    val mass = mass_ match {
      //case double: Array[Double] => double
      case float: Array[Float] => float
      //case int: Array[Int] => int
      case _ => throw new InternalError()
    }

    //time
    val times_ = time_values.toArray()
    val time = times_ match {
      case double: Array[Double] => double
      case _ => throw new InternalError()
    }
    //point_counts
    val point_ = point_values.toArray()
    val point_counts = point_ match {
      case int: Array[Int] => int
      case _ => throw new InternalError()
    }
    //intensity
    val intensity_ = intensity_values.toArray()
    val intensity = intensity_ match {
      case float: Array[Float] => float
      //case double: Array[Double] => double
      //case int: Array[Int] => int
      case _ => throw new InternalError()
    }
    //spark
    val spark = SparkSession.builder().appName("SimpleApp").config("spark.some.config.option", "some-value").master("local").getOrCreate()
    spark.conf.set("spark.executor.memory", "2g")
    import spark.implicits._
    //前処理用にRDD化
    //val conf = new SparkConf().setMaster("local").set("spark.executor.memory", "1g")
    val sc = spark.sparkContext
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import spark.implicits._
    val intensity_int = intensity.map(_.toInt)
    var intensity_rdd = sc.makeRDD(intensity_int)
    //負の値を除去
    val intensity_replace = intensity_rdd.map { line =>
      var result: Array[Int] = null
      if (line < 0)
        result = Array(0)
      else
        result = Array(line)
      result.mkString("")
    }
    val intensity_array = intensity_replace.collect().map(_.toInt)
    //粒子化（少数第二位四捨五入＋10倍）
    var mass_rdd = sc.makeRDD(mass)
    val mass_replace = mass_rdd.map { line =>
      var mass_result: Array[Int] = null
      val line_10 = BigDecimal(line * 10).setScale(0,BigDecimal.RoundingMode.HALF_UP).toInt
      //val line_10 = BigDecimal(line).setScale(0, BigDecimal.RoundingMode.HALF_UP).toInt
      mass_result = Array(line_10)
      mass_result.mkString(",")
    }
    val mass_array = mass_replace.collect().map(_.toInt)
    //val mass_array_check = mass_array.tail :+0
    //旧重複チェック
//    val mass_array_check = 0 +: mass_array.init
//
//    val dupe = (0 until mass_array.size).map(i => mass_array(i) - mass_array_check(i))
//    val dupe_add = 1+:dupe.init
//    val dupe_result = (0 until dupe.size).map{i =>
//      if (dupe(i) == 0) 0
//      else if (dupe_add(i) == 0) 0
//      else i+1
//    }.toArray

    //時間テーブルの作成
    val time2 = time.map(_*10)
    val time = time2.map(_.toInt)
    val temp = Seq(time,point_counts ).transpose
    val rdd_time = sc.parallelize(temp).map(y => Row(y(0), y(1)))
    val rdd_time_ = rdd_time.map { x =>
      var result = Array[Int]()
      val x1 = x.getAs[Int](0)
      for (i <- 1 to x.getAs[Int](1)) {
        result = result :+  x1
      }
      result
    }
    val time_calc = rdd_time_.collect.flatten
    //val time_calc2 = time_calc.map(_.toString)
    //    //Dataftame作成
    //    val xs = Seq(mass_array, mass_array_check, intensity_array).transpose
    //    val rdd = sc.parallelize(xs).map(ys => Row(ys(0), ys(1), ys(2)))
    //    val colname = Array("mass", "mass_check", "intensity")
    //    val schema = StructType(colname.map(name => StructField(name, IntegerType, true)))
    //    val df = sqlContext.createDataFrame(rdd, schema)
    //    val result = df.withColumn("check", $"mass" - $"mass_check")
    //    //インデックス付与
    //    val result_id = result.withColumn("idx", monotonicallyIncreasingId())

    //val result_dupe = spark.sql("SELECT * FROM global_temp.result_id where idx =180")

    //Dataftame作成
    val xs = Seq(time_calc,mass_array, intensity_array).transpose
    val rdd = sc.parallelize(xs).map(ys => Row(ys(0), ys(1), ys(2)))
    val colname = Array("time","mass","intensity")
    val schema = StructType(colname.map{ name =>
      if (name == "time") StructField(name, IntegerType, true)
      else StructField(name, IntegerType, true)
    })
    val df = sqlContext.createDataFrame(rdd, schema)

    //val df2 = df.filter(df("mass") <= "200")
    //val df3 = df2.filter(df2("time") <= "2000")
    val result_df = df.groupBy("time").pivot("mass").sum("intensity").sort("time")
    //result_df.write.mode(SaveMode.Overwrite).csv("./result")
    result_df.coalesce(1).write.mode(SaveMode.Overwrite).option("header","true").csv("result.csv")









    df.createGlobalTempView("df")
    val df_check1 = spark.sql("select * from global_temp.df where Dupe_num == 0")

    //df.select(concat($"time",$"dupe_num") as result).show()

    val df_check = df.withColumn("check_str",concat_ws("-",'time,'mass))
    df_check.createGlobalTempView("df_check")
    //val result = df_check.groupBy("check_str").agg(sum("time","mass","Dype_num"))
    val df_check5 = spark.sql("select check_str,count(*) from global_temp.df_check group by check_str")
    val df_check5 = spark.sql("select check_str,count(*) from global_temp.df_check group by check_str")
    val result_df = df.groupBy($"mass",$"intensity",$"Dupe_num",$"check_str").pivot("time")
    //val df_result =df_check.groupBy($"check_str").agg(mass)
    val result_dupe = spark.sql("SELECT CONCAT (time,-,Dupe_num) FROM global_temp.df as resule")



    //    val result_dupe = spark.sql("SELECT * FROM global_temp.df where Dupe_num =0")

//    //インデックス付与
//    val result_id = result.withColumn("idx", monotonicallyIncreasingId())
    //    val result_dupe = spark.sql("SELECT * FROM global_temp.result_id where idx =180")
//    //重複位置の取得
//    result_id.createGlobalTempView("result")
//    val dupe_id = spark.sql("SELECT idx FROM global_temp.result where check =0").collect()
//    //インデックスで降順へ
//    val result_id_desc = result_id.sort(desc("idx"))
//    //重複位置の削除、強度の統合
//    println(dupe_id)

    //時間テーブルの作成
    val time2 = time.map(_*10)
    val time = time2.map(_.toInt)
    val temp = Seq(time,point_counts ).transpose
    val rdd_time = sc.parallelize(temp).map(y => Row(y(0), y(1)))
    val rdd_time_ = rdd_time.map { x =>
      var result = Array[Int]()
      val x1 = x.getAs[Int](0)
      for (i <- 1 to x.getAs[Int](1)) {
        result = result :+  x1
      }
      result
    }





    //遅い
//    val time2 = time.map(_*10)
//    val time = time2.map(_.toInt)
//    val count_num = (0 to time.length - 1).toArray
//    var result = Array[Int]()
//    for (i <- count_num) {
//      var j = point_counts(i).toInt
//      for (k <- (1 to j)) {
//        result = result :+ time(i)
//      }
//    }
    //ここまで

//    val result_cut_dupe = result_id_desc.map{ row =>
//      val row3 = row.getAs[Int](3)
//      val row1 = row.getAs[Int](1)
//      var temp = 0
//
//      for (x <- dupe_id.reverse) {
//        var id = x.map(_.toInt)
//        if (row3 == id) {
//          Row(0, 0, 0, row3)
//          temp = temp + row.getAs[Int](1)
//        }
//        else if (row3 == id-1) {
//          row(1)
//          Row(row(0), row1 + temp, row(2), row(3))
//          temp = 0
//        }
//        else {
//          Row(row(0), row(1), row(2), row(3))
//        }
//        }
//    }
    //配列の復元
//    val mass_cheked = args
//    val intensity_cheked = args
//    //配列の編成
//    var num = 0
//    var mass_sep = List[Any]()
//    var intensity_sep = List[Any]()
//    for (x <- point_counts){
//      mass_sep :+= mass_cheked.slice(num,num+x)
//      intensity_sep :+=intensity_cheked.slice(num,num+x)
//      num = num + x
//    }
//    //再度初期化
//    num = 0
//    //テーブル操作
//    df.createGlobalTempView("cdf2")
//    val check = spark.sql("SELECT mass-mass_check FROM global_temp.cdf2").collect()
//    //結果の保存
//    //result_dupe.write.mode(SaveMode.Overwrite).csv("./result.csv")
//
//    spark.stop()
//    //print
//    println(s"--------------------------------------------\n")
//    println(mass,"\n")
//    println(time,"\n")
//    println(point_counts,"\n")
//    println(intensity,"\n")
//    println(s"--------------------------------------------\n")
//    val logFile = "C:\\Users\\tatab\\OneDrive\\Programing\\scala\\scala-test\\iris.csv"
//    val logData = spark.read.csv(logFile).cache()
//    //val df5 = spark.read .format("csv").option("header", "true").option("mode", "DROPMALFORMED").option("inferSchema","True").load("C:\\Users\\tatab\\OneDrive\\Programing\\scala\\scala-test\\iris.csv")    println(s"--------------------------------------------\n")
//    //println(logFile,"\n")
//    println(s"--------------------------------------------\n")

    //spark.stop()

  }
}