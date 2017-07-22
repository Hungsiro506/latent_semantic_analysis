package preprocess

/**
  * Created by hungdv on 20/07/2017.
  */

import java.util

import breeze.linalg.{DenseMatrix => BDenseMatrix, SparseVector => BSparseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.{Matrices, Matrix, SingularValueDecomposition, Vectors, Vector => MLLibVector}
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.KryoSerializer
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

object RunLSA {


  def main(args: Array[String]): Unit = {
    //val k = 50
    val numTerms = 200000
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder().config("spark.serializer", classOf[KryoSerializer].getName)
      //.master("local[3]")
      .appName("LSA").getOrCreate()
    val assembleMatrix = new BuildDocumentTermMatrix(spark)
    import assembleMatrix._
    import spark.implicits._

    val rawNormalCmtPath = "normal_comments.txt"
    val rawSaraCmtPath = "sara_comments.txt"
    //Server
    val normalCmtPath = "/user/hungvd8/normal_pps.txt"
    val saraCmtPath = "/user/hungvd8/sara_pps.txt"
    val dictionaryPath = "/user/hungvd8/id_full.txt"
    val stopwordsPath = "/user/hungvd8/stop_words.txt"
    val saraTestPath = "/user/hungvd8/sara_test_pps.txt"
    val normalTestPath = "/user/hungvd8/normal_test_pps.txt"
    //Local
    //val normalCmtPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/preprocessed/normal_pps.txt"
/*    val normalCmtPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/preprocessed/normal_pps.txt"
    val saraCmtPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/preprocessed/sara_pps.txt"
    val dictionaryPath = "id_full.txt"
    val stopwordsPath = "stop_words.txt"*/
    /*    val saraTestPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/preprocessed/sara_test_pps.txt"
    val normalTestPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/preprocessed/normal_test_pps.txt"*/

    /*val normalTitle: (String => String) = (agrs: String) => { "normal"}
    val saraTitle: (String => String) = (agrs: String) => { "sara"}
    val sqlNormalTitle = org.apache.spark.sql.functions.udf(normalTitle)
    val sqlSaraTitle = org.apache.spark.sql.functions.udf(saraTitle)*/

    val normalCmt = spark.sparkContext.textFile(normalCmtPath).map(text => ("normal",text))
    val saraCmt = spark.sparkContext.textFile(saraCmtPath).map(text => ("sara",text))

    val saraTest= spark.sparkContext.textFile(saraTestPath).map(text => ("sara",text.split(" ")))
    val normalTest = spark.sparkContext.textFile(normalTestPath).map(text => ("normal",text.split(" ")))
    val testSet = saraTest.union(normalTest)

    val docTexts = spark.createDataset(normalCmt.union(saraCmt))

    val (docTermMatrix, termIds, docIds, termIdfs) = documentTermMatrix(docTexts, numTerms)

    docTermMatrix.cache()

    val vecRdd = docTermMatrix.select("tfidfVec").rdd.map { row =>
      Vectors.fromML(row.getAs[MLVector]("tfidfVec"))
    }

    vecRdd.cache()
    val mat = new RowMatrix(vecRdd)
    val turnning = for(k <- 200 to 1000 by 200) yield {
    //val k = 150
      println("k :" + k)
      val svd = mat.computeSVD(k, computeU=true)

      val queryEngine = new LSAQueryEngine(svd, termIds, docIds, termIdfs)
      val bEngine = spark.sparkContext.broadcast[LSAQueryEngine](queryEngine)

      val result = testSet.map{
        case(topic,terms) => (topic,terms,bEngine.value.calculateTopicScore(terms,10,"sara"))
      }.toDF("label","terms","prediction_score")

      result.show()
/*
      val predictionLabel: (Double => String) = (agr: Double) => { if(agr > 0.8) "sara" else "normal"}
      val sqlpredictionLabel = org.apache.spark.sql.functions.udf(predictionLabel)
*/

      val result_labeled = result.withColumn("predicted_label",when($"prediction_score" > lit(0.5),"sara").otherwise("normal")).cache()
      result_labeled.show(30)

      result_labeled.createOrReplaceTempView("resutl_labeled")
      //result_labeled.write.format("com.databricks.spark.csv").save("/user/hungvd8/result_labeled")
      //val result_labeled = result.withColumn("predicted_label",sqlpredictionLabel(col("prediction_score")))
      //result_labeled.createOrReplaceTempView("result_labeled")
      //Can't use ML pipeline here :((
      val TPFP = 2816
      val TPFN_df = spark.sql("SELECT count(*) as count from resutl_labeled where (predicted_label = 'sara')")
      TPFN_df.show()
      val TPFN = TPFN_df.head(1)(0).getAs[Int]("count")
      val TP_df = spark.sql("SELECT count(*) as count from resutl_labeled where (label = 'sara' AND predicted_label = 'sara')")
      val TP = TP_df.head(1)(0).getAs[Int]("count")
      TP_df.show()
      //val TP_df = result_labeled.where(col("label") === "sara" && col("predicted_label") === "sara")
      //val TP_df = result_labeled.where(col("label") === "sara" && col("predicted_label") === "sara")
      //TP_df.show()
      //val TP = TP_df.count()
      //println("TP :"  + TP)
      //val TPFP = result_labeled.where(col("label") === "sara").count()
      //val TPFN = result_labeled.where(col("predicted_label") === "sara").count()

      println( " Precision " + TP/TPFP)
      println( " Recall " + TP/TPFN)

      result_labeled.unpersist(true)
      bEngine.unpersist()
      bEngine.destroy()
      (k,TP/TPFP,TP/TPFN)
    }
    docTermMatrix.unpersist()
    vecRdd.unpersist()
    turnning.sortBy(_._2).reverse.foreach(println(_))

  }

  /**
    * The top concepts are the concepts that explain the most variance in the dataset.
    * For each top concept, finds the terms that are most relevant to the concept.
    *
    * @param svd A singular value decomposition.
    * @param numConcepts The number of concepts to look at.
    * @param numTerms The number of terms to look at within each concept.
    * @param termIds The mapping of term IDs to terms.
    * @return A Seq of top concepts, in order, each with a Seq of top terms, in order.
    */
  def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                            numTerms: Int, termIds: Array[String]): Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map {case (score, id) => (termIds(id), score) }
    }
    topTerms
  }

  /**
    * The top concepts are the concepts that explain the most variance in the dataset.
    * For each top concept, finds the documents that are most relevant to the concept.
    *
    * @param svd A singular value decomposition.
    * @param numConcepts The number of concepts to look at.
    * @param numDocs The number of documents to look at within each concept.
    * @param docIds The mapping of document IDs to terms.
    * @return A Seq of top concepts, in order, each with a Seq of top terms, in order.
    */
  def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                           numDocs: Int, docIds: Map[Long, String]): Seq[Seq[(String, Double)]] = {
    val u  = svd.U
    val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId
      topDocs += docWeights.top(numDocs).map { case (score, id) => (docIds(id), score) }
    }
    topDocs
  }

  def loadDictionary(path: String): util.HashSet[String] = {
    val lines: Iterator[String] = scala.io.Source.fromFile(path).getLines()
    val dictionary = new util.HashSet[String]
    lines.foreach(line => dictionary.add(line.split(" ")(0)))
    dictionary
  }
}

class LSAQueryEngine(
                      val svd: SingularValueDecomposition[RowMatrix, Matrix],
                      val termIds: Array[String],
                      val docIds: Map[Long, String],
                      val termIdfs: Array[Double]) extends Serializable{

  val VS: BDenseMatrix[Double] = multiplyByDiagonalMatrix(svd.V, svd.s)
  val normalizedVS: BDenseMatrix[Double] = rowsNormalized(VS)
  val US: RowMatrix = multiplyByDiagonalRowMatrix(svd.U, svd.s)
  val normalizedUS: RowMatrix = distributedRowsNormalized(US)

  val idTerms: Map[String, Int] = termIds.zipWithIndex.toMap
  val idDocs: Map[String, Long] = docIds.map(_.swap)

  /**
    * Finds the product of a dense matrix and a diagonal matrix represented by a vector.
    * Breeze doesn't support efficient diagonal representations, so multiply manually.
    */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: MLLibVector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs { case ((r, c), v) => v * sArr(c) }
  }

  /**
    * Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
    */
  def multiplyByDiagonalRowMatrix(mat: RowMatrix, diag: MLLibVector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map { vec =>
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    })
  }

  /**
    * Returns a matrix where each row is divided by its length.
    */
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).foreach(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  /**
    * Returns a distributed matrix where each row is divided by its length.
    */
  def distributedRowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map { vec =>
      val array = vec.toArray
      val length = math.sqrt(array.map(x => x * x).sum)
      Vectors.dense(array.map(_ / length))
    })
  }

  /**
    * Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest
    * relevance scores to the given term.
    */
  def topDocsForTerm(termId: Int): Seq[(Double, Long)] = {
    val rowArr = (0 until svd.V.numCols).map(i => svd.V(termId, i)).toArray
    val rowVec = Matrices.dense(rowArr.length, 1, rowArr)

    // Compute scores against every doc
    val docScores = US.multiply(rowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  /**
    * Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest
    * relevance scores to the given term.
    */
  def topTermsForTerm(termId: Int): Seq[(Double, Int)] = {
    // Look up the row in VS corresponding to the given term ID.
    val rowVec = normalizedVS(termId, ::).t

    // Compute scores against every term
    val termScores = (normalizedVS * rowVec).toArray.zipWithIndex

    // Find the terms with the highest scores
    termScores.sortBy(-_._1).take(10)
  }

  /**
    * Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest
    * relevance scores to the given doc.
    */
  def topDocsForDoc(docId: Long): Seq[(Double, Long)] = {
    // Look up the row in US corresponding to the given doc ID.
    val docRowArr = normalizedUS.rows.zipWithUniqueId.map(_.swap).lookup(docId).head.toArray
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    // Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  /**
    * Builds a term query vector from a set of terms.
    */
  def termsToQueryVector(terms: Seq[String]): BSparseVector[Double] = {
    val indices: Array[Int] = terms.map(idTerms.getOrElse(_,-1)).filter(_ >= 0).toArray
    val values: Array[Double] = indices.map(termIdfs(_))
    //val values: Array[Double] = indices.map(id => if(id  >= 0 )termIdfs(id) else 0)
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  /**
    * Finds docs relevant to a term query, represented as a vector with non-zero weights for the
    * terms in the query.
    */
  def topDocsForTermQuery(query: BSparseVector[Double],topN: Int): Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores: RowMatrix = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights: RDD[(Double, Long)] = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(topN)

  }

  def printTopTermsForTerm(term: String): Unit = {
    val idWeights = topTermsForTerm(idTerms(term))
    println(idWeights.map { case (score, id) => (termIds(id), score) }.mkString(", "))
  }

  def printTopDocsForDoc(doc: String): Unit = {
    val idWeights = topDocsForDoc(idDocs(doc))
    println(idWeights.map { case (score, id) => (docIds(id), score) }.mkString(", "))
  }

  def printTopDocsForTerm(term: String): Unit = {
    val idWeights = topDocsForTerm(idTerms(term))
    println(idWeights.map { case (score, id) => (docIds(id), score) }.mkString(", "))
  }

  def printTopDocsForTermQuery(terms: Seq[String]): Unit = {
    val queryVec = termsToQueryVector(terms)
    val idWeights = topDocsForTermQuery(queryVec,10)
    println(idWeights.map { case (score, id) => (docIds(id), score) }.mkString(", "))
  }
  def getTopSimilarDocs(terms: Seq[String],topN: Int): Seq[(String,Double)] ={
    val queryVec = termsToQueryVector(terms)
    val idWeights = topDocsForTermQuery(queryVec,topN)
    val result = idWeights.map({
      case (score,id) => (docIds(id),score)
    })
    result
  }
  def calculateTopicScore(terms: Seq[String],topN: Int,topic: String): Double ={
    var sum = 0.0
    var scoreTopic = 0.0
    //var x = 1
    val topSimilarDocs: Seq[(String,Double)] = getTopSimilarDocs(terms,topN)
    topSimilarDocs.sortBy(_._2).foreach{ tuple =>
      sum = sum + tuple._2
      //sum = sum + tuple._2*x
      if(tuple._1 == topic) scoreTopic = scoreTopic + tuple._2
      //if(tuple._1 == topic) scoreTopic = scoreTopic + tuple._2*x
      //x = x+1
    }
    scoreTopic/sum
  }
}



object test{
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val sparkSession = SparkSession.builder().master("local[1]").appName("test").getOrCreate()
    import sparkSession.implicits._
    val df = sparkSession.sparkContext.parallelize(Seq(1234)).toDF("count")
    df.show
    val TF = df.head(1)(0).getAs[Int]("count")
    println(TF)

  }
}