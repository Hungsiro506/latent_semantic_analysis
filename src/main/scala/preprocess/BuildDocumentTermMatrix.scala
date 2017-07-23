package preprocess

import java.util

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import java.util.Properties
import jsastrawi.morphology.{DefaultLemmatizer, Lemmatizer}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.ml.feature.{CountVectorizer, IDF}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
/**
  * Created by hungdv on 22/07/2017.
  */
class BuildDocumentTermMatrix(private val spark: SparkSession) extends Serializable {
  import spark.implicits._

  /**
    * Load stopwords
    * @param path
    * @return
    */
  def loadStopWords(path:String) : Set[String] = {
    scala.io.Source.fromFile(path).getLines().toSet
  }

  /**
    * Load dictionary
    * @param path
    * @return
    */
  def loadDictionary(path: String): util.HashSet[String] = {
    val lines: Iterator[String] = scala.io.Source.fromFile(path).getLines()
    val dictionary = new util.HashSet[String]
    lines.foreach(line => dictionary.add(line.split(" ")(0)))
    dictionary
  }

  /**
    * Create a StanfordCoreNLP pipeline object to lemmatize documents
    * @return StanfordCoreNLP
    */
  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos")
    new StanfordCoreNLP(props)
  }

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }
  def createIndonesiaLemmatizer(dictionnary: util.HashSet[String]) = {
    val lemma : Lemmatizer = new DefaultLemmatizer(dictionnary)
    lemma
  }

  /**
    * extract sentences, tokenizer, lammatizer, remove stopwords, get characterOnly
    * @param text
    * @param stopWords
    * @param pipeline
    * @param indoLemmatizer
    * @return
    */
  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP, indoLemmatizer :Lemmatizer)
  : Seq[String] = {
    import scala.collection.JavaConverters._
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala; token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = indoLemmatizer.lemmatize(token.toString())
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }
  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP)
  : Seq[String] = {
    import scala.collection.JavaConverters._
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala; token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = token.get(classOf[LemmaAnnotation])
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }


  /**
    *
    * @param docs
    * @param stopwords
    * @param lemmalizer
    * @return
    */
  def contentsToTerms(docs: Dataset[(String,String)],stopwords: Set[String],lemmalizer: Lemmatizer): Dataset[(String, Seq[String])] ={
    val bStopWords = spark.sparkContext.broadcast(stopwords)
    docs.mapPartitions{ iter =>
      val pipeline = createNLPPipeline()
      iter.map{case(title,contents) => (title,plainTextToLemmas(contents,bStopWords.value,pipeline,lemmalizer))}
    }
  }
  def contentsToTerms(docs: Dataset[(String,String)],stopwords: Set[String]): Dataset[(String, Seq[String])] ={
    val bStopWords = spark.sparkContext.broadcast(stopwords)
    docs.mapPartitions{ iter =>
      val pipeline = createNLPPipeline()
      iter.map{case(title,contents) => (title,plainTextToLemmas(contents,bStopWords.value,pipeline))}
    }
  }
  def contentsToTerms(docs: Dataset[(String,String)]): Dataset[(String,Seq[String])]= {
    docs.mapPartitions{
      iter =>
        iter.map{case(title,contents) => (title,contents.split(" "))}
    }
  }


  /**
    * Returns a document-term matrix where each element is the TF-IDF of the row's document and
    * the column's term.
    * @param docTexts
    * @param stopWordsFile
    * @param numTerms
    * @param dictionnaryPath
    * @return
    */
  def documentTermMatrix(docTexts: Dataset[(String, String)], stopWordsFile: String, numTerms: Int,dictionnaryPath: String)
  : (DataFrame, Array[String], Map[Long, String], Array[Double]) = {
    //val lemma: Lemmatizer = createIndonesiaLemmatizer(loadDictionary(path = dictionnaryPath))
    val stopwords = loadStopWords(stopWordsFile)
    val terms = contentsToTerms(docTexts, stopwords)

    val termsDF = terms.toDF("title", "terms")
    val filtered = termsDF.where(size($"terms") > 1)

    val countVectorizer = new CountVectorizer()
      .setInputCol("terms").setOutputCol("termFreqs").setVocabSize(numTerms)

    val vocabModel = countVectorizer.fit(filtered)
    val docTermFreqs = vocabModel.transform(filtered)
    val termIds = vocabModel.vocabulary
    docTermFreqs.cache()

    val docIds = docTermFreqs.rdd.map(_.getString(0)).zipWithUniqueId().map(_.swap).collect().toMap
    val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
    val idfModel = idf.fit(docTermFreqs)
    val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")
    (docTermMatrix, termIds, docIds, idfModel.idf.toArray)
  }


  /**
    *Returns a document-term matrix where each element is the TF-IDF of the row's document and
    * the column's term.
    *
    * @param docTexts
    * @param numTerms
    * @return
    */
  def documentTermMatrix(docTexts: Dataset[(String, String)], numTerms: Int)
  : (DataFrame, Array[String], Map[Long, String], Array[Double]) = {
    val terms = contentsToTerms(docTexts)
    val termsDF = terms.toDF("title", "terms")
    val filtered = termsDF.where(size($"terms") > 1)

    val countVectorizer = new CountVectorizer()
      .setInputCol("terms").setOutputCol("termFreqs").setVocabSize(numTerms)
    val vocabModel = countVectorizer.fit(filtered)
    val docTermFreqs = vocabModel.transform(filtered)

    val termIds = vocabModel.vocabulary

    docTermFreqs.cache()

    val docIds = docTermFreqs.rdd.map(_.getString(0)).zipWithUniqueId().map(_.swap).collect().toMap

    val idf = new IDF().setInputCol("termFreqs").setOutputCol("tfidfVec")
    val idfModel = idf.fit(docTermFreqs)
    val docTermMatrix = idfModel.transform(docTermFreqs).select("title", "tfidfVec")

    (docTermMatrix, termIds, docIds, idfModel.idf.toArray)
  }

}
