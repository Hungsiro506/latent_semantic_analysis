package preprocess

import java.io.{File, FileOutputStream, PrintWriter}
import java.util
import java.util.Properties

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import java.util.Properties

import breeze.io.TextWriter.FileWriter
import jsastrawi.morphology.{DefaultLemmatizer, Lemmatizer}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by hungdv on 20/07/2017.
  */
object LocalPreprocess {
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

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }
  def getLattersOnly(str: String): String = {
    val result = new ArrayBuffer[Char]()
    str.foreach(c => if(Character.isLetter(c)) result += c )
    result.mkString("")
  }
  def createIndonesiaLemmatizer(dictionnary: util.HashSet[String]) = {
    val lemma : Lemmatizer = new DefaultLemmatizer(dictionnary)
    lemma
  }

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators", "tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def loadContent(path: String) :Iterator[String] =  scala.io.Source.fromFile(path).getLines()

  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP, indoLemmatizer :Lemmatizer)
  : Seq[String] = {
    import scala.collection.JavaConverters._
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences.asScala; token <- sentence.get(classOf[TokensAnnotation]).asScala) {
      val lemma = indoLemmatizer.lemmatize(getLattersOnly(token.lemma()))
      if (lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
        lemmas += lemma.toLowerCase
      }
    }
    lemmas
  }


  def preprocess(dictionaryPath: String, stopwordsPath: String,normalPath: String,saraPath:String) = {
    val dictionary = loadDictionary(dictionaryPath)
    val stopwords = loadStopWords(stopwordsPath)
    val normal = loadContent(normalPath)
    val sara = loadContent(saraPath)
    val lemmatizer = createIndonesiaLemmatizer(dictionary)
    val pipeline = createNLPPipeline()

    val normalPreprocessedPath = "/home/hungdv/preprocess/normal_test_pps.txt"
    //val normalPreprocessedPath = "/home/hungdv/preprocess/normal_pps.txt"
    var nomalPreprocesses = new scala.collection.mutable.ArrayBuffer[String]
    normal.foreach{
      line =>
        val lemmas: Seq[String] = plainTextToLemmas(line,stopwords,pipeline,lemmatizer)
          nomalPreprocesses += lemmas.mkString(" ")
        //lemmas.foreach(println(_))
    }
    writeFile(normalPreprocessedPath,nomalPreprocesses)
    println(" --- ---  got normal comments clearned ! ")
    val saraPreprocessedPath = "/home/hungdv/preprocess/sara_test_pps.txt"
    //val saraPreprocessedPath = "/home/hungdv/preprocess/sara_pps.txt"
    var saraPreprocesses = new scala.collection.mutable.ArrayBuffer[String]
    sara.foreach{
      line =>
        val lemmas: Seq[String] = plainTextToLemmas(line,stopwords,pipeline,lemmatizer)
        saraPreprocesses += lemmas.mkString(" ")
      //lemmas.foreach(println(_))
    }
    writeFile(saraPreprocessedPath,saraPreprocesses)

    println("Done")
  }

  def writeFile(path: String, records: ArrayBuffer[String]) = {
    val f: File = new File(path)
    f.getParentFile().mkdirs()
    f.createNewFile()
    try{
      val writer: java.io.FileWriter = new java.io.FileWriter(f)
      for(record <- records){
        //println(record)
        writer.write(record + "\n")
      }
      writer.flush();
      writer.close();
    }
    catch {
      case e: Throwable => println("IDGAF! :))")
    }
  }

  def main(args: Array[String]): Unit = {
    //val normalCmtPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/first1000_normal.txt"
    val normalCmtPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/normal_comments.txt"
    val saraCmtPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/sara_comments.txt"
    val dictionaryPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/id_full.txt"
    val stopwordsPath = "/home/hungdv/workspace/Babe_challenge/src/main/resources/stop_words.txt"

    val normalTestPath = "/home/hungdv/Downloads/interview-test/data/data/test_data/nornal_comments.txt"
    val saraTestPath = "/home/hungdv/Downloads/interview-test/data/data/test_data/sara_comments.txt"

    preprocess(dictionaryPath,stopwordsPath,normalTestPath,saraTestPath)
    //preprocess(dictionaryPath,stopwordsPath,normalCmtPath,saraCmtPath)
  }


}
