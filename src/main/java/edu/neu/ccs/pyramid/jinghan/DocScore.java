package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.patterns.Data;

import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 11/24/16.
 */
public class DocScore {
    public DataSet doc2word;
    public DataSet word2vec;
    public WordVectorRegression wordVectorRegression;
    public int numWords;
    public int numDocs;
    public double[] docScores;
    public double[] wordScores;



    public DocScore(WordVectorRegression wordVectorRegression, DataSet doc2word, DataSet word2vec, double[] wordScores){
        this.wordScores = wordScores;
        this.doc2word = doc2word;
        this.word2vec = word2vec;
        this.wordVectorRegression = wordVectorRegression;
        this.numDocs = doc2word.getNumDataPoints();
        this.numWords = word2vec.getNumDataPoints();
        updateWordScores();
        updataDocScores();
    }

    public void updataDocScores(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocScores);}

    public void updateDocScores(int docIndex) {
        this.docScores[docIndex] = wordVectorRegression.predict(doc2word.getRow(docIndex));
    }

    public void updateWordScores(){
        for (int i=0; i<numWords;i++){
            wordVectorRegression.wordScores.set(i, wordScores[i]);
        }
    }
}
