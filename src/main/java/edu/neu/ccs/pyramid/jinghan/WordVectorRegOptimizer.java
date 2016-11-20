package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 11/3/16.
 */
public class WordVectorRegOptimizer extends GBOptimizer {
    // size = num docs * num words
    private DataSet doc2word;
    // size = num words * num vec dim
    private WordVectorRegression wordVectorRegression;
    private int numWords;
    private int numDocs;
    private double[] labels;
    private double[] docScores;


    public WordVectorRegOptimizer(WordVectorRegression wordVectorRegression, RegressorFactory factory,
                                  DataSet doc2word, DataSet word2vec, double[] labels, double[] weights) {
        //super(wordVectorRegression, word2vec, factory);
        super(wordVectorRegression, word2vec, factory, weights);
        this.doc2word = doc2word;
        this.wordVectorRegression = wordVectorRegression;
        this.numWords = word2vec.getNumDataPoints();
        this.numDocs = doc2word.getNumDataPoints();
        this.labels = labels;
        this.docScores = new double[numDocs];
    }

    @Override
    protected void addPriors() {
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        updateDocScores();
        double[] gradient = new double[numWords];
        for (int i=0;i<gradient.length;i++){
            gradient[i] = gradient(ensembleIndex, i);
        }
        System.out.println("gradient is");
        System.out.println(Arrays.toString(gradient));

        return gradient;
    }


    private double gradient(int ensembleIndex, int wordIndex){
        double sum = 0;
        for (int i=0;i<numDocs;i++){
            sum += (docScores[i]
                    -labels[i])*doc2word.getRow(i).get(wordIndex)/numDocs;
        }
        return -2*sum;
    }

    private void updateDocScores(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocScore);
    }

    private void updateDocScore(int docIndex){
        this.docScores[docIndex] = wordVectorRegression.predict(doc2word.getRow(docIndex));
    }

    @Override
    protected void initializeOthers() {
    }

    @Override
    protected void updateOthers() {
        System.out.println("word scores\n");
        for (int i=0;i<numWords;i++){
            System.out.println(scoreMatrix.getScoresForData(i)[0]);
            System.out.println(" ");
            wordVectorRegression.wordScores.set(i,scoreMatrix.getScoresForData(i)[0]);
        }
//        System.out.println(wordVectorRegression.wordScores);
    }
}
