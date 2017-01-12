package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.BackTrackingLineSearcher;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.stanford.nlp.patterns.Data;
import org.apache.mahout.math.Arrays;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 12/4/16.
 */
public class WordVectorClaOptimizer extends GBOptimizer {
    // size = num docs * num words
    public DataSet doc2word;
    public WordVectorRegression wordVectorRegression;
    public int numWords;
    public int numDocs;
    public int[] labels;
    public double[] docScores;
    public double[] docProb;
    public double bias;
    public double shrinkageTuned;
    public double lam;
    public boolean isDocScoresCacheValid;
    public boolean isProbCacheValid;
    public Vector wordScores;


    public WordVectorClaOptimizer(WordVectorRegression wordVectorRegression, RegressorFactory factory,
                                  DataSet doc2word, DataSet word2vec, int[] labels, double[] weights, double bias, double lam) {
        super(wordVectorRegression, word2vec, factory, weights);
        this.doc2word = doc2word;
        this.wordVectorRegression = wordVectorRegression;
        this.numWords = word2vec.getNumDataPoints();
        this.numDocs = doc2word.getNumDataPoints();
        this.labels = labels;
        this.docScores = new double[numDocs];
        this.docProb = new double[numDocs];
        this.wordScores = wordVectorRegression.wordScores;
        this.bias = bias;
        this.lam = lam;
        this.isDocScoresCacheValid=false;
        this.isProbCacheValid=false;
    }


    @Override
    protected double computeLearningRate(double[] searchDir) {
        System.out.println("initial learning rate = "+shrinkage);
        // switch back to real gradient
//        System.out.println("loss before line search");

        WordVecClaLoss loss = new WordVecClaLoss(doc2word, labels, wordScores, bias, lam);
//        System.out.println(loss.getValue());
//        System.out.println("bias = "+bias);
        BackTrackingLineSearcher lineSearcher = new BackTrackingLineSearcher(loss);
        lineSearcher.setInitialStepLength(shrinkage);
        double learningRate = lineSearcher.computeLearningRate(new DenseVector(searchDir));
        System.out.println("tuned learning rate = "+learningRate);

        this.shrinkageTuned = learningRate;
        return learningRate;
    }


    public void updateDocScores(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocScores);
    }

    public void updateDocScores(int docIndex){
        this.docScores[docIndex] = wordScores.dot(doc2word.getRow(docIndex))+bias;
    }

    public void updateDocProb(){IntStream.range(0, numDocs).parallel().forEach(this::updateDocProb);
    }

    public void updateDocProb(int docIndex){

        this.docProb[docIndex] = 1/(1 + Math.exp(-docScores[docIndex]));
    }


    public double gradientForWord(int wordIndex){
        double sum = 0;
        Vector column = doc2word.getColumn(wordIndex);
        for (Vector.Element element: column.nonZeroes()){

            int docId = element.index();
            double proportion = element.get();

            sum += (labels[docId] - docProb[docId])*proportion;

        }
        sum += -2*lam*wordVectorRegression.wordScores.get(wordIndex);

//        return sum/numDocs;
        return sum;
    }

    @Override
    protected void addPriors() {

    }


    @Override
    protected double[] gradient(int ensembleIndex) {
        double[] gradient = new double[numWords];
        if (isDocScoresCacheValid&&isProbCacheValid){
            IntStream.range(0, gradient.length).parallel()
                    .forEach(i->gradient[i]=gradientForWord(i));
        }
        updateDocScores();
        updateDocProb();
        this.isDocScoresCacheValid=true;
        this.isProbCacheValid=true;
        IntStream.range(0, gradient.length).parallel()
                .forEach(i->gradient[i]=gradientForWord(i));
        return gradient;
    }

    @Override
    protected void initializeOthers() {
    }

    @Override
    protected void updateOthers() {

        WordVecClaLoss loss = new WordVecClaLoss(doc2word, labels, wordVectorRegression.wordScores, bias, lam);
        System.out.println("before updateOthers");
        System.out.println("bias = " + bias);
        System.out.println("loss "+ loss.getValue());
        System.out.println("lam = " + lam);
        System.out.println("wordScores before updateOthers is ");
        for(int j=0; j<5; j++){
            System.out.println(wordScores.get(j));
        }

        for (int i=0;i<numWords;i++){
            wordVectorRegression.wordScores.set(i,scoreMatrix.getScoresForData(i)[0]);
        }

        gradient(0);
        bias += shrinkageTuned*gradientForBias();

        System.out.println("after update others");
        System.out.println("bias = "+bias);
        System.out.println("loss = "+ loss.getValue());

        /*
        check word score
                */
        System.out.println("word score");
        for(int j=0; j<10; j++){
            System.out.println(wordScores.get(j));
        }
        System.out.println("loss after update others");
        System.out.println(loss.getValue());
        System.out.println("lam = "+lam);


//        System.out.println(wordVectorRegression.wordScores);
//        System.out.println(wordVectorRegression.wordScores);
    }
    public double gradientForBias(){
        if (isProbCacheValid&&isDocScoresCacheValid){
            return IntStream.range(0, doc2word.getNumDataPoints()).parallel().mapToDouble(i->(labels[i]-
                    docProb[i])*docProb[i]).sum();
        }

        updateDocScores();
        updateDocProb();
        this.isDocScoresCacheValid=true;
        this.isProbCacheValid=true;
        return IntStream.range(0, doc2word.getNumDataPoints()).parallel().mapToDouble(i->(labels[i]-
        docProb[i])*docProb[i]).sum();
    }

}
