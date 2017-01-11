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
        this.bias = bias;
        this.lam = lam;

    }


    @Override
    protected double computeLearningRate(double[] searchDir) {
        System.out.println("initial learning rate = "+shrinkage);
        double[] gradient = gradient(0);
        // switch back to real gradient
        WordVecClaLoss loss = new WordVecClaLoss(doc2word, labels, wordVectorRegression.wordScores, bias, lam);

//        WordVecRegLoss loss = new WordVecClaLoss(doc2word, labels, wordVectorRegression.wordScores, new DenseVector(gradient).times(-1));
        BackTrackingLineSearcher lineSearcher = new BackTrackingLineSearcher(loss);
        lineSearcher.setInitialStepLength(shrinkage);
        double learningRate = lineSearcher.computeLearningRate(new DenseVector(searchDir));
        System.out.println("tuned learning rate = "+learningRate);

        /*
        print loss
         */

        double lossValue = loss.getValue();

        this.shrinkageTuned = learningRate;
        return learningRate;
    }

    public void updateDocScores(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocScores);
    }

    public void updateDocScores(int docIndex){
        this.docScores[docIndex] = wordVectorRegression.predict(doc2word.getRow(docIndex))+bias;
    }

    public void updateDocProb(){IntStream.range(0, numDocs).parallel().forEach(this::updateDocProb);
    }

    public void updateDocProb(int docIndex){

        this.docProb[docIndex] = 1/(1 + Math.exp(-docScores[docIndex]));
    }


    public double gradientForWord(int wordIndex){
        double sum = 0;
//        for (int i=0; i<numDocs; i++){

////            sum += (labels[i] - 1/donomi)*doc2word.getRow(i).get(wordIndex)/numDocs;
//            sum += (labels[i] - 1/donomi)*doc2word.getRow(i).get(wordIndex);
        Vector column = doc2word.getColumn(wordIndex);
        for (Vector.Element element: column.nonZeroes()){

            int docId = element.index();
            double proportion = element.get();
            // average
//            sum += (labels[docId] - 1/donomi)*proportion/numDocs;
            // without average
            sum += (labels[docId] - docProb[docId])*proportion;

        }
        sum += -2*lam*wordVectorRegression.wordScores.get(wordIndex);



        return sum/numDocs;
//        return sum;
    }

    @Override
    protected void addPriors() {

    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        updateDocScores();
        updateDocProb();
        /*
        probability
         */
//        System.out.println("document scores");
//        for(int prob_i=0; prob_i<10; prob_i++){
//            System.out.println(docScores[prob_i]);
//        }

        double[] gradient = new double[numWords];
//        for (int i=0; i<gradient.length; i++){
//            gradient[i] = gradientForWord(i);
//        }
//        System.out.println("gradient is ");
//        System.out.println(Arrays.toString(gradient));
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
//        System.out.println("loss before updateOthers");
//        System.out.println(loss.getValue());

        for (int i=0;i<numWords;i++){
            wordVectorRegression.wordScores.set(i,scoreMatrix.getScoresForData(i)[0]);
        }

        /*
        check word score
         */
//        System.out.println("word score check 2");
//        for(int j=0; j<10; j++){
//            System.out.println(scoreMatrix.getScoresForData(j)[0]);
//        }
//        System.out.println("loss after update others");
//        System.out.println(loss.getValue());


        bias += shrinkageTuned*gradientForBias();

//        System.out.println(wordVectorRegression.wordScores);
//        System.out.println(wordVectorRegression.wordScores);
    }
    public double gradientForBias(){

        return IntStream.range(0, doc2word.getNumDataPoints()).parallel().mapToDouble(i->(labels[i]-
        docProb[i])*docProb[i]).average().getAsDouble();
//        return IntStream.range(0, doc2word.getNumDataPoints()).parallel().mapToDouble(i->(labels[i]-
//        docProb[i])*docProb[i]).sum();
    }

}
