package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.BackTrackingLineSearcher;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 11/3/16.
 */
public class WordVectorRegOptimizer extends GBOptimizer {
    // size = num docs * num words
    private DataSet doc2word;
    // size = num words * num vec dim
    public WordVectorRegression wordVectorRegression;
    private int numWords;
    private int numDocs;
    private double[] labels;
    private double[] docScores;
    public double bias;
    public double shrinkageTuned;
    public double lam;
//    private double[] wordSumSquare;


    public WordVectorRegOptimizer(WordVectorRegression wordVectorRegression, RegressorFactory factory,
                                  DataSet doc2word, DataSet word2vec, double[] labels, double[] weights, double bias, double lam) {
        super(wordVectorRegression, word2vec, factory, weights);
        this.doc2word = doc2word;
        this.wordVectorRegression = wordVectorRegression;
        this.numWords = word2vec.getNumDataPoints();
        this.numDocs = doc2word.getNumDataPoints();
        this.labels = labels;
        this.docScores = new double[numDocs];
        this.bias = bias;
        this.lam = lam;

    }

    @Override
    protected void addPriors() {
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        updateDocScores();
        double[] gradient = new double[numWords];

//        double[] gradient_newton = new double[numWords];
        IntStream.range(0, gradient.length).parallel()
                .forEach(i->gradient[i]=gradientForWord(i));


        return gradient;
    }

//    public void getPrint(int i){
//        // i is the index of the word
//        double[] gradient = gradient(0);
//        System.out.println("gradient = " + gradient[i]);
//    }

    @Override
    protected double computeLearningRate(double[] searchDir) {
        System.out.println("initial learning rate = "+shrinkage);
//        double[] gradient = gradient(0);
        double bias = gradientForBias();
//        double[] gradient_addbias = new double[numWords+1];
        double[] gradient = gradient(0);
//
//        IntStream.range(0,numWords).parallel().forEach(i->gradient_addbias[i]=gradient[i]);
//        gradient_addbias[numWords] = bias;
        WordVecRegLoss loss = new WordVecRegLoss(doc2word, labels, wordVectorRegression.wordScores, new DenseVector(gradient).times(-1), bias, lam) ;
        System.out.println("loss before line search");
        System.out.println(loss.getValue());

//        System.out.println("excellent gradient is" + gradient[3206]);
        // switch back to real gradient
//        WordVecRegLoss loss = new WordVecRegLoss(doc2word, labels, wordVectorRegression.wordScores, new DenseVector(gradient).times(-1), bias);

        BackTrackingLineSearcher lineSearcher = new BackTrackingLineSearcher(loss);
        lineSearcher.setInitialStepLength(shrinkage);
        BackTrackingLineSearcher.MoveInfo moveInfo = lineSearcher.moveAlongDirection(new DenseVector(searchDir));
        double learningRate = moveInfo.getStepLength();
        System.out.println("tuned learning rate = "+learningRate);
        this.shrinkageTuned = learningRate;
        System.out.println("loss after line search");
        System.out.println(loss.getValue());
        return learningRate;
    }

    private double gradientForWord(int wordIndex){
        double sum = 0;
        Vector column = doc2word.getColumn(wordIndex);
        for (Vector.Element element: column.nonZeroes()){
            int docId = element.index();
            double proportion = element.get();
//            sum += (docScores[docId]
//                    -labels[docId])*proportion/(numDocs*numDocs);
            sum += (docScores[docId]
                    -labels[docId])*proportion;


        }
        sum += lam*wordVectorRegression.wordScores.get(wordIndex);
//        return -2*sum;
        return -sum/numDocs;
    }

    public double gradientForBias(){
        return IntStream.range(0, doc2word.getNumDataPoints()).parallel().mapToDouble(i->(labels[i]-
        docScores[i])).sum();
    }


    private void updateDocScores(){ IntStream.range(0, numDocs).parallel().forEach(this::updateDocScore);
    }

    private void updateDocScore(int docIndex){
        this.docScores[docIndex] = wordVectorRegression.predict_addBias(doc2word.getRow(docIndex));
    }

    @Override
    protected void initializeOthers() {
    }

    @Override
    protected void updateOthers() {
//        System.out.println("word scores\n");
        for (int i=0;i<numWords;i++){
//            System.out.println(scoreMatrix.getScoresForData(i)[0]);
//            System.out.println(" ");
            wordVectorRegression.wordScores.set(i,scoreMatrix.getScoresForData(i)[0]);
        }
//        System.out.println(wordVectorRegression.wordScores);
//        System.out.println(wordVectorRegression.wordScores);

            System.out.println("bias is ");
            bias += shrinkageTuned*gradientForBias();
            System.out.println(bias);
    }



}
