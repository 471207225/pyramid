package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.BackTrackingLineSearcher;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import org.apache.mahout.math.DenseVector;

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
    private double[] wordSumSquare;


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
        this.wordSumSquare = new double[numWords];
        for (int w=0;w<numWords;w++){
            for (int i=0;i<numDocs;i++){
                wordSumSquare[w] += Math.pow(doc2word.getRow(i).get(w),2);
            }
        }
    }

    @Override
    protected void addPriors() {
    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        updateDocScores();
        double[] gradient = new double[numWords];
        for (int i=0;i<gradient.length;i++){
            gradient[i] = gradientForWord(i);
        }
        System.out.println("gradient is");
        System.out.println(Arrays.toString(gradient));

        return gradient;
    }

    @Override
    protected double computeLearningRate(double[] searchDir) {
        System.out.println("initial learning rate = "+shrinkage);
        double[] gradient = gradient(0);
        // switch back to real gradient
        WordVecRegLoss loss = new WordVecRegLoss(doc2word, labels, wordVectorRegression.wordScores, new DenseVector(gradient).times(-1));
        BackTrackingLineSearcher lineSearcher = new BackTrackingLineSearcher(loss);
        lineSearcher.setInitialStepLength(shrinkage);
        BackTrackingLineSearcher.MoveInfo moveInfo = lineSearcher.moveAlongDirection(new DenseVector(searchDir));
        double learningRate = moveInfo.getStepLength();
        System.out.println("tuned learning rate = "+learningRate);
        return learningRate;
    }

        private double gradientForWord(int wordIndex){
        double sum = 0;
        for (int i=0;i<numDocs;i++){
            sum += (docScores[i]
                    -labels[i])*doc2word.getRow(i).get(wordIndex)/numDocs;
        }
        return -2*sum;
    }

//    // newton
//    private double gradientForWord(int wordIndex){
//        double numerator = 0;
//        for (int i=0;i<numDocs;i++){
//            numerator += (labels[i]-docScores[i])*doc2word.getRow(i).get(wordIndex);
//        }
//
//        return numerator/wordSumSquare[wordIndex];
//    }

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
//            System.out.println(scoreMatrix.getScoresForData(i)[0]);
//            System.out.println(" ");
            wordVectorRegression.wordScores.set(i,scoreMatrix.getScoresForData(i)[0]);
        }
        System.out.println(wordVectorRegression.wordScores);
//        System.out.println(wordVectorRegression.wordScores);
    }
}
