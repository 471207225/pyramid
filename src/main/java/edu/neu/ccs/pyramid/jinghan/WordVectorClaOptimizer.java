package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.BackTrackingLineSearcher;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GBOptimizer;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.stanford.nlp.patterns.Data;
import org.apache.mahout.math.Arrays;
import org.apache.mahout.math.DenseVector;

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

    public WordVectorClaOptimizer(WordVectorRegression wordVectorRegression, RegressorFactory factory,
                                  DataSet doc2word, DataSet word2vec, int[] labels, double[] weights) {
        super(wordVectorRegression, word2vec, factory, weights);
        this.doc2word = doc2word;
        this.wordVectorRegression = wordVectorRegression;
        this.numWords = word2vec.getNumDataPoints();
        this.numDocs = doc2word.getNumDataPoints();
        this.labels = labels;
        this.docScores = new double[numDocs];
        this.docProb = new double[numDocs];
    }


    @Override
    protected double computeLearningRate(double[] searchDir) {
        System.out.println("initial learning rate = "+shrinkage);
        double[] gradient = gradient(0);
        // switch back to real gradient
        WordVecClaLoss loss = new WordVecClaLoss(doc2word, labels, wordVectorRegression.wordScores, new DenseVector(gradient).times(-1));
//        WordVecRegLoss loss = new WordVecClaLoss(doc2word, labels, wordVectorRegression.wordScores, new DenseVector(gradient).times(-1));
        BackTrackingLineSearcher lineSearcher = new BackTrackingLineSearcher(loss);
        lineSearcher.setInitialStepLength(shrinkage);
        BackTrackingLineSearcher.MoveInfo moveInfo = lineSearcher.moveAlongDirection(new DenseVector(searchDir));
        double learningRate = moveInfo.getStepLength();
        System.out.println("tuned learning rate = "+learningRate);
        return learningRate;
    }

    public void updateDocScores(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocScores);
    }

    public void updateDocScores(int docIndex){
        this.docScores[docIndex] = wordVectorRegression.predict(doc2word.getRow(docIndex));
    }

    public void updateDocProb(){IntStream.range(0, numDocs).parallel().forEach(this::updateDocProb);
    }

    public void updateDocProb(int docIndex){

        this.docProb[docIndex] = Math.exp(-docScores[docIndex]);
    }


    public double gradientForWord(int wordIndex){
        double sum = 0;
        for (int i=0; i<numDocs; i++){
            double donomi = 1+docProb[i];
//            sum += (labels[i] - 1/donomi)*doc2word.getRow(i).get(wordIndex)/numDocs;
            sum += (labels[i] - 1/donomi)*doc2word.getRow(i).get(wordIndex);

        }

        return sum;
    }

    @Override
    protected void addPriors() {

    }

    @Override
    protected double[] gradient(int ensembleIndex) {
        updateDocScores();
        updateDocProb();
        double[] gradient = new double[numWords];
        for (int i=0; i<gradient.length; i++){
            gradient[i] = gradientForWord(i);
        }
        System.out.println("gradient is ");
        System.out.println(Arrays.toString(gradient));
        return gradient;
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
//        System.out.println(wordVectorRegression.wordScores);
        System.out.println(wordVectorRegression.wordScores);
    }
}
