package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.dataset.DataSet;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Matrices;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 12/6/16.
 */
public class WordVecClaLoss implements Optimizable.ByGradientValue{
    Vector wordScores;
    public double [][] targetDistribution;
    public DataSet doc2word;
    public int [] labels;
    public Vector gradient;
    public Vector parameters;
    public double[][] logEstimatedDistribution;
    public double bias;
    public double lam;
    public int numDocs;
    public double[] docProb;
    public double[] docScores;
    public int numWords;
    public boolean isValueCacheValid;
    public double value;
    public boolean isDocScoresCacheValid;
    public boolean isProbCacheValid;
    public boolean isGradientCacheValid;
    public boolean isBiasCacheValid;


    public WordVecClaLoss(DataSet doc2word, int [] labels, Vector wordScores, double bias, double lam) {
        this.doc2word = doc2word;
        this.labels = labels;
        this.bias = bias;
        this.lam = lam;
        this.numDocs = labels.length;
        this.docProb = new double[numDocs];
        this.docScores = new double[numDocs];
        this.numWords = doc2word.getNumFeatures();
        this.gradient = new DenseVector(numWords+1);
        this.parameters = new DenseVector(numWords+1);
        this.wordScores=wordScores;
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
        this.isDocScoresCacheValid=false;
        this.isProbCacheValid=false;

    }

    @Override
    public Vector getParameters() {
        combineParameters();
        return parameters;
    }

    @Override
    public void setParameters(Vector parameters) {
        this.parameters=parameters;
        this.bias = parameters.get(0);
        IntStream.range(0,numWords).parallel().forEach(i->wordScores.set(i, parameters.get(i+1)));
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
        this.isDocScoresCacheValid=false;
        this.isProbCacheValid=false;
    }

    @Override
    public Vector getGradient() {
        if (isGradientCacheValid){
            return gradient;
        }
        if (isDocScoresCacheValid&&isProbCacheValid){
            updateGradient();
            this.isDocScoresCacheValid=true;
            this.isProbCacheValid=true;
            return gradient;
        }
        updateDocScores();
        updateDocProb();
        updateGradient();
        this.isDocScoresCacheValid=true;
        this.isProbCacheValid=true;
        this.isGradientCacheValid=true;
        return gradient;
    }


    @Override
    public double getValue() {
        if (isValueCacheValid){
            return this.value;
        }
        if (isDocScoresCacheValid&&isProbCacheValid){
         this.value = calculateValue();
        }
        updateDocScores();
        updateDocProb();
        System.out.println("wordScores");
        for (int i=0; i<5; i++){
            System.out.println(wordScores.get(i));
        }
        System.out.println("bias = "+bias);

        System.out.println("docScores");
        for (int i=0; i<5; i++){
            System.out.println(docScores[i]);
        }
        System.out.println("probability");
        for (int i=0; i<5; i++){
            System.out.println(docProb[i]);
        }
        this.value = calculateValue();
        this.isDocScoresCacheValid=true;
        this.isProbCacheValid=true;
        return this.value;
    }

    public double calculateValue(){
        getTargetDistribution();
        getlogEstimatedDistribution();
        double part1 = IntStream.range(0, doc2word.getNumDataPoints()).parallel()
                .mapToDouble(i->(KLDivergence.klGivenPLogQ(targetDistribution[i],logEstimatedDistribution[i]))).sum();

        double part2 = IntStream.range(0, doc2word.getNumFeatures()).parallel().mapToDouble(i->Math.pow(wordScores.get(i), 2)).sum();
        part2 = part2*lam;
        return part1+part2;
    }

    public void getTargetDistribution(){
        this.targetDistribution = new double[labels.length][2];
        for(int i=0; i<labels.length; i++){
            if (labels[i]==0){

                this.targetDistribution[i][0]=1;
                this.targetDistribution[i][1]=0;

            }else {
                this.targetDistribution[i][0]=0;
                this.targetDistribution[i][1]=1;
            }
        }
    }

    public void getlogEstimatedDistribution() {
        this.logEstimatedDistribution = new double[labels.length][2];
        for(int i=0; i<labels.length; i++){
            double [] scores = new double[2];
            scores[0] = 0;
            scores[1] = docScores[i];
            double logprobability_don = MathUtil.logSumExp(scores);

            this.logEstimatedDistribution[i][0]=-logprobability_don;
            this.logEstimatedDistribution[i][1]=scores[1]-logprobability_don;
        }
    }

    // document scores
    public void updateDocScores(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocScores);
    }

    public void updateDocScores(int docIndex){
        this.docScores[docIndex] = wordScores.dot(doc2word.getRow(docIndex))+bias;
    }

    // probability
    public void updateDocProb(){IntStream.range(0, numDocs).parallel().forEach(this::updateDocProb);
    }

    public void updateDocProb(int docIndex){

        this.docProb[docIndex] = 1/(1 + Math.exp(-docScores[docIndex]));
    }

    // gradient

    public double gradientForWord(int wordIndex){
        double sum = 0;

        Vector column = doc2word.getColumn(wordIndex);
        for (Vector.Element element: column.nonZeroes()){

            int docId = element.index();
            double proportion = element.get();

            sum += (labels[docId] - docProb[docId])*proportion;

        }
        sum += -2*lam*wordScores.get(wordIndex);

//        return sum/numDocs;
        return -sum;
    }

    public double gradientForBias(){

        return -IntStream.range(0, doc2word.getNumDataPoints()).parallel().mapToDouble(i->(labels[i]-
                docProb[i])*docProb[i]).sum();
    }

    public void updateGradient(){
        IntStream.range(0, numWords).parallel().forEach(i->gradient.set(i+1, gradientForWord(i)));
        gradient.set(0, gradientForBias());
    }

    public void combineParameters(){
        parameters.set(0, bias);
        IntStream.range(0, numWords).parallel().forEach(i->parameters.set(i+1, wordScores.get(i)));
    }


}
