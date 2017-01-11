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
    public double[][] logEstimatedDistribution;
    public double bias;
    public double lam;
    public int numDocs;
    public double[] docProb;
    public double[] docScores;
    public int numWords;

    public WordVecClaLoss(DataSet doc2word, int [] labels, Vector wordScores, double bias, double lam) {
        this.doc2word = doc2word;
//        this.wordScores = wordScores;
        this.labels = labels;
        this.bias = bias;
        this.lam = lam;
        this.numDocs = labels.length;
        this.docProb = new double[numDocs];
        this.docScores = new double[numDocs];
        this.numWords = doc2word.getNumFeatures();
        this.gradient = new DenseVector(numWords);
//        this.wordScores = new DenseVector(numWords);
        this.wordScores=wordScores;


    }
    /*
    get TargetDistribution and EstimateDistribution.
     */

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
            scores[1] = wordScores.dot(doc2word.getRow(i))+bias;
            double logprobability_don = MathUtil.logSumExp(scores);

            this.logEstimatedDistribution[i][0]=-logprobability_don;
            this.logEstimatedDistribution[i][1]=scores[1]-logprobability_don;
        }
    }


    /*
    Update doc scores and probability and gradient
     */

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
        return sum;
    }


    public void gradient(){
        updateDocScores();
        updateDocProb();
        IntStream.range(0, numWords).parallel().forEach(i->gradient.set(i, gradientForWord(i)));
    }

    @Override
    public Vector getGradient() {
        return gradient;
    }


    @Override
    public double getValue() {
//        return IntStream.range(0, doc2word.getNumDataPoints()).parallel()
//                .mapToDouble(i->(-labels[i]*Math.log(1/(1+Math.exp(-wordScores.dot(doc2word.getRow(i)))))
//                +(labels[i]-1)*Math.log(1-1/(1+Math.exp(-wordScores.dot(doc2word.getRow(i))))))).sum();
//        double los = IntStream.range(0, doc2word.getNumDataPoints()).parallel()
//                .mapToDouble(i->(-labels[i]*Math.log(1/(1+Math.exp(-wordScores.dot(doc2word.getRow(i)))))
//                +(labels[i]-1)*Math.log(1-1/(1+Math.exp(-wordScores.dot(doc2word.getRow(i))))))).average().getAsDouble();
        gradient();
        getTargetDistribution();
        getlogEstimatedDistribution();

        System.out.println("wordScores");
        for (int i=0; i<10; i++){
            System.out.println(wordScores.get(i));
        }

        System.out.println("Probability");
        for (int i=0; i<10; i++){
            System.out.println(docProb[i]);
            System.out.println(1/(1 + Math.exp(-docScores[i])));
            System.out.println("\n");
        }


        System.out.println("\n");
        System.out.println("check for logEstimateDistribution ");
        for (int j=0; j<10;j++){
            System.out.println(Arrays.toString(logEstimatedDistribution[j]));
//            System.out.println(logEstimatedDistribution[j].toString());
//            System.out.printf(" %f ",logEstimatedDistribution[j][0]);
//            System.out.printf(" %f ",logEstimatedDistribution[j][1]);
        }
        System.out.println("\n");
        for (int j=12500; j<10;j++){
//            System.out.println(logEstimatedDistribution[j].toString());
            System.out.println(Arrays.toString(logEstimatedDistribution[j]));
        }
//            System.out.printf(" %f ",logEstimatedDistribution[j][0]);
//            System.out.printf(" %f ",logEstimatedDistribution[j][1]);
//            System.out.printf("\t");
//        }


        double part1 = IntStream.range(0, doc2word.getNumDataPoints()).parallel()
                .mapToDouble(i->(KLDivergence.klGivenPLogQ(targetDistribution[i],logEstimatedDistribution[i]))).sum();

        double part2 = IntStream.range(0, doc2word.getNumFeatures()).parallel().mapToDouble(i->Math.pow(wordScores.get(i), 2)).sum();
        part2 = part2*lam;

//        System.out.println("loss is ");
//        System.out.println(los);
        return (part1+ part2)/numDocs;
//        return part1+part2;
    }

    @Override
    public Vector getParameters() {
        return wordScores;
    }

    @Override
    public void setParameters(Vector parameters) {
        this.wordScores = parameters;

    }
}
