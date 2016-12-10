package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.eval.KLDivergence;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.dataset.DataSet;

import edu.neu.ccs.pyramid.util.MathUtil;
import edu.neu.ccs.pyramid.util.Matrices;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

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

    public WordVecClaLoss(DataSet doc2word, int [] labels, Vector wordScores, Vector gradient) {
        this.doc2word = doc2word;
        this.wordScores = wordScores;
        this.wordScores = new DenseVector(wordScores);
        this.gradient = gradient;
        this.labels = labels;

        getTargetDistribution();

    }


//    public void getTargetDistribution(){
//
////        void setFeatureValue(int dataPointIndex,
////        int featureIndex, double featureValue)
////        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(numData).numFeatures(numfeatures).build();
//        this.targetDistribution = (DataSet) DataSetBuilder.getBuilder().numDataPoints(labels.length).numFeatures(2);
//        for(int i=0; i<labels.length; i++){
//            if(labels[i]==0){
//                this.targetDistribution.setFeatureValue(i,0,1);
//                this.targetDistribution.setFeatureValue(i,1,0);
//            } else {
//                this.targetDistribution.setFeatureValue(i,0,0);
//                this.targetDistribution.setFeatureValue(i,1,1);
//            }
//        }
//    }

    public void getTargetDistribution(){
        this.targetDistribution = new double[labels.length][2];
        for(int i=0; i<labels.length; i++){
            if (labels[i]==0){

                this.targetDistribution[i][0]=1;
                this.targetDistribution[i][1]=0;

//                System.out.print(targetDistribution[i][0]+"\n");
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
            scores[1] = wordScores.dot(doc2word.getRow(i));
            double logprobability_don = MathUtil.logSumExp(scores);

            this.logEstimatedDistribution[i][0]=-logprobability_don;
            this.logEstimatedDistribution[i][1]=scores[1]-logprobability_don;
//            System.out.println(logEstimatedDistribution[i][1]+"\n");
        }
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
        getlogEstimatedDistribution();

        double los = IntStream.range(0, doc2word.getNumDataPoints()).parallel()
                .mapToDouble(i->(KLDivergence.klGivenPLogQ(targetDistribution[i],logEstimatedDistribution[i]))).average().getAsDouble();

        System.out.println("loss is ");
        System.out.println(los);
        return los;

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
