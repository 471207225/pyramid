package edu.neu.ccs.pyramid.regression.linear_regression;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.util.MathUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 11/20/16.
 */
public class SquareLoss implements Optimizable.ByGradientValue{
    private static final Logger logger = LogManager.getLogger();
    private LinearRegression linearRegression;
    private DataSet dataSet;
    private double[] labels;
    private double value;
    private boolean isGradientCacheValid;
    private boolean isValueCacheValid;
    private Vector gradient;
    private double priorVariance;


    public SquareLoss(LinearRegression linearRegression, DataSet dataSet, double[] labels, double priorVariance) {
        this.linearRegression = linearRegression;
        this.dataSet = dataSet;
        this.labels = labels;
        this.priorVariance = priorVariance;
        this.isGradientCacheValid = false;
        this.isValueCacheValid = false;
        this.gradient = new DenseVector(dataSet.getNumFeatures()+1);
    }

    @Override
    public Vector getParameters() {
        return linearRegression.getWeights().getWeights();
    }

    @Override
    public void setParameters(Vector parameters) {
        this.linearRegression.getWeights().setWeightVector(parameters);
        this.isValueCacheValid=false;
        this.isGradientCacheValid=false;
    }

    @Override
    public double getValue() {
        if (isValueCacheValid){
            return this.value;
        }
        this.value =  computeLoss()+penalty();
        this.isValueCacheValid = true;
        return this.value;
    }

    @Override
    public Vector getGradient() {
        if (isGradientCacheValid){
            return this.gradient;
        }
        updateGradient();
        this.isGradientCacheValid = true;
        return this.gradient;
    }


    private double computeLoss(){
        return IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .mapToDouble(i->Math.pow(linearRegression.predict(dataSet.getRow(i))-labels[i],2))
                .average().getAsDouble();
    }

    private double penalty(){
        return Math.pow(linearRegression.getWeights().getWeightsWithoutBias().norm(2),2)/(2*priorVariance);
    }

    private void updateGradient(){
        double[] residual = new double[dataSet.getNumDataPoints()];
        IntStream.range(0,dataSet.getNumDataPoints()).parallel()
                .forEach(i-> residual[i] = labels[i] - linearRegression.predict(dataSet.getRow(i)));
        updateGradientForBias(residual);
        IntStream.range(0,dataSet.getNumFeatures()).parallel()
                .forEach(j->updateGradient(j, residual));
    }

    private void updateGradientForBias(double[] residual){
        double g = MathUtil.arraySum(residual)*2*(-1)/dataSet.getNumDataPoints();
        gradient.set(0,g);
    }

    private void updateGradient(int featureIndex, double[] residual){
        Vector featureColumn = dataSet.getColumn(featureIndex);
        // sparse
        double sum = 0;
        for (Vector.Element nonzeros: featureColumn.nonZeroes()){
            int i = nonzeros.index();
            double v = nonzeros.get();
            sum += 2*residual[i]*(-v);
        }
        sum /= dataSet.getNumDataPoints();
        sum += linearRegression.getWeights().getWeightsWithoutBias().get(featureIndex)/priorVariance;
        gradient.set(featureIndex+1, sum);
    }
}
