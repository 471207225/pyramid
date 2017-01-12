package edu.neu.ccs.pyramid.optimization.gradient_boosting;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.GradientMatrix;
import edu.neu.ccs.pyramid.dataset.ScoreMatrix;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.RegressorFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import org.apache.commons.io.output.ByteArrayOutputStream;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Created by chengli on 10/1/15.
 */
public abstract class GBOptimizer implements Serializable{
    protected ScoreMatrix scoreMatrix;

    protected GradientBoosting boosting;
    protected RegressorFactory factory;
    protected DataSet dataSet;
    protected double[] weights;
    protected boolean isInitialized;
    protected double shrinkage = 1;
    public Regressor regressor_original = null;
    public double[] gradients;
    public static final long serialVersionUID = 1L;


    protected GBOptimizer(GradientBoosting boosting, DataSet dataSet, RegressorFactory factory, double[] weights) {
        this.boosting = boosting;
        this.factory = factory;
        this.dataSet = dataSet;
        this.weights = weights;
//        this.regressor = fitRegressor(0);
        boosting.featureList = dataSet.getFeatureList();
    }

    protected GBOptimizer(GradientBoosting boosting, DataSet dataSet,RegressorFactory factory){
        this(boosting, dataSet, factory, defaultWeights(dataSet.getNumDataPoints()));
    }


    /**
     * model specific initialization
     * should be called after constructor
     */
    public void initialize(){
        if (boosting.getEnsemble(0).getRegressors().size()==0){
            addPriors();
        }
        this.scoreMatrix = new ScoreMatrix(dataSet.getNumDataPoints(),boosting.getNumEnsembles());
        this.initStagedScores();
        initializeOthers();
        updateOthers();
        this.isInitialized = true;
    }

    protected abstract void addPriors();

    protected abstract double[] gradient(int ensembleIndex);

    /**
     * e.g. probability matrix
     */
    protected abstract void initializeOthers();

    protected Regressor fitRegressor(int ensembleIndex){
        double[] gradients = gradient(ensembleIndex);
        this.gradients = gradients;
        Regressor regressor = factory.fit(dataSet,gradients, weights);
        return regressor;
    }

    //todo make it more general
    protected void shrink(Regressor regressor, double[] searchDir){
        double learningRate = computeLearningRate(searchDir);
        System.out.println("tree learning rate"+learningRate);
        if (regressor instanceof RegressionTree){
            ((RegressionTree)regressor).shrink(learningRate);
        }
    }

    protected double computeLearningRate(double[] searchDir){
        return shrinkage;
    }

    protected void updateStagedScore(Regressor regressor, int ensembleIndex,
                                   int dataIndex){
        Vector vector = dataSet.getRow(dataIndex);
        double score = regressor.predict(vector);
        this.scoreMatrix.increment(dataIndex,ensembleIndex,score);
    }

    protected void updateStagedScores(Regressor regressor, int ensembleIndex){
        int numDataPoints = dataSet.getNumDataPoints();
        IntStream.range(0, numDataPoints).parallel()
                .forEach(dataIndex -> this.updateStagedScore(regressor,ensembleIndex,dataIndex));
    }

    public void iterate(){
        if (!isInitialized){
            throw new RuntimeException("GBOptimizer is not initialized");
        }
        for (int k=0;k<boosting.getNumEnsembles();k++){
            Regressor regressor = fitRegressor(k);


            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            ObjectOutputStream out = null;
            try {
                out = new ObjectOutputStream(bos);
                out.writeObject(regressor);

                //De-serialization of object
                ByteArrayInputStream bis = new ByteArrayInputStream(bos.toByteArray());
                ObjectInputStream in = new ObjectInputStream(bis);
                Regressor copied = (Regressor) in.readObject();
                this.regressor_original = copied;
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ClassNotFoundException e) {
                e.printStackTrace();
            }


//            System.out.println("regressor = "+regressor);
            double[] searchDir = regressor.predict(dataSet);
            shrink(regressor, searchDir);
            boosting.getEnsemble(k).add(regressor);
            updateStagedScores(regressor,k);
        }
        updateOthers();
    }

    public void iterate(int numIterations){
        for (int i=0;i<numIterations;i++){
            iterate();
        }
    }

    protected void initStagedScores(){
        for (int k=0;k<boosting.getNumEnsembles();k++){
            for (Regressor regressor: boosting.getEnsemble(k).getRegressors()){
                this.updateStagedScores(regressor,k);
            }
        }
    }

    /**
     * e.g. probability matrix
     */
    protected abstract void updateOthers();



    public void setShrinkage(double shrinkage) {
        this.shrinkage = shrinkage;
    }

    public RegressorFactory getRegressorFactory() {
        return factory;
    }

    protected static double[] defaultWeights(int numData){
        double[] weights = new double[numData];
        Arrays.fill(weights,1.0);
        return weights;
    }
}
