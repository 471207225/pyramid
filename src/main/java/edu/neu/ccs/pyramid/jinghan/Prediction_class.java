package edu.neu.ccs.pyramid.jinghan;

import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 12/5/16.
 */
public class Prediction_class {
    public double[] prob;
    public double[] predictions;
    public double threshold;
    public int numDocs;
    public int[] classPred;

    public Prediction_class(double[] predictions, double threshold) {

        this.numDocs = predictions.length;
        this.prob = new double[numDocs];
        this.threshold = threshold;
        this.predictions = predictions;
        this.classPred = new int[predictions.length];
        updateDocProb();
        updateDocClass();
    }

    public void updateDocProb(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocProb);
    }

    public void updateDocProb(int docIndex){
        double donominator = Math.exp(-predictions[docIndex])+1;
        this.prob[docIndex] = 1/donominator;
    }


    public void updateDocClass(int docIndex){
        if(prob[docIndex] >= threshold){
            this.classPred[docIndex] = 1;
        }
    }

    public void updateDocClass(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocClass);
    }

}

