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
        this.threshold = threshold;
        this.predictions = predictions;
        this.prob = new double[numDocs];
        this.classPred = new int[predictions.length];


//        System.out.println("prediction is");
//        for(int i=0; i<10; i++){
//            System.out.println(predictions[i]);
//        }


        updateProb();
        updateDocClass();

    }

    public void updateProb(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateProb);
    }

    public void updateProb(int docIndex){
        this.prob[docIndex] = 1/(1+Math.exp(-predictions[docIndex]));
    }


    public void updateDocClass(int docIndex){
        if(prob[docIndex] >= threshold){
            this.classPred[docIndex] = 1;
        }else {
            this.classPred[docIndex] = 0;
        }
    }

    public void updateDocClass(){
        IntStream.range(0, numDocs).parallel().forEach(this::updateDocClass);
    }

}

