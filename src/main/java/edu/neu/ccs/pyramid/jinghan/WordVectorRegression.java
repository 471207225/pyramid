package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.IOException;

/**
 * Created by jinghanyang on 11/3/16.
 */
public class WordVectorRegression extends GradientBoosting implements Regressor {
    private static final long serialVersionUID = 1L;
    public Vector wordScores;
    public double bias;

//    static Vector wordScores;

    public WordVectorRegression(int numWords, double bias) {
        super(1);
        this.bias = bias;
        this.wordScores = new DenseVector(numWords);
    }
    /**
     *
     * @param vector document; elements are word proportions
     * @return
     */



    @Override
    public double predict(Vector vector) {
//        System.out.println("word scores");
//        System.out.println(wordScores.toString());
//        System.out.println("data point");
//        System.out.println(vector.toString());
        return vector.dot(wordScores);
    }

    public double predict_addBias(Vector vector){
        return vector.dot(wordScores) + bias;
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }

    private void writeObject(java.io.ObjectOutputStream out)
            throws IOException {
        double[] serializableWeights = new double[wordScores.size()];
        for (int i=0;i<serializableWeights.length;i++){
            serializableWeights[i] = wordScores.get(i);
        }
        out.writeObject(serializableWeights);

    }
    private void readObject(java.io.ObjectInputStream in)
            throws IOException, ClassNotFoundException{
        double[] serializableWeights = (double[])in.readObject();
        wordScores = new DenseVector(serializableWeights);
    }
}
