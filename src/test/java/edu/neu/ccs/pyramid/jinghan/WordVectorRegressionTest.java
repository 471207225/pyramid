package edu.neu.ccs.pyramid.jinghan;



import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import static org.junit.Assert.*;

/**
 * Created by jinghanyang on 11/3/16.
 */
public class WordVectorRegressionTest {
    public static void main(String[] args) {
        Vector wordScores = new DenseVector(3);
        wordScores.set(0, 0.5);
        wordScores.set(1, -0.6);
        wordScores.set(2, 0.7);


        Vector vector = new DenseVector(3);
        vector.set(0,2);
        vector.set(1,1);

//        WordVectorRegression wordVectorRegression = new WordVectorRegression(3);
//        System.out.println(wordVectorRegression.predict(vector));
    }

}