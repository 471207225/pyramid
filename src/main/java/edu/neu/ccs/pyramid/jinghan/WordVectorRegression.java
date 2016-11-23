package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.GradientBoosting;
import edu.neu.ccs.pyramid.regression.Regressor;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/**
 * Created by jinghanyang on 11/3/16.
 */
public class WordVectorRegression extends GradientBoosting implements Regressor {
    private static final long serialVersionUID = 1L;
    Vector wordScores;

    public WordVectorRegression(int numWords) {
        super(1);
        this.wordScores = new DenseVector(numWords);
    }
    /**
     *
     * @param vector document; elements are word proportions
     * @return
     */
    @Override
    public double predict(Vector vector) {

        return vector.dot(wordScores);
    }

    @Override
    public FeatureList getFeatureList() {
        return null;
    }
}
