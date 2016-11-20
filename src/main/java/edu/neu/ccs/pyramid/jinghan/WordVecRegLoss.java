package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by chengli on 11/20/16.
 */
public class WordVecRegLoss implements Optimizable.ByGradientValue{
    Vector wordScores;
    private DataSet doc2word;
    private double[] labels;
    private Vector gradient;


    public WordVecRegLoss(DataSet doc2word, double[] labels, Vector wordScores, Vector gradient) {
        this.doc2word = doc2word;
        this.labels = labels;
        this.wordScores = new DenseVector(wordScores);
        this.gradient = gradient;
    }

    @Override
    public Vector getParameters() {
        return wordScores;
    }

    @Override
    public void setParameters(Vector parameters) {
        this.wordScores = parameters;
    }

    @Override
    public double getValue() {
        return IntStream.range(0, doc2word.getNumDataPoints()).parallel()
                .mapToDouble(i->Math.pow(labels[i]-wordScores.dot(doc2word.getRow(i)),2))
                .average().getAsDouble();
    }

    @Override
    public Vector getGradient() {
        return gradient;
    }
}
