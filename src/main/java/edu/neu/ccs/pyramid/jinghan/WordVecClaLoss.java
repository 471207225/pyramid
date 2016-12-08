package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.optimization.Optimizable;
import edu.neu.ccs.pyramid.dataset.DataSet;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 12/6/16.
 */
public class WordVecClaLoss implements Optimizable.ByGradientValue{
    Vector wordScores;
    public DataSet doc2word;
    public int [] labels;
    public Vector gradient;

    public WordVecClaLoss(DataSet doc2word, int [] labels, Vector wordScores, Vector gradient) {
        this.doc2word = doc2word;
        this.wordScores = wordScores;
        this.wordScores = new DenseVector(wordScores);
        this.gradient = gradient;
        this.labels = labels;
    }

    @Override
    public Vector getGradient() {
        return gradient;
    }

    @Override
    public double getValue() {
        return IntStream.range(0, doc2word.getNumDataPoints()).parallel()
                .mapToDouble(i->(-labels[i]*Math.log(1/(1+Math.exp(-wordScores.dot(doc2word.getRow(i)))))
                +(labels[i]-1)*Math.log(1-1/(1+Math.exp(-wordScores.dot(doc2word.getRow(i))))))).sum();
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
