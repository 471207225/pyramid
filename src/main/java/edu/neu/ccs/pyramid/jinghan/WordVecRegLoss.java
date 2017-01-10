package edu.neu.ccs.pyramid.jinghan;

import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.optimization.Optimizable;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 1/4/16.
 */
public class WordVecRegLoss implements Optimizable.ByGradientValue{
    public Vector wordScores;
    private DataSet doc2word;
    private double[] labels;
    private Vector gradient;
    public double bias;
    public double lam;


    public WordVecRegLoss(DataSet doc2word, double[] labels, Vector wordScores, Vector gradient, double bias, double lam) {
        this.doc2word = doc2word;
        this.labels = labels;
        this.wordScores = new DenseVector(wordScores);
        this.gradient = gradient;
        this.bias = bias;
        this.lam = lam;
    }

    @Override
    public Vector getParameters() {
        return wordScores;
    }

    @Override
    public void setParameters(Vector parameters) {
        this.wordScores = parameters;
    }


    public double getValue() {
//        return IntStream.range(0, doc2word.getNumDataPoints()).parallel()
//                .mapToDouble(i->Math.pow(labels[i]-wordScores.dot(doc2word.getRow(i)),2))
//                .average().getAsDouble();
        double part1 = IntStream.range(0, doc2word.getNumDataPoints()).parallel()
                .mapToDouble(i->Math.pow(labels[i]-wordScores.dot(doc2word.getRow(i)),2)).sum()/2;
        double part2 = IntStream.range(0, doc2word.getNumFeatures()).parallel().mapToDouble(i->Math.pow(wordScores.get(i),2)).sum();
        part2 = part2*lam/2;
        return part1+ part2;
    }

    @Override
    public Vector getGradient() {
        return gradient;
    }
}
