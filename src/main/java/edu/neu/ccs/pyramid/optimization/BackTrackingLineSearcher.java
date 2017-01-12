package edu.neu.ccs.pyramid.optimization;

import edu.neu.ccs.pyramid.jinghan.WordVecClaLoss;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;

import java.util.stream.IntStream;

/**
 * minimize the function along the line
 * Numerical Optimization, Second Edition, Jorge Nocedal Stephen J. Wright
 * algorithm: page 37
 * default parameters: page 142
 * Created by chengli on 12/9/14.
 */
public class BackTrackingLineSearcher {
    private static final Logger logger = LogManager.getLogger();
    private Optimizable.ByGradientValue function;
    private double initialStepLength = 1;
    private double shrinkage = 0.5;
    private double c = 1e-4;

    public BackTrackingLineSearcher(Optimizable.ByGradientValue function) {
        this.function = function;
    }

    /**
     * move to a new position along the direction
     */
    public MoveInfo moveAlongDirection(Vector searchDirection){
        Vector localSearchDir;
        if (logger.isDebugEnabled()){
            logger.debug("start line search");
            // don't want to show too much; only show it on small problems
            if (searchDirection.size()<100){
                logger.debug("direction="+searchDirection);
            }
        }
        MoveInfo moveInfo = new MoveInfo();

        double stepLength = initialStepLength;
        double value = function.getValue();
        moveInfo.setOldValue(value);
        Vector gradient = function.getGradient();
        double product = gradient.dot(searchDirection);
        if (product < 0){
            localSearchDir = searchDirection;
        } else {
            if (logger.isWarnEnabled()) {
                logger.warn("Bad search direction! Use negative gradient instead. Product of gradient and search direction = " + product);
            }

            localSearchDir = gradient.times(-1);
        }

        Vector initialPosition;
        // keep a copy of initial parameters
        if (function.getParameters().isDense()){
            initialPosition = new DenseVector(function.getParameters());
        } else {
            initialPosition = new RandomAccessSparseVector(function.getParameters());
        }
        while(true){

            Vector step = localSearchDir.times(stepLength);
            Vector target = initialPosition.plus(step);
            function.setParameters(target);

            double targetValue = function.getValue();
            if (logger.isDebugEnabled()){
                logger.debug("step length = "+stepLength+", target value = "+targetValue);
//                logger.debug("requirement = "+(value + c*stepLength*product));
            }
            // todo: if equal ok?
            if ((targetValue <= value + c*stepLength*product && value<Double.POSITIVE_INFINITY )|| stepLength==0){
                moveInfo.setStep(step);
                moveInfo.setStepLength(stepLength);
                moveInfo.setNewValue(targetValue);
                break;
            }
            stepLength *= shrinkage;
        }
        if (logger.isDebugEnabled()){
            logger.debug("line search done. "+moveInfo);
        }
        return moveInfo;
    }

    public double computeLearningRate(Vector searchDirection) {
        Vector localSearchDir;
        int numWords = searchDirection.size();
        Vector searchDirectionWithBias = new DenseVector(numWords+1);
        IntStream.range(0, numWords).parallel().forEach(i->searchDirectionWithBias.set(i+1, searchDirection.get(i)));
        if (logger.isDebugEnabled()) {
            logger.debug("start line search");
            // don't want to show too much; only show it on small problems
            if (searchDirectionWithBias.size() < 100) {
                logger.debug("direction=" + searchDirectionWithBias);
            }
        }
        MoveInfo moveInfo = new MoveInfo();

        double stepLength = initialStepLength;

        System.out.println("starting point of one iteration");

        double value = function.getValue();
        System.out.println("loss = " + value);
        moveInfo.setOldValue(value);

        Vector gradient = function.getGradient();
        searchDirectionWithBias.set(0,-gradient.get(0));

        System.out.println("check whole gradient here");
        for (int i=0; i<5; i++){
            System.out.println(gradient.get(i));
        }


        double product = gradient.dot(searchDirectionWithBias);
        if (product < 0) {
            localSearchDir = searchDirectionWithBias;
        } else {
            if (logger.isWarnEnabled()) {
                logger.warn("Bad search direction! Use negative gradient instead. Product of gradient and search direction = " + product);
            }

            localSearchDir = gradient.times(-1);
        }


        Vector initialPosition;
        // keep a copy of initial parameters
        initialPosition = new DenseVector(function.getParameters());
        System.out.println("First, check initial position");
        for (int i=0; i<5; i++){

            System.out.println(initialPosition.get(i));

        }


        int ct = 0;

        while (true) {
            ct = ct+1;
            Vector step = localSearchDir.times(stepLength);
            System.out.println("step length " + stepLength);

            System.out.println("localSearchDir");
            for (int i=0; i<5; i++){
                System.out.println(localSearchDir.get(i));
            }

            Vector target = initialPosition.plus(step);
            System.out.println("length of initial postion " + initialPosition.size());

            System.out.println("step");
            for (int i=0; i<5; i++){
                System.out.println(step.get(i));
            }

            System.out.println("initial position");
            for (int i=0; i<5; i++){
                System.out.println(initialPosition.get(i));
            }

            System.out.println("target");
            for (int i=0; i<5; i++){
                System.out.println(target.get(i));
            }










            function.setParameters(target);

            double targetValue = function.getValue();
            if (logger.isDebugEnabled()) {
                logger.debug("step length = " + stepLength + ", target value = " + targetValue);
//                logger.debug("requirement = "+(value + c*stepLength*product));
            }

            // todo: if equal ok?
            if ((targetValue <= value + c * stepLength * product && value < Double.POSITIVE_INFINITY) || stepLength == 0) {
                moveInfo.setStep(step);
                moveInfo.setStepLength(stepLength);
                moveInfo.setNewValue(targetValue);
                break;
            }
            stepLength *= shrinkage;
        }
        if (logger.isDebugEnabled()) {
            logger.debug("line search done. " + moveInfo);
        }

        //stay at initial position
        function.setParameters(initialPosition);
//
        return moveInfo.getStepLength();
    }

    public void setInitialStepLength(double initialStepLength) {
        this.initialStepLength = initialStepLength;
    }

    public void setShrinkage(double shrinkage) {
        this.shrinkage = shrinkage;
    }

    public void setC(double c) {
        this.c = c;
    }

    public static class MoveInfo{
        private double oldValue;
        private double newValue;
        private Vector step;
        private double stepLength;

        public Vector getStep() {
            return step;
        }

        public void setStep(Vector step) {
            this.step = step;
        }

        public double getStepLength() {
            return stepLength;
        }

        public void setStepLength(double stepLength) {
            this.stepLength = stepLength;
        }

        public double getOldValue() {
            return oldValue;
        }

        public void setOldValue(double oldValue) {
            this.oldValue = oldValue;
        }

        public double getNewValue() {
            return newValue;
        }

        public void setNewValue(double newValue) {
            this.newValue = newValue;
        }

        @Override
        public String toString() {
            final StringBuilder sb = new StringBuilder();
            sb.append("oldValue=").append(oldValue);
            sb.append(", newValue=").append(newValue);
            sb.append(", stepLength=").append(stepLength);
            return sb.toString();
        }
    }
}
