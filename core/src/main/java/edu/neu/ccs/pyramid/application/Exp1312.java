package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.calibration.*;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.*;
import edu.neu.ccs.pyramid.multilabel_classification.*;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;

import edu.neu.ccs.pyramid.multilabel_classification.cbm.PluginF1;
import edu.neu.ccs.pyramid.multilabel_classification.predictor.IndependentPredictor;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.Serializable;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * ranker for br lr
 */
public class Exp1312 {


    public static void main(Config config) throws Exception{


        MultiLabelClfDataSet train = TRECFormat.loadMultiLabelClfDataSet(config.getString("train"),DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet valid = TRECFormat.loadMultiLabelClfDataSet(config.getString("valid"),DataSetType.ML_CLF_SPARSE,true);
        MultiLabelClfDataSet cal = TRECFormat.loadMultiLabelClfDataSet(config.getString("cal"),DataSetType.ML_CLF_SPARSE,true);

        MultiLabelClfDataSet test = TRECFormat.loadMultiLabelClfDataSet(config.getString("test"),DataSetType.ML_CLF_SPARSE,true);
        CBM cbm = (CBM) Serialization.deserialize(config.getString("cbm"));
        cbm.setAllowEmpty(config.getBoolean("allowEmpty"));



        MultiLabelClfDataSet labelCalData = cal;

        MultiLabelClfDataSet setCalData = cal;

        if (config.getBoolean("splitCalibrationData")){
            List<Integer> labelCalIndices = IntStream.range(0, cal.getNumDataPoints()).filter(i->i%2==0).boxed().collect(Collectors.toList());
            List<Integer> setCalIndices = IntStream.range(0, cal.getNumDataPoints()).filter(i->i%2==1).boxed().collect(Collectors.toList());
            labelCalData = DataSetUtil.sampleData(cal, labelCalIndices);
            setCalData = DataSetUtil.sampleData(cal, setCalIndices);
        }


        List<MultiLabel> support = DataSetUtil.gatherMultiLabels(train);
        LabelCalibrator labelCalibrator = null;
        switch (config.getString("labelCalibrator")){
            case "isotonic":
                labelCalibrator = new IsoLabelCalibrator(cbm, labelCalData);
                break;
            case "none":
                labelCalibrator = new IdentityLabelCalibrator();
                break;
                //todo add platt
        }

        PredictionVectorizer predictionVectorizer = PredictionVectorizer.newBuilder()
                .brProb(config.getBoolean("brProb"))
                .setPrior(config.getBoolean("setPrior"))
                .cardPrior(config.getBoolean("cardPrior"))
                .card(config.getBoolean("card"))
                .pairPrior(config.getBoolean("pairPrior"))
                .encodeLabel(config.getBoolean("encodeLabel"))
                .f1Prior(config.getBoolean("f1Prior"))
                .cbmProb(config.getBoolean("cbmProb"))
                .implication(config.getBoolean("implication"))
                .labelProbs(config.getBoolean("labelProbs"))
                .position(config.getBoolean("position"))
                .logScale(config.getBoolean("logScale"))
                .hierarchy(config.getBoolean("hierarchy"))
                .weight(config.getString("weight"))
                .cdf(config.getBoolean("cdf"))
                .build(train,labelCalibrator);

        if (config.getBoolean("hierarchy")){
            Hierarchy hierarchy = (Hierarchy)Serialization.deserialize(Paths.get(config.getString("train"),"hierarchy.ser"));
            predictionVectorizer.setHierarchyRelation(hierarchy);
        }


        PredictionVectorizer.TrainData weightedCalibratorTrainData = predictionVectorizer.createCaliTrainingData(setCalData,cbm,  config.getInt("numTrainCandidates"));
        RegDataSet calibratorTrainData = weightedCalibratorTrainData.regDataSet;
        double[] weights = weightedCalibratorTrainData.instanceWeights;

        VectorCalibrator setCalibrator = null;

        switch (config.getString("setCalibrator")){
            case "trivial":
                setCalibrator = new VectorTrivialCalibrator(calibratorTrainData);
                break;
            case "cardinality_isotonic":
                setCalibrator = new VectorCardIsoSetCalibrator(calibratorTrainData, 1, 3);
                break;
            case "GB":
                RerankerTrainer rerankerTrainer = RerankerTrainer.newBuilder()
                        .numCandidates(config.getInt("numPredictCandidates"))
                        .monotonic(config.getBoolean("monotonic"))
                        .numIterations(config.getInt("numIterations"))
                        .numLeaves(config.getInt("numLeaves"))
                        .shrinkage(config.getDouble("shrinkage"))
                        .build();
                setCalibrator = rerankerTrainer.train(calibratorTrainData, weights,cbm,predictionVectorizer);

                break;

            case "lambdaMART":
                RerankerTrainer rerankerTrainer1 = RerankerTrainer.newBuilder()
                        .numCandidates(config.getInt("numCandidates"))
                        .monotonic(config.getBoolean("monotonic"))
                        .numIterations(config.getInt("numIterations"))
                        .numLeaves(config.getInt("numLeaves"))
                        .shrinkage(config.getDouble("shrinkage"))
                        .build();
                setCalibrator = rerankerTrainer1.trainLambdaMART(weightedCalibratorTrainData, cbm,predictionVectorizer, config.getInt("ndcgTruncationLevel"));
                break;
            case "isotonic":
                setCalibrator = new VectorIsoSetCalibrator(calibratorTrainData,1);
                break;
            case "none":
                setCalibrator = new VectorIdentityCalibrator(1);
                break;
            default:
                throw new IllegalArgumentException("illegal setCalibrator, given="+config.getString("setCalibrator"));
        }

//        System.out.println(setCalibrator);



//        System.out.println("performance on calibration set");
//
//        showPerformance(config, cal, cbm, labelCalibrator, setCalibrator, predictionVectorizer, support,"cal");
        System.out.println("=================================");
        System.out.println("on validation set");

        showPerformance(config, valid, cbm, labelCalibrator, setCalibrator, predictionVectorizer, support,"valid");
        System.out.println("=================================");
        System.out.println("on test set");

        showPerformance(config, test, cbm, labelCalibrator, setCalibrator, predictionVectorizer, support,"test");
    }

    private static CaliRes eval(List<PredictionVectorizer.Instance> predictions, VectorCalibrator calibrator){
        double mse = CalibrationEval.mse(generateStream(predictions,calibrator));
        double ace = CalibrationEval.absoluteError(generateStream(predictions,calibrator),10);
        double sharpness = CalibrationEval.sharpness(generateStream(predictions,calibrator),10);
        System.out.println("calibration performance");
        System.out.println("mse="+mse);
//        System.out.println("absolute calibration error="+ace);
        System.out.println("alignment error="+CalibrationEval.squareError(generateStream(predictions,calibrator),10));
        System.out.println("sharpness="+sharpness);
        System.out.println("uncertainty="+CalibrationEval.variance(generateStream(predictions,calibrator)));
        System.out.println(Displayer.displayCalibrationResult(generateStream(predictions,calibrator)));
        CaliRes caliRes = new CaliRes();
        caliRes.mse = mse;
        caliRes.ace= ace;
        caliRes.sharpness = sharpness;

//        List<Pair<Double,Double>> pairs = generateStream(predictions,calibrator).map(p-> new Pair<>(p.getFirst(),1.0*p.getSecond())).collect(Collectors.toList());
//        Bucketer.Result predictRes = Bucketer.groupWithEqualSize(pairs,100);

//        System.out.println("equal sized buckets");
//        System.out.println("x=");
//        System.out.println(Arrays.toString(predictRes.getAverageX()));
//        System.out.println("y=");
//        System.out.println(Arrays.toString(predictRes.getAverageY()));

        return caliRes;
    }

    private static Stream<Pair<Double,Integer>> generateStream(List<PredictionVectorizer.Instance> predictions, VectorCalibrator vectorCalibrator){
        return predictions.stream()
                .parallel().map(pred->new Pair<>(vectorCalibrator.calibrate(pred.vector),(int)pred.correctness));
    }


    public static class CaliRes implements Serializable {
        public static final long serialVersionUID = 446782166720638575L;
        public double mse;
        public double ace;
        public double sharpness;
    }
    
    private static void showPerformance(Config config, MultiLabelClfDataSet dataset, CBM cbm, LabelCalibrator labelCalibrator,
                                        VectorCalibrator setCalibrator, PredictionVectorizer predictionVectorizer,
                                        List<MultiLabel> support, String dataName) throws Exception{
        
        
        LabelCalibrator identity = new IdentityLabelCalibrator();
        

        MultiLabelClassifier classifier = null;
        switch (config.getString("predict.mode")){
            case "independent_original":
                classifier = new IndependentPredictor(cbm,identity);
                break;
            case "independent":
                classifier = new IndependentPredictor(cbm,labelCalibrator);
                break;
            case "support":
                classifier = new edu.neu.ccs.pyramid.multilabel_classification.predictor.SupportPredictor(cbm, labelCalibrator, support);
                break;
            case "support_original":
                classifier = new edu.neu.ccs.pyramid.multilabel_classification.predictor.SupportPredictor(cbm, identity, support);
                break;

            case "rerank":
                classifier = (Reranker)setCalibrator;
                break;

            default:
                throw new IllegalArgumentException("illegal predict.mode");
        }
        MultiLabel[] predictions = classifier.predict(dataset);

        MLMeasures mlMeasures =new MLMeasures(dataset.getNumClasses(),dataset.getMultiLabels(), predictions);
        System.out.println("classification performance");
        System.out.println(mlMeasures);

        Paths.get(config.getString("output")).toFile().mkdirs();
        File accResult = Paths.get(config.getString("output"),dataName+"_accuracy.txt").toFile();
        FileUtils.writeStringToFile(accResult,""+mlMeasures.getInstanceAverage().getAccuracy());

        File f1Result = Paths.get(config.getString("output"),dataName+"_f1.txt").toFile();
        FileUtils.writeStringToFile(f1Result,""+mlMeasures.getInstanceAverage().getF1());
//        PluginF1 pluginF1 = new PluginF1(cbm, support);
//        System.out.println("performance with uncalibrated probability, support GFM");
//        System.out.println(new MLMeasures(pluginF1, dataset));
//
//        if (config.getString("setCalibrator").equals("reranker")){
//            RerankerGFM rerankerGFM = new RerankerGFM((Reranker) setCalibrator);
//            System.out.println("performance with reranker GFM");
//            System.out.println(new MLMeasures(rerankerGFM, dataset));
//        }


        if (true) {

            List<PredictionVectorizer.Instance> instances = IntStream.range(0, dataset.getNumDataPoints()).parallel()
                    .boxed().map(i -> predictionVectorizer.createInstance(cbm, dataset.getRow(i),predictions[i],dataset.getMultiLabels()[i]))
                    .collect(Collectors.toList());

            eval(instances, setCalibrator);
        }





//        Hierarchy hierarchy = (Hierarchy)Serialization.deserialize(Paths.get(config.getString("train"),"hierarchy.ser"));
//        System.out.println("mistakes due to hierarchy="+Arrays.stream(predictions).filter(p->!hierarchy.satisfy(p)).count());

        
    }

}
