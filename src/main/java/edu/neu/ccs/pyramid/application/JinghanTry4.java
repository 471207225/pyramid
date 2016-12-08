package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.Prediction_class;
import edu.neu.ccs.pyramid.jinghan.WordVectorClaOptimizer;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegression;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.DenseVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.regex.Pattern;

/**
 * Created by jinghanyang on 12/4/16.
 */
public class JinghanTry4 {

    public static void main(String[] args) throws Exception{
        if (args.length!=1){
            throw new IllegalArgumentException("Please specify a properties file");
        }
        Config config = new Config(args[0]);
        System.out.println(config);
        if (config.getBoolean("train")){
            train(config);
        }
    }

    public static void train(Config config) throws Exception {
        DataSet trainWord2vec = loadword2vecMatrix(config.getString("input.trainWord2vec"));
        DataSet testWord2vec = loadword2vecMatrix(config.getString("input.testWord2vec"));

        DataSet train_docFeatures = loadDocMatrix(config.getString("input.train_docFeatures"), config);
        DataSet test_docFeatures = loadDocMatrix(config.getString("input.test_docFeatures"), config);

        int [] train_docLabels = loadlabels(config.getString("input.train_docLabels"), config);
        int [] test_docLabels = loadlabels(config.getString("input.test_docLabels"), config);

        int numWords_train = trainWord2vec.getNumDataPoints();

        double[] train_weights;
        if (config.getBoolean("useWeights")) {
            train_weights = loadweights(config.getString("input.trainWeights"), config);
        } else {
            train_weights = new double[numWords_train];
            Arrays.fill(train_weights, 1);
        }

        String output = config.getString("output.folder");
        String modelName = "models";
        File path = Paths.get(output, modelName).toFile();
        path.mkdirs();

        WordVectorRegression wordVectorRegression = loadModel(config, trainWord2vec).getFirst();
        int interationStart = loadModel(config, trainWord2vec).getSecond();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves")).setMinDataPerLeaf(0);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        WordVectorClaOptimizer optimizer = new WordVectorClaOptimizer(wordVectorRegression, regTreeFactory, train_docFeatures, trainWord2vec, train_docLabels, train_weights);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();
        double threshold = config.getDouble("threshold");
        int progressInterval = config.getInt("train.showProgress.interval");
        int numIterations = config.getInt("train.numIterations");

        int saveModelInterval = config.getInt("saveModelInterval");

        for (int i = 1; i <= numIterations; i++) {
            int iteration_num = i + interationStart;
            System.out.println("iteration" + iteration_num);
            optimizer.iterate();
            int size = wordVectorRegression.getEnsemble(0).getRegressors().size();
//            // print tree
//            System.out.println(wordVectorRegression.getEnsemble(0).get(size-1));
            if (config.getBoolean("train.showTrainProgress") && (i % progressInterval == 0 || i == numIterations)) {
                double[] train_prediction_score = wordVectorRegression.predict(train_docFeatures);
                System.out.println("training prediction score is ");
                System.out.println(Arrays.toString(train_prediction_score));
                Prediction_class prediction_class_train = new Prediction_class(train_prediction_score, threshold);
//                System.out.println("training prediction for probability is ");
//                System.out.println(Arrays.toString(prediction_class_train.prob));
                int[] train_prediciton_class = prediction_class_train.classPred;
//                int[] train_docLabels_int = dToint(train_docLabels);
                System.out.println("training accuracy = " + Accuracy.accuracy(train_docLabels, train_prediciton_class));
            }
            if (config.getBoolean("train.showTestProgress") && (i % progressInterval == 0 || i == numIterations)) {
                LinearRegression testLinearReg = getLinearReg(wordVectorRegression, testWord2vec);
                double[] test_prediction_score = testLinearReg.predict(test_docFeatures);
//                System.out.println("testing prediction score");
//                System.out.println(Arrays.toString(test_prediction_score));
                Prediction_class prediction_class_test = new Prediction_class(test_prediction_score, threshold);
                int[] test_prediction_class = prediction_class_test.classPred;
//                int[] test_docLabels_int = dToint(test_docLabels);
                System.out.println("testing accuaray = " + Accuracy.accuracy(test_docLabels, test_prediction_class));
            }

            if (i % saveModelInterval == 0) {
                File serializeModel = new File(path, "iter." + wordVectorRegression.getEnsemble(0).getRegressors().size() + ".model");
                Serialization.serialize(wordVectorRegression, serializeModel);
            }

        }

//
//        File reportFile_train = new File(output, "train_predictions.txt");
//        report_train(prediction_class, train_docFeatures, reportFile_train);
//        System.out.println("predictions on the training set are written to"+reportFile_train.getAbsolutePath());
//
//        File reportFile_test = new File(output, "test_predictions.txt");
//        LinearRegression linearRegression = getLinearReg(wordVectorRegression, trainWord2vec);
//        report_test(linearRegression, test_docFeatures, reportFile_test);
//        System.out.println("predictions on the training set are written to"+reportFile_test.getAbsolutePath());
    }

    public static DataSet loadword2vecMatrix(String path) throws Exception{
        int numData = (int) Files.lines(Paths.get(path)).count();
        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;

        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(numData).numFeatures(numFeatures).build();
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int dataIndex = 0;
            while ((line = br.readLine())!=null){
                String [] lineEle = line.split(" ");
                for (int j=0; j<denseDataSet.getNumFeatures(); j++){
                    denseDataSet.setFeatureValue(dataIndex, j, Double.parseDouble(lineEle[j]));
                }
                dataIndex +=1;
            }
        }
        return denseDataSet;
    }

    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
        int numData = (int) Files.lines(Paths.get(path)).count() -1;
        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;
        DataSet dataSet = new SparseDataSet(numData, numFeatures, false);
        try (BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int dataIndex =0;
            while ((line = br.readLine())!=null){
                if (dataIndex == 0){
                    dataIndex += 1;
                    String[] featureNames = line.split(" ");
                    FeatureList featureList = new FeatureList();
                    for (String featureName: featureNames){
                        Feature feature = new Feature();
                        feature.setName(featureName);
                        featureList.add(feature);
                    }
                    dataSet.setFeatureList(featureList);
                    continue;
                }
                String [] lineEle = line.split(" ");
                for (int j=0; j<dataSet.getNumFeatures(); j++){
                    dataSet.setFeatureValue(dataIndex-1, j, Double.valueOf(lineEle[j]));
                }
                dataIndex += 1;
            }
        }
        return dataSet;
    }


    public static double[] loadweights(String path, Config config) throws Exception{
        int numFeatures = (int) Files.lines(Paths.get(path)).count();
        double [] weights = new double[numFeatures];
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int lineIndex = 0;
            while ((line = br.readLine()) != null){
                weights[lineIndex] = Double.parseDouble(line);
                lineIndex += 1;
            }
        }
        return weights;
    }

    public static int[] loadlabels(String path, Config config) throws Exception{
        int numData = (int) Files.lines(Paths.get(path)).count();
        int [] labels = new int[numData];
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int lineIndex = 0;
            while((line = br.readLine())!=null){
                labels[lineIndex] = Integer.parseInt(line);
//                labels[lineIndex] = Double.parseDouble(line);
                lineIndex += 1;
            }
        }
        return labels;
    }

    public static Pair<WordVectorRegression, Integer> loadModel(Config config, DataSet trainWord2Vec) throws Exception{
        int numWords = trainWord2Vec.getNumDataPoints();
        int completedIterations = 0;
        String output = config.getString("output.folder");
        String modelName = "models";
        File folder = Paths.get(output, modelName).toFile();
        File[] modelFiles = folder.listFiles((dir,name) -> name.startsWith("iter.") && (name.endsWith(".model")));
        File lastFile = null;
        int lastIter = -1;
        for (File file: modelFiles){
            String[] split = file.getName().split(Pattern.quote("."));
            int iter = Integer.parseInt(split[1]);
            if (iter>lastIter){
                lastIter = iter;
                lastFile = file;
                completedIterations = lastIter;
            }
        }

        WordVectorRegression wordVectorRegression;

        if (lastIter==-1){
            wordVectorRegression = new WordVectorRegression(numWords);
        }else {
            wordVectorRegression = (WordVectorRegression) Serialization.deserialize(lastFile);
        }

        System.out.println("wordVectorRegression loaded with " + completedIterations+ " iterations completed");
        return new Pair<> (wordVectorRegression,completedIterations);
    }

    private static LinearRegression getLinearReg (WordVectorRegression wordVectorRegression, DataSet word2Vec){
        int numWords = word2Vec.getNumDataPoints();
        org.apache.mahout.math.Vector vector = new DenseVector(numWords+1);
        for (int j=0;j<numWords;j++){
            vector.set(j+1, wordVectorRegression.score(word2Vec.getRow(j),0));
        }
        LinearRegression linearRegression = new LinearRegression(numWords, vector);
        return linearRegression;
    }

    public static int[] dToint(double[] doubleArray){
        int[] intArray = new int[doubleArray.length];
        for(int i=0; i<intArray.length; i++){
            intArray[i] = (int) doubleArray[i];
        }
        return intArray;
    }

    public static void report_train(WordVectorRegression wordVectorRegression, DataSet dataSet, File reportFile) throws IOException {
        double [] prediction = wordVectorRegression.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }

    public static void report_test(LinearRegression linearRegression,DataSet dataSet, File reportFile) throws IOException {
        double [] prediction = linearRegression.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }

}
