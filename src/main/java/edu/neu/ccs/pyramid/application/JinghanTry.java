package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegOptimizer;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegression;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.stanford.nlp.patterns.Data;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.elasticsearch.river.RiverIndexName;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

import static edu.neu.ccs.pyramid.application.App2.report;


/**
 * Created by jinghanyang on 11/6/16.
 */
public class JinghanTry {


    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");

        }
        Config config = new Config(args[0]);
        System.out.println(config);
        if (config.getBoolean("train")) {
            train(config);
        }
    }

    public static void train(Config config) throws Exception{

        DataSet trainWord2vec = loadword2vecMatrix(config.getString("input.trainWord2vec"));
        DataSet testWord2vec = loadword2vecMatrix(config.getString("input.testWord2vec"));


        DataSet train_docFeatures = loadDocMatrix(config.getString("input.train_docFeatures"), config);
        DataSet test_docFeatures = loadDocMatrix(config.getString("input.test_docFeatures"), config);


        double [] train_docLabels = loadlabels(config.getString("input.train_docLabels"), config);
        double [] test_docLabels = loadlabels(config.getString("input.test_docLabels"), config);

        int numWords_train = trainWord2vec.getNumDataPoints();

        double [] train_weights;
        if (config.getBoolean("useWeights")){
            train_weights = loadweights(config.getString("input.trainWeights"), config);
        } else {
            train_weights = new double[numWords_train];
            Arrays.fill(train_weights, 1);
        }

        String output = config.getString("output.folder");
        String modelName = "models";
        File path = Paths.get(output, modelName).toFile();
        path.mkdirs();
        
        WordVectorRegression wordVectorRegression = loadModel(config, trainWord2vec);
        // it is essential to set mindataperleave = 0
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves")).setMinDataPerLeaf(0);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        WordVectorRegOptimizer optimizer = new WordVectorRegOptimizer(wordVectorRegression, regTreeFactory, train_docFeatures, trainWord2vec, train_docLabels, train_weights);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();        
        int progressInterval = config.getInt("train.showProgress.interval");
        
        int numIterations = config.getInt("train.numIterations");



        int saveModelInterval = config.getInt("saveModelInterval");

        for(int i=1; i<=numIterations;i++){
            System.out.println("iteration"+i);
//            optimizer.setShrinkage(config.getDouble("train.shrinkage")/numIterations);
            optimizer.iterate();
            if (config.getBoolean("train.showTrainProgress")&&(i%progressInterval==0 || i== numIterations)){

                double [] train_prediction = wordVectorRegression.predict(train_docFeatures);
//                System.out.println("training prediction is");
//                System.out.println(Arrays.toString(train_prediction));

                System.out.println("train RMSE = " + RMSE.rmse(train_docLabels, train_prediction));
            }
            if (config.getBoolean("train.showTestProgress")&&(i%progressInterval==0 || i==numIterations)){
                LinearRegression testLinearReg = getLinearReg(wordVectorRegression, testWord2vec, config);
                double [] test_prediction = testLinearReg.predict(test_docFeatures);
                System.out.println("test RMSE = " + RMSE.rmse(test_docLabels,test_prediction));
                System.out.println("*******************");
            }

            if (i%saveModelInterval==0){
                File serializeModel = new File(path,  "iter." + wordVectorRegression.getEnsemble(0).getRegressors().size() + ".model");
                Serialization.serialize(wordVectorRegression, serializeModel);
            }
        }


        File reportFile_train = new File(output, "train_predictions.txt");
        report_train(wordVectorRegression, train_docFeatures, reportFile_train);
        System.out.println("predictions on the training set are written to"+reportFile_train.getAbsolutePath());

        File reportFile_test = new File(output, "test_predictions.txt");
        LinearRegression linearRegression = getLinearReg(wordVectorRegression, trainWord2vec, config);
        report_test(linearRegression, test_docFeatures, reportFile_test);
        System.out.println("predictions on the training set are written to"+reportFile_test.getAbsolutePath());
    }


    public static DataSet loadword2vecMatrix(String path) throws Exception{

        int numData = (int) Files.lines(Paths.get(path)).count();
        int numfeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;

        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(numData).numFeatures(numfeatures).build();
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int dataIndex = 0;
            while ((line = br.readLine()) !=null){
                String [] lineEle = line.split(" ");
                for (int j=0; j<denseDataSet.getNumFeatures();j++){
                    denseDataSet.setFeatureValue(dataIndex,j,Double.parseDouble(lineEle[j]));
                }
                dataIndex += 1;
            }
        }

        return denseDataSet;
    }

    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
        int numData = (int) Files.lines(Paths.get(path)).count()-1;
        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;
        DataSet dataSet = new SparseDataSet(numData, numFeatures, false);
        try (BufferedReader br = new BufferedReader(new FileReader(path))
        ) {
            String line = null;
            int dataIndex = 0;
            while ((line = br.readLine()) != null) {

                if(dataIndex == 0){
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

                for (int j=0;j<dataSet.getNumFeatures();j++){
                    try {
                        dataSet.setFeatureValue(dataIndex - 1, j, Double.valueOf(lineEle[j]));
                    } catch (Exception e){ System.out.println(lineEle[j]);}
                }
                dataIndex += 1;
            }
        }
        return dataSet;
    }
    public static double[] loadlabels(String path, Config config) throws Exception{
        int numData = (int)Files.lines(Paths.get(path)).count();
        double [] labels = new double[numData];
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {

                labels[lineIndex] = Double.parseDouble(line);
                lineIndex += 1;
            }
        }
        return labels;
    }

    public static double[] loadweights(String path, Config config) throws Exception{
        int numFeatures = (int) Files.lines(Paths.get(path)).count();
        double [] weights = new double[numFeatures];
        try(BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int lineIndex = 0;
            while ((line = br.readLine()) != null){
                weights[lineIndex] = Double.parseDouble(line);
                        lineIndex +=1;
            }
        }
        return weights;
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


    private static WordVectorRegression loadModel(Config config, DataSet trainWord2Vec) throws Exception{
        int numWords = trainWord2Vec.getNumDataPoints();
        int completedIterations = 0;
        String output = config.getString("output.folder");
        String modelName = "models";
        File folder = Paths.get(output,modelName).toFile();
        File[] modeFiles = folder.listFiles((dir, name) -> name.startsWith("iter.") && (name.endsWith(".model")));
        File lastFile = null;
        int lastIter = -1;
        for (File file: modeFiles){
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
        } else {
            wordVectorRegression = (WordVectorRegression) Serialization.deserialize(lastFile);
        }

        System.out.println("wordVectorRegression loaded, with "+completedIterations+ " iterations completed");
        return wordVectorRegression;
    }

    private static LinearRegression getLinearReg (WordVectorRegression wordVectorRegression, DataSet word2Vec, Config config){
        int numWords = word2Vec.getNumDataPoints();
        Vector vector = new DenseVector(numWords+1);
        IntStream.range(0, numWords).parallel()
                .forEach(j->vector.set(j+1, wordVectorRegression.score(word2Vec.getRow(j),0)));

//        for (int j=0;j<numWords;j++){
//            vector.set(j+1, wordVectorRegression.score(word2Vec.getRow(j),0));
//        }
        LinearRegression linearRegression = new LinearRegression(numWords, vector);
        return linearRegression;
    }


}


