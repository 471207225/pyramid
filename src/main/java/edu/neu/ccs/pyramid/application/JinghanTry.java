package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegOptimizer;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegression;
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
import java.util.Arrays;

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
        String sparsity = config.getString("input.matrixType");
        DataSetType dataSetType = null;
        switch (sparsity){
            case "dense":
                dataSetType = DataSetType.REG_DENSE;
                break;
            case "sparse":
                dataSetType = DataSetType.REG_SPARSE;
                break;
            default:
                throw new IllegalArgumentException("input.matrixType should be dense or sparse");
        }
        DataSet train_docstoword = loadDocMatrix(config.getString("input.trainData"));
        DataSet test_docstoword = null;
        if (config.getBoolean("train.showTestProgress")){
            test_docstoword = loadDocMatrix(config.getString("input.testData"));
        }
        double [] train_labels = loadlabels(config.getString("input.trainLabel"));
        double [] test_labels = loadlabels(config.getString("input.testLabel"));


        double [] train_weights;
        if (config.getBoolean("useWeights")){
            train_weights = loadweights(config.getString("input.trainWeights"));
        } else {
            train_weights = new double[5000];
            Arrays.fill(train_weights, 1);
        }


        DataSet word2vec = loadword2vecMatrix(config.getString("input.word2vec"));

        DenseVector initialScores = new DenseVector(5000);
        WordVectorRegression wordVectorRegression = new WordVectorRegression(initialScores);
        // it is essential to set mindataperleave = 0
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves")).setMinDataPerLeaf(0);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        WordVectorRegOptimizer optimizer = new WordVectorRegOptimizer(wordVectorRegression, regTreeFactory, train_docstoword, word2vec, train_labels, train_weights);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();        
        int progressInterval = config.getInt("train.showProgress.interval");
        
        int numIterations = config.getInt("train.numIterations");
        for(int i=1; i<=numIterations;i++){
            System.out.println("iteration"+i);
//            optimizer.setShrinkage(config.getDouble("train.shrinkage")/numIterations);
            optimizer.iterate();
            if (config.getBoolean("train.showTrainProgress")&&(i%progressInterval==0 || i== numIterations)){

                double [] train_prediction = wordVectorRegression.predict(train_docstoword);
                System.out.println("training prediction is");
                System.out.println(Arrays.toString(train_prediction));

                System.out.println("training RMSE = " + RMSE.rmse(train_labels, train_prediction));
            }
            if (config.getBoolean("train.showTestProgress")&&(i%progressInterval==0 || i==numIterations)){
                double [] test_prediction = wordVectorRegression.predict(test_docstoword);
                System.out.println("test RMSE = " + RMSE.rmse(test_labels,test_prediction));
                System.out.println("*******************");
            }
        }
        System.out.println("training done!");
        String output = config.getString("output.folder");
        new File(output).mkdir();
        File serializedModel = new File(output, "model");
        Serialization.serialize(wordVectorRegression, serializedModel);
        System.out.println("model saved to"+serializedModel.getAbsolutePath());
        File reportFile = new File(output, "train_predictions.txt");
        report(wordVectorRegression, train_docstoword, reportFile);
        System.out.println("predictions on the training set are written to"+reportFile.getAbsolutePath());
    }


    public static DataSet loadDocMatrix(String path) throws Exception{
        DataSet dataSet = new SparseDataSet(25000, 5000, false);
        try (BufferedReader br = new BufferedReader(new FileReader(path));
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
                    dataSet.setFeatureValue(dataIndex-1, j, Double.valueOf(lineEle[j]));
                }
                dataIndex += 1;
            }
        }
        return dataSet;
    }
    public static double[] loadlabels(String Path) throws Exception{
        double [] labels = new double[25000];
        try(BufferedReader br = new BufferedReader(new FileReader(Path))){
            String line = null;
            int lineIndex = 0;
            while ((line = br.readLine()) != null) {

                    labels[lineIndex] = Double.parseDouble(line);
                    lineIndex += 1;
                }
            }
        return labels;
    }
    public static double[] loadweights(String Path) throws Exception{
        double [] weights = new double[5000];
        try(BufferedReader br = new BufferedReader(new FileReader(Path))){
            String line = null;
            int lineIndex = 0;
            while ((line = br.readLine()) != null){
                weights[lineIndex] = Double.parseDouble(line);
                        lineIndex +=1;
            }
        }
        return weights;
    }

    public static DataSet loadword2vecMatrix(String path) throws Exception{
        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(5000).numFeatures(300).build();
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
    public static void report(WordVectorRegression wordVectorRegression, DataSet dataSet, File reportFile) throws IOException {
        double [] prediction = wordVectorRegression.predict(dataSet);
        String str = PrintUtil.toMutipleLines(prediction);
        FileUtils.writeStringToFile(reportFile, str);
    }


}


