package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.classification.logistic_regression.LogisticRegression;
import edu.neu.ccs.pyramid.classification.logistic_regression.RidgeLogisticOptimizer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.optimization.Terminator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

/**
 * Created by jinghanyang on 12/9/16.
 */
public class JinghanTry_logistic {
    public static void main(String[] args) throws Exception{
        if (args.length!=1){
            throw new IllegalArgumentException("Please specify a properties file");
        }
        Config config = new Config(args[0]);
        System.out.println(config);
        train(config);
    }

    public static void train(Config config) throws Exception{
        DataSet train_docFeatures = loadDocMatrix(config.getString("input.train_docFeatures"), config);
        DataSet test_docFeatures = loadDocMatrix(config.getString("input.test_docFeatures"), config);

        int [] train_docLabels = loadlabels(config.getString("input.train_docLabels"), config);
        int [] test_docLabels = loadlabels(config.getString("input.test_docLabels"), config);

        double variance = config.getDouble("variance");
        LogisticRegression logisticRegression = new LogisticRegression(config.getInt("numClass"), train_docFeatures.getNumFeatures());
        RidgeLogisticOptimizer optimizer = new RidgeLogisticOptimizer(logisticRegression,train_docFeatures,train_docLabels,variance,true);
//        optimizer.getOptimizer().getTerminator().setMaxIteration(config.getInt("iterations")).setMode(Terminator.Mode.STANDARD);
//        optimizer.optimize();
//
//        System.out.println("train acc = " + Accuracy.accuracy(train_docLabels, logisticRegression.predict(train_docFeatures)));
//        System.out.println("test acc = " + Accuracy.accuracy(test_docLabels, logisticRegression.predict(test_docFeatures)));
        int numIterations = config.getInt("iterations");
        for(int i =1; i<= numIterations; i++){
            optimizer.optimize();
            System.out.println("iteration" + i);
            System.out.println("train acc = " + Accuracy.accuracy(train_docLabels, logisticRegression.predict(train_docFeatures)));
            System.out.println("test acc = " + Accuracy.accuracy(test_docLabels, logisticRegression.predict(test_docFeatures)));
        }
//        new File(config.getString("outPath_weights")).mkdir();
//        File weightReport = new File(config.getString("outPath_weights")+"/weights.txt");
//        System.out.println(optimizer.getOptimizer().getTerminator().getHistory());
        System.out.println("bias for 0 class is "+ logisticRegression.getWeights().getBiasForClass(0));
        System.out.println("bias for 1 class is "+ logisticRegression.getWeights().getBiasForClass(1));
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








}
