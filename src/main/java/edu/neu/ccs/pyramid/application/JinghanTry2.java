package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.DataSetType;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegOptimizer;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegression;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientValueOptimizer;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.linear_regression.SquareLoss;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
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

/**
 * Created by chengli on 11/20/16.
 */
public class JinghanTry2 {
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            throw new IllegalArgumentException("Please specify a properties file.");

        }
        Config config = new Config(args[0]);
        System.out.println(config);
        train(config);
    }

    public static void train(Config config) throws Exception{

        DataSet train_docstoword = loadDocMatrix(config.getString("input.trainData"), config);
        DataSet test_docstoword = loadDocMatrix(config.getString("input.testData"), config);

        double [] train_labels = loadlabels(config.getString("input.trainLabel"), config);
        double [] test_labels = loadlabels(config.getString("input.testLabel"), config);

        LinearRegression linearRegression = new LinearRegression(train_docstoword.getNumFeatures());

        SquareLoss squareLoss = new SquareLoss(linearRegression, train_docstoword, train_labels, config.getDouble("variance"));

        GradientValueOptimizer optimizer = null;
        switch (config.getString("optimizer")){
            case "GD":
                optimizer = new GradientDescent(squareLoss);
                break;
            case "LBFGS":
                optimizer = new LBFGS(squareLoss);
                break;
        }

        for (int iter=0;iter<config.getInt("iterations");iter++){
            optimizer.iterate();
            System.out.println("iteration "+iter);
            System.out.println("training RMSE = "+ RMSE.rmse(train_labels, linearRegression.predict(train_docstoword)));
            System.out.println("test RMSE = "+ RMSE.rmse(test_labels, linearRegression.predict(test_docstoword)));
        }
        new File(config.getString("outPath_weights")).mkdir();
        File weightReport = new File(config.getString("outPath_weights")+"/weights.txt");
        StringBuilder sb = new StringBuilder();
        for (int j=0;j<train_docstoword.getNumFeatures();j++){
            sb.append(linearRegression.getWeights().getWeightsWithoutBias().get(j));
            sb.append("\n");
        }


        FileUtils.writeStringToFile(weightReport, sb.toString());
    }

    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
        int numData = (int)Files.lines(Paths.get(path)).count()-1;
        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;
        DataSet dataSet = new SparseDataSet(numData, numFeatures, false);
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

}
