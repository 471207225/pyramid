package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.optimization.GradientDescent;
import edu.neu.ccs.pyramid.optimization.GradientValueOptimizer;
import edu.neu.ccs.pyramid.optimization.LBFGS;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoostOptimizer;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.linear_regression.SquareLoss;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;

import java.io.BufferedReader;
import java.io.FileReader;

/**
 * Created by chengli on 11/20/16.
 */
public class JinghanTry3 {
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

        LSBoost lsBoost = new LSBoost();

        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(2);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        LSBoostOptimizer optimizer = new LSBoostOptimizer(lsBoost, train_docstoword, regTreeFactory, train_labels);
        optimizer.setShrinkage(0.1);
        optimizer.initialize();

        for (int iter=0;iter<config.getInt("iterations");iter++){
            optimizer.iterate();
            System.out.println("iteration "+iter);
            System.out.println("training RMSE = "+ RMSE.rmse(train_labels, lsBoost.predict(train_docstoword)));
            System.out.println("test RMSE = "+ RMSE.rmse(test_labels, lsBoost.predict(test_docstoword)));
        }


    }


    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
        DataSet dataSet = new SparseDataSet(config.getInt("numData"), config.getInt("numfeature"), false);
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
    public static double[] loadlabels(String Path, Config config) throws Exception{
        double [] labels = new double[config.getInt("numData")];
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
}
