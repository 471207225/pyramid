package edu.neu.ccs.pyramid.application;

import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegOptimizer;
import edu.neu.ccs.pyramid.jinghan.WordVectorRegression;

import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;

import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 12/11/16.
 */
public class Jinghan_predict_reg {
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


        String output = config.getString("output.folder");
        String resFolder = config.getString("predict.folder");
        String modelName = "models";
        File path = Paths.get(output, modelName).toFile();
        path.mkdirs();

        WordVectorRegression wordVectorRegression = loadModel(config, trainWord2vec);
        System.out.println("getting word scores for training");


        double[] wordScores_train = new double[trainWord2vec.getNumDataPoints()];

        IntStream.range(0, trainWord2vec.getNumDataPoints()).parallel()
                .forEach(i->wordScores_train[i] = wordVectorRegression.score(trainWord2vec.getRow(i),0));
        String str_train = PrintUtil.toMutipleLines(wordScores_train);
        File reportFile_wordScores_train = new File(resFolder,"wordScores_train.txt");
        FileUtils.writeStringToFile(reportFile_wordScores_train, str_train);

        double[] wordScores_test = new double[testWord2vec.getNumDataPoints()];

        IntStream.range(0, testWord2vec.getNumDataPoints()).parallel()
                .forEach(i->wordScores_test[i] = wordVectorRegression.score(testWord2vec.getRow(i),0));
        String str_test = PrintUtil.toMutipleLines(wordScores_test);
        File reportFile_wordScores_test = new File(resFolder,"wordScores_test.txt");
        FileUtils.writeStringToFile(reportFile_wordScores_test, str_test);

        System.out.println("");
        File reportFile_train = new File(resFolder, "train_predictions.txt");
        report_train(wordVectorRegression, train_docFeatures, reportFile_train);
        System.out.println("predictions on the training set are written to"+reportFile_train.getAbsolutePath());

        File reportFile_test = new File(resFolder, "test_predictions.txt");
        LinearRegression linearRegression = getLinearReg(wordVectorRegression, testWord2vec, config);
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
        double bias = config.getDouble("bias");
        if (lastIter==-1){
            wordVectorRegression = new WordVectorRegression(numWords, bias);
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

