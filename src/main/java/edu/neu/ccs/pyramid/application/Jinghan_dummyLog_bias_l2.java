package edu.neu.ccs.pyramid.application;

import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.DataSet;
import edu.neu.ccs.pyramid.dataset.DataSetBuilder;
import edu.neu.ccs.pyramid.dataset.SparseDataSet;
import edu.neu.ccs.pyramid.eval.Accuracy;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.*;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.Ensemble;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * Created by jinghanyang on 12/20/16.
 */
public class Jinghan_dummyLog_bias_l2 {
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
        double bias = config.getDouble("bias");
        double lam = config.getDouble("lam");

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
        WordVectorClaOptimizer optimizer = new WordVectorClaOptimizer(wordVectorRegression, regTreeFactory, train_docFeatures, trainWord2vec, train_docLabels, train_weights, bias, lam);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();
        double threshold = config.getDouble("threshold");
        int progressInterval = config.getInt("train.showProgress.interval");
        int numIterations = config.getInt("train.numIterations");

        int saveModelInterval = config.getInt("saveModelInterval");

        for (int i = 1; i <= numIterations; i++) {
            System.out.println("begin iterate");
            int iteration_num = i + interationStart;
            System.out.println("iteration" + iteration_num);
//            System.out.println("check word scores before iterate");
//            for (int j=1; j<wordVectorRegression.wordScores.size(); j++){
//                if (j%1000==0){
//                    System.out.println(wordVectorRegression.wordScores.get(j));
//                }
//            }
            optimizer.iterate();
//            System.out.println("check word scores after iterate");
//            for (int j=1; j<wordVectorRegression.wordScores.size(); j++){
//                if (j%1000==0){
//                    System.out.println(wordVectorRegression.wordScores.get(j));
//                }
//            }


            int size = wordVectorRegression.getEnsemble(0).getRegressors().size();
//            // print tree
//            System.out.println(wordVectorRegression.getEnsemble(0).get(size-1));
            if (config.getBoolean("train.showTrainProgress") && (i % progressInterval == 0 || i == numIterations)) {
                Ensemble ensemble = wordVectorRegression.getEnsemble(0);

                Regressor regressor_original = optimizer.regressor_original;
                Regressor regressor_shrunk = ensemble.get(iteration_num-1);






                RegressionTree regressionTree = (RegressionTree) regressor_shrunk;
                List<List<Integer>> wordIDs = new ArrayList<List<Integer>>();
                List<List<Integer>> dummyIDs = new ArrayList<List<Integer>>();
                List<List<Integer>> dummyFeaturesInd = new ArrayList<List<Integer>>();
                List<Integer> leafIDs = new ArrayList<Integer>();
                for (int itr=0; itr<trainWord2vec.getNumDataPoints(); itr++){
                    TreeInfo treeInfo = new TreeInfo(regressionTree, trainWord2vec.getRow(itr));
                    wordIDs.add(treeInfo.wordVecIds);
                    dummyIDs.add(treeInfo.dummyIds);
                    dummyFeaturesInd.add(treeInfo.dummyFeatureIndexes);
                    leafIDs.add(treeInfo.leafId);
                }
                LeafNodeCount leafNodeCount = new LeafNodeCount(leafIDs);
                Map<Integer, List<Integer>> map_count = leafNodeCount.map;
                for (Integer name: map_count.keySet()){
                    String key = name.toString();
                    String value = map_count.get(name).toString();
                    int leafCount = map_count.get(name).size();

                }

                double[] train_prediction_score = wordVectorRegression.predict(train_docFeatures);
                double[] gradients = optimizer.gradients;
                System.out.println("gradient");
                for(int pr_i=0; pr_i<=10; pr_i++){
                    System.out.println(gradients[pr_i]);
                }
                ArrayIndexComparator comparator = new ArrayIndexComparator(gradients);
                Integer[] indexes = comparator.createIndexArray();
                Arrays.sort(indexes, comparator);

                FeatureList wordNames = train_docFeatures.getFeatureList();
                double[] predictGradient_shrunk = regressor_shrunk.predict(trainWord2vec);
                int checkNum = 100;
                double shrinkage_tuned = optimizer.shrinkageTuned;
                if(i%1==0){
                    try{
                        PrintWriter writer = new PrintWriter(config.getString("res.path")+"/"+i+ ".txt", "UTF-8");
                        writer.println("skrinkage "+shrinkage_tuned);
                        for (int ind_checkWord = 0; ind_checkWord<checkNum; ind_checkWord++) {
                            int wordInd_write = indexes[ind_checkWord];

                            // get dummy feature indexes for this word
                            List<Integer> dummyFeaturesInd_word = dummyFeaturesInd.get(wordInd_write);
                            List<String> dummyFeaturesName_word = new  ArrayList<String>();
                            for (int ind_featureName=0; ind_featureName<dummyFeaturesInd_word.size(); ind_featureName++){
                                int wordIdIndex = dummyFeaturesInd_word.get(ind_featureName);
//                                System.out.println(wordNames.get(wordIdIndex-300).toString());
                                dummyFeaturesName_word.add(wordNames.get(wordIdIndex-300).getName());
                            }
                            String dummyFeaturesName_word_string = dummyFeaturesName_word.toString();
                            String wvfeatures = wordIDs.get(wordInd_write).toString();
                            String dyfeatures = "noWordID";
                            if (!dummyIDs.get(wordInd_write).isEmpty()) {
                                dyfeatures = dummyIDs.get(wordInd_write).toString();
                            }
                            int lfID = leafIDs.get(wordInd_write);
                            int dataCount = map_count.get(lfID).size();
                            writer.println("");
                            writer.println(" word: \""+ wordNames.get(wordInd_write).getName()+ " \" \t weight "+ train_weights[wordInd_write] +" \t target gradient: " + gradients[wordInd_write]*shrinkage_tuned + " \t shrunked predict gradient: " +
                                    predictGradient_shrunk[wordInd_write]
                                    +" \t leaf ID: " + lfID+" \t wv path features: " + wvfeatures + " \t wordID path features: " + dyfeatures+ " \t wordID features "+dummyFeaturesName_word_string+ " \t data count: "+ dataCount );
                        }

                        writer.close();
                    } catch (IOException e) {
                        // do something
                    }
                }
//                System.out.println("training prediction score is ");
//                System.out.println(Arrays.toString(train_prediction_score));

//                System.out.println("training prediction for probability is ");
//                System.out.println(Arrays.toString(prediction_class_train.prob));


                /*
                print score and class
                 */
//                System.out.println("train_predict_score");
//                for(int pr_prediction=0; pr_prediction<10; pr_prediction++){
//                    System.out.println(train_prediction_score[pr_prediction]);
//                }
//
//                System.out.println("train_predict_class");
//                for(int pr_class=0; pr_class<10; pr_class++){
//                    System.out.println(train_prediciton_class[pr_class]);
//                }
//                int[] train_docLabels_int = dToint(train_docLabels);

//                System.out.println("tree is ");
//                System.out.println(regressor_shrunk);


                Vector wordScores = optimizer.wordVectorRegression.wordScores;
//                System.out.println("wordScores");
//                for(int ws_idx = 0; ws_idx<1000; ws_idx++){
//                    System.out.println(wordScores.get(ws_idx));
//                }

                LinearRegression trainLinearReg = getLinearReg(wordVectorRegression,trainWord2vec,bias);
                double[] train_prediciton_score = trainLinearReg.predict(train_docFeatures);
                Prediction_class prediction_class_train = new Prediction_class(train_prediciton_score, threshold);
                int[] train_prediciton_class = prediction_class_train.classPred;
                System.out.println("training accuracy = " + Accuracy.accuracy(train_docLabels, train_prediciton_class));
            }
            if (config.getBoolean("train.showTestProgress") && (i % progressInterval == 0 || i == numIterations)) {
                LinearRegression testLinearReg = getLinearReg(wordVectorRegression, testWord2vec, bias);
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
        DataSet dataSet = new SparseDataSet(numData, numFeatures,false);
        try (BufferedReader br = new BufferedReader(new FileReader(path))){
            String line = null;
            int dataIndex = 0;
            while ((line = br.readLine())!=null){
                String [] lineEle = line.split(" ");
                for (int j=0; j<numFeatures; j++){
                    dataSet.setFeatureValue(dataIndex,j,Double.valueOf(lineEle[j]));
                }
                dataIndex += 1;
            }
            return dataSet;
        }
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
        double bias = config.getDouble("bias");
        if (lastIter==-1){
            wordVectorRegression = new WordVectorRegression(numWords, bias);
        }else {
            wordVectorRegression = (WordVectorRegression) Serialization.deserialize(lastFile);
        }

        System.out.println("wordVectorRegression loaded with " + completedIterations+ " iterations completed");
        return new Pair<> (wordVectorRegression,completedIterations);
    }

    private static LinearRegression getLinearReg (WordVectorRegression wordVectorRegression, DataSet word2Vec, double bias){
        int numWords = word2Vec.getNumDataPoints();
        Vector vector = new DenseVector(numWords+1);
//
//        for (int j=0;j<numWords;j++){
//            vector.set(j+1, wordVectorRegression.score(word2Vec.getRow(j),0));
//        }
        vector.set(0,bias);
        IntStream.range(0, numWords).parallel()
                .forEach(j->vector.set(j+1, wordVectorRegression.score(word2Vec.getRow(j),0)));


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
