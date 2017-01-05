package edu.neu.ccs.pyramid.application;

import com.fasterxml.jackson.databind.util.TypeKey;
import edu.neu.ccs.pyramid.configuration.Config;
import edu.neu.ccs.pyramid.dataset.*;
import edu.neu.ccs.pyramid.eval.RMSE;
import edu.neu.ccs.pyramid.feature.Feature;
import edu.neu.ccs.pyramid.feature.FeatureList;
import edu.neu.ccs.pyramid.jinghan.*;
import edu.neu.ccs.pyramid.multilabel_classification.cbm.CBM;
import edu.neu.ccs.pyramid.optimization.gradient_boosting.Ensemble;
import edu.neu.ccs.pyramid.regression.Regressor;
import edu.neu.ccs.pyramid.regression.least_squares_boost.LSBoost;
import edu.neu.ccs.pyramid.regression.linear_regression.LinearRegression;
import edu.neu.ccs.pyramid.regression.regression_tree.Node;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeConfig;
import edu.neu.ccs.pyramid.regression.regression_tree.RegTreeFactory;
import edu.neu.ccs.pyramid.regression.regression_tree.RegressionTree;
import edu.neu.ccs.pyramid.util.Pair;
import edu.neu.ccs.pyramid.util.PrintUtil;
import edu.neu.ccs.pyramid.util.Serialization;
import edu.stanford.nlp.patterns.Data;
import org.apache.commons.io.FileUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.elasticsearch.river.RiverIndexName;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.*;

/**
 * Created by jinghanyang on 1/3/16.
 */
public class Jinghan_dummyLR_bias_l2 {
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


        double bias = config.getDouble("bias");
        double lam = config.getDouble("lambda.l2");

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

        WordVectorRegression wordVectorRegression = loadModel(config, trainWord2vec).getFirst();
        // it is essential to set mindataperleave = 0
        int interationStart = loadModel(config, trainWord2vec).getSecond();
        RegTreeConfig regTreeConfig = new RegTreeConfig().setMaxNumLeaves(config.getInt("train.numLeaves")).setMinDataPerLeaf(0);
        RegTreeFactory regTreeFactory = new RegTreeFactory(regTreeConfig);
        WordVectorRegOptimizer optimizer = new WordVectorRegOptimizer(wordVectorRegression, regTreeFactory, train_docFeatures, trainWord2vec, train_docLabels, train_weights, bias, lam);
        optimizer.setShrinkage(config.getDouble("train.shrinkage"));
        optimizer.initialize();
        int progressInterval = config.getInt("train.showProgress.interval");

        int numIterations = config.getInt("train.numIterations");

        int saveModelInterval = config.getInt("saveModelInterval");

        for(int i=1; i<=numIterations;i++){
            int iteration_num = i + interationStart;
            System.out.println("iteration" + iteration_num);
//            optimizer.setShrinkage(config.getDouble("train.shrinkage")/numIterations);
            optimizer.iterate();
            if (config.getBoolean("train.showTrainProgress")&&(i%progressInterval==0 || i== numIterations)){

                Ensemble ensemble = wordVectorRegression.getEnsemble(0);

                /**
                 * regressor: tree (latest one)
                 */
                // We just want to know the result of the latest regressor
                //before shrinking is
//                System.out.println("the regressor before shrinking is");
                Regressor regressor_original = optimizer.regressor_original;

//                System.out.println(regressor_original);

                // shrunk
//                System.out.println("check start from here");
//                System.out.println("the latest shrunk regressor is ");
                Regressor regressor_shrunk = ensemble.get(iteration_num-1);
                RegressionTree regressionTree = (RegressionTree) regressor_shrunk;
                System.out.println("tree is ");
                System.out.println(regressionTree);

                List<List<Integer>> wordIDs = new ArrayList<List<Integer>>();
                List<List<Integer>> dummyIDs = new ArrayList<List<Integer>>();
                List<List<Integer>> dummyFeaturesInd = new ArrayList<List<Integer>>();
                List<Integer> leafIDs = new ArrayList<Integer>();
                for(int itr=0; itr<trainWord2vec.getNumDataPoints(); itr++){
                    TreeInfo treeInfo = new TreeInfo(regressionTree, trainWord2vec.getRow(itr));
                    wordIDs.add(treeInfo.wordVecIds);
//                    System.out.println("wordID list is ");
//                    for (int itr_wordID = 0; itr_wordID<wordIDs.get(itr).size(); itr_wordID++){
////                        System.out.println(wordIDs.get(itr).get(itr_wordID));
//                    }
//                    System.out.println("dummy"+treeInfo.dummyIds);
                    dummyIDs.add(treeInfo.dummyIds);
//                    System.out.println("dummyID list is ");
//                    for (int itr_dummyID = 0; itr_dummyID<wordIDs.get(itr).size(); itr_dummyID++){
//                        System.out.println(dummyIDs.get(itr).get(itr_dummyID));
//                    }
                    dummyFeaturesInd.add(treeInfo.dummyFeatureIndexes);
                    leafIDs.add(treeInfo.leafId);
//                    System.out.println("this leaf ID is" + leafIDs.get(itr));
                }
                LeafNodeCount leafNodeCount = new LeafNodeCount(leafIDs);
                Map<Integer, List<Integer>> map_count = leafNodeCount.map;
//                System.out.println("leaf Id");
                for (Integer name: map_count.keySet()) {
                    String key = name.toString();
                    String value = map_count.get(name).toString();
                    int leafCount = map_count.get(name).size();
//                    System.out.println(key+" "+ value);
//                    System.out.println(key+" "+ leafCount);
                }

                /**
                 * properties of tree
                 */
//                Node root = regressionTree.getRoot();
//                System.out.println(root);
//                // properties of the root node
//                int id_root = root.getId();
//                double value_root = root.getValue();
//                int featureIndex_root = root.getFeatureIndex();
//                double threshold_root = root.getThreshold();
//                Node leftChild_root = root.getLeftChild();
//                Node rightChird_root = root.getRightChild();
//                boolean leaf_root = root.leaf;
//                boolean splitable_root = root.splitable;
//
//                System.out.println("id"+ id_root);
//                System.out.println("value"+value_root);
//                System.out.println("featureIndex"+featureIndex_root);
//                System.out.println("threshold"+threshold_root);
//                System.out.println("left child"+leftChild_root);
//                System.out.println("right child"+rightChird_root);
//                System.out.println("is leaf ?" + leaf_root);
//                System.out.println("is splitable?"+ splitable_root);
//
//                System.out.println("check stop here");



                double[] train_prediction = wordVectorRegression.predict(train_docFeatures);
//                System.out.println("training prediction is");
//                System.out.println(Arrays.toString(train_prediction));


                /**
                 * sorted gradients by absolute value
                 */
                double[] gradients = optimizer.gradients;
                ArrayIndexComparator comparator = new ArrayIndexComparator(gradients);
                Integer[] indexes = comparator.createIndexArray();
                Arrays.sort(indexes, comparator);
//                System.out.println("sorted index is "+ Arrays.toString(indexes));
//                System.out.println("sorted element is");
//                for(int ind: indexes){
//                    System.out.println(gradients[ind]);
//                }

                FeatureList wordNames = train_docFeatures.getFeatureList();
                double[] predictGradient_original = regressor_original.predict(trainWord2vec);
                double[] predictGradient_shrunk = regressor_shrunk.predict(trainWord2vec);
                /**
                 * print word, gradient, predicted gradient (the first 100)
                 */
                int checkNum = 100;
                double shrinkage_tuned = optimizer.shrinkageTuned;
                if(i%2==0){
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
                                System.out.println(wordNames.get(wordIdIndex-300).toString());
                                dummyFeaturesName_word.add(wordNames.get(wordIdIndex-300).getName());
                            }
                            String dummyFeaturesName_word_string = dummyFeaturesName_word.toString();
                            String wvfeatures = wordIDs.get(wordInd_write).toString();
                            String dyfeatures = "noDummy";
                            if (!dummyIDs.get(wordInd_write).isEmpty()) {
                                dyfeatures = dummyIDs.get(wordInd_write).toString();
                            }
                            int lfID = leafIDs.get(wordInd_write);
                            int dataCount = map_count.get(lfID).size();
                            writer.println("");
                            writer.println(" word: \""+ wordNames.get(wordInd_write).getName()+ " \" \t weight "+ train_weights[wordInd_write] +" \t word score"+ optimizer.wordVectorRegression.score(testWord2vec.getRow(wordInd_write),0)+" \t target gradient: " + gradients[wordInd_write]*shrinkage_tuned + " \t shrunked predict gradient: " +
                                    predictGradient_shrunk[wordInd_write]
                                    +" \t leaf ID: " + lfID+" \t wv path features: " + wvfeatures + " \t wordID path features: " + dyfeatures+ " \t wordID features "+dummyFeaturesName_word_string+ " \t data count: "+ dataCount );
                        }

                        writer.close();
                    } catch (IOException e) {
                        // do something
                    }
                }
//                System.out.println("shrinkage "+ shrinkage_tuned);
                for(int j=0; j<checkNum; j++){
                    int wordInd = indexes[j];
//                    System.out.println(" word "+ wordNames.get(wordInd).getName()+ " target gradient " + gradients[wordInd]+ " predicted gradient "+ predictGradient[wordInd]);
                }
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
        LinearRegression linearRegression = getLinearReg(wordVectorRegression, testWord2vec, config);
        report_test(linearRegression, test_docFeatures, reportFile_test);
        System.out.println("predictions on the training set are written to"+reportFile_test.getAbsolutePath());
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




    /**
     * load dense word2vec
     * @param path
     * @return
     * @throws Exception
     */
//    public static DataSet loadword2vecMatrix(String path) throws Exception{
//
//        int numData = (int) Files.lines(Paths.get(path)).count();
//        int numfeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;
//
//        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(numData).numFeatures(numfeatures).build();
//        try(BufferedReader br = new BufferedReader(new FileReader(path))){
//            String line = null;
//            int dataIndex = 0;
//            while ((line = br.readLine()) !=null){
//                String [] lineEle = line.split(" ");
//                for (int j=0; j<denseDataSet.getNumFeatures();j++){
//                    denseDataSet.setFeatureValue(dataIndex,j,Double.parseDouble(lineEle[j]));
//                }
//                dataIndex += 1;
//            }
//        }
//
//        return denseDataSet;
//    }
    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
        int numData = (int) Files.lines(Paths.get(path)).count()-1;
        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;

        DataSet denseDataSet = DataSetBuilder.getBuilder().numDataPoints(numData).numFeatures(numFeatures).build();
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
                    denseDataSet.setFeatureList(featureList);
                    continue;
                }
                String [] lineEle = line.split(" ");

                for (int j=0;j<denseDataSet.getNumFeatures();j++){
                    try {
                        denseDataSet.setFeatureValue(dataIndex - 1, j, Double.valueOf(lineEle[j]));
                    } catch (Exception e){ System.out.println(lineEle[j]);}
                }
                dataIndex += 1;
            }
        }
        return denseDataSet;
    }

    /*
    load sparse word matirx
     */
//    public static DataSet loadDocMatrix(String path, Config config) throws Exception{
//        int numData = (int) Files.lines(Paths.get(path)).count()-1;
//        int numFeatures = Files.lines(Paths.get(path)).findFirst().get().split(" ").length;
//        DataSet dataSet = new SparseDataSet(numData, numFeatures, false);
//        try (BufferedReader br = new BufferedReader(new FileReader(path))
//        ) {
//            String line = null;
//            int dataIndex = 0;
//            while ((line = br.readLine()) != null) {
//
//                if(dataIndex == 0){
//                    dataIndex += 1;
//                    String[] featureNames = line.split(" ");
//                    FeatureList featureList = new FeatureList();
//                    for (String featureName: featureNames){
//                        Feature feature = new Feature();
//                        feature.setName(featureName);
//                        featureList.add(feature);
//                    }
//                    dataSet.setFeatureList(featureList);
//                    continue;
//                }
//                String [] lineEle = line.split(" ");
//
//                for (int j=0;j<dataSet.getNumFeatures();j++){
//                    try {
//                        dataSet.setFeatureValue(dataIndex - 1, j, Double.valueOf(lineEle[j]));
//                    } catch (Exception e){ System.out.println(lineEle[j]);}
//                }
//                dataIndex += 1;
//            }
//        }
//        return dataSet;
//    }
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
        double lam = config.getDouble("lambda.l2");
        double bias = config.getDouble("bias");
        if (lastIter==-1){
            wordVectorRegression = new WordVectorRegression(numWords, bias);
        }else {
            wordVectorRegression = (WordVectorRegression) Serialization.deserialize(lastFile);
        }

        System.out.println("wordVectorRegression loaded with " + completedIterations+ " iterations completed");
        return new Pair<> (wordVectorRegression,completedIterations);
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
